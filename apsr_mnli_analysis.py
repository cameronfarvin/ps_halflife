import re
import random
import logging
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
from apsr_utils import APSRUtils


class MNLIAnalysis:
    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        module_seed_value: int = 42,
        load_existing_cache: bool = True,
        unified_cache_path: str = "./cache/filtered_unified_cambcore_crossref_cache.pkl",
        mnli_cache_path: str = "./cache/mnli_analysis_cache.pkl",
        output_csv_path: str = "./output_data/mnli_analysis_output.csv",
        log_path: str = "./logs/mnli_analysis_log.txt",
    ):
        self.model_name = model_name
        self.module_seed_value = module_seed_value
        self.unified_cache_path = unified_cache_path
        self.mnli_cache_path = mnli_cache_path
        self.output_csv_path = output_csv_path
        self.utils = APSRUtils(log_path=log_path)
        self.load_existing_cache = load_existing_cache

        self.device = None
        self.tokenizer = None
        self.model = None
        self.seeds_planted = False

        self.unified_cache_df = None
        self.mnli_analysis_cache = {}  # Dict keyed by (apsr_title, citing_doi)

        if self.load_existing_cache:
            self.LoadUnifiedCache()
            self.LoadMNLICache()

        self.InitModelDevice()

        self.mnli_max_observed: int = 0

    def LoadUnifiedCache(self) -> None:
        try:
            self.utils.Log(
                "info", *self.utils.GetFuncLine(), "Loading unified cache..."
            )
            self.unified_cache_df = pd.read_pickle(self.unified_cache_path)
            if self.unified_cache_df.empty:
                raise ValueError("Unified cache DataFrame is empty.")
            self.utils.Log(
                "info",
                *self.utils.GetFuncLine(),
                f"Loaded {len(self.unified_cache_df):,} rows from unified cache.",
            )
        except Exception as e:
            self.utils.Log(
                "error",
                *self.utils.GetFuncLine(),
                f"Failed to load unified cache from {self.unified_cache_path}: {e}",
            )
            raise e

    def LoadMNLICache(self) -> None:
        existing_cache = self.utils.LoadExistingCache(self.mnli_cache_path)
        if existing_cache is None:
            self.mnli_analysis_cache = {}
            self.utils.Log(
                "info", *self.utils.GetFuncLine(), "No existing MNLI cache found."
            )
        else:
            self.mnli_analysis_cache = existing_cache
            self.utils.Log(
                "info",
                *self.utils.GetFuncLine(),
                f"Loaded MNLI cache with {len(self.mnli_analysis_cache)} records.",
            )

    def InitModelDevice(self) -> None:
        if self.seeds_planted and self.model and self.tokenizer and self.device:
            return

        try:
            random.seed(self.module_seed_value)
            np.random.seed(self.module_seed_value)
            torch.manual_seed(self.module_seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.module_seed_value)

            self.seeds_planted = True

            # Device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                torch.cuda.empty_cache()
                self.utils.Log("info", *self.utils.GetFuncLine(), "Using CUDA.")
            else:
                self.device = torch.device("cpu")
                self.utils.Log("info", *self.utils.GetFuncLine(), "Using CPU.")

            # Model & Tokenizer
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            self.model.eval()  # Turn off model training; inference only.

        except Exception as e:
            self.utils.Log(
                "error", *self.utils.GetFuncLine(), f"Model initialization failed: {e}"
            )
            raise e

    import re

    def EstimateTokenQuant(self, text: str) -> int:
        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        estimate = 0
        for word in words:
            if re.match(r"^\W+$", word):  # punctuation-only (e.g., . , ; --)
                estimate += len(word)  # Each symbol likely gets its own token.
            elif re.match(r"^\d+$", word):  # Pure numbers (e.g., 2023)
                estimate += 1
            elif re.match(r"^\d+[\.,]\d+$", word):  # Decimals like 3.14 or 1,000.2
                estimate += 2
            elif len(word) <= 3:
                estimate += 1
            elif len(word) <= 6:
                estimate += 1.3
            elif len(word) <= 10:
                estimate += 1.7
            else:
                estimate += 2.3
        return int(round(estimate))

    def SubmitMNLIWork(self, premise: str, hypotheses: List[str]) -> F.Tensor:
        """
        premise  = APSR abstract
        hypotheses = list of citing abstracts
        Returns a Tensor of shape [len(hypotheses), 3].
        """
        try:
            if not hypotheses or not premise:
                raise ValueError("Abstracts missing for MNLI inference.")

            MAX_TOKENS = 512  # 512 is model limit; truncate beyond.
            hypotheses_max: str = max(hypotheses)
            iteration_max_str: str = max(premise, hypotheses_max)
            iteration_max_len: int = len(iteration_max_str)
            if iteration_max_len > self.mnli_max_observed:
                self.mnli_max_observed = iteration_max_len
                token_est = self.EstimateTokenQuant(iteration_max_str)
                if token_est > MAX_TOKENS:
                    self.utils.Log(
                        "warning",
                        *self.utils.GetFuncLine(),
                        f"Estimated token count exceeded by {token_est - MAX_TOKENS} for MNLI inference.",
                    )

            tokens = self.tokenizer(
                [premise] * len(hypotheses),
                hypotheses,
                max_length=MAX_TOKENS,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                logits = self.model(**tokens).logits
                probs = F.softmax(logits, dim=1)  # shape: [batch_size, 3]

            if probs.shape[0] != len(hypotheses):
                raise ValueError("Batch size mismatch between input and output.")

            return probs
        except Exception as e:
            self.utils.Log(
                "error",
                *self.utils.GetFuncLine(),
                f"MNLI inference failed: {e}",
            )
            raise e

    def Analyze(self, batch_size: int = 16) -> None:
        try:
            # Check if we have a unified cache; testing runs without it.
            if not self.load_existing_cache:
                self.utils.Log(
                    "error",
                    *self.utils.GetFuncLine(),
                    "MNLI analysis requires a unified cache.",
                )
                return

            total_rows = len(self.unified_cache_df)
            pbar_update_interval = max(1, total_rows // 100)
            pbar_prefix = "MNLI Abstract Analysis"
            if total_rows == 0:
                self.utils.Log(
                    "error", *self.utils.GetFuncLine(), "No data to analyze."
                )
                return

            self.utils.Log(
                "info", *self.utils.GetFuncLine(), "Starting MNLI analysis..."
            )
            completed = 0
            rows_for_batch = []
            premise_text = None
            for idx, row in self.unified_cache_df.iterrows():
                key = (row["apsr_title"], row["citing_doi"])

                # Check if we already computed probabilities for this pair.
                if key in self.mnli_analysis_cache:
                    completed += 1
                    if completed % pbar_update_interval == 0 or completed == total_rows:
                        self.utils.ProgressBar(
                            completed, total_rows, prefix=pbar_prefix
                        )
                    continue

                if not premise_text:
                    premise_text = row["apsr_title_abstract"]
                    rows_for_batch = []

                if row["apsr_title_abstract"] != premise_text:
                    self.ProcessBatch(premise_text, rows_for_batch)
                    premise_text = row["apsr_title_abstract"]
                    rows_for_batch = []

                rows_for_batch.append((key, row["citing_abstract"]))

                # If we reached batch_size, process the batch
                if len(rows_for_batch) >= batch_size:
                    self.ProcessBatch(premise_text, rows_for_batch)
                    rows_for_batch = []
                    premise_text = None

                completed += 1
                if completed % pbar_update_interval == 0 or completed == total_rows:
                    self.utils.ProgressBar(completed, total_rows, prefix=pbar_prefix)
                    self.utils.PickleOut(self.mnli_analysis_cache, self.mnli_cache_path)

            # Process any leftover rows in the final batch
            if rows_for_batch and premise_text:
                self.ProcessBatch(premise_text, rows_for_batch)
                self.utils.ProgressBar(
                    completed + len(rows_for_batch), total_rows, prefix=pbar_prefix
                )
                self.utils.PickleOut(self.mnli_analysis_cache, self.mnli_cache_path)

            self.utils.Log("info", *self.utils.GetFuncLine(), "MNLI analysis complete.")
            self.OutputMergedResults()

        except Exception as e:
            self.utils.Log(
                "error", *self.utils.GetFuncLine(), f"MNLI analysis failed: {e}"
            )
            raise e

    def ProcessBatch(
        self, premise_abstract: str, batch_rows: List[Tuple[Tuple[str, str], str]]
    ) -> None:
        """
        premise_abstract: APSR abstract string for the entire batch
        batch_rows: list of ((apsr_title, citing_doi), citing_abstract)
        """
        try:
            if not batch_rows:
                return
            citing_texts = [x[1] for x in batch_rows]
            probabilities = self.SubmitMNLIWork(premise_abstract, citing_texts)

            # For each item, store in self.mnli_analysis_cache
            for i, (key, _) in enumerate(batch_rows):
                self.mnli_analysis_cache[key] = {
                    "contradiction_prob": round(probabilities[i, 0].item(), 4),
                    "neutral_prob": round(probabilities[i, 1].item(), 4),
                    "entailment_prob": round(probabilities[i, 2].item(), 4),
                }
        except Exception as e:
            self.utils.Log(
                "error",
                *self.utils.GetFuncLine(),
                f"Batch processing failed: {e}",
            )

    def OutputMergedResults(self) -> None:
        """
        Merges the newly computed probabilities with the columns from the unified cache
        and writes them to a CSV with the first 3 columns being the MNLI probabilities.
        """
        try:
            records = []
            for idx, row in self.unified_cache_df.iterrows():
                key = (row["apsr_title"], row["citing_doi"])
                if key not in self.mnli_analysis_cache:
                    # If we missed a pair, fill with empty or 0.0000
                    cp, np_, ep = 0.0, 0.0, 0.0
                    self.utils.Log(
                        "error",
                        *self.utils.GetFuncLine(),
                        f"Key {key} not found in cache.",
                    )
                else:
                    cp = self.mnli_analysis_cache[key]["contradiction_prob"]
                    np_ = self.mnli_analysis_cache[key]["neutral_prob"]
                    ep = self.mnli_analysis_cache[key]["entailment_prob"]

                rec = {
                    "contradiction_prob": cp,
                    "neutral_prob": np_,
                    "entailment_prob": ep,
                    "apsr_title": row["apsr_title"],
                    "apsr_title_doi": row["apsr_title_doi"],
                    "apsr_title_pub_year": row["apsr_title_pub_year"],
                    "apsr_title_pub_month": row["apsr_title_pub_month"],
                    "apsr_title_total_cited_by_count": row[
                        "apsr_title_total_cited_by_count"
                    ],
                    "apsr_title_filtered_cited_by_count": row[
                        "apsr_title_filtered_cited_by_count"
                    ],
                    "apsr_title_abstract": row["apsr_title_abstract"],
                    "citing_title": row["citing_title"],
                    "citing_doi": row["citing_doi"],
                    "citing_pub_year": row["citing_pub_year"],
                    "citing_pub_month": row["citing_pub_month"],
                    "citing_abstract": row["citing_abstract"],
                }
                records.append(rec)

            columns_in_order = [
                "contradiction_prob",
                "neutral_prob",
                "entailment_prob",
                "apsr_title",
                "apsr_title_doi",
                "apsr_title_pub_year",
                "apsr_title_pub_month",
                "apsr_title_total_cited_by_count",
                "apsr_title_filtered_cited_by_count",
                "apsr_title_abstract",
                "citing_title",
                "citing_doi",
                "citing_pub_year",
                "citing_pub_month",
                "citing_abstract",
            ]
            df_out = pd.DataFrame(records, columns=columns_in_order)
            self.utils.OutputCSV(df_out, self.output_csv_path)
        except Exception as e:
            self.utils.Log(
                "error",
                *self.utils.GetFuncLine(),
                f"Merged output failed: {e}",
            )
            raise e


if __name__ == "__main__":
    analysis = MNLIAnalysis()

    try:
        analysis.Analyze(batch_size=4)
    except Exception as e:
        analysis.utils.Log(
            "error", *analysis.utils.GetFuncLine(), f"MNLI analysis failed: {e}"
        )
        raise e
    finally:
        analysis.utils.WriteLog()
