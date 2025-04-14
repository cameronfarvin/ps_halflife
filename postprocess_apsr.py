import threading
import concurrent.futures
import pandas as pd
import requests
import time
import csv
import os
import re

from typing import Dict, List, Any
from apsr_utils import APSRUtils
from bs4 import BeautifulSoup
from io import StringIO


class PostProcessAPSR:
    def __init__(self, attempt_cache_load: bool = False):
        # Semaphores for post-processing.
        self.cc_apsr_citations_processed = False
        self.cc_apsr_dois_processed = False
        self.crossref_processed = False

        # Data caches.
        self.cc_apsr_citations_cache: Dict[str, List[Any]] = {}
        self.cc_apsr_dois_cache: Dict = {}
        self.crossref_cache: Dict[str, Dict[str, Any]] = {}

        # (Input Path): Cambridge Core Web Interface Results + Credentials
        self.cambcore_web_ifc_csv_path: str = "./input_data/apsr_results.csv"
        self.credentials_path: str = "./credentials.txt"

        # (Output Paths): CSV Output Paths
        self.cc_apsr_citations_output_csv_path: str = (
            "./output_data/cc_apsr_citations.csv"
        )
        self.crossref_output_csv_path: str = "./output_data/crossref.csv"
        self.unified_cambcore_crossref_output_csv_path: str = (
            "./output_data/unified_cambcore_crossref.csv"
        )
        self.cc_apsr_dois_output_csv_path: str = "./output_data/cc_apsr_dois.csv"

        # (Input/Output Paths): Cache Paths (.pkl)
        self.cc_apsr_citations_cache_output_path: str = (
            "./cache/cc_apsr_citations_cache.pkl"
        )
        self.cc_apsr_dois_cache_output_path: str = "./cache/cc_apsr_dois_cache.pkl"
        self.crossref_cache_output_path: str = "./cache/crossref_cache.pkl"
        self.filtered_unified_cambcore_crossref_cache_output_path: str = (
            "./cache/filtered_unified_cambcore_crossref_cache.pkl"
        )

        # (Output Path): Log File
        self.log_path: str = "./logs/postprocessing_log.txt"

        # Initialize APSRUtils
        self.utils = APSRUtils(log_path=self.log_path)

        # Cache flags.
        self.loaded_cc_apsr_citations_cache = False
        self.loaded_cc_apsr_dois_cache = False
        self.loaded_crossref_cache = False

        # DataFrame for input CSV from Cambridge Core.
        self.camb_core_df: pd.DataFrame = None
        self.LoadCamCoreCSV()

        # Compute DOI prefix length.
        self.doi_prefix_length: int = len("https://doi.org/")

        # Load credentials from the credentials file, if present.
        self.credentials = self.LoadCredentials()

        # Create a persistent session for HTTP requests.
        self.session = requests.Session()
        self.client_errors = [400, 401, 403, 404]

        # Locks for thread-safe updates.
        self.crossref_lock = threading.Lock()
        self.cc_apsr_citation_lock = threading.Lock()
        self.cc_apsr_dois_lock = threading.Lock()

        # Load cached data if the flag is set.
        if attempt_cache_load:
            self.LoadCachedData()

        # Initialize the log with a message of type "info" indicating the start date/time of the process.
        self.utils.Log(
            "info",
            "",
            None,
            f"Post-processing started at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        )

    def CleanUp(self) -> None:
        if self.session:
            self.session.close()

    def LoadCachedData(self) -> None:
        # Load cached cc_apsr citation data.
        c_data = self.utils.LoadExistingCache(self.cc_apsr_citations_cache_output_path)
        if c_data != None:
            self.cc_apsr_citations_cache = c_data
            self.loaded_cc_apsr_citations_cache = True

        # Load cached cc_apsr doi data.
        d_data = self.utils.LoadExistingCache(self.cc_apsr_dois_cache_output_path)
        if d_data != None:
            self.cc_apsr_dois_cache = d_data
            self.loaded_cc_apsr_dois_cache = True

        # Load cached crossref data.
        a_data = self.utils.LoadExistingCache(self.crossref_cache_output_path)
        if a_data != None:
            self.crossref_cache = a_data
            self.loaded_crossref_cache = True

    def LoadCredentials(self) -> Dict[str, str]:
        if os.path.exists(self.credentials_path):
            credentials = {}
            try:
                os.makedirs(os.path.dirname(self.credentials_path), exist_ok=True)
                with open(self.credentials_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        key, value = line.split(":", 1)
                        credentials[key.strip()] = value.strip()
            except Exception as e:

                self.utils.Log(
                    "error",
                    *self.utils.GetFuncLine(),
                    f"Failed to load credentials: {e}",
                )
                return None
            return credentials

    def WebFetch(
        self, url: str, base_timeout: int = 15, max_attempts: int = 5
    ) -> requests.Response:
        wait = base_timeout
        for attempt in range(max_attempts):
            try:
                response = self.session.get(url, timeout=wait)
                if response.status_code == 429:
                    # Use safe conversion for the Retry-After header (capped at 180 sec).
                    retry_after_header = response.headers.get("Retry-After")
                    if retry_after_header is not None:
                        try:
                            retry_after = int(retry_after_header)

                        except ValueError:
                            retry_after = wait
                    else:
                        retry_after = wait

                    # Both log and immediately print 429 response.
                    msg = f"Server responded with 429 ({attempt + 1} times); This session is being throttled. Retrying in {wait} seconds."

                    self.utils.Log("warning", *self.utils.GetFuncLine(), msg)
                    print(msg)

                    time.sleep(min(retry_after, 180))
                    continue
                    # Check for other HTTP errors.
                return response

            except requests.RequestException as e:

                self.utils.Log(
                    "error",
                    *self.utils.GetFuncLine(),
                    f"Webfetch failed for {url}; Exception: {e}",
                )

            # Back off and retry.
            wait = min(wait * 2, 180)
            time.sleep(wait)

        # If all attempts fail, log the error and return None.

        self.utils.Log(
            "error",
            *self.utils.GetFuncLine(),
            f"All {max_attempts} attempts failed for {url}.",
        )
        return None

    def LoadCamCoreCSV(self) -> None:
        try:
            print("Loading input CSV...")
            self.camb_core_df = pd.read_csv(self.cambcore_web_ifc_csv_path)

        except Exception as e:

            self.utils.Log(
                "error",
                *self.utils.GetFuncLine(),
                f"Failed to load Cambridge Core CSV: {e}",
            )
            self.camb_core_df = None
            raise e

    def PostProcessCambridgeCoreCitations(self) -> None:

        def QueryCambridgeCoreCitations(citation_row):
            paper_title = citation_row["title"].strip()
            cambridge_core_citations_url = citation_row[
                "all_citing_papers_link"
            ].strip()

            returned_citations = []
            error = None
            res = self.WebFetch(cambridge_core_citations_url)
            if res is None:
                return paper_title, returned_citations, error
            elif res.status_code in self.client_errors:
                error = f"[ error ]: Server Response: {res.status_code}; Cambridge Core Citations URL: {cambridge_core_citations_url}"
                return paper_title, returned_citations, error

            reader = csv.reader(StringIO(res.text))

            # Ensure the returned CSV has exactly two columns.
            for entry in reader:
                if len(entry) != 2:

                    self.utils.Log(
                        "error",
                        *self.utils.GetFuncLine(),
                        f"Invalid CSV entry: {entry} for {paper_title}; skipping.",
                    )
                    continue

                # Extract the DOI from the second column.
                doi = entry[1].strip()
                if doi == "DOI":
                    continue
                elif len(doi) > self.doi_prefix_length:
                    returned_citations.append(doi.strip())
                else:

                    self.utils.Log(
                        "error",
                        *self.utils.GetFuncLine(),
                        f"Invalid DOI: '{doi}' for {paper_title}; skipping.",
                    )
                    if error is None:
                        error = "Invalid DOIs encountered: "
                    else:
                        error += f"; {doi}"
                    returned_citations.append("")
            return paper_title, returned_citations, error

        try:
            if self.camb_core_df is None:

                self.utils.Log(
                    "error",
                    *self.utils.GetFuncLine(),
                    "No input data loaded. Cannot process citations.",
                )
                return

            if self.loaded_cc_apsr_citations_cache:
                pre_update_len = len(self.cc_apsr_citations_cache)
                with self.cc_apsr_citation_lock:
                    self.cc_apsr_citations_cache.update(
                        {
                            title: []
                            for title in self.camb_core_df["title"]
                            if title not in self.cc_apsr_citations_cache
                        }
                    )
                if len(self.cc_apsr_citations_cache) == pre_update_len:

                    self.utils.Log(
                        "info",
                        *self.utils.GetFuncLine(),
                        "No new citation data to process.",
                    )
                    return
            else:
                with self.cc_apsr_citation_lock:
                    self.cc_apsr_citations_cache = {
                        title: [] for title in self.camb_core_df["title"]
                    }

            # Threadpool: Query Cambridge Core for APSR citations.
            pbar_completed = 0
            pbar_total = len(self.camb_core_df)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(QueryCambridgeCoreCitations, row)
                    for _, row in self.camb_core_df.iterrows()
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        paper_title, citations, error = future.result()
                        if paper_title:
                            with self.cc_apsr_citation_lock:
                                self.cc_apsr_citations_cache[paper_title] = citations
                            if error:

                                self.utils.Log(
                                    "warning",
                                    *self.utils.GetFuncLine(),
                                    f"error for {paper_title}: {error}",
                                )

                    except Exception as e:

                        self.utils.Log(
                            "error",
                            *self.utils.GetFuncLine(),
                            f"Failed to process citation row: {e}",
                        )

                    pbar_completed += 1
                    if pbar_completed % 10 == 0 or pbar_completed == pbar_total:
                        self.utils.ProgressBar(
                            pbar_completed,
                            pbar_total,
                            prefix=f"Querying Cambridge Core Citations ({pbar_total:,} total)",
                        )

            with self.cc_apsr_citation_lock:
                # Pad all lists in citation_data to the same length
                max_len = max(
                    len(citations)
                    for citations in self.cc_apsr_citations_cache.values()
                )
                for key in self.cc_apsr_citations_cache:
                    self.cc_apsr_citations_cache[key].extend(
                        [""] * (max_len - len(self.cc_apsr_citations_cache[key]))
                    )

                # Output the citation data to a CSV file
                self.utils.OutputCSV(
                    pd.DataFrame(self.cc_apsr_citations_cache),
                    self.cc_apsr_citations_output_csv_path,
                )
                self.utils.PickleOut(
                    self.cc_apsr_citations_cache,
                    self.cc_apsr_citations_cache_output_path,
                )

            self.cc_apsr_citations_processed = True

        except Exception as e:

            self.utils.Log(
                "error",
                *self.utils.GetFuncLine(),
                f"Failed to process Cambridge Core citations: {e}",
            )
            raise e

    def PostProcessCambridgeCoreDOIs(self) -> None:

        month_map = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12,
        }
        re_compiled_str = re.compile(rf",\s*({'|'.join(month_map.keys())})\s+(\d{{4}})")

        def QueryCambridgeCoreDOIs(citation_row):
            ret = {
                "apsr_paper_title": citation_row["title"].strip(),
                "apsr_paper_doi": None,
                "apsr_paper_pub_year": None,
                "apsr_paper_pub_month": None,
                "error": "",
            }

            apsr_paper_link = citation_row["article_link"].strip()
            if not apsr_paper_link:
                ret["error"] += f"[ error ]: No APSR paper link found for citation row."
                return ret

            res = self.WebFetch(apsr_paper_link)
            if res is None:
                ret[
                    "error"
                ] += f"[ error ]: No server response after maximum attempts; APSR paper link: {apsr_paper_link}"
                return ret
            elif res.status_code in self.client_errors:
                ret[
                    "error"
                ] += f"[ error ]: Server Response: {res.status_code}; APSR paper link: {apsr_paper_link}"
                return ret

            soup = BeautifulSoup(res.text, "html.parser")

            # Parse the response to find the DOI.
            apsr_doi = next(
                (
                    a["href"]
                    for a in soup.find_all("a", href=True)
                    if a["href"].startswith("https://doi.org/")
                ),
                None,
            )
            if apsr_doi:
                ret["apsr_paper_doi"] = apsr_doi
            else:
                ret[
                    "error"
                ] += f"[ error ]: No DOI found for APSR paper link: {apsr_paper_link}"

            date_span = soup.find("span", string=re_compiled_str)
            if date_span:
                match = re_compiled_str.search(date_span.string)
                if match:
                    month_str, year_str = match.groups()
                    ret["apsr_paper_pub_month"] = month_map[month_str]
                    ret["apsr_paper_pub_year"] = int(year_str)
                else:
                    ret[
                        "error"
                    ] += f"[ error ]: No publication date found for APSR paper link: {apsr_paper_link}"

            return ret

        try:
            if self.camb_core_df is None:

                self.utils.Log(
                    "error",
                    *self.utils.GetFuncLine(),
                    "No input data loaded. Cannot process DOIs.",
                )
                return

            if self.loaded_cc_apsr_dois_cache:
                pre_update_len = len(self.cc_apsr_dois_cache)
                with self.cc_apsr_dois_lock:
                    self.cc_apsr_dois_cache.update(
                        {
                            title: {}
                            for title in self.camb_core_df["title"]
                            if title not in self.cc_apsr_dois_cache
                        }
                    )
                if len(self.cc_apsr_dois_cache) == pre_update_len:

                    self.utils.Log(
                        "info",
                        *self.utils.GetFuncLine(),
                        "No new APSR DOI data to process.",
                    )
                    return
            else:
                with self.cc_apsr_dois_lock:
                    self.cc_apsr_dois_cache = {
                        title: {} for title in self.camb_core_df["title"]
                    }

            # Threadpool: Query Cambridge Core for APSR DOIs.
            pbar_completed = 0
            pbar_total = len(self.camb_core_df)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(QueryCambridgeCoreDOIs, row)
                    for _, row in self.camb_core_df.iterrows()
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        ret = future.result()
                        if ret["apsr_paper_title"] is not None:
                            with self.cc_apsr_dois_lock:
                                self.cc_apsr_dois_cache[ret["apsr_paper_title"]] = {
                                    "apsr_paper_doi": ret["apsr_paper_doi"],
                                    "apsr_paper_pub_year": ret["apsr_paper_pub_year"],
                                    "apsr_paper_pub_month": ret["apsr_paper_pub_month"],
                                    "error": ret["error"],
                                }
                            if ret["error"]:

                                self.utils.Log(
                                    "warning",
                                    *self.utils.GetFuncLine(),
                                    f"error for {ret['apsr_paper_title']}: {ret['error']}",
                                )

                    except Exception as e:

                        self.utils.Log(
                            "error",
                            *self.utils.GetFuncLine(),
                            f"Failed to process citation row: {e}",
                        )

                    pbar_completed += 1
                    if pbar_completed % 10 == 0 or pbar_completed == pbar_total:
                        self.utils.ProgressBar(
                            pbar_completed,
                            pbar_total,
                            prefix=f"Querying Cambridge Core DOIs ({pbar_total:,} total)",
                        )

            with self.cc_apsr_dois_lock:
                # Output the APSR Doi data to a CSV file.
                self.utils.OutputCSV(
                    pd.DataFrame(self.cc_apsr_dois_cache),
                    self.cc_apsr_dois_output_csv_path,
                )
                self.utils.PickleOut(
                    self.cc_apsr_dois_cache,
                    self.cc_apsr_dois_cache_output_path,
                )

            self.cc_apsr_dois_processed = True

        except Exception as e:

            self.utils.Log(
                "error",
                *self.utils.GetFuncLine(),
                f"Failed to process Cambridge Core DOIs: {e}",
            )
            raise e

    def PostProcessCrossRef(self, min_abstract_char_length: int = 25) -> None:
        def QueryCrossRefByDOI(pair) -> Dict:
            apsr_title, doi = pair
            ret = {
                "apsr_title": apsr_title,
                "citing_doi": None,
                "citing_title": None,
                "citing_abstract": None,
                "citing_pub_year": None,
                "citing_pub_month": None,
                "error": "",
            }

            # Format the doi string.
            if doi.startswith("https://doi.org/"):
                crossref_doi = (doi[self.doi_prefix_length :]).replace(";", "").strip()
            else:
                crossref_doi = doi.replace(";", "").strip()
            ret["citing_doi"] = crossref_doi

            # Build the CrossRef API query; fetch.
            url = [f"https://api.crossref.org/works/{crossref_doi}"]
            if self.credentials and self.credentials.get("email", ""):
                url.append(f"?mailto={self.credentials['email']}")
            res = self.WebFetch("".join(url), base_timeout=30)
            if res is None:
                ret[
                    "error"
                ] += f"[ error ]: No server response after maximum attempts; CrossRef DOI: {crossref_doi}"
                return ret
            elif res.status_code in self.client_errors:
                ret[
                    "error"
                ] += f"[ error ]: Server Response: {res.status_code}; CrossRef DOI: {crossref_doi}"
                return ret

            data = res.json().get("message", {})

            # Extract and clean citing title.
            title_val = data.get("title", "")
            if title_val:
                if isinstance(title_val, list):
                    title_val = " ".join(title_val).strip()
                ret["citing_title"] = (
                    BeautifulSoup(title_val, "html.parser").get_text().strip()
                )
            else:
                ret["error"] += f"[ error ]: No title found for DOI: {doi}"

            # Extract and clean citing abstract.
            abstract_val = data.get("abstract", "")
            if abstract_val:
                if isinstance(abstract_val, list):
                    abstract_val = " ".join(abstract_val).strip()
                ret["citing_abstract"] = (
                    BeautifulSoup(abstract_val, "html.parser").get_text().strip()
                )
            else:
                ret["error"] += f"[ error ]: No abstract found for DOI: {doi}"

            # Get the publication information.
            date_fields = ["published-print", "published-online"]
            for field in date_fields:
                date_list_ymd = data.get(field, {}).get("date-parts", [])
                if (
                    date_list_ymd
                    and isinstance(date_list_ymd, list)
                    and date_list_ymd[0]
                ):
                    ymd = date_list_ymd[0]
                    ret["citing_pub_year"] = int(ymd[0])
                    ret["citing_pub_month"] = int(ymd[1]) if len(ymd) > 1 else None
                    break
                else:
                    ret[
                        "error"
                    ] += f"[ error ]: No publication date found for DOI: {doi}"

            # Return the processed data.
            return ret

        try:
            if (
                not self.cc_apsr_citations_processed
                and not self.loaded_cc_apsr_citations_cache
            ):
                self.PostProcessCambridgeCoreCitations()

            # Build list of (apsr_title, doi) pairs that need processing.
            to_process = []
            for apsr_title, doi_list in self.cc_apsr_citations_cache.items():
                for doi in doi_list:
                    d = doi[self.doi_prefix_length :].strip()
                    if not d or len(d) < 1:
                        continue
                    record = self.crossref_cache.get(d, {})
                    if (not record.get("error")) and (
                        not record.get("citing_title")
                        or not record.get("citing_abstract")
                        or not record.get("apsr_title")
                        or not record.get("citing_doi")
                    ):
                        to_process.append((apsr_title, d))

            if len(to_process) == 0:

                self.utils.Log(
                    "info", *self.utils.GetFuncLine(), "No new abstracts to process."
                )
                self.crossref_processed = True
                return

            # Threadpool: Query CrossRef API by doi.
            pbar_completed = 0
            pbar_total = len(to_process)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(QueryCrossRefByDOI, pair): pair
                    for pair in to_process
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        ret = future.result()
                        with self.crossref_lock:
                            self.crossref_cache[ret["citing_doi"]] = {
                                "apsr_title": ret["apsr_title"],
                                "citing_doi": ret["citing_doi"],
                                "citing_title": ret["citing_title"],
                                "citing_abstract": ret["citing_abstract"],
                                "citing_pub_year": ret["citing_pub_year"],
                                "citing_pub_month": ret["citing_pub_month"],
                                "error": ret["error"],
                            }

                    except Exception as e:

                        self.utils.Log(
                            "error",
                            *self.utils.GetFuncLine(),
                            f"Failed to process abstract row: {e}",
                        )

                    # Update the progress bar on every 10th processed abstract.
                    pbar_completed += 1
                    if pbar_completed % 10 == 0 or pbar_completed == pbar_total:
                        self.utils.ProgressBar(
                            pbar_completed,
                            pbar_total,
                            prefix=f"Querying CrossRef API ({pbar_total:,} total)",
                        )

                    # (Batch update) Every 100 processed abstracts, flush CSV and pickle out.
                    if pbar_completed % 100 == 0:
                        with self.crossref_lock:
                            self.utils.OutputCSV(
                                pd.DataFrame.from_dict(
                                    self.crossref_cache, orient="index"
                                ),
                                self.crossref_output_csv_path,
                            )
                            self.utils.PickleOut(
                                self.crossref_cache, self.crossref_cache_output_path
                            )

            # Final CSV and pickle update after processing is complete.
            with self.crossref_lock:
                self.utils.OutputCSV(
                    pd.DataFrame.from_dict(self.crossref_cache, orient="index"),
                    self.crossref_output_csv_path,
                    print_notification=True,
                )
                self.utils.PickleOut(
                    self.crossref_cache, self.crossref_cache_output_path
                )
            self.crossref_processed = True

        except Exception as e:

            self.utils.Log(
                "error", *self.utils.GetFuncLine(), f"Failed to process abstracts: {e}"
            )
            raise e

    def FinalOutput(self, min_abstract_char_length: int = 25) -> None:
        """
        Outputs a .csv and .pkl file containing processed data with no errors
        and abstracts at least `min_abstract_char_length` characters long.
        Populates missing fields from caches or input CSV as needed.
        """
        try:
            # Filter records with no errors and abstracts meeting the length requirement
            valid_records = [
                record
                for record in self.crossref_cache.values()
                if not record.get("error")
                and len(record.get("citing_abstract", "")) >= min_abstract_char_length
            ]

            if not valid_records:
                raise ValueError("No valid records found for output.")

            # Create a DataFrame from the filtered records
            final_output_df = pd.DataFrame(valid_records)

            # Populate missing fields from apsr_doi_cache
            final_output_df["apsr_title_doi"] = final_output_df["apsr_title"].map(
                lambda title: self.cc_apsr_dois_cache.get(title, {}).get(
                    "apsr_paper_doi", "N/A"
                )
            )
            final_output_df["apsr_title_pub_year"] = final_output_df["apsr_title"].map(
                lambda title: self.cc_apsr_dois_cache.get(title, {}).get(
                    "apsr_paper_pub_year", "N/A"
                )
            )
            final_output_df["apsr_title_pub_month"] = final_output_df["apsr_title"].map(
                lambda title: self.cc_apsr_dois_cache.get(title, {}).get(
                    "apsr_paper_pub_month", "N/A"
                )
            )

            # Populate missing fields from camb_core_df
            if self.camb_core_df is not None:
                apsr_title_data = self.camb_core_df.set_index("title")[
                    ["cited_by_count", "abstract"]
                ].to_dict(orient="index")
                final_output_df["apsr_title_total_cited_by_count"] = final_output_df[
                    "apsr_title"
                ].map(
                    lambda title: apsr_title_data.get(title, {}).get(
                        "cited_by_count", "N/A"
                    )
                )
                final_output_df["apsr_title_abstract"] = final_output_df[
                    "apsr_title"
                ].map(
                    lambda title: apsr_title_data.get(title, {}).get("abstract", "N/A")
                )

            # Populate citing DOI publication date from crossref_cache
            final_output_df["citing_pub_year"] = final_output_df["citing_doi"].map(
                lambda doi: self.crossref_cache.get(doi, {}).get(
                    "citing_pub_year", "N/A"
                )
            )
            final_output_df["citing_pub_month"] = final_output_df["citing_doi"].map(
                lambda doi: self.crossref_cache.get(doi, {}).get(
                    "citing_pub_month", "N/A"
                )
            )

            # Calculate filtered cited-by count
            final_output_df["apsr_title_filtered_cited_by_count"] = final_output_df[
                "apsr_title"
            ].map(final_output_df["apsr_title"].value_counts())

            # Ensure the output DataFrame has the required columns
            required_columns = [
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
            final_output_df = final_output_df.reindex(columns=required_columns)

            # Sort the DataFrame by APSR title
            final_output_df.sort_values(by="apsr_title", ascending=True, inplace=True)

            # Write the output to a CSV file
            self.utils.OutputCSV(
                final_output_df,
                self.unified_cambcore_crossref_output_csv_path,
                print_notification=True,
            )

            # Write the output to a pickle file
            self.utils.PickleOut(
                final_output_df,
                self.filtered_unified_cambcore_crossref_cache_output_path,
            )

            print(f"Output generated: {final_output_df.size:,} cells in final output.")

        except Exception as e:

            self.utils.Log(
                "error",
                *self.utils.GetFuncLine(),
                f"Failed to produce final output: {e}",
            )
            raise e


if __name__ == "__main__":
    pproc = PostProcessAPSR(attempt_cache_load=True)

    try:
        pproc.PostProcessCambridgeCoreCitations()
        pproc.PostProcessCambridgeCoreDOIs()
        pproc.PostProcessCrossRef()
        pproc.FinalOutput()
        pproc.CleanUp()
    except KeyboardInterrupt:
        if pproc:
            if (
                pproc.cc_apsr_citations_cache
                and len(pproc.cc_apsr_citations_cache.keys()) > 0
            ):
                pproc.utils.PickleOut(
                    pproc.cc_apsr_citations_cache,
                    pproc.cc_apsr_citations_cache_output_path,
                )
            if pproc.crossref_cache and len(pproc.crossref_cache.keys()) > 0:
                pproc.utils.PickleOut(
                    pproc.crossref_cache, pproc.crossref_cache_output_path
                )
            pproc.CleanUp()

            fn, line = pproc.utils.GetFuncLine()
            pproc.utils.Log(
                "info",
                *pproc.utils.GetFuncLine(),
                "Post-processing interrupted by user.",
            )

    except Exception as e:
        if pproc:
            if (
                pproc.cc_apsr_citations_cache
                and len(pproc.cc_apsr_citations_cache.keys()) > 0
            ):
                pproc.utils.PickleOut(
                    pproc.cc_apsr_citations_cache,
                    pproc.cc_apsr_citations_cache_output_path,
                )
            if pproc.crossref_cache and len(pproc.crossref_cache.keys()) > 0:
                pproc.utils.PickleOut(
                    pproc.crossref_cache, pproc.crossref_cache_output_path
                )
            if pproc.cc_apsr_dois_cache and len(pproc.cc_apsr_dois_cache.keys()) > 0:
                pproc.utils.PickleOut(
                    pproc.cc_apsr_dois_cache,
                    pproc.cc_apsr_dois_cache_output_path,
                )
        pproc.CleanUp()

        fn, line = pproc.utils.GetFuncLine()
        pproc.utils.Log(
            "error", *pproc.utils.GetFuncLine(), f"Post-processing failed: {e}"
        )
        raise e

    finally:
        pproc.CleanUp()

        # Print and save out the log messages, if any.
        pproc.utils.WriteLog()
