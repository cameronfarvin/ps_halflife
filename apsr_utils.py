import os
import pickle
import time
import inspect
import pandas as pd

from typing import List


class APSRUtils:
    def __init__(self, log_path, clear_existing_log=True):
        # Initialize logging utillity.
        self.log_path = log_path
        self.log_messages = {"info": [], "warning": [], "error": []}

        # Delete the log file if the clear_existing_log flag is set.
        if clear_existing_log:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            if os.path.exists(self.log_path):
                os.remove(self.log_path)

    def Log(self, log_type, function_name, line_number, message, stream_to_file=False):
        if message:
            if log_type not in self.log_messages:
                self.log_messages[log_type] = []

            log_message = (
                f"[ {log_type} ]: ({function_name}, ln {line_number}): {message}"
            )
            self.log_messages[log_type].append(log_message)

            # If the stream flag is set, stream to file.
            if stream_to_file:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                with open(self.log_path, "a") as log_file:
                    log_file.write(log_message + "\n")

    def WriteLog(self):
        if len(self.log_messages.items()) > 0:
            print(
                f"Note: {sum(len(v) for v in self.log_messages.values())} log messages written to {self.log_path}"
            )

            # Write all logs to the log file.
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "w") as log_file:
                for log_type, messages in self.log_messages.items():
                    for message in messages:
                        log_file.write(message + "\n")

    def GetFuncLine(self):
        return (
            inspect.currentframe().f_back.f_code.co_name,
            inspect.currentframe().f_back.f_lineno,
        )

    def PickleOut(self, data, cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def LoadExistingCache(self, cache_path) -> any:
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        else:
            return None

    def OutputCSV(self, data, output_path, print_notification=False):
        try:
            if print_notification:
                print(f"Writing {output_path}...")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_path, index=False)
            else:
                pd.DataFrame(data).to_csv(output_path, index=False)

        except Exception as e:
            function_name, line_number = self.GetFuncLine()
            self.Log("error", function_name, line_number, f"Failed to write CSV: {e}")

    def ProgressBar(self, iteration, total, prefix="", length=25):
        percent = "{0:.1f}".format(100 * (iteration / float(max(1, total))))
        filled_length = int(length * iteration // max(1, total))
        bar = "█" * filled_length + "-" * (length - filled_length)
        print(f"\r{prefix} |{bar}| {percent}% Complete", end="\r")
        if iteration == total:
            print()
