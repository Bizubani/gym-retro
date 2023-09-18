"""
A simple logging utility. Writes the log to file as a CSV.
"""

import csv
import os


class Logger:
    def __init__(self, log_dir, log_name, log_headers):
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_headers = log_headers
        self.log_path = os.path.join(self.log_dir, self.log_name)
        self.log_file = open(self.log_path, "w")
        self.log_writer = csv.DictWriter(self.log_file, fieldnames=self.log_headers)
        self.log_writer.writeheader()
        self.log_file.flush()

    def log(self, log_data):
        self.log_writer.writerows(log_data)
        self.log_file.flush()

    def close(self):
        print("Closing log file")
        self.log_file.close()
