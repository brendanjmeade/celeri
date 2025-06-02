#!/usr/bin/env python3

import os
import sys
import subprocess
import datetime
import glob
import papermill
from pathlib import Path

# Create directory for logs
Path("notebook_run_logs").mkdir(exist_ok=True)

# Get all .ipynb files in the current directory
notebooks = glob.glob("*.ipynb")

# Timestamp for log files
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Results dictionary
results = {}

# Run each notebook
for notebook in notebooks:
    print(f"Running {notebook}...")
    log_file = f"notebook_run_logs/{notebook}_{timestamp}.log"

    try:
        # Run the notebook with papermill, using /dev/null (or NUL on Windows) as output path
        # This effectively discards the output notebook
        # import papermill
        output_path = os.devnull

        with open(log_file, 'w') as log:
            result = papermill.execute_notebook(
                notebook,
                output_path,
                progress_bar=True,
                stdout_file=log,
                stderr_file=log
            )
        status = "Success"
        print(f"{notebook} completed successfully")

    except Exception as e:
        status = "Failed"
        print(f"{notebook} failed: {str(e)}")

        # Extract error message
        with open(log_file, 'a') as log:
            log.write(f"\nError: {str(e)}\n")

    # Record the result
    results[notebook] = status
    print("----------------------------------------")

# Write results to CSV
with open("notebook_results.csv", 'w') as f:
    f.write("Notebook,Status\n")
    for notebook, status in results.items():
        f.write(f"{notebook},{status}\n")

# Print summary
success_count = list(results.values()).count("Success")
failed_count = list(results.values()).count("Failed")
print(f"\nSummary:")
print(f"Successful notebooks: {success_count}")
print(f"Failed notebooks: {failed_count}")
print(f"Results saved to notebook_results.csv")
print(f"Detailed logs are available in the notebook_run_logs directory")
