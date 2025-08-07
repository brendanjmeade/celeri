#!/usr/bin/env python3

import argparse
import csv
import datetime
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import papermill


def setup_logging(log_dir: str) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Ensure log directory exists
    Path(log_dir).mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Jupyter notebooks and log results."
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Specific notebook files to run (*.ipynb). If none provided, all notebooks in current directory will be run.",
        type=Path,
    )
    parser.add_argument(
        "--log-dir",
        default="notebook_run_logs",
        help="Directory to store logs (default: notebook_run_logs)",
        type=Path,
    )
    parser.add_argument(
        "--output-csv",
        default="notebook_results.csv",
        help="Path for the CSV results file (default: notebook_results.csv)",
        type=Path,
    )
    parser.add_argument(
        "--kernel",
        default="python",
        help="Kernel to use for notebook execution (default: python)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude notebooks containing this substring. Can be used multiple times.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of notebooks to run in parallel (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum execution time in seconds for each notebook (default: no timeout)",
    )
    return parser.parse_args()


def get_notebooks(args: argparse.Namespace) -> list[Path]:
    """Get the list of notebooks to run based on arguments."""
    if args.notebooks:
        notebooks = [Path(notebook) for notebook in args.notebooks]
        # Check if provided notebooks exist
        for notebook in notebooks:
            if not notebook.exists():
                logging.error(f"Notebook {notebook} not found")
                sys.exit(1)
    else:
        # Get all .ipynb files in the current directory
        notebooks = list(Path.cwd().glob("*.ipynb"))

    # Filter out excluded notebooks
    if args.exclude:
        original_count = len(notebooks)
        for exclude_pattern in args.exclude:
            notebooks = [nb for nb in notebooks if exclude_pattern not in nb]
        excluded_count = original_count - len(notebooks)
        if excluded_count > 0:
            logging.info(f"Excluded {excluded_count} notebook(s) based on pattern(s)")

    if not notebooks:
        logging.warning("No notebooks found to process")

    return notebooks


def execute_notebook(
    notebook: str, args: argparse.Namespace, timestamp: str
) -> tuple[str, str, str | None]:
    """Execute a single notebook and return its result."""
    log_file = Path(args.log_dir) / f"{notebook}_{timestamp}.log"
    logging.info(f"Running {notebook}...")

    try:
        # Run the notebook with papermill
        output_path = os.devnull

        # Capture standard output to prevent papermill progress bars from cluttering parallel output
        with log_file.open("w") as log:
            # Disable progress bar in parallel mode to avoid messy output
            show_progress = args.parallel <= 1

            papermill.execute_notebook(
                notebook,
                output_path,
                progress_bar=show_progress,
                stdout_file=log,
                stderr_file=log,
                kernel_name=args.kernel,
                timeout=args.timeout,
            )
        status = "Success"
        error_msg = None
        logging.info(f"{notebook} completed successfully")

    except Exception as e:
        status = "Failed"
        error_msg = str(e)
        # Use a cleaner one-line error message for parallel execution
        if args.parallel > 1:
            logging.error(f"{notebook} failed: {type(e).__name__}")
        else:
            logging.error(f"{notebook} failed: {error_msg}")

        # Extract error message
        with log_file.open("a") as log:
            log.write(f"\nError: {error_msg}\n")

    return notebook, status, error_msg


def write_results_to_csv(results: dict[str, dict[str, str]], csv_path: Path) -> None:
    """Write results to CSV file with additional metadata."""
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Notebook", "Status", "Error"])
        for notebook, data in results.items():
            writer.writerow([notebook, data["status"], data.get("error", "")])


def main() -> None:
    args = parse_args()
    setup_logging(args.log_dir)

    # Get notebooks to run
    notebooks = get_notebooks(args)

    if not notebooks:
        return

    # Timestamp for log files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Results dictionary with more information
    results = {}

    # Parallel execution
    if args.parallel > 1:
        logging.info(f"Running notebooks with {args.parallel} workers")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_to_notebook = {
                executor.submit(execute_notebook, notebook, args, timestamp): notebook
                for notebook in notebooks
            }

            for future in as_completed(future_to_notebook):
                notebook, status, error = future.result()
                results[notebook] = {"status": status, "error": error}
                print("----------------------------------------")
    else:
        # Sequential execution
        for notebook in notebooks:
            notebook, status, error = execute_notebook(notebook, args, timestamp)
            results[notebook] = {"status": status, "error": error}
            print("----------------------------------------")

    # Write results to CSV
    write_results_to_csv(results, args.output_csv)

    # Print summary
    success_count = sum(1 for data in results.values() if data["status"] == "Success")
    failed_count = len(results) - success_count

    logging.info("\nSummary:")
    logging.info(f"Successful notebooks: {success_count}")
    logging.info(f"Failed notebooks: {failed_count}")
    logging.info(f"Results saved to {args.output_csv}")
    logging.info(f"Detailed logs are available in the {args.log_dir} directory")


if __name__ == "__main__":
    main()
