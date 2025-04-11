import sys

import pandas as pd


def add_columns_to_segment_csv(input_file, output_file):
    """Reads a CSV file, adds several new columns with default values, and writes the modified data to a new CSV file.

    Parameters
    ----------
    input_file (str): The path to the input CSV file.
    output_file (str): The path to the output CSV file.

    The following columns are added to the DataFrame with default values:
    - ss_rate_bound_flag: 0
    - ss_rate_bound_min: -1.0
    - ss_rate_bound_max: 1.0
    - ds_rate_bound_flag: 0
    - ds_rate_bound_min: -1.0
    - ds_rate_bound_max: 1.0
    - ts_rate_bound_flag: 0
    - ts_rate_bound_min: -1.0
    - ts_rate_bound_max: 1.0

    If there is an error reading the input file or writing to the output file, an error message is printed.

    Returns
    -------
    None
    """
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    df["ss_rate_bound_flag"] = 0
    df["ss_rate_bound_min"] = -1.0
    df["ss_rate_bound_max"] = 1.0
    df["ds_rate_bound_flag"] = 0
    df["ds_rate_bound_min"] = -1.0
    df["ds_rate_bound_max"] = 1.0
    df["ts_rate_bound_flag"] = 0
    df["ts_rate_bound_min"] = -1.0
    df["ts_rate_bound_max"] = 1.0
    df.to_csv(output_file, index=False)

    try:
        df.to_csv(output_file, index=False)
        print(f"Successfully wrote to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python add_slip_rate_constraint_cols_to_segment_csv_file.py <input_file> <output_file>"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    add_columns_to_segment_csv(input_file, output_file)
