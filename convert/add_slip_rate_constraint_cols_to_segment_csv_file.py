import sys
import pandas as pd


def add_columns_to_segment_csv(input_file, output_file):
    df = pd.read_csv(input_file)
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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python add_slip_rate_constraint_cols_to_segment_csv_file.py <input_file> <output_file>"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    add_columns_to_segment_csv(input_file, output_file)
