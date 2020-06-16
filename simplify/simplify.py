import pandas
import os


# Columns we do not use
drop_cols = ["chisq", "chisqDof", "confidence", "GPStime", "ifo", "imgUrl", "id"]


def simplify_csv(csv_in: str, n: int, csv_out: str = "csv_out.csv", save_to_file: bool = True):

    # Make sure input variables are correct
    assert csv_in.split(".")[-1] == "csv", "csv_in must be a path to a CSV file"
    assert csv_out.split(".")[-1] == "csv", "csv_out must be a path to a CSV file"
    assert os.path.exists(csv_in), "CSV file path does not exist"

    # Load CSV into panda data frame, dropping unnecessary columns
    data = pandas.read_csv(csv_in).drop(columns=drop_cols)
    # Initialize output data frame
    output = pandas.DataFrame()

    # Get list of all labels
    glitch_labels = data["label"].unique()

    # Iterate through glitches
    for i, glitch in enumerate(glitch_labels, start=1):
        # Extract all rows with particular glitch
        glitch_rows = data[data["label"] == glitch]

        try:
            # Try to extract n rows
            new_rows = glitch_rows.sample(n)
        except ValueError:
            # If n is too high, return all rows
            new_rows = glitch_rows

        # Append new rows to output data frame
        output = output.append(new_rows)
        # Print success message
        # print(f"{i}. Generated {len(new_rows)} (of {len(glitch_rows)}) rows for {glitch}.")

    # Save output data frame to csv
    if save_to_file:
        output.to_csv(csv_out)

    return output
