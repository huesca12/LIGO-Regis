# IMPORTS
# import external packages
import pandas as pd
import os


def simplify_csv(csv_in: str, n: int, csv_out: str = "csv_out.csv", save_to_file: bool = False):

    # make sure input variables are correct
    assert csv_in.split(".")[-1] == "csv", "csv_in must be a path to a CSV file"
    if save_to_file:
        assert csv_out is not None, "to save to a file, csv_out must not be None"
    if csv_out is not None:  # if csv_out is None, implicitly understand that a file is not being saved
        assert csv_out.split(".")[-1] == "csv", "csv_out must be a path to a CSV file"
    assert os.path.exists(csv_in), "CSV file path does not exist"

    # load CSV into panda data frame
    data = pd.read_csv(csv_in)
    # initialize output data frame
    output = pd.DataFrame()

    # get list of all labels
    glitch_labels = data["label"].unique()

    # iterate through glitches
    for i, glitch in enumerate(glitch_labels, start=1):
        # extract all rows with particular glitch
        glitch_rows = data[data["label"] == glitch]

        try:
            # try to extract n rows
            new_rows = glitch_rows.sample(n)
        except ValueError:
            # if n is too high, return all rows
            new_rows = glitch_rows

        # append new rows to output data frame
        output = output.append(new_rows)
        # print success message
        print(f"{i}. Extracted {len(new_rows)} (of {len(glitch_rows)}) rows for {glitch}.")
    print(f"The simplified data set has {len(output)} total entries.")

    # save output data frame to csv
    if save_to_file:
        output.to_csv(csv_out)

    return output
