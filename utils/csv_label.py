"""
Adds labels for morphology and intensity bias
"""

import pandas as pd
import os

dir = "./splits2/exp198"

def check_condition(filename, condition_set):
    # Split the filename by underscores
    parts = filename.split('_')
    # Check if the length of parts is sufficient to have the 5th and 6th parts
    if len(parts) > 5:
        # Extract the part between the 5th and 6th underscores
        target_part = parts[5]
        # Check if any character in the target_part is in the condition_set
        return int(any(char in condition_set for char in target_part))
    return 0

for split in ["train", "val", "test"]:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(dir, split + ".csv"))

    # Add a new column 'col_0_or_2' based on the presence of '0' or '2' between 5th and 6th underscores
    df['morph_bias'] = df['filename'].apply(lambda x: check_condition(x, {'0', '2'}))

    # Add another column 'col_0_or_1' based on the presence of '0' or '1' between 5th and 6th underscores
    df['intensity_bias'] = df['filename'].apply(lambda x: check_condition(x, {'0', '1'}))

    # Print the first few rows to verify the new columns
    print(df.head())

    # Save the updated DataFrame back to the CSV file
    df.to_csv(os.path.join(dir, split + ".csv"), index=False)
