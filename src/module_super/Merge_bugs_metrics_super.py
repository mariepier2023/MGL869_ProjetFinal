import json
import os
import math
import logging
import sys
from datetime import time
import time


import numpy as np
import pandas as pd

from pathlib import Path



def merge_bugs_metrics(version="3.0.0", recalculate_models=True, plot_images=True):
    # Set directories
    base_dir = Path(os.path.realpath(__file__)).parent.parent.parent
    data_dir = base_dir / "data"
    version_output_dir = base_dir / "output" / "super" / version
    version_output_dir.mkdir(exist_ok=True)

    #Set Logger
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    file_handler = logging.FileHandler(version_output_dir / f"logs_{version}.log", mode='w')
    logger = logging.getLogger()
    for handler in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(handler)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"VERSION: {version}")
    logging.info("")

    # Load dataset
    all_metrics_path = data_dir / ("commit_all_metrics_" + version + ".csv")
    filtered_dataset = pd.read_csv(all_metrics_path, low_memory=False)
    filtered_dataset = filtered_dataset.drop("Kind", axis=1)

    # Read the files with bugs
    files_with_bugs = pd.read_csv(data_dir / f"Bugs_{version}_Super.csv")
    files_with_bugs = files_with_bugs.drop("key", axis=1)
    files_with_bugs = files_with_bugs.drop("sha", axis=1)
    files_with_bugs = files_with_bugs.drop("message", axis=1)
    files_with_bugs = files_with_bugs.drop_duplicates()

    # Add "Bugs" column
    bugs = pd.DataFrame(np.zeros(len(filtered_dataset)), columns=["Bugs"])
    filtered_dataset = pd.concat([filtered_dataset, bugs], axis=1)
    filtered_dataset["Bugs"] = "0"
    for index, row in files_with_bugs.iterrows():
        file_path = row['filename']
        file_name = Path(file_path).name
        priority = row['priority']
        if file_path.endswith(".java"):
            for idx, data in filtered_dataset.loc[filtered_dataset["Name"] == file_name].iterrows():
                bug_levels = ["Blocker", "Critical", "Major", "Minor", "Trivial"]
                if data["Bugs"] not in bug_levels:
                    filtered_dataset.at[idx, "Bugs"] = priority
                    break
                elif bug_levels.index(data["Bugs"]) < bug_levels.index(priority):
                    filtered_dataset.at[idx, "Bugs"] = priority

    java_files_names = [Path(file_path).name for file_path in files_with_bugs["filename"] if file_path.endswith(".java")]
    logging.info(f"Total number of .java files: {len(filtered_dataset)}")
    logging.info(f"Number of .java files in the Bugs_{version}_Super.csv: {len(java_files_names)}")
    logging.info(
        f"Number of .java files with bug in the filtered_dataset: {len(filtered_dataset.loc[filtered_dataset["Bugs"] != 0, "Bugs"])}")
    logging.info(f"Missing .java files in the filtered_dataset:")
    missing_java_file = 0
    for file in java_files_names:
        if file not in list(filtered_dataset["Name"]):
            logging.info(f"    {file}")
            missing_java_file = missing_java_file + 1
    logging.info(f"Total number of missing .java files: {missing_java_file}")
    logging.info("")

    # Save all metrics and bugs to file
    filtered_dataset.to_csv(data_dir / f"und_hive_all_metrics_and_bugs_{version}_Super.csv", index=False)

    logging.info(f"Number of combined rows for current and previous versions without duplicates")
    logging.info(f"Initial number of metric columns: {filtered_dataset.iloc[:, 1:-1].shape[1]}")
    logging.info(f"Initial number of rows: {filtered_dataset.iloc[:, 1:-1].shape[0]}")
    # Count the number of cells containing text that is not "0" in the "Bugs" column
    text_cells_count = filtered_dataset["Bugs"].apply(lambda x: isinstance(x, str) and x != "0").sum()
    # Display the result
    print(f"Number of cells with text (not '0') in the 'Bugs' column: {text_cells_count}")
    logging.info("")



if __name__ == "__main__":
    start_time = time.time()
    with open(Path(os.path.realpath(__file__)).parent.parent/'module/version_metadata.json','r', encoding='utf-8') as file:
        version_metadata = json.load(file)
    for tag, metadata in version_metadata.items():
        version = tag[-5:]
        print(f"Analyzing version {version}...")
        merge_bugs_metrics(version)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")