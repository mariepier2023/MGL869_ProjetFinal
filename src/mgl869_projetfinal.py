import csv
import json
import os
from pathlib import Path
import time
from module.extractor_commit import get_commits
from module.extractor_jira import extract_issues
from module.commit_metriques import find_metrics
from module.préparation_métrique import prepare_all_metrics


def write_commits_to_csv(commit_list, output_file):
    # Check if the file exists
    if os.path.exists(output_file):
        # Delete the file if it exists
        os.remove(output_file)
        print(f"Deleted existing file: {output_file}")

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['key', 'sha', 'message', 'filename'])

        for commit in commit_list:
            key = commit['key']
            sha = commit['sha']
            message = commit['message']
            for bug_file in commit['files']:
                writer.writerow([key, sha, message, bug_file])

def main(version, metadata):
    VERSION = version
    OUTPUT_FILE_NAME = "Bugs_" + VERSION + ".csv"
    OUTPUT_FILE = Path(os.path.realpath(__file__)).parent.parent/"data"/OUTPUT_FILE_NAME
    JIRA_SEARCH_FILTER = "project = HIVE AND issuetype = Bug AND status in (Resolved, Closed) AND resolution = Fixed AND affectedVersion = " + VERSION

    if not os.path.exists(OUTPUT_FILE):
        print("Extracting jira issues...")
        issues = {}
        for issue in extract_issues(JIRA_SEARCH_FILTER, ["versions"]):
            issues[issue["key"]] = {"affectedVersions": [e["name"] for e in issue["fields"]["versions"]]}
        print(f"\tFound {len(issues)} issues.\n")


        print("Extracting github commits...")
        REPO_PATH = r'C:\Users\lafor\Desktop\ETS - Cours\MGL869-01_Sujets speciaux\Laboratoire\Hive\hive'
        commits_list = get_commits(REPO_PATH, issues)
        write_commits_to_csv(commits_list, OUTPUT_FILE)

        del commits_list
        del issues
    else:
        print(f"file {OUTPUT_FILE} already exists")

    print("Finding metrics...")
    print(Path(os.path.realpath(__file__)).parent.parent/"data")
    metric_file = Path(os.path.realpath(__file__)).parent.parent / "data" / ("combined_metriques_PF_" + version + ".csv")
    if not os.path.exists(metric_file):
        print(f"File not found: {metric_file}")
        find_metrics(version, metadata)
        return
    else:
        print(f"File found: {metric_file}")

    print("Merge metrics...")
    commit_metriques_PF_file_name = 'combined_metriques_PF_' + version + '.csv'
    commit_metriques_und_file_name = 'und_hive_all_metrics_' + version + '.csv'
    output_file_name = 'commit_all_metrics_' + version + '.csv'
    prepare_all_metrics(version,commit_metriques_PF_file_name,commit_metriques_und_file_name,output_file_name)



if __name__ == "__main__":
    start_time = time.time()
    with open(Path(os.path.realpath(__file__)).parent/'module/version_metadata.json','r', encoding='utf-8') as file:
        version_metadata = json.load(file)
    for tag, metadata in version_metadata.items():
        version = tag[-5:]
        print(f"Analyzing version {version}...")
        main(version, metadata)
    end_time = time.time()
    execution_time = end_time - start_time

