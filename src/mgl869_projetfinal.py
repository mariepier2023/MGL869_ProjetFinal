import csv
import os
from pathlib import Path

from src.module.commit_extractor import get_commits
from src.module.jira_extrator import extract_issues


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


VERSION = "2.0.0"
OUTPUT_FILE_NAME = "Bugs_" + VERSION + ".csv"
OUTPUT_FILE = Path(os.path.realpath(__file__)).parent.parent/"data"/OUTPUT_FILE_NAME
JIRA_SEARCH_FILTER = "project = HIVE AND issuetype = Bug AND status in (Resolved, Closed) AND resolution = Fixed AND affectedVersion = " + VERSION

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
