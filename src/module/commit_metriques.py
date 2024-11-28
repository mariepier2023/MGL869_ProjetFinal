import csv
import subprocess
import os
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed



def write_commits_to_csv(results, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Deleted existing file: {output_file}")

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['file', 'lines_added', 'lines_deleted', 'num_commits', 'num_bug_fixes', 'num_all_commits',
                         'num_developers', 'num_all_developers', 'avg_time_between_changes',
                         'avg_time_all_versions', 'avg_expertise', 'min_expertise', 'comment_change_commits', 'non_comment_change_commits'])

        for result in results:
            writer.writerow([result['file'], result['lines_added'], result['lines_deleted'], result['num_commits'],
                             result['num_bug_fixes'], result['num_all_commits'], result['num_developers'],
                             result['num_all_developers'], result['avg_time_between_changes'],
                             result['avg_time_all_versions'], result['avg_expertise'], result['min_expertise'],
                             result['comment_change_commits'], result['non_comment_change_commits']])
    print(f"Written results to CSV file: {output_file}")

def run_git_command(repo_path, command):
    result = subprocess.run(
        ['git', '-C', repo_path] + command,
        capture_output=True,
        text=True,
        encoding='utf-8',
        check=True
    )
    return result.stdout.strip()

def get_lines_added_deleted(repo_path, start_commit, end_commit, file_path):
    output = run_git_command(repo_path, ['diff', '--numstat', f'{start_commit}..{end_commit}', '--', file_path])
    if output:
        parts = output.split('\t')
        if len(parts) == 3:
            added, deleted, _ = parts
            added = 0 if added == '-' else int(added)
            deleted = 0 if deleted == '-' else int(deleted)
            return added, deleted
    return 0, 0

def get_commits(repo_path, file_path, start_commit, end_commit):
    output = run_git_command(repo_path, ['log', '--pretty=format:%H %s', f'{start_commit}..{end_commit}', '--', file_path])
    if not output:
        return []
    commits = [line.split(' ', 1) for line in output.split('\n') if ' ' in line]
    return [(sha, message) for sha, message in commits]

def get_bug_fixing_commits(commits, keywords):
    return [commit for commit in commits if any(keyword in commit[1].lower() for keyword in keywords)]

def get_developers(repo_path, commits):
    developers = set()
    for sha, _ in commits:
        author = run_git_command(repo_path, ['show', '-s', '--format=%an', sha])
        developers.add(author)
    return developers

def get_commit_dates(repo_path, commits):
    dates = []
    for sha, _ in commits:
        date_str = run_git_command(repo_path, ['show', '-s', '--format=%ci', sha])
        dates.append(datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S %z'))
    return sorted(dates)

def calculate_average_time(dates):
    if len(dates) < 2:
        return 0
    intervals = [(dates[i] - dates[i-1]).total_seconds() for i in range(1, len(dates))]
    return sum(intervals) / len(intervals)

def get_developer_expertise(repo_path, developers, end_commit):
    expertise = defaultdict(int)
    for developer in developers:
        output = run_git_command(repo_path, ['log', '--author', developer, '--pretty=format:%H', '-n', '1', end_commit])
        expertise[developer] = len(output.split('\n')) if output else 0
    return expertise

def get_comment_change_commits(repo_path, file_path, start_commit, end_commit):
    output = run_git_command(repo_path, ['log', '-p', '--pretty=format:%H', f'{start_commit}..{end_commit}', '--', file_path])
    if not output:
        return 0, 0

    commits = output.split('\ncommit ')
    comment_change_commits = 0
    non_comment_change_commits = 0

    for commit in commits:
        if not commit.strip():
            continue
        lines = commit.split('\n')
        sha = lines[0].strip()
        diff_output = '\n'.join(lines[1:])
        if any(line.strip().startswith(('+', '-')) and line.strip()[1:].strip().startswith('#') for line in diff_output.split('\n')):
            comment_change_commits += 1
        else:
            non_comment_change_commits += 1

    return comment_change_commits, non_comment_change_commits

def process_file(repo_path, file_path, start_commit, end_commit):
    added, deleted = get_lines_added_deleted(repo_path, start_commit, end_commit, file_path)
    commits = get_commits(repo_path, file_path, start_commit, end_commit)
    bug_fixing_commits = get_bug_fixing_commits(commits, ['fix', 'bug', 'error', 'issue'])
    all_commits = get_commits(repo_path, file_path, '', end_commit)
    developers = get_developers(repo_path, commits)
    all_developers = get_developers(repo_path, all_commits)
    commit_dates = get_commit_dates(repo_path, commits)
    all_commit_dates = get_commit_dates(repo_path, all_commits)
    average_time = calculate_average_time(commit_dates)
    all_average_time = calculate_average_time(all_commit_dates)
    expertise = get_developer_expertise(repo_path, developers, end_commit)
    avg_expertise = sum(expertise.values()) / len(expertise) if expertise else 0
    min_expertise = min(expertise.values()) if expertise else 0
    comment_change_commits, non_comment_change_commits = get_comment_change_commits(repo_path, file_path, start_commit, end_commit)

    return {
        "file": file_path,
        "lines_added": added,
        "lines_deleted": deleted,
        "num_commits": len(commits),
        "num_bug_fixes": len(bug_fixing_commits),
        "num_all_commits": len(all_commits),
        "num_developers": len(developers),
        "num_all_developers": len(all_developers),
        "avg_time_between_changes": average_time,
        "avg_time_all_versions": all_average_time,
        "avg_expertise": avg_expertise,
        "min_expertise": min_expertise,
        "comment_change_commits": comment_change_commits,
        "non_comment_change_commits": non_comment_change_commits
    }

def main(repo_path, start_commit, end_commit):
    print("Starting main function")
    subprocess.run(["git", "-C", repo_path, "checkout", "master"], check=True)
    files = run_git_command(repo_path, ['ls-tree', '-r', '--name-only', end_commit]).split('\n')
    results = []

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, repo_path, file_path, start_commit, end_commit): file_path for file_path in files}
        for future in as_completed(futures):
            results.append(future.result())

    print("Finished main function")
    return results

if __name__ == "__main__":
    start_time = time.time()
    version_start = "2.0.0"
    version_end = "3.0.0"
    repo_path = r'C:\Users\lafor\Desktop\ETS - Cours\MGL869-01_Sujets speciaux\Laboratoire\Hive\hive'
    start_commit = "7f9f1fcb8697fb33f0edc2c391930a3728d247d7"
    end_commit = "ce61711a5fa54ab34fc74d86d521ecaeea6b072a"
    output_file_name = "Commit_Metriques_" + version_end + ".csv"
    output_file = Path(os.path.realpath(__file__)).parent.parent.parent / "data" / output_file_name
    print(output_file)
    results = main(repo_path, start_commit, end_commit)
    write_commits_to_csv(results, output_file)

    del results

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")