import subprocess
import os

def is_valid_path(path):
    try:
        if not os.path.exists(path):
            print("Path does not exist.")
            return False
        if not os.access(path, os.R_OK):
            print("Path is not readable.")
            return False
        return True
    except Exception as e:
        print(f"Invalid path: {e}")
        return False

def get_commits(repo_path, issues):
    issues=issues
    found = {}
    try:
        print(f"Repo path: {repo_path}")
        if not is_valid_path(repo_path):
            return []
        # Navigate to the repository path
        subprocess.run(["git", "-C", repo_path, "checkout", "master"], check=True)
        # Run the git log command with specified encoding
        result = subprocess.run(
            ['git', '-C', repo_path, 'log', '--pretty=format:%H %s'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        result.check_returncode()  # Check if the command was successful

        # Split the output into lines
        commits = result.stdout.strip().split('\n')
        commit_list = []
        for line in commits:
            sha, message = line.split(' ', 1)

            for key in issues:
                if key in message:
                    # Get the list of modified files for each commit with specified encoding
                    files_result = subprocess.run(
                        ['git', '-C', repo_path, 'show', '--pretty=format:', '--name-only', sha],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        check=True
                    )
                    files_result.check_returncode()  # Check if the command was successful
                    files = files_result.stdout.strip().split('\n')
                    commit_list.append({'key': key,'sha': sha, 'message': message, 'files': files})
                    found [key] = line
                    del issues[key]
                    break
        print("Commits extracted successfully")
        return commit_list
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while fetching commits:\n{e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred:\n{e}")
        return []
