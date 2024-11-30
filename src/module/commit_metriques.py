import csv
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import time


def run_git_log(version_start, version_end, repo_path):
    """Exécute git log pour extraire les données nécessaires entre deux versions avec les patches."""
    command = (
        f"git log {version_start}..{version_end} --stat --numstat --patch "
        f'--pretty=format:"Commit: %H%nAuthor: %an <%ae>%nDate: %ad%nSubject: %s%n" --date=short'
    )
    result = subprocess.run(command, shell=True, text=True, capture_output=True, cwd=repo_path, encoding='utf-8')
    return result.stdout.strip() if result.returncode == 0 else ""


def run_git_log_until(version_start, repo_path):
    """Exécute git log jusqu'à version_start pour collecter les informations d'expertise."""
    command = (
        f"git log {version_start} --stat --numstat "
        f'--pretty=format:"Commit: %H%nAuthor: %an <%ae>%nDate: %ad%nSubject: %s%n" --date=short'
    )
    result = subprocess.run(command, shell=True, text=True, capture_output=True, cwd=repo_path, encoding='utf-8')
    return result.stdout.strip() if result.returncode == 0 else ""


def run_git_log_global(version_end, repo_path):
    """Exécute git log pour tous les commits jusqu'à version_end."""
    command = (
        f"git log --stat --numstat {version_end} "
        f'--pretty=format:"Commit: %H%nAuthor: %an <%ae>%nDate: %ad%nSubject: %s%n" --date=short'
    )
    result = subprocess.run(command, shell=True, text=True, capture_output=True, cwd=repo_path, encoding='utf-8')
    return result.stdout.strip() if result.returncode == 0 else ""


def parse_git_log(log_output):
    """Analyse les données de git log pour extraire les métriques."""
    commits = []
    current_commit = {}
    for line in log_output.splitlines():
        if line.startswith("Commit:"):
            if current_commit:  # Ajoute le commit précédent
                commits.append(current_commit)
            current_commit = {
                "hash": line.split(" ", 1)[1],
                "author": "",
                "date": "",
                "subject": "",
                "files": []
            }
        elif line.startswith("Author:"):
            current_commit["author"] = line.split(": ", 1)[1]
        elif line.startswith("Date:"):
            current_commit["date"] = datetime.strptime(line.split(": ", 1)[1], "%Y-%m-%d")
        elif line.startswith("Subject:"):
            current_commit["subject"] = line.split(": ", 1)[1]
        elif re.match(r"^\d+\t\d+\t", line):  # Ligne numstat : ajoutées, supprimées, fichier
            parts = line.split("\t")
            current_commit["files"].append({
                "added": int(parts[0]),
                "deleted": int(parts[1]),
                "file": parts[2]
            })
    if current_commit:  # Ajoute le dernier commit
        commits.append(current_commit)
    return commits


def parse_git_log_with_comments(log_output):
    """Analyse la sortie de git log avec patch pour extraire les ajouts/suppressions de lignes et les commentaires."""
    commits = []
    current_commit = None
    current_file = None

    for line in log_output.splitlines():
        if line.startswith("Commit:"):
            if current_commit:
                commits.append(current_commit)
            current_commit = {
                "hash": line.split(" ", 1)[1],
                "author": "",
                "date": "",
                "subject": "",
                "files": defaultdict(lambda: {
                    "lines_added": 0,
                    "lines_deleted": 0,
                    "comments_added": 0,
                    "comments_deleted": 0
                }),
            }
        elif line.startswith("Author:"):
            current_commit["author"] = line.split(": ", 1)[1]
        elif line.startswith("Date:"):
            current_commit["date"] = datetime.strptime(line.split(": ", 1)[1], "%Y-%m-%d")
        elif line.startswith("Subject:"):
            current_commit["subject"] = line.split(": ", 1)[1]
        elif re.match(r"^diff --git a/", line):  # Nouveau fichier
            current_file = line.split(" b/")[-1]
        elif re.match(r"^\+\+\+", line) or re.match(r"^---", line):  # Ignore les headers des fichiers
            continue
        elif re.match(r"^\+[^+]", line):  # Ligne ajoutée (pas un "+++" header)
            current_commit["files"][current_file]["lines_added"] += 1
            if "#" in line:
                current_commit["files"][current_file]["comments_added"] += 1
        elif re.match(r"^-", line):
            current_commit["files"][current_file]["lines_deleted"] += 1
            if "#" in line:
                current_commit["files"][current_file]["comments_deleted"] += 1

    if current_commit:  # Ajouter le dernier commit
        commits.append(current_commit)

    return commits


def get_all_files_at_version(version, repo_path):
    """Récupère tous les fichiers présents à une version donnée."""
    command = f"git ls-tree -r --name-only {version}"
    result = subprocess.run(command, shell=True, text=True, capture_output=True, cwd=repo_path)
    return result.stdout.strip().splitlines() if result.returncode == 0 else []


def calculate_developer_expertise(commits):
    """Calcule le nombre de commits réalisés par chaque développeur jusqu'à version_start."""
    dev_expertise = defaultdict(int)
    for commit in commits:
        dev_expertise[commit["author"]] += 1
    return dev_expertise


def analyze_metrics(commits, all_files, dev_expertise):
    """Analyse les métriques, y compris les changements de commentaires."""
    file_metrics = defaultdict(lambda: {
        "lines_added": 0,
        "lines_deleted": 0,
        "comments_added": 0,
        "comments_deleted": 0,
        "commits": 0,
        "bug_fixes": 0,
        "developers": set(),
        "avg_time_between_changes": 0,
        "dev_expertise_avg": 0,
        "dev_expertise_min": 0,
        "developer_count": 0
    })

    # Analyse des commits pour chaque fichier
    for commit in commits:
        is_bug_fix = any(keyword in commit["subject"].lower() for keyword in
                         ["fix", "resolve", "repair", "patch", "debug", "correct", "rectify", "remedy ", "mend",
                          "amend"])
        for file, file_data in commit["files"].items():
            if file not in all_files:
                continue  # Ignore les fichiers absents de la version_end

            metrics = file_metrics[file]
            metrics["lines_added"] += file_data["lines_added"]
            metrics["lines_deleted"] += file_data["lines_deleted"]
            metrics["comments_added"] += file_data["comments_added"]
            metrics["comments_deleted"] += file_data["comments_deleted"]
            metrics["commits"] += 1
            metrics["developers"].add(commit["author"])
            if is_bug_fix:
                metrics["bug_fixes"] += 1

    # Calcul des métriques finales
    for file_name, metrics in file_metrics.items():
        if metrics["commits"] > 1:
            commit_dates = sorted(
                [c["date"] for c in commits if file_name in c["files"]]
            )
            time_diffs = [(commit_dates[i] - commit_dates[i - 1]).days for i in range(1, len(commit_dates))]
            metrics["avg_time_between_changes"] = sum(time_diffs) / len(time_diffs) if time_diffs else 0

        metrics["developer_count"] = len(metrics["developers"])
        metrics["dev_expertise_avg"] = (
            sum(dev_expertise[dev] for dev in metrics["developers"]) / len(metrics["developers"])
            if metrics["developers"] else 0
        )
        metrics["dev_expertise_min"] = (
            min(dev_expertise[dev] for dev in metrics["developers"])
            if metrics["developers"] else 0
        )
        del metrics["developers"]  # Supprimer les données intermédiaires inutiles

    return file_metrics


def analyze_global_metrics(commits, all_files):
    """Analyse les métriques globales en incluant tous les fichiers de all_files."""
    file_metrics = defaultdict(lambda: {
        "commits_global": 0,
        "developers_global": set(),
        "avg_time_between_changes_global": 0,
        "developer_count_global": 0,
    })
    dev_expertise = defaultdict(int)

    # Analyse des commits
    for commit in commits:
        for file in commit["files"]:
            if file["file"] not in all_files:
                continue  # Ignore les fichiers absents de all_files

            file_name = file["file"]
            file_metrics[file_name]["commits_global"] += 1
            file_metrics[file_name]["developers_global"].add(commit["author"])

        # Mettre à jour l'expertise des développeurs
        dev_expertise[commit["author"]] += 1

    # Ajouter les fichiers non modifiés avec des valeurs par défaut
    for file in all_files:
        if file not in file_metrics:
            file_metrics[file]  # Initialisation implicite avec les valeurs par défaut

    # Calculer les métriques finales
    for file_name, metrics in file_metrics.items():
        if metrics["commits_global"] > 1:
            commit_dates = sorted(
                [c["date"] for c in commits if file_name in {f["file"] for f in c["files"]}]
            )
            time_diffs = [(commit_dates[i] - commit_dates[i - 1]).days for i in range(1, len(commit_dates))]
            if len(time_diffs) > 0:  # Vérification pour éviter la division par zéro
                metrics["avg_time_between_changes_global"] = sum(time_diffs) / len(time_diffs)
            else:
                metrics["avg_time_between_changes_global"] = 0
        else:
            metrics["avg_time_between_changes_global"] = 0

        metrics["developer_count_global"] = len(metrics["developers_global"])
        # Supprimer la clé 'developers'
        del metrics["developers_global"]

    return file_metrics


def export_combined_metrics_to_csv(metrics_local, metrics_global, output_path):
    """Exporte les métriques locales et globales dans un même fichier CSV."""

    # Vérifier si le fichier existe déjà
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Le fichier {output_path} existait déjà, il a été supprimé.")

    # Créer le fichier CSV
    with open(output_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        # Obtenir toutes les clés des métriques
        if metrics_local:
            local_headers = list(next(iter(metrics_local.values())).keys())
        else:
            local_headers = []

        if metrics_global:
            global_headers = list(next(iter(metrics_global.values())).keys())
        else:
            global_headers = []

        # Créer un en-tête combiné
        headers = ["Name"] + [f"{h}" for h in local_headers] + [f"{h}" for h in global_headers]
        writer.writerow(headers)

        # Inclure tous les fichiers présents dans les deux jeux de métriques
        all_files = set(metrics_local.keys()).union(set(metrics_global.keys()))

        # Écrire les données
        for file in sorted(all_files):
            local_data = metrics_local.get(file, {key: 0 for key in local_headers})
            global_data = metrics_global.get(file, {key: 0 for key in global_headers})
            row = [file] + list(local_data.values()) + list(global_data.values())
            writer.writerow(row)

    print(f"Les métriques combinées ont été exportées vers {output_path}")


def main():
    # Define Git versions and repository path
    version_start = "release-2.0.0"
    version_end = "rel/release-3.0.0"
    version = "3.0.0"
    repo_path = R"C:\Users\lafor\Desktop\ETS - Cours\MGL869-01_Sujets speciaux\Laboratoire\Hive\hive"
    repo_output = Path(os.path.realpath(__file__)).parent.parent.parent / "data"
    repo_name_output = "combined_metriques_PF_" + version + ".csv"
    output_path = os.path.join(repo_output, repo_name_output)
    print(output_path)

    # Run Git logs to get commits
    log_output = run_git_log(version_start, version_end, repo_path)
    commits_with_comments = parse_git_log_with_comments(log_output)
    precedant_log_output = run_git_log_until(version_start, repo_path)
    precedant_commits = parse_git_log(precedant_log_output)
    global_log_output = run_git_log_global(version_end, repo_path)
    global_commits = parse_git_log(global_log_output)

    # Get the list of files at version_end
    all_files = get_all_files_at_version(version_end, repo_path)
    print("nombre de fichier :" + str(len(all_files)))

    # Calculate developer expertise
    dev_expertise = calculate_developer_expertise(precedant_commits)

    # Analyze metrics
    file_metrics = analyze_metrics(commits_with_comments, all_files, dev_expertise)
    mid_time = time.time()
    print(f"Temps d'exécution de l'analyse des métriques locales: {mid_time - start_time} secondes")
    global_metrics = analyze_global_metrics(global_commits, all_files)
    mid_time_global = time.time()
    print(f"Temps d'exécution de l'analyse des métriques globales: {mid_time_global - mid_time} secondes")

    # Export combined metrics to CSV
    export_combined_metrics_to_csv(file_metrics, global_metrics, output_path)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
