import subprocess
import json
from datetime import datetime

REPO_PATH = r"C:\Users\lafor\Desktop\ETS - Cours\MGL869-01_Sujets speciaux\Laboratoire\Hive\hive"  # Chemin vers votre dépôt Hive

def get_tag_details(tag):
    """Retourne les détails d'un tag (hash et date) sans les métadonnées supplémentaires."""
    try:
        details = subprocess.check_output(
            ["git", "-C", REPO_PATH, "log", "-1", "--pretty=format:%H %ci", tag],
            text=True
        ).strip()
        commit_hash, commit_date = details.split(" ", 1)
        return {"hash": commit_hash, "date": commit_date}
    except subprocess.CalledProcessError:
        return {"error": f"Tag '{tag}' not found"}

def collect_tag_details(tags):
    """Collecte les détails pour une liste de tags."""
    results = {}
    for tag in tags:
        results[tag] = get_tag_details(tag)
    return results

def find_merge_base(base_branch, target_branch):
    """Trouve le commit commun le plus récent entre deux branches."""
    try:
        merge_base = subprocess.check_output(
            ["git", "-C", REPO_PATH, "merge-base", base_branch, target_branch], text=True
        ).strip()
        return merge_base
    except subprocess.CalledProcessError:
        return None

def get_commit_date(commit_hash):
    """Retourne la date d'un commit spécifique."""
    try:
        date = subprocess.check_output(
            ["git", "-C", REPO_PATH, "show", "-s", "--format=%ci", commit_hash], text=True
        ).strip()
        return date
    except subprocess.CalledProcessError:
        return None

def analyze_branch_origins(branches, base_branch="master"):
    """Analyse le commit de divergence pour une liste de branches par rapport à une branche de base."""
    results = {}
    for branch in branches:
        merge_base = find_merge_base(base_branch, branch)
        if merge_base:
            commit_date = get_commit_date(merge_base)
            results[branch] = {
                "parent_commit": merge_base,
                "divergence_date": commit_date,
            }
        else:
            results[branch] = {
                "error": "Parent commit not found"
            }
    return results

def parse_date(date_str):
    """Convertit une chaîne de date en objet datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z")

def find_closest_lower_tag(branch_date, tags_data):
    """
    Trouve le tag le plus proche en termes de temps,
    avec une date strictement inférieure à celle de la branche donnée.
    """
    closest_tag = None
    closest_time_diff = None
    branch_date_parsed = parse_date(branch_date)

    for tag, tag_info in tags_data.items():
        tag_date_parsed = parse_date(tag_info["date"])

        if tag_date_parsed >= branch_date_parsed:
            continue

        time_diff = (branch_date_parsed - tag_date_parsed).total_seconds()

        if closest_time_diff is None or time_diff < closest_time_diff:
            closest_time_diff = time_diff
            closest_tag = tag

    return closest_tag

def associate_branches_to_lower_tags(tags_data, branches_data):
    """Associe chaque branche au tag le plus proche ayant une date inférieure."""
    results = {}
    for branch, branch_info in branches_data.items():
        branch_date = branch_info.get("divergence_date")
        if not branch_date:
            results[branch] = {"error": "No divergence date found"}
            continue

        closest_tag = find_closest_lower_tag(branch_date, tags_data)
        results[branch] = {
            "branch_date": branch_date,
            "commit_branchement": branch_info.get("parent_commit"),
            "hash_version": tags_data.get(branch, {}).get("hash"),
            "closest_lower_tag": closest_tag,
            "hash_closest_lower_tag": tags_data.get(closest_tag, {}).get("hash") if closest_tag else None,
            "tag_date": tags_data[closest_tag]["date"] if closest_tag else None
        }

    return results

if __name__ == "__main__":
    # Liste des tags et branches
    tags = [
        "release-2.0.0",
        "rel/release-2.1.0",
        "rel/release-2.2.0",
        "rel/release-2.3.0",
        "rel/release-3.0.0",
        "rel/release-3.1.0"
    ]
    branches = tags

    tag_details = collect_tag_details(tags)
    branch_origins = analyze_branch_origins(branches)
    combined_results = associate_branches_to_lower_tags(tag_details, branch_origins)

    output_file = "version_metadata.json"
    with open(output_file, "w") as json_file:
        json.dump(combined_results, json_file, indent=4)

    print(json.dumps(combined_results, indent=4))
