import pandas as pd
from pathlib import Path
import os
import time


def prepare_all_metrics(version):
    start_time = time.time()
    # Chemins des fichiers
    commit_metriques_PF_file_name = 'combined_metriques_PF_' + version + '.csv'
    commit_metriques_und_file_name = 'und_hive_all_metrics_' + version + '.csv'
    output_file_name = 'commit_all_metrics_' + version + '.csv'
    file_path = Path(os.path.realpath(__file__)).parent.parent.parent / 'data'
    commit_metriques_PF_file = file_path / commit_metriques_PF_file_name
    commit_metriques_und_file = file_path / commit_metriques_und_file_name
    output_file = file_path / output_file_name

    # Lire les métriques des commits
    dataset1 = pd.read_csv(commit_metriques_PF_file)
    dataset2 = pd.read_csv(commit_metriques_und_file)
    print(f"Nombre de lignes dans le dataset1: {len(dataset1)}")
    print(f"Nombre de lignes dans le dataset2: {len(dataset2)}")
    print(f"La différence entre dataset1 et dataset2: {len(dataset1) - len(dataset2)}")

    # Extraire uniquement le nom de fichier de la colonne 'Name' dans dataset1
    dataset1['Name'] = dataset1['Name'].apply(lambda x: os.path.basename(x))

    # Fusionner les datasets sur la colonne 'Name'
    merged_dataset = pd.merge(dataset1, dataset2, on='Name', how='outer')

    # Compter les lignes contenant des NaN
    nan_count = merged_dataset.isna().any(axis=1).sum()
    print(f"Nombre de lignes contenant des NaN avant remplacement: {nan_count}")

    # Remplacer les valeurs NaN par 0
    merged_dataset = merged_dataset.fillna(0)

    # Vérifier si le fichier de sortie existe déjà et le supprimer
    if os.path.exists(output_file):
        os.remove(output_file)
    # Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
    merged_dataset.to_csv(output_file, index=False)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
