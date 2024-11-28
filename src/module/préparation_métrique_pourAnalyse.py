from pathlib import Path
import pandas as pd
import os
import time

start_time = time.time()
# Chemins des fichiers
commit_metriques_file_name = 'commit_metriques_3.0.0.csv'
bugs_file_name = 'Bugs_3.0.0.csv'
output_file_name = 'commit_metriques_with_bugs_3.0.0.csv'
file_path = Path(os.path.realpath(__file__)).parent.parent.parent / 'data'
commit_metriques_file = file_path / commit_metriques_file_name
bugs_file = file_path / bugs_file_name
output_file = file_path / output_file_name
print(file_path / commit_metriques_file_name)

# Lire les fichiers avec des bugs
bugs_set = set(pd.read_csv(bugs_file, usecols=[3]).squeeze().tolist())

# Lire les métriques des commits
dataset = pd.read_csv(commit_metriques_file)

# Ajouter une colonne 'Bug' indiquant 'Yes' ou 'No'
dataset['Bug'] = dataset['file'].apply(lambda x: 1 if x in bugs_set else 0)
dataset = dataset.drop(columns=['file'])
# Vérifier si le fichier de sortie existe déjà et le supprimer
if os.path.exists(output_file):
    os.remove(output_file)

# Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
dataset.to_csv(output_file, index=False)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")