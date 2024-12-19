import json
import os
import math
import logging
import pickle
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter

import warnings
from statistics import stdev
from sklearn.metrics import recall_score, make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as sklearn_auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


def model(version="3.0.0", recalculate_models=True, plot_images=True):
    # Set directories
    base_dir = Path(os.path.realpath(__file__)).parent.parent.parent
    data_dir = base_dir / "data"
    version_output_dir = base_dir / "output" / "super" / version
    version_output_dir.mkdir(exist_ok=True)

    # Set Logger
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    file_handler = logging.FileHandler(version_output_dir / f"logs_{version}_model.log", mode='w')
    logger = logging.getLogger()
    for handler in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(handler)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"VERSION: {version}")
    logging.info("")

    # Load dataset
    all_metrics_path = data_dir / ("und_hive_all_metrics_and_bugs_" + version + "_Super.csv")
    filtered_dataset = pd.read_csv(all_metrics_path, low_memory=False)
    if "Kind" in filtered_dataset.columns:
        filtered_dataset = filtered_dataset.drop("Kind", axis=1)
        logging.info("Colonne 'Kind' supprimée.")
    else:
        logging.warning("La colonne 'Kind' n'existe pas dans le DataFrame.")

    def divided_count_path(dataset, operation):
        """Change "CountPath" scale because numbers are too big in regard to other columns.
        The numbers of the new scale will have a maximum of 4 digits."""
        count_path_operation = f"CountPath{operation}"
        max_nb_of_digits = math.floor(math.log10(max(dataset[count_path_operation]))) + 1
        division_factor = 10 ** (max_nb_of_digits - 3)
        if division_factor == 1:
            return dataset
        dataset[count_path_operation] = dataset[count_path_operation].apply(
            lambda x: x if math.isnan(x) else int(round(x / division_factor, 0)))
        return dataset.rename(columns={count_path_operation: f"{count_path_operation}-divided-by-{division_factor:,}"})

    for operation in ["Min", "Max", "Mean"]:
        filtered_dataset = divided_count_path(filtered_dataset, operation)

        # Display initial variable columns
    initial_columns = list(filtered_dataset.columns[1:-1])
    logging.info(f"Initial variable columns: {len(initial_columns)}")
    for column in initial_columns:
        logging.info(f"    {column}")
    logging.info("")

    # Drop columns with all NaN
    filtered_dataset = filtered_dataset.dropna(axis=1, how='all')
    remaining_columns = list(filtered_dataset.columns[1:-1])
    logging.info("Drop all NaN columns")
    logging.info(f"Remaining columns ({len(remaining_columns)}):")
    for column in remaining_columns:
        logging.info(f"    {column}")
    dropped_columns = [column for column in initial_columns if column not in remaining_columns]
    logging.info(f"Dropped all NaN columns ({len(dropped_columns)}):")
    for column in dropped_columns:
        logging.info(f"    {column}")
    logging.info("")

    # Check for missing values
    logging.info("Columns with missing values:")
    missing_values = filtered_dataset.iloc[:, 1:-1].isnull().sum()
    for column in missing_values.index:
        logging.info(f"    {column}: {missing_values[column]}")
    logging.info(
        f"Total rows with missing values removed: {len(filtered_dataset[~(filtered_dataset.index.isin(filtered_dataset.dropna().index))])}")
    filtered_dataset = filtered_dataset.dropna()
    logging.info(f"Total rows remaining: {len(filtered_dataset)}")
    logging.info("")

    # Remove correlated columns
    # Ref.: https://www.kaggle.com/code/prashant111/comprehensive-guide-on-feature-selection#2.6-Correlation-Matrix-with-Heatmap-
    corr_matrix = filtered_dataset.iloc[:, 1:-1].corr()

    if plot_images:
        # Create correlation heatmap
        plt.figure(figsize=(77, 75))
        plt.title(f'Correlation Heatmap version {version}')
        a = sns.heatmap(corr_matrix, square=True, annot=True, fmt='.2f', linecolor='black')
        a.set_xticklabels(a.get_xticklabels(), rotation=30)
        a.set_yticklabels(a.get_yticklabels(), rotation=30)
        plt.savefig(version_output_dir / f"correlation_heatmap_{version}.png")

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.7
    correlation_treshold = 0.7
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_treshold)]
    logging.info("Correlated columns to be dropped:")
    for column in to_drop:
        correlated_to = list(upper[upper[column].abs() > correlation_treshold].index)
        logging.info(f"    {column}, correlated to: {correlated_to}")
    logging.info("")

    # Drop correlated columns
    filtered_dataset = filtered_dataset.drop(to_drop, axis=1)

    # Checking boxplots (ref.: https://www.kaggle.com/code/marcinrutecki/gridsearchcv-kfold-cv-the-right-way)
    def boxplots_custom(filtered_dataset, columns_list, rows, cols, suptitle):
        fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(13, 50))
        fig.suptitle(suptitle, y=1, size=25)
        axs = axs.flatten()
        for i, data in enumerate(columns_list):
            sns.boxplot(data=filtered_dataset[data], orient='h', ax=axs[i])
            axs[i].set_title(data + ', skewness is: ' + str(round(filtered_dataset[data].skew(axis=0, skipna=True), 2)))

    columns_list = list(filtered_dataset.columns[1:-1])
    if plot_images:
        boxplots_custom(filtered_dataset=filtered_dataset, columns_list=columns_list,
                        rows=math.ceil(len(columns_list) / 3), cols=3,
                        suptitle='Boxplots for each variable')
        plt.tight_layout()
        plt.savefig(version_output_dir / f"boxplots_{version}.png")

    def IQR_method(df, n, features):
        """
        Takes a dataframe and returns an index list corresponding to the observations
        containing more than n outliers according to the Tukey IQR method.
        Ref.: https://www.kaggle.com/code/marcinrutecki/gridsearchcv-kfold-cv-the-right-way
        """
        outlier_list = []

        for column in features:
            # 1st quartile (25%)
            Q1 = np.percentile(df[column], 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(df[column], 75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            # Determining a list of indices of outliers
            outlier_list_column = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)].index
            # appending the list of outliers
            outlier_list.extend(outlier_list_column)

        # selecting observations containing more than x outliers
        outlier_list = Counter(outlier_list)
        multiple_outliers = list(k for k, v in outlier_list.items() if v > n)

        return multiple_outliers

    # Remove outliers (save the outliers to disk)
    # Adjust the `n` argument of `IQR_method` to allow more outliers to be kept, otherwise most of the files with bugs
    # where being removed
    n = 20
    logging.info("Remove outliers:")
    logging.info(f"    Initial number of rows in the filtered_dataset: {len(filtered_dataset)}")
    logging.info(
        f"    Initial number of .java files with bug in the filtered_dataset: {len(filtered_dataset.loc[filtered_dataset["Bugs"] == 1, "Bugs"])}")
    logging.info(f"    IQR_method n argument: {n}")
    outliers_IQR = IQR_method(filtered_dataset, n, columns_list)
    outliers = filtered_dataset.loc[outliers_IQR].reset_index(drop=True)
    logging.info(f"    Total number of outliers is: {len(outliers_IQR)}")
    # Drop outliers
    filtered_dataset = filtered_dataset.drop(outliers_IQR, axis=0).reset_index(drop=True)
    logging.info(f"    Final number of rows in the filtered_dataset: {len(filtered_dataset)}")
    logging.info(
        f"    Final number of .java files with bug in the filtered_dataset: {len(filtered_dataset.loc[filtered_dataset["Bugs"] == 1, "Bugs"])}")
    logging.info("")

    # Drop columns with all same value
    initial_columns = list(filtered_dataset.columns[1:-1])
    logging.info("Drop same value columns after outliers removal")
    logging.info(f"Initial columns: {len(initial_columns)}")
    for column in initial_columns:
        logging.info(f"    {column}")
    number_unique = filtered_dataset.nunique()
    cols_to_drop = number_unique[number_unique == 1].index
    filtered_dataset = filtered_dataset.drop(cols_to_drop, axis=1)
    outliers_dataset = outliers.drop(cols_to_drop, axis=1)
    remaining_columns = list(filtered_dataset.columns[1:-1])
    logging.info(f"Remaining columns ({len(remaining_columns)}):")
    for column in remaining_columns:
        logging.info(f"    {column}")
    dropped_columns = [column for column in initial_columns if column not in remaining_columns]
    logging.info(f"Dropped same value columns ({len(dropped_columns)}):")
    for column in dropped_columns:
        logging.info(f"    {column}")
    logging.info("")

    # Print variables range
    logging.info("Variables range:")
    for column in filtered_dataset.columns[1:-1]:
        logging.info(
            f"    {column}: {round(min(filtered_dataset[column]), 1)} - {round(max(filtered_dataset[column]), 1)}")
    logging.info("")

    # Save preprocessed data to file
    # filtered_dataset.to_csv(version_output_dir / f"und_hive_metrics_preprocessed_{current_version}.csv", index=False)

    # Save outliers data to file
    # outliers.to_csv(version_output_dir / f"outliers_{current_version}.csv", index=False)

    # Drop "Name" column
    if "Name" in filtered_dataset.columns:
        filtered_dataset = filtered_dataset.drop("Name", axis=1)
        outliers = outliers.drop("Name", axis=1)
        logging.info("Colonne 'Name' supprimée.")
    else:
        logging.warning("Colonne 'Name' n'existe pas dans le DataFrame.")

    x = filtered_dataset.drop(columns=["Bugs"], axis=1)
    y = filtered_dataset["Bugs"]

    # Encodage de la variable cible
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.3, random_state=0, shuffle=True, stratify=y_encoded
    )
    print("Classes uniques dans y_train:", np.unique(y_train))
    # Initialiser le répertoire
    version_output_dir.mkdir(parents=True, exist_ok=True)
    model_filename = version_output_dir / f"logistic_regression_model_{version}_super.pkl"

    # KFold 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    #kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Paramètres de GridSearch
    param_grid = {
        "solver": ["saga"],  # Le solver saga gère les multi-classes par défaut
        "C": [1],  # Paramètres de régularisation
        "max_iter": [10000],  # Plus d'itérations
        "class_weight": ["balanced"],  # Prendre en compte les déséquilibres de classes
        "random_state": [42],
    }

    scoring = make_scorer(precision_score, average="weighted", zero_division=0)
    # Charger ou entraîner le modèle
    existing_model = True
    try:
        with open(model_filename, "rb") as f:
            logistic_clf = pickle.load(f)
        logging.info("Modèle existant chargé.")
    except FileNotFoundError:
        existing_model = False
    if not existing_model or recalculate_models:
        logging.info("Entraînement du modèle avec GridSearchCV...")
        # Scoring ajusté pour éviter les problèmes de classes rares

        # GridSearchCV avec les correctifs
        logistic_grid = GridSearchCV(
            LogisticRegression(), param_grid=param_grid, cv=kf, scoring=scoring, verbose=3, n_jobs=-1
        )
        # Résultat optimal
        logistic_grid.fit(x_train, y_train)
        logistic_clf = logistic_grid.best_estimator_
        print("Meilleurs paramètres :", logistic_clf.get_params())
        with open(model_filename, "wb") as f:
            pickle.dump(logistic_clf, f, protocol=5)

    logging.info(f"Meilleurs paramètres : {logistic_clf.get_params()}")

    # Calculer la précision pondérée avec 10-fold
    precision_scores = cross_val_score(
        logistic_clf, x_train, y_train, cv=kf, scoring=scoring
    )
    precision_mean = precision_scores.mean()
    precision_std = stdev(precision_scores)
    logging.info(f"Précision (10-fold) : Moyenne = {precision_mean:.4f}, Écart-type = {precision_std:.4f}")

    # Définir le scorer pour le rappel pondéré
    recall_scorer = make_scorer(recall_score, average="weighted", zero_division=0)

    # Calculer les scores de rappel en 10-fold cross-validation avec le modèle optimisé
    logging.info("Calcul des scores de rappel (10-fold cross-validation) avec le modèle optimisé...")
    recall_scores = cross_val_score(
        logistic_clf, x_train, y_train, cv=kf, scoring=recall_scorer
    )
    # Calculer la moyenne et l'écart-type
    recall_mean = recall_scores.mean()
    recall_std = stdev(recall_scores)

    # Afficher les résultats
    logging.info(f"Rappel (10-fold) : Moyenne = {recall_mean:.4f}, Écart-type = {recall_std:.4f}")

    # Évaluation finale sur l'ensemble de test
    y_pred = logistic_clf.predict(x_test)
    logging.info("Performance sur l'ensemble de test :")
    logging.info(f"Confusion Matrix :\n{confusion_matrix(y_test, y_pred)}")
    logging.info(f"Classification Report :\n{classification_report(y_test, y_pred)}")

    # Visualiser la matrice de confusion
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    plt.title("Matrice de Confusion " + version)
    plt.savefig(version_output_dir / f"matrice de Confusion_{version}.png")
    plt.show()

    # # Binariser les classes (requis pour AUC multiclasse)
    # y_test_binarized = label_binarize(y_test, classes=range(len(le.classes_)))
    # y_pred_proba = logistic_clf.predict_proba(x_test)
    # n_classes = y_test_binarized.shape[1]
    #
    # # Calculer l'AUC One-vs-Rest pour chaque classe
    # auc_scores = []
    # supports = []  # Nombre d'exemples pour chaque classe
    # for i in range(len(le.classes_)):
    #     # AUC pour la classe i
    #     auc_score = roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i])
    #     auc_scores.append(auc_score)
    #     supports.append(y_test_binarized[:, i].sum())
    #     print(f"AUC for class {le.classes_[i]}: {auc_score:.4f}")
    #
    # # Moyenne pondérée des AUC
    # weighted_auc = sum(
    #     auc * (support / len(y_test)) for auc, support in zip(auc_scores, supports)
    # )
    # print(f"Weighted AUC: {weighted_auc:.4f}")
    #
    # # Calcul des courbes ROC et des AUC
    # fpr = {}
    # tpr = {}
    # roc_auc = {}
    #
    # for i in range(n_classes):
    #     # Vérifiez les valeurs binarisées pour NaN/Inf
    #     if np.isnan(y_test_binarized[:, i]).any() or np.isinf(y_test_binarized[:, i]).any():
    #         print(f"Problème avec y_test_binarized pour la classe {i}")
    #         continue
    #
    #     # Vérifiez les probabilités prédites pour NaN/Inf
    #     if np.isnan(y_pred_proba[:, i]).any() or np.isinf(y_pred_proba[:, i]).any():
    #         print(f"Problème avec y_pred_proba pour la classe {i}")
    #         continue
    #
    #     # Calcul des courbes ROC
    #     fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    #
    #     # Nettoyez les valeurs si nécessaire
    #     fpr[i] = np.nan_to_num(fpr[i], nan=0.0, posinf=1.0, neginf=0.0)
    #     tpr[i] = np.nan_to_num(tpr[i], nan=0.0, posinf=1.0, neginf=0.0)
    #
    #     # Calcul de l'AUC
    #     roc_auc[i] = sklearn_auc(fpr[i], tpr[i])
    #
    # # Tracer les courbes ROC pour chaque classe
    # plt.figure(figsize=(10, 8))
    # colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red"])
    #
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(
    #         fpr[i],
    #         tpr[i],
    #         color=color,
    #         lw=2,
    #         label=f"{le.classes_[i]} (AUC = {roc_auc[i]:.2f})"
    #     )
    #
    # # Tracer la ligne "no skill"
    # plt.plot([0, 1], [0, 1], "k--", lw=2, label="No Skill")
    #
    # # Configurer le graphique
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("Taux de faux positifs (FPR)")
    # plt.ylabel("Taux de vrais positifs (TPR)")
    # plt.title("Courbe ROC Multiclasse "+version)
    # plt.legend(loc="lower right")
    # plt.savefig(version_output_dir / f"AUC_{version}.png")
    # plt.show()

if __name__ == "__main__":
    start_time = time.time()
    # with open(Path(os.path.realpath(__file__)).parent.parent / 'module' / 'version_metadata.json', 'r',
    #           encoding='utf-8') as file:
    #     version_metadata = json.load(file)
    # for tag, metadata in version_metadata.items():
    #     version = tag[-5:]
    #     print(f"Modeling version {version}...")
    #     model(version)
    model("3.0.0")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Temps d'exécution : {execution_time:.2f} secondes.")
