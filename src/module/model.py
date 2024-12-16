import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import precision_recall_fscore_support as score, make_scorer, precision_score
from sklearn import metrics
from collections import Counter
import matplotlib.pyplot as plt
from pickle import dump, load
from simpleNomo import nomogram
import xlsxwriter
from statistics import stdev

# Define version and directories
version = "3_0_0"
dots_separated_version = ".".join(version.split("_"))
base_dir = Path(os.path.realpath(__file__)).parent.parent.parent
data_dir = base_dir / "data"
version_output_dir = base_dir / "output" / version
version_output_dir.mkdir(exist_ok=True)

# Load dataset
all_metrics_path = data_dir / f"commit_all_metrics_3.0.0.csv"
filtered_dataset = pd.read_csv(all_metrics_path, low_memory=False)
filtered_dataset = filtered_dataset.drop("Kind", axis=1)

# Function to divide large "CountPath" values
def divided_count_path(dataset, operation):
    count_path_operation = f"CountPath{operation}"
    max_nb_of_digits = math.floor(math.log10(max(dataset[count_path_operation]))) + 1
    multiples_of_1000 = max_nb_of_digits // 3
    division_factor = 10 ** (3 * (multiples_of_1000 - 1))
    if division_factor == 1:
        return dataset
    dataset[count_path_operation] = dataset[count_path_operation].apply(lambda x: round(x / division_factor, 0))
    return dataset.rename(columns={count_path_operation: f"{count_path_operation}-divided-by-{division_factor:,}"})

# Apply division to "CountPath" columns
for operation in ["Min", "Max", "Mean"]:
    filtered_dataset = divided_count_path(filtered_dataset, operation)

# Read files with bugs and add "Bugs" column
files_with_bugs = pd.read_csv(data_dir / 'Bugs_3.0.0.csv')
files_with_bugs['Name'] = files_with_bugs['Name'].apply(lambda x: os.path.basename(x))
bugs = pd.DataFrame(np.zeros(len(filtered_dataset)), columns=["Bugs"])
filtered_dataset = pd.concat([filtered_dataset, bugs], axis=1)
java_files = [Path(file).name for file in files_with_bugs if Path(file).suffix == ".java"]
filtered_dataset.loc[filtered_dataset["Name"].isin(java_files), "Bugs"] = 1

# Drop columns with all NaN values
filtered_dataset = filtered_dataset.dropna(axis=1, how='all')

# Create correlation heatmap and drop highly correlated columns
corr_matrix = filtered_dataset.iloc[:, 1:-1].corr()
plt.figure(figsize=(77,75))
plt.title(f'Correlation Heatmap version {dots_separated_version}')
a = sns.heatmap(corr_matrix, square=True, annot=True, fmt='.2f', linecolor='black')
a.set_xticklabels(a.get_xticklabels(), rotation=30)
a.set_yticklabels(a.get_yticklabels(), rotation=30)
plt.savefig(version_output_dir / f"correlation_heatmap_{version}.png")
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.9)]
filtered_dataset = filtered_dataset.drop(to_drop, axis=1)

# Function to create custom boxplots
def boxplots_custom(filtered_dataset, columns_list, rows, cols, suptitle):
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(13, 50))
    fig.suptitle(suptitle, y=1, size=25)
    axs = axs.flatten()
    for i, data in enumerate(columns_list):
        sns.boxplot(data=filtered_dataset[data], orient='h', ax=axs[i])
        axs[i].set_title(data + ', skewness is: ' + str(round(filtered_dataset[data].skew(axis=0, skipna=True), 2)))

# Create boxplots for each variable
columns_list = list(filtered_dataset.columns[1:-1])
boxplots_custom(filtered_dataset=filtered_dataset, columns_list=columns_list, rows=math.ceil(len(columns_list) / 3), cols=3, suptitle='Boxplots for each variable')
plt.tight_layout()
plt.savefig(version_output_dir / f"boxplots_{version}.png")

# Function to identify outliers using IQR method
def IQR_method(df, n, features):
    outlier_list = []
    for column in features:
        Q1 = np.percentile(df[column], 25)
        Q3 = np.percentile(df[column], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_column = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)].index
        outlier_list.extend(outlier_list_column)
    outlier_list = Counter(outlier_list)
    multiple_outliers = list(k for k, v in outlier_list.items() if v > n)
    return multiple_outliers

# Remove outliers
n = 20
outliers_IQR = IQR_method(filtered_dataset, n, columns_list)
outliers = filtered_dataset.loc[outliers_IQR].reset_index(drop=True)
filtered_dataset = filtered_dataset.drop(outliers_IQR, axis=0).reset_index(drop=True)

# Drop columns with all same values
number_unique = filtered_dataset.nunique()
cols_to_drop = number_unique[number_unique == 1].index
filtered_dataset = filtered_dataset.drop(cols_to_drop, axis=1)
outliers_dataset = outliers.drop(cols_to_drop, axis=1)

# Save preprocessed data and outliers to files
filtered_dataset.to_csv(version_output_dir / f"und_hive_metrics_preprocessed_{version}.csv", index=False)
outliers.to_csv(version_output_dir / f"outliers_{version}.csv", index=False)

# Drop "Name" column and separate data from labels
filtered_dataset = filtered_dataset.drop("Name", axis=1)
outliers = outliers.drop("Name", axis=1)
X = filtered_dataset.iloc[:, :-1]
y = filtered_dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Set 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=False)

# Define parameter grid for Logistic Regression
param_grid = {
    "penalty": [None, 'l2', 'l1', 'elasticnet'],
    "solver": ['newton-cg', 'newton-cholesky', 'lbfgs', 'sag', 'saga'],
    "max_iter": [1000, 2000, 3000]  # Increase the number of iterations
}

# Use 'macro' average for multiclass classification
precision_scorer = make_scorer(precision_score, average='macro')

# Load or train Logistic Regression model
existing_model = True
try:
    with open(version_output_dir / f"logistic_regression_model_{version}.pkl", "rb") as f:
        logistic_regression_clf = load(f)
except FileNotFoundError:
    existing_model = False
if not existing_model:
    logistic_regression_grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=kf, scoring=precision_scorer, verbose=3)
    logistic_regression_grid.fit(X_train, y_train)
    logistic_regression_clf = logistic_regression_grid.best_estimator_
    with open(version_output_dir / f"logistic_regression_model_{version}.pkl", "wb") as f:
        dump(logistic_regression_clf, f, protocol=5)

# Calculate cross-validation scores for Logistic Regression
score_lr = cross_val_score(logistic_regression_clf, X_train, y_train, cv=kf, scoring=precision_scorer)
lr_score = score_lr.mean()
lr_stdev = stdev(score_lr)
lr_predicted = logistic_regression_clf.predict(X_test)
lr_predicted_probs = logistic_regression_clf.predict_proba(X_test)[:, 1]
lr_precision, lr_recall, lr_fscore, lr_support = score(y_test, lr_predicted, average='macro')

# Calculate and plot AUC for Logistic Regression
lr_fpr, lr_tpr, lr_thresholds = metrics.roc_curve(y_test, lr_predicted_probs, pos_label=1)
lr_auc = metrics.auc(lr_fpr, lr_tpr)
plt.figure(figsize=(8, 6))
plt.plot(lr_fpr, lr_tpr, color="blue", label=f"AUC = {lr_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Logistic Regression AUC Curve - version {dots_separated_version}")
plt.legend(loc="lower right")
plt.grid()
plt.savefig(version_output_dir / f"logistic_regression_auc_{version}.png")

# Generate nomogram configuration file
workbook = xlsxwriter.Workbook(version_output_dir / f"nomogram_config_{version}.xlsx")
worksheet = workbook.add_worksheet()
worksheet.write("A1", "feature")
worksheet.write("B1", "coef")
worksheet.write("C1", "min")
worksheet.write("D1", "max")
worksheet.write("E1", "type")
worksheet.write("F1", "position")
worksheet.write("A2", "intercept")
worksheet.write("B2", round(logistic_regression_clf.intercept_[0], 4))
worksheet.write("A3", "threshold")
worksheet.write("B3", 0.5)
for i, column in enumerate(filtered_dataset.columns[:-1], start=0):
    worksheet.write(f"A{i + 4}", column)
    worksheet.write(f"B{i + 4}", round(logistic_regression_clf.coef_[0][i], 4))
    worksheet.write(f"C{i + 4}", round(min(filtered_dataset[column]), 1))
    worksheet.write(f"D{i + 4}", round(max(filtered_dataset[column]), 1))
    worksheet.write(f"E{i + 4}", "continuous")
workbook.close()

# Print nomogram for Logistic Regression
nomogram_fig = nomogram(str(version_output_dir / f"nomogram_config_{version}.xlsx"), result_title="Bug risk", fig_width=50, single_height=0.45, dpi=300, ax_para={"c": "black", "linewidth": 1.3, "linestyle": "-"}, tick_para={"direction": 'in', "length": 3, "width": 1.5, }, xtick_para={"fontsize": 10, "fontfamily": "Songti Sc", "fontweight": "bold"}, ylabel_para={"fontsize": 12, "fontname": "Songti Sc", "labelpad": 100, "loc": "center", "color": "black", "rotation": "horizontal"}, total_point=100)
nomogram_fig.savefig(version_output_dir / f"nomogram_{version}.png")

# Define parameter grid for Random Forest
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 4, 8, 16],
    "min_samples_split": [2, 4],
    "min_samples_leaf": [1, 2],
    "max_features": ["auto", "sqrt", "log2"],
    "random_state": [0],
}

# Load or train Random Forest model
existing_model = True
try:
    with open(version_output_dir / f"random_forest_model_{version}.pkl", "rb") as f:
        random_forest_clf = load(f)
except FileNotFoundError:
    existing_model = False
if not existing_model:
    random_forest_grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=kf, scoring=precision_scorer, verbose=3)
    random_forest_grid.fit(X_train, y_train)
    random_forest_clf = random_forest_grid.best_estimator_
    with open(version_output_dir / f"random_forest_model_{version}.pkl", "wb") as f:
        dump(random_forest_clf, f, protocol=5)

# Calculate cross-validation scores for Random Forest
score_rf = cross_val_score(random_forest_clf, X_train, y_train, cv=kf, scoring=precision_scorer)
rf_score = score_rf.mean()
rf_stdev = stdev(score_rf)
rf_predicted = random_forest_clf.predict(X_test)
rf_predicted_probs = random_forest_clf.predict_proba(X_test)[:, 1]
rf_precision, rf_recall, rf_fscore, rf_support = score(y_test, rf_predicted, average='macro')

# Calculate and plot AUC for Random Forest
rf_fpr, rf_tpr, rf_thresholds = metrics.roc_curve(y_test, rf_predicted_probs, pos_label=1)
rf_auc = metrics.auc(rf_fpr, rf_tpr)
plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, color="blue", label=f"AUC = {rf_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Random Forest AUC Curve - version {dots_separated_version}")
plt.legend(loc="lower right")
plt.grid()
plt.savefig(version_output_dir / f"random_forest_auc_{version}.png")