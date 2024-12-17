import json
import os
from datetime import datetime

# Construct the file path
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'version_metadata.json')

# Load the JSON data
with open(file_path, 'r', encoding='utf-8') as file:
    version_metadata = json.load(file)

# Function to calculate the difference between two dates
def calculate_date_difference(date1_str, date2_str):
    if date1_str and date2_str:
        date_format = "%Y-%m-%d %H:%M:%S %z"
        date1 = datetime.strptime(date1_str, date_format)
        date2 = datetime.strptime(date2_str, date_format)
        return abs((date2 - date1).days)
    return None

# Calculate the difference for each version
date_differences = {}
for version, metadata in version_metadata.items():
    branch_date = metadata.get('branch_date')
    tag_date = metadata.get('tag_date')
    date_difference = calculate_date_difference(branch_date, tag_date)
    date_differences[version] = date_difference

# Print the date differences
for version, difference in date_differences.items():
    print(f"Version: {version}, Date Difference: {difference} days")