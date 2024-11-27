import json
import requests

# Documentation de l'API de JIRA: https://developer.atlassian.com/cloud/jira/platform/rest/v2/intro

JIRA_URL = "https://issues.apache.org/jira/rest/api/2"
HEADERS = {"Accept": "application/json"}

def extract_issues(filter: str, fields: list[str] = []):
    try:
        params = {
          "jql": filter,
          "maxResults": "5000",
          "fieldsByKeys": "true",
          "fields": ",".join(fields)
        }
        response = requests.get(f"{JIRA_URL}/search", headers=HEADERS, params=params)
        response.raise_for_status()
        return [{"key": e["key"], "fields": e["fields"]} for e in json.loads(response.text)["issues"]]
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        return []