from flask import Flask, render_template, request
import pandas as pd
import re
from collections import defaultdict
from transformers import pipeline
import json
import difflib

app = Flask(__name__)

# Initialize the TQA pipeline
tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")

# Load CSV into a DataFrame
df = pd.read_csv('sample jira extract.csv')

# Dictionary to store question-answer pairs
qa_pairs = {}

# Synonyms dictionary for column names
column_synonyms = {
    "priority": ["importance", "urgency"],
    "summary": ["title", "heading"],
    "description": ["details", "info"],
    "issue_type": ["type", "category"],
    "assignee": ["assigned_to", "responsible"],
    "created_date": ["date_created", "creation_date"],
    "labels": ["tags"]
}

def map_synonym_to_column(word):
    word_lower = word.lower()
    for column, synonyms in column_synonyms.items():
        if word_lower == column or word_lower in synonyms:
            return column
    return None

def ask_question(query):
    condition_pattern = re.compile(r'(\w+)\s+as\s+(\w+)')
    keyword_pattern = re.compile(r'related\s+to\s+(\w+)')

    conditions = defaultdict(list)
    keyword = None

    for match in condition_pattern.finditer(query.lower()):
        column = match.group(1)
        value = match.group(2)
        mapped_column = map_synonym_to_column(column)
        if mapped_column:
            conditions[mapped_column].append(value)

    for match in keyword_pattern.finditer(query.lower()):
        keyword = match.group(1)

    filtered_df = df.copy()
    for column, values in conditions.items():
        if column in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[column].str.lower().isin(values)]

    if keyword:
        filtered_df = filtered_df[filtered_df.apply(lambda row: keyword in row.astype(str).str.lower().values, axis=1)]

    return filtered_df.to_dict(orient='records')

def learn_question_answer_pairs_from_file(file_path):
    with open(file_path, 'r') as f:
        qa_data = json.load(f)
        for item in qa_data['question_answers']:
            question = item['question']
            answers = item['answers']
            learn_question_answer_pair(question, answers)

def learn_question_answer_pair(question, answers):
    qa_pairs[question.lower()] = answers

def calculate_similarity(query, text):
    matcher = difflib.SequenceMatcher(None, query.lower(), text.lower())
    return round(matcher.ratio() * 100, 2)

def get_response(question):
    if question.lower() in qa_pairs:
        return qa_pairs[question.lower()]
    else:
        response = ask_question(question)
        return response

def handle_simple_queries(query):
    issue_key_pattern = re.compile(r'\b([A-Za-z0-9]+-[0-9]+)\b')

    match = issue_key_pattern.search(query)
    if match:
        issue_key = match.group(1)
        filtered_df = df[df['issue_key'] == issue_key]

        if filtered_df.empty:
            return f"No information found for issue key {issue_key}"

        if 'summary' in query.lower():
            return filtered_df['summary'].iloc[0]
        elif 'description' in query.lower():
            return filtered_df['description'].iloc[0]
        elif 'issue_type' in query.lower():
            return filtered_df['issue_type'].iloc[0]
        elif 'priority' in query.lower():
            return filtered_df['priority'].iloc[0]
        elif 'assignee' in query.lower():
            return filtered_df['assignee'].iloc[0]
        elif 'created_date' in query.lower():
            return filtered_df['created_date'].iloc[0]
        elif 'labels' in query.lower():
            return filtered_df['labels'].iloc[0]
        else:
            return "Query does not specify a valid field (summary, description, issue_type, priority, assignee, created_date, labels)."
    else:
        return "No issue key found in the query."

def handle_list_queries(query):
    list_keywords = {
        'standby': ['standby'],
        'server': ['server'],
        'high priority': ['high priority', 'high prio'],
        'bug': ['bug'],
        'ui': ['ui']
    }

    keyword_pattern = re.compile(r'related\s+to\s+(\w+)')
    match = keyword_pattern.search(query.lower())

    if match:
        keyword = match.group(1)
        filtered_rows = df[df.apply(lambda row: keyword in row['summary'].lower() or keyword in row['description'].lower(), axis=1)]
        if not filtered_rows.empty:
            return filtered_rows.to_dict(orient='records')
        else:
            return "No matching issues found."

    column_pattern = re.compile(r'(\w+)\s+as\s+(\w+)')
    match = column_pattern.search(query.lower())

    if match:
        column_name = match.group(1)
        column_content = match.group(2)

        filtered_rows = df[df[column_name].str.lower() == column_content.lower()]
        if not filtered_rows.empty:
            return filtered_rows.to_dict(orient='records')
        else:
            return f"No issues found with {column_name} as {column_content}."

    return "No relevant keywords or values found in the query."

def handle_complex_queries(query):
    query = query.lower()

    pattern = re.compile(r'what\s+are\s+the\s+(.+?)\s+(.+?)\s+issues\s+related\s+to\s+(.+)')

    column_normalization = {}

    for column in df.columns:
        if df[column].dtype == 'object':
            unique_values = df[column].str.lower().unique()
            normalization_map = {value.lower(): value for value in unique_values}
            column_normalization[column.lower()] = normalization_map

    match = pattern.match(query)
    
    if match:
        column_content = match.group(1).strip()
        column_name = match.group(2).strip()
        keyword = match.group(3).strip()

        if column_name.lower() in column_normalization:
            column_content_normalized = column_normalization[column_name.lower()].get(column_content.lower(), column_content)
        else:
            return "Query not recognized or conditions not met."

        if column_name.lower() in df.columns:
            filtered_df = df[(df[column_name].str.lower() == column_content_normalized.lower()) & 
                             (df.apply(lambda row: keyword.lower() in row['summary'].lower() or keyword.lower() in row['description'].lower(), axis=1))]
        else:
            return "Query not recognized or conditions not met."

        if not filtered_df.empty:
            return filtered_df.to_dict(orient='records')
        else:
            return f"No matching issues found with {column_content_normalized} {column_name} related to {keyword}."
    
    return "Query not recognized or conditions not met."

def detect_query_type(query):
    simple_keywords = ['summary', 'description', 'issue_type', 'priority', 'assignee', 'created_date', 'labels']
    list_pattern = re.compile(r'list\s+all\s+issues\s+related\s+to\s+(.+)', re.IGNORECASE)
    complex_pattern = re.compile(r'what\s+are\s+the\s+(.+?)\s+(.+?)\s+issues\s+related\s+to\s+(.+)', re.IGNORECASE)

    # Check for complex query first
    if complex_pattern.match(query.lower()):
        return "complex"

    # Check for list query pattern
    if list_pattern.match(query.lower()):
        return "list"

    # Check for simple query keywords
    for keyword in simple_keywords:
        if keyword in query.lower():
            return "simple"

    # Default to simple if no specific pattern is matched
    return "simple"



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_query = request.form['query']
    query_type = detect_query_type(user_query)

    if query_type == "simple":
        response = handle_simple_queries(user_query)
    elif query_type == "list":
        response = handle_list_queries(user_query)
    elif query_type == "complex":
        response = handle_complex_queries(user_query)
    else:
        response = "Unable to determine query type."

    if isinstance(response, list):
        return json.dumps(response, indent=4)
    else:
        return response

if __name__ == '__main__':
    learn_question_answer_pairs_from_file('qa_pairs.json')
    app.run(debug=True)
