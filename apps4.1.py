import pandas as pd
import re
from collections import defaultdict
from transformers import pipeline
import json
import difflib
from flask import Flask, render_template, jsonify, request, redirect, url_for, session

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
    # Initialize regex patterns
    condition_pattern = re.compile(r'(\w+)\s+as\s+(\w+)')
    keyword_pattern = re.compile(r'related\s+to\s+(\w+)')

    # Initialize conditions and keyword
    conditions = defaultdict(list)
    keyword = None

    # Extract conditions and keyword from query
    for match in condition_pattern.finditer(query.lower()):
        column = match.group(1)
        value = match.group(2)
        mapped_column = map_synonym_to_column(column)
        if mapped_column:
            conditions[mapped_column].append(value)

    for match in keyword_pattern.finditer(query.lower()):
        keyword = match.group(1)

    # Filter the DataFrame based on conditions
    filtered_df = df.copy()
    for column, values in conditions.items():
        if column in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[column].str.lower().isin(values)]

    # Apply keyword filter
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
    # Calculate similarity percentage using difflib
    matcher = difflib.SequenceMatcher(None, query.lower(), text.lower())
    return round(matcher.ratio() * 100, 2)

def handle_simple_queries(query):
    # Regex pattern to match simple queries with issue key
    issue_key_pattern = re.compile(r'\b([A-Za-z0-9]+-[0-9]+)\b')

    match = issue_key_pattern.search(query)
    if match:
        issue_key = match.group(1)
        # Find the row in the DataFrame corresponding to the issue key
        filtered_df = df[df['issue_key'] == issue_key]

        if filtered_df.empty:
            return f"No information found for issue key {issue_key}"

        # Extract specific information based on user query
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
    # List of keywords to identify list queries
    list_keywords = {
        'standby': ['standby'],
        'server': ['server'],
        'high priority': ['high priority', 'high prio'],
        'bug': ['bug'],
        'ui': ['ui']
    }

    # Extract keyword from query
    keyword_pattern = re.compile(r'related\s+to\s+(\w+)')
    match = keyword_pattern.search(query.lower())

    if match:
        keyword = match.group(1)
        # Filter DataFrame based on keyword in summary or description
        filtered_rows = df[df.apply(lambda row: keyword in row['summary'].lower() or keyword in row['description'].lower(), axis=1)]
        if not filtered_rows.empty:
            return filtered_rows.to_dict(orient='records')
        else:
            return "No matching issues found."

    # Extract column-based queries
    column_pattern = re.compile(r'(\w+)\s+as\s+(\w+)')
    match = column_pattern.search(query.lower())

    if match:
        column_name = match.group(1)
        column_content = match.group(2)

        # Filter DataFrame based on column content value
        filtered_rows = df[df[column_name].str.lower() == column_content.lower()]
        if not filtered_rows.empty:
            return filtered_rows.to_dict(orient='records')
        else:
            return f"No issues found with {column_name} as {column_content}."

    return "No relevant keywords or values found in the query."

def handle_complex_queries(query):
    query = query.lower()

    # Regular expression pattern to match the query format
    pattern = re.compile(r'what\s+are\s+the\s+(.+?)\s+(.+?)\s+issues\s+related\s+to\s+(.+)')

    # Initialize an empty dictionary for column normalization
    column_normalization = {}

    # Populate column normalization mappings based on unique values in each column
    for column in df.columns:
        if df[column].dtype == 'object':  # Process only object/string columns
            unique_values = df[column].str.lower().unique()
            normalization_map = {value.lower(): value for value in unique_values}
            column_normalization[column.lower()] = normalization_map

    # Attempt to match the query pattern
    match = pattern.match(query)
    
    if match:
        column_content = match.group(1).strip()  # Extract column content (e.g., high)
        column_name = match.group(2).strip()     # Extract column name (e.g., priority)
        keyword = match.group(3).strip()         # Extract keyword (e.g., standby)

        # Normalize column content dynamically
        if column_name.lower() in column_normalization:
            column_content_normalized = column_normalization[column_name.lower()].get(column_content.lower(), column_content)
        else:
            return "Query not recognized or conditions not met."

        # Filter DataFrame based on the extracted criteria
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

# Learn question-answer pairs from JSON file
def learn_question_answer_pairs_from_file(file_path):
    with open(file_path, 'r') as f:
        qa_data = json.load(f)
        for item in qa_data['question_answers']:
            question = item['question']
            answers = item['answers']
            learn_question_answer_pair(question, answers)

def learn_question_answer_pair(question, answers):
    qa_pairs[question.lower()] = answers

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# Endpoint to render index1.html
@app.route('/')
def index():
    return render_template('index1.html')

# Endpoint to handle queries and redirect to result1.html
@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        user_query = data['query']

        # Depending on the type of query, invoke the appropriate handler function
        if 'high priority' in user_query or 'medium priority' in user_query or 'story' in user_query:
            response = handle_complex_queries(user_query)
        elif 'related to' in user_query:
            response = handle_list_queries(user_query)
        else:
            response = handle_simple_queries(user_query)

        # Prepare response data
        if isinstance(response, str):
            response = {'message': response}
        else:
            response = {'message': response}

        # Store response in session for result1.html
                # Store response in session for result1.html
        session['query_result'] = response

        # Redirect to result1.html
        return redirect(url_for('show_result'))

    except Exception as e:
        return jsonify({'message': str(e)}), 500  # Return the error message and status code 500

# Endpoint to render result1.html with query results
@app.route('/result')
def show_result():
    # Retrieve query result from session
    query_result = session.get('query_result', None)

    # Clear session data after retrieving
    session.pop('query_result', None)

    # Render result1.html with query result data
    return render_template('result1.html', query_result=query_result)

# Main driver function
if __name__ == '__main__':
    app.run(debug=True)

