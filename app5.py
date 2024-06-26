import pandas as pd
import re
from collections import defaultdict
from transformers import pipeline
import json
import difflib
from flask import Flask, request, jsonify, render_template

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
    try:
        with open(file_path, 'r') as f:
            qa_data = json.load(f)
            for item in qa_data['question_answers']:
                question = item['question']
                answers = item['answers']
                learn_question_answer_pair(question, answers)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON file: '{file_path}'")

def learn_question_answer_pair(question, answers):
    qa_pairs[question.lower()] = answers

def calculate_similarity(query, text):
    # Calculate similarity percentage using difflib
    matcher = difflib.SequenceMatcher(None, query.lower(), text.lower())
    return round(matcher.ratio() * 100, 2)

def get_response(question):
    # Check if the question exists in learned pairs
    if question.lower() in qa_pairs:
        return qa_pairs[question.lower()]
    else:
        # If not found, use the existing logic to answer the question
        response = ask_question(question)
        return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/retrieve', methods=['POST'])
def retrieve():
    user_query = request.form['query']
    results = get_response(user_query)
    return render_template('result.html', result=results, query_type='retrieval')

@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    user_query = request.form['query']
    
    # Check for similarity in summary and description columns
    combined_text = df['summary'] + " " + df['description']
    similarities = combined_text.apply(lambda x: calculate_similarity(user_query, x))
    max_similarity = similarities.max()
    index_of_max = similarities.idxmax()

    similarity_threshold = 20  # Adjust as needed
    if max_similarity >= similarity_threshold:
        response = df.iloc[index_of_max].to_dict()
        return render_template('result.html', result=response, similarity=max_similarity, query_type='similarity')
    else:
        return render_template('result.html', result=None, similarity=max_similarity, query_type='similarity')

if __name__ == '__main__':
    learn_question_answer_pairs_from_file('qa_pairs.json')
    app.run(debug=True)
