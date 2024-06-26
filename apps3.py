import pandas as pd
import re
from collections import defaultdict
from transformers import pipeline
import json

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

def get_response(question):
    # Check if the question exists in learned pairs
    if question.lower() in qa_pairs:
        return qa_pairs[question.lower()]
    else:
        # If not found, use the existing logic to answer the question
        response = ask_question(question)
        return response

if __name__ == '__main__':
    # Learn question-answer pairs from JSON file
    learn_question_answer_pairs_from_file('qa_pairs.json')

    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Bot: Goodbye!")
            break
        response = get_response(user_query)
        if isinstance(response, list):
            for item in response:
                print("Bot:", item)
        else:
            print("Bot:", response)
