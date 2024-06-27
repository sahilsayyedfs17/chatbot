from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
from collections import defaultdict
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering
import json
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize Flask app
app = Flask(__name__)

# Load models and data
model_path = "./models/tapas"
model = TapasForQuestionAnswering.from_pretrained(model_path)
tokenizer = TapasTokenizer.from_pretrained(model_path)
tqa = pipeline("table-question-answering", model=model, tokenizer=tokenizer)

df = pd.read_csv('sample jira extract.csv')

# Load the fine-tuned SentenceTransformer model
sentence_model = SentenceTransformer('./output/fine-tuned-model')

# Dictionary to store question-answer pairs
qa_pairs = {}

# Synonyms dictionary for column names
column_synonyms = {}
for column in df.columns:
    synonyms = [column.lower(), column.capitalize(), column.upper(), column]
    column_synonyms[column.lower()] = synonyms

# Manually update additional synonyms
column_synonyms.update({
    "priority": ["importance", "urgency", "Priority"],
    "summary": ["title", "heading", "Summary"],
    "description": ["details", "info", "Description"],
    "issue_type": ["type", "category", "Issue_type", "Issue Type", "issue type", "Issue type"],
    "assignee": ["assigned_to", "responsible", "assigned to", "assignee", "asign"],
    "created_date": ["date_created", "creation_date", "it created"],
    "labels": ["tags", "label"],
    "issue_key": ["issue key", "issue", "Issue Key", "Issue key", "ticket no", "Ticket no", "Ticket", "Ticked Id", "ticket Id", "ticket id"],
    "reporter": ["Reporter", "reported", "has raised"],
    "status": ["Status", "state"]
})

def map_synonym_to_column(word):
    word_lower = word.lower()
    for column, synonyms in column_synonyms.items():
        if word_lower == column or word_lower in map(str.lower, synonyms):
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

def get_response(question):
    if question.lower() in qa_pairs:
        return qa_pairs[question.lower()]
    else:
        response = ask_question(question)
        return response

def handle_query(query):
    query_lower = query.lower()
    
    issue_key_pattern = re.compile(r'\b([A-Za-z0-9]+-[0-9]+)\b')
    list_keyword_pattern = re.compile(r'related\s+to\s+(\w+)')
    column_pattern = re.compile(r'(\w+)\s+as\s+(\w+)')
    complex_pattern = re.compile(r'what\s+are\s+the\s+(.+?)\s+(.+?)\s+issues\s+related\s+to\s+(.+)')

    match = issue_key_pattern.search(query)
    if match:
        issue_key = match.group(1)
        filtered_df = df[df['issue_key'] == issue_key]

        if filtered_df.empty:
            return f"No information found for issue key {issue_key}"

        for word in query.split():
            column = map_synonym_to_column(word)
            if column:
                return filtered_df[column].iloc[0]

        return "Query does not specify a valid field (summary, description, issue_type, priority, assignee, created_date, labels)."
    
    match = complex_pattern.match(query_lower)
    if match:
        column_content = match.group(1).strip()
        column_name = match.group(2).strip()
        keyword = match.group(3).strip()

        column_mapped = map_synonym_to_column(column_name)
        if not column_mapped:
            return f"Column '{column_name}' not recognized."

        column_normalization = {}
        for column in df.columns:
            if df[column].dtype == 'object':
                unique_values = df[column].str.lower().unique()
                normalization_map = {value.lower(): value for value in unique_values}
                column_normalization[column.lower()] = normalization_map

        if column_mapped in column_normalization:
            column_content_normalized = column_normalization[column_mapped].get(column_content.lower(), column_content)
        else:
            return "Query not recognized or conditions not met."

        if column_mapped in df.columns:
            filtered_df = df[(df[column_mapped].str.lower() == column_content_normalized.lower()) & 
                             (df.apply(lambda row: keyword.lower() in row['summary'].lower() or keyword.lower() in row['description'].lower(), axis=1))]
        else:
            return "Query not recognized or conditions not met."

        if not filtered_df.empty:
            return filtered_df.to_dict(orient='records')
        else:
            return f"No matching issues found with {column_content_normalized} {column_mapped} related to {keyword}."

    match = list_keyword_pattern.search(query_lower)
    if match:
        keyword = match.group(1)
        filtered_rows = df[df.apply(lambda row: keyword in row['summary'].lower() or keyword in row['description'].lower(), axis=1)]
        if not filtered_rows.empty:
            return filtered_rows.to_dict(orient='records')
        else:
            return "No matching issues found."

    match = column_pattern.search(query_lower)
    if match:
        column_name = match.group(1)
        column_content = match.group(2)

        column_name_mapped = map_synonym_to_column(column_name)
        if column_name_mapped:
            filtered_rows = df[df[column_name_mapped].str.lower() == column_content.lower()]
            if not filtered_rows.empty:
                return filtered_rows.to_dict(orient='records')
            else:
                return f"No matching issues found with {column_name_mapped} as {column_content}."

    # Use the fine-tuned model for similarity search
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    corpus = list(qa_pairs.keys())
    corpus_embeddings = sentence_model.encode(corpus, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)

    if hits and hits[0]:
        most_similar_question = corpus[hits[0][0]['corpus_id']]
        similarity_score = hits[0][0]['score']
        if similarity_score > 0.7:
            return qa_pairs[most_similar_question]
    
    return get_response(query)

@app.route("/")
def home():
    return render_template("index3temp.html")

@app.route("/get_answer", methods=["POST"])
def get_answer():
    user_input = request.form.get("query")
    if user_input:
        response = handle_query(user_input)
        return jsonify(response)
    return jsonify({"error": "No input provided"})

if __name__ == "__main__":
    learn_question_answer_pairs_from_file('qa_pairs.json')
    app.run(debug=True)
