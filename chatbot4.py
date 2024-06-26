from flask import Flask, render_template, request
import pandas as pd
import re
from collections import defaultdict
from transformers import pipeline
import json
import difflib
from sentence_transformers import SentenceTransformer, util

# Initialize Flask app
app = Flask(__name__)

# Initialize the TQA pipeline
tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")

# Load CSV into a DataFrame
df = pd.read_csv('sample jira extract.csv')

# Initialize the SentenceTransformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

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

def get_response(question):
    # Check if the question exists in learned pairs
    if question.lower() in qa_pairs:
        return qa_pairs[question.lower()]
    else:
        # If not found, use the existing logic to answer the question
        response = ask_question(question)
        return response

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

def handle_similarity_check(user_input, dataframe):
    # Encode the user input and the 'summary' and 'description' columns of the DataFrame
    user_input_embedding = sentence_model.encode(user_input, convert_to_tensor=True)
    summary_embeddings = sentence_model.encode(dataframe['summary'].tolist(), convert_to_tensor=True)
    description_embeddings = sentence_model.encode(dataframe['description'].tolist(), convert_to_tensor=True)
    
    # Calculate cosine similarity between user input and each row in the DataFrame
    summary_similarities = util.pytorch_cos_sim(user_input_embedding, summary_embeddings)
    description_similarities = util.pytorch_cos_sim(user_input_embedding, description_embeddings)
    
    # Find the most similar row based on the maximum similarity score
    combined_similarities = (summary_similarities + description_similarities) / 2
    most_similar_index = combined_similarities.argmax().item()
    most_similar_row = dataframe.iloc[most_similar_index]
    similarity_score = combined_similarities[0][most_similar_index].item() * 100  # Convert to percentage
    
    return most_similar_row, similarity_score

if __name__ == '__main__':
    # Learn question-answer pairs from JSON file
    learn_question_answer_pairs_from_file('qa_pairs.json')

    while True:
        print("Options:")
        print("1. Simple Queries (using TAPAS)")
        print("2. List Queries")
        print("3. Complex Queries")
        print("4. Exit")
        print("5. Check Similarity")
        user_choice = input("Your choice (1/2/3/4/5): ")

        if user_choice == "1":
            user_query = input("You: ")
            
            if user_query.lower() == "exit":
                print("Bot: Goodbye!")
                break
            
            response = handle_simple_queries(user_query)
            print(f"Bot: {response}")

        elif user_choice == "2":
            user_query = input("You: ")
            
            if user_query.lower() == "exit":
                print("Bot: Goodbye!")
                break
            
            response = handle_list_queries(user_query)
            if isinstance(response, list):
                for item in response:
                    print(f"Bot: {item}")
            else:
                print(f"Bot: {response}")

        elif user_choice == "3":
            user_query = input("You: ")
            
            if user_query.lower() == "exit":
                print("Bot: Goodbye!")
                break
            
            response = handle_complex_queries(user_query)
            if isinstance(response, list):
                for item in response:
                    print(f"Bot: {item}")
            else:
                print(f"Bot: {response}")

        elif user_choice == "4":
            print("Bot: Goodbye!")
            break
        
        elif user_choice == "5":
            user_query = input("Enter your issue to check similarity: ")
            
            if user_query.lower() == "exit":
                print("Bot: Goodbye!")
                break

            #most_similar_row, similarity_score = handle_similarity_check(user_query)
            most_similar_row, similarity_score = handle_similarity_check(user_query, df)
            
             
            print(f"Most similar issue: {most_similar_row.to_dict()}")
            print(f"Similarity score: {similarity_score:.2f}%")

        else:
            print("Bot: Invalid choice. Please choose between 1, 2, 3, 4, or 5.")
