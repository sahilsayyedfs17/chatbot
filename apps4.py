import pandas as pd
import re
from transformers import pipeline
import json
import difflib

# Initialize the TAPAS pipeline
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
    "labels": ["tags"],
    "issue_key": ["key", "id"]
}

def map_synonym_to_column(word):
    word_lower = word.lower()
    for column, synonyms in column_synonyms.items():
        if word_lower == column or word_lower in synonyms:
            return column
    return None

def parse_conditions(query):
    conditions = []
    keywords = re.split(r'\s+&&\s+|\s+\|\|\s+', query)
    operators = re.findall(r'\s+&&\s+|\s+\|\|\s+', query)
    
    for keyword in keywords:
        negated = False
        if keyword.strip().startswith('not '):
            negated = True
            keyword = keyword.strip()[4:]
        
        column, value = None, None
        if ' ' in keyword:
            parts = keyword.split(' ', 1)
            column = map_synonym_to_column(parts[0])
            value = parts[1].strip().strip("'")
        else:
            value = keyword.strip().strip("'")
        
        if column:
            conditions.append((column, value, negated))
        else:
            conditions.append((None, value, negated))
    
    return conditions, operators

def filter_dataframe(df, conditions, operators):
    results = []
    for column, value, negated in conditions:
        if column:
            if negated:
                results.append(~df[column].str.contains(value, case=False, na=False))
            else:
                results.append(df[column].str.contains(value, case=False, na=False))
        else:
            if negated:
                results.append(~df.apply(lambda row: row.astype(str).str.contains(value, case=False).any(), axis=1))
            else:
                results.append(df.apply(lambda row: row.astype(str).str.contains(value, case=False).any(), axis=1))
    
    if not results:
        return df
    
    final_result = results[0]
    for i in range(1, len(results)):
        if operators[i-1].strip() == '&&':
            final_result = final_result & results[i]
        elif operators[i-1].strip() == '||':
            final_result = final_result | results[i]
    
    return df[final_result]

def ask_question(query):
    conditions, operators = parse_conditions(query)
    
    filtered_df = filter_dataframe(df, conditions, operators)
    
    return filtered_df.to_dict(orient='records')

def calculate_similarity(query, text):
    # Calculate similarity percentage using difflib
    matcher = difflib.SequenceMatcher(None, query.lower(), text.lower())
    return round(matcher.ratio() * 100, 2)

def get_response(question):
    # Check if the question exists in learned pairs
    if question.lower() in qa_pairs:
        return qa_pairs[question.lower()]
    else:
        # Use TAPAS model for answering tabular questions
        inputs = {
            "query": question,
            "table": {
                "columns": df.columns.tolist(),
                "rows": df.values.tolist()
            }
        }
        response = tqa(**inputs)
        
        # Extract the relevant data from the response
        answers = [answer['answer'] for answer in response]
        return answers

def learn_question_answer_pairs_from_file(file_path):
    with open(file_path, 'r') as f:
        qa_data = json.load(f)
        for item in qa_data['question_answers']:
            question = item['question']
            answers = item['answers']
            learn_question_answer_pair(question, answers)

def learn_question_answer_pair(question, answers):
    qa_pairs[question.lower()] = answers

if __name__ == '__main__':
    # Learn question-answer pairs from JSON file
    learn_question_answer_pairs_from_file('qa_pairs.json')

    while True:
        print("Options:")
        print("1. Retrieval")
        print("2. Checking similarity of issue")
        user_choice = input("Your choice (1/2): ")

        if user_choice == "1":
            user_query = input("You: ")
            
            if user_query.lower() == "exit":
                print("Bot: Goodbye!")
                break
            
            response = get_response(user_query)
            if response:
                if isinstance(response, list):
                    if len(response) > 1 or not any('issue_key' in item for item in response):
                        # Display the whole row in a tabular form
                        print("Bot: Here are the matching issues:")
                        print(f"{'Issue Key':<10} {'Summary':<40} {'Description':<50} {'Issue Type':<10} {'Priority':<10} {'Assignee':<30} {'Created Date':<12} {'Labels':<15}")
                        for item in response:
                            print(f"{item['issue_key']:<10} {item['summary']:<40} {item['description']:<50} {item['issue_type']:<10} {item['priority']:<10} {item['assignee']:<30} {item['created_date']:<12} {item['labels']:<15}")
                    else:
                        for item in response:
                            if 'issue_key' in item and 'summary' in item:
                                print(f"Bot: The summary of {item['issue_key']} is '{item['summary']}'.")
                            elif 'issue_key' in item and 'assignee' in item:
                                print(f"Bot: Issue {item['issue_key']} is assigned to {item['assignee']}.")
                            elif 'issue_key' in item and 'priority' in item:
                                print(f"Bot: Issue {item['issue_key']} has priority '{item['priority']}'.")
                            else:
                                print(f"Bot: {item}")
                else:
                    print(f"Bot: {response}")
            else:
                print("Bot: No matching results found.")
        
        elif user_choice == "2":
            user_query = input("You: ")
            
            if user_query.lower() == "exit":
                print("Bot: Goodbye!")
                break
            
            # Check for similarity in summary and description columns
            combined_text = df['summary'] + " " + df['description']
            similarities = combined_text.apply(lambda x: calculate_similarity(user_query, x))
            max_similarity = similarities.max()
            index_of_max = similarities.idxmax()

            # Adjust the similarity threshold
            similarity_threshold = 20  # Adjust as needed
            if max_similarity >= similarity_threshold:
                print(f"Bot: Similarity to the closest entry: {max_similarity}%")
                print("Bot: Here is the closest matching issue:")
                item = df.iloc[index_of_max]
                print(f"{'Issue Key':<10} {'Summary':<40} {'Description':<50} {'Issue Type':<10} {'Priority':<10} {'Assignee':<30} {'Created Date':<12} {'Labels':<15}")
                print(f"{item['issue_key']:<10} {item['summary']:<40} {item['description']:<50} {item['issue_type']:<10} {item['priority']:<10} {item['assignee']:<30} {item['created_date']:<12} {item['labels']:<15}")
            else:
                print(f"Bot: No similar entry found.")

        else:
            print("Bot: Invalid choice. Please choose either 1 or 2.")
