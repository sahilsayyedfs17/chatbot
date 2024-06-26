import torch
from transformers import pipeline
import pandas as pd

# Initialize the TQA pipeline
tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")

def ask_question(query, table):
    # Ensure table is a DataFrame
    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame([table])
    
    # Attempt to process the query for complex cases
    filtered_table = process_complex_query(query, table)
    if filtered_table is not None and not filtered_table.empty:
        return filtered_table.to_dict(orient='records')
    else:
        # Fallback to using the pipeline for simple questions
        answer = tqa(table=table, query=query)["answer"]
        return answer

def process_complex_query(query, table):
    try:
        words = query.lower().split()
        columns = table.columns.str.lower()
        conditions = {}
        keyword = None

        i = 0
        while i < len(words):
            word = words[i].strip()
            if word in columns and i + 2 < len(words) and words[i + 1].strip() == 'as':
                column_name = word
                column_value = words[i + 2].strip()
                conditions[column_name] = column_value
                i += 3
            elif word == 'related' and i + 2 < len(words) and words[i + 1].strip() == 'to':
                keyword = words[i + 2].strip()
                i += 3
            else:
                i += 1

        # Apply column-specific filters
        filtered_table = table.copy()
        for column, value in conditions.items():
            filtered_table = filtered_table[filtered_table[column].str.lower().str.strip() == value.lower()]

        # Apply keyword filter across all columns if keyword is present
        if keyword:
            mask = filtered_table.apply(lambda row: row.astype(str).str.contains(keyword, case=False, na=False).any(), axis=1)
            filtered_table = filtered_table[mask]

        return filtered_table

    except Exception as e:
        print(f"Error processing complex query: {e}")
        return None
    return None

if __name__ == '__main__':
    # Read CSV file into a DataFrame
    table = pd.read_csv('sample jira extract.csv')
    table = table.astype(str)  # Convert all data to strings to ensure consistency

    # Interactive chat loop
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Bot: Goodbye!")
            break
        response = ask_question(user_query, table)
        if isinstance(response, list):
            if response:
                for item in response:
                    print(f"Bot: {item}")
            else:
                print("Bot: No matching records found.")
        else:
            print(f"Bot: {response}")
