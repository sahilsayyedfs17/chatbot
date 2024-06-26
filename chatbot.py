import torch
from transformers import pipeline
import pandas as pd

# Initialize the TQA pipeline
tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")

def ask_question(query, table):
    # Ensure table is a DataFrame
    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame([table])
    answer = tqa(table=table, query=query)["answer"]
    return answer

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
        print(f"Bot: {response}")
