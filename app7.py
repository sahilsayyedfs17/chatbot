import pandas as pd
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Parse the CSV file with normalized column names
def parse_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.lower() for col in df.columns]
    return df

jira_issues_df = parse_csv('sample jira extract.csv')

# Parse the JSON file
def parse_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

qa_pairs = parse_json('qa_pairs.json')

# Prepare the QA dataset
from datasets import Dataset

def prepare_qa_dataset(qa_pairs):
    data = {
        'question': [],
        'answer': []
    }
    
    for qa_pair in qa_pairs['question_answers']:
        question = qa_pair['question']
        for answer in qa_pair['answers']:
            data['question'].append(question)
            formatted_answer = (f"Issue Key: {answer['issue_key']}\n"
                                f"Summary: {answer['summary']}\n"
                                f"Description: {answer['description']}\n"
                                f"Issue Type: {answer['issue_type']}\n"
                                f"Priority: {answer['priority']}\n"
                                f"Assignee: {answer['assignee']}\n"
                                f"Created Date: {answer['created_date']}\n"
                                f"Labels: {answer['labels']}")
            data['answer'].append(formatted_answer)
    
    return Dataset.from_dict(data)

qa_dataset = prepare_qa_dataset(qa_pairs)

# Fine-tune the model
model_name = "t5-small"  # You can use any other suitable model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = [q for q in examples["question"]]
    targets = [a for a in examples["answer"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids

    model_inputs["labels"] = labels
    return model_inputs

tokenized_qa_dataset = qa_dataset.map(preprocess_function, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_qa_dataset,
    eval_dataset=tokenized_qa_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Define utility functions for querying and formatting issues
def find_issues(jira_issues_df, condition):
    results = []
    for _, issue in jira_issues_df.iterrows():
        if condition(issue):
            results.append(issue)
    return results

def format_issue(issue):
    return (f"Issue Key: {issue.get('issue key', 'N/A')}\n"
            f"Summary: {issue.get('summary', 'N/A')}\n"
            f"Description: {issue.get('description', 'N/A')}\n"
            f"Issue Type: {issue.get('issue type', 'N/A')}\n"
            f"Priority: {issue.get('priority', 'N/A')}\n"
            f"Assignee: {issue.get('assignee', 'N/A')}\n"
            f"Created Date: {issue.get('created', 'N/A')}\n"
            f"Labels: {issue.get('labels', 'N/A')}")

def generate_response(question, model, tokenizer, jira_issues_df):
    inputs = tokenizer(question, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Add logic to dynamically query the CSV based on the answer or question
    question_lower = question.lower()
    if "high priority issues" in question_lower:
        def condition(issue):
            return (
                'priority' in issue and issue['priority'].lower() == 'high' and
                (
                    ('summary' in issue and 'standby' in issue['summary'].lower()) if "standby" in question_lower else
                    ('labels' in issue and 'security' in issue['labels'].lower()) if "security" in question_lower else
                    True
                )
            )
        results = find_issues(jira_issues_df, condition)
        dynamic_answer = "\n\n".join([format_issue(issue) for issue in results])
        return dynamic_answer

    # Add more dynamic querying logic based on the patterns learned from qa_pairs.json
    
    return answer

# Example usage
question = "What is the summary of AAA-789?"
response = generate_response(question, model, tokenizer, jira_issues_df)
print(response)
