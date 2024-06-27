#tp prepare datase in a format that can be used by sentence tranformer model (datais in qa_PAIRS.json)
import json
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

def prepare_data(file_path):
    with open(file_path, 'r') as f:
        qa_data = json.load(f)

    examples = []
    for item in qa_data['question_answers']:
        question = item['question']
        for answer in item['answers']:
            text = f"Summary: {answer['summary']} Description: {answer['description']}"
            examples.append(InputExample(texts=[question, text], label=1.0))

    return examples

examples = prepare_data('qa_pairs.json')
with open('training_data.pkl', 'wb') as f:
    import pickle
    pickle.dump(examples, f)
