import json
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets
from torch.utils.data import DataLoader

# Load the question-answer pairs
with open('qa_pairs.json', 'r') as f:
    qa_data = json.load(f)

# Prepare the data for training
train_examples = []
for item in qa_data['question_answers']:
    question = item['question']
    answers = item['answers']
    for answer in answers:
        train_examples.append(InputExample(texts=[question, answer['summary']]))

# Define the model
model_name = 'all-MiniLM-L6-v2'
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Define the loss function
train_loss = losses.CosineSimilarityLoss(model)

# Create a DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Train the model
num_epochs = 4
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path='./output/fine-tuned-model'
)

print("Model training completed and saved at './output/fine-tuned-model'")
