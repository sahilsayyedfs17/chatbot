from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import models, datasets
from torch.utils.data import DataLoader
import pickle

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the training data
with open('training_data.pkl', 'rb') as f:
    train_examples = pickle.load(f)

# Create a DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define the loss function
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

# Save the fine-tuned model
model.save('./output/fine-tuned-model')
