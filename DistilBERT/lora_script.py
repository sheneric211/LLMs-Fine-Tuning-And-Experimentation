import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import Dataset, load_metric
import pandas as pd
import numpy as np

# Define low-rank adaptation layer
class LowRankAdapter(nn.Module):
    def __init__(self, input_dim, rank):
        super(LowRankAdapter, self).__init__()
        self.down_proj = nn.Linear(input_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, input_dim, bias=False)
    
    def forward(self, x):
        return self.up_proj(self.down_proj(x))

# Function to add low-rank adapters to the model
def add_adapters_to_model(model, rank=8):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            adapter = LowRankAdapter(module.in_features, rank)
            module.add_module("adapter", adapter)
        elif len(list(module.children())) > 0:
            add_adapters_to_model(module, rank)
    return model

# Clear GPU cache
torch.cuda.empty_cache()

# Load the datasets from CSV files
train_df = pd.read_csv("train_subset.csv")
test_df = pd.read_csv("test_subset.csv")

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['content'], truncation=True, padding='max_length')

# Load the tokenizer and model
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Add low-rank adapters to the model
model = add_adapters_to_model(model, rank=8)

# Tokenize the dataset
train_tokenized = train_dataset.map(preprocess_function, batched=True)
test_tokenized = test_dataset.map(preprocess_function, batched=True)

# Define the data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Define compute_metrics function
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")
f1_metric = load_metric("f1")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    precision = precision_metric.compute(predictions=preds, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=preds, references=labels, average="weighted")
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,  # Adjust based on available GPU memory
        per_device_eval_batch_size=4,  # Adjust based on available GPU memory
        gradient_accumulation_steps=2,  # Simulate larger batch size
        learning_rate=2e-5,
        num_train_epochs=5,
        logging_steps=10,
        save_steps=10,
        evaluation_strategy="steps",
        fp16=True,  # Enable mixed precision training
        report_to="none",  # Disable logging to avoid issues
    ),
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Evaluate the model
predictions = trainer.predict(test_tokenized)

# Process outputs to get class labels
outputs = np.argmax(predictions.predictions, axis=1)

# Calculate and save metrics
metrics = compute_metrics(predictions)
with open("metrics.txt", "w") as f:
    f.write("Base Model Performance:\n")
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")