import os
import torch
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

# Set the environment variable to limit GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Clear GPU cache
torch.cuda.empty_cache()
# File to store metrics
metrics_file = "lora_metrics.txt"

# Training configurations
epochs_list = [1, 3, 5, 10]
learning_rates = [2e-5, 3e-5, 5e-5]
batch_sizes = [1, 4, 8]
# Load the datasets from CSV files
train_df = pd.read_csv("train_subset.csv")
test_df = pd.read_csv("test_subset.csv")

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

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


# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['content'], truncation=True, padding='max_length')

# Load the tokenizer and model

# Define the data collator
# data_collator = DataCollatorWithPadding(tokenizer)


with open(metrics_file, "a") as f:
    for num_epochs in epochs_list:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                print(f"Starting LoRA training: Epochs={num_epochs}, Learning Rate={lr}, Batch Size={batch_size}")
                try:
                    # Load the tokenizer and model
                    model_name = "albert-base-v2"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)

                    # Add low-rank adapters to the model
                    model = add_adapters_to_model(model, rank=8)

                    # Tokenize the dataset
                    train_tokenized = train_dataset.map(preprocess_function, batched=True)
                    test_tokenized = test_dataset.map(preprocess_function, batched=True)

                    # Define the data collator
                    data_collator = DataCollatorWithPadding(tokenizer)

                    # Initialize Trainer
                    trainer = Trainer(
                        model=model,
                        args=TrainingArguments(
                            output_dir="./results",
                            overwrite_output_dir=True,
                            per_device_train_batch_size=batch_size,
                            per_device_eval_batch_size=batch_size,
                            gradient_accumulation_steps=2,
                            learning_rate=lr,
                            num_train_epochs=num_epochs,
                            logging_steps=10,
                            save_steps=10,
                            evaluation_strategy="steps",
                            fp16=True,
                            report_to="none",
                        ),
                        train_dataset=train_tokenized,
                        eval_dataset=test_tokenized,
                        compute_metrics=compute_metrics,
                        data_collator=data_collator,
                    )

                    # Train the model
                    trainer.train()

                    # Evaluate the model
                    predictions = trainer.predict(test_tokenized)

                    # Process outputs to get class labels
                    outputs = np.argmax(predictions.predictions, axis=1)

                    # Calculate and save metrics
                    metrics = compute_metrics(predictions)
                    f.write(f"Epochs: {num_epochs}, Learning Rate: {lr}, Batch Size: {batch_size}\n")
                    for key, value in metrics.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")

                    print(f"Completed LoRA: Epochs={num_epochs}, LR={lr}, Batch Size={batch_size}")
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f"Skipping: Epochs={num_epochs}, LR={lr}, Batch Size={batch_size} due to OOM")
                    else:
                        raise e

                # Clear GPU cache after each run
                torch.cuda.empty_cache()

print("All LoRA configurations have been processed.")