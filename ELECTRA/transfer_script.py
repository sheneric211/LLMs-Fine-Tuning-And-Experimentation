import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import Dataset, load_metric
import pandas as pd
import numpy as np

# Define a custom ELECTRA model with task-specific layers
class CustomElectraModel(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(CustomElectraModel, self).__init__()
        self.electra = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.electra.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.electra.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[0][:,0,:]  # Use the [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.electra.config.num_labels), labels.view(-1))
        return (loss, logits) if loss is not None else logits

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
model_name = "google/electra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = CustomElectraModel(model_name)

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

    # Ensure the number of predictions matches the number of references
    if len(preds) != len(labels):
        min_len = min(len(preds), len(labels))
        preds = preds[:min_len]
        labels = labels[:min_len]

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

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
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
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the model
model.electra.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Evaluate the model
predictions = trainer.predict(test_tokenized)

# Process outputs to get class labels
outputs = np.argmax(predictions.predictions, axis=1)

# Calculate and save metrics
metrics = compute_metrics(predictions)
with open("metrics.txt", "w") as f:
    f.write("Transfer Learning Model Performance:\n")
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")

print("Transfer learning model evaluation completed.")
