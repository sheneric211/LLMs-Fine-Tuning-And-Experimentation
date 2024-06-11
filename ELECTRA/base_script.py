import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import Dataset, load_metric
import pandas as pd

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
model = AutoModelForSequenceClassification.from_pretrained(model_name)

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
        per_device_eval_batch_size=4,  # Use a standard batch size for evaluation
        report_to="none",
    ),
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

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

print("Base model evaluation completed.")
