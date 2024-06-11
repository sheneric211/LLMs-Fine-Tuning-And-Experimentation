import pandas as pd
from datasets import load_dataset
import numpy as np

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Load the amazon-polarity dataset
dataset = load_dataset("amazon_polarity")

# Select 1000 examples for training and testing with a random seed
train_subset = dataset["train"].shuffle(seed=SEED).select(range(100))
test_subset = dataset["test"].shuffle(seed=SEED).select(range(100))

# Convert datasets to pandas DataFrame
train_df = pd.DataFrame(train_subset)
test_df = pd.DataFrame(test_subset)

# Save to CSV files
train_df.to_csv("train_subset.csv", index=False)
test_df.to_csv("test_subset.csv", index=False)

print("CSV files for train and test subsets have been created.")
