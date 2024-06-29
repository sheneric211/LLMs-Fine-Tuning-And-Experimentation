# LLM Fine-Tuning Experimentation

This project explores fine-tuning various pre-trained language models for sentiment analysis on the Amazon Polarity dataset. Models such as ELECTRA, GPT, and DistilBERT are fine-tuned using different techniques including transfer learning, LoRA (Low-Rank Adaptation), and supervised learning. The performance of these models is evaluated and compared across different hyperparameters.

## Setup

### Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Datasets
- scikit-learn
- pandas

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sheneric211/LLM-Fine-Tuning-Experimentation.git
    cd LLM-Fine-Tuning-Experimentation
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv myenv
    source myenv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install torch transformers datasets scikit-learn pandas
    ```

## Data Preparation

Generate the train and test CSV files from the Amazon Polarity dataset:

```bash
python dataset/create_csv.py
