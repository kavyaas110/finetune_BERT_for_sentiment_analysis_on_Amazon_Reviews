# BERT Fine-Tuning for Sentiment Analysis

Fine-tuning a pretrained **BERT** model for multi-class sentiment analysis on the **Stanford Amazon Reviews dataset**.

The project explores using BERT to classify Amazon product reviews as **negative, neutral, or positive**, with a focus on data preprocessing, BERT tokenization, optimizer selection, fine-tuning, learning-rate tuning, and model evaluation.

## Overview

BERT (Bidirectional Encoder Representations from Transformers) is a pretrained transformer-based language model that learns contextual representations of text.

Instead of training a language model from scratch, this project fine-tunes a pretrained BERT model for sentiment classification.

The overall pipeline consists of:

1. Dataset preprocessing
2. BERT tokenization
3. Model and optimizer setup
4. Fine-tuning BERT on Amazon reviews
5. Model evaluation
6. Hyperparameter tuning and convergence analysis

## Dataset

The project uses the **Stanford Amazon Reviews dataset**.

For sentiment classification, the primary fields used are:

- `review/text` — textual content of the review
- `review/score` — Amazon star rating

The original dataset is approximately **11.7 GB**, so it is processed in streaming mode and converted into smaller subsets for experimentation.

| Dataset Size | Number of Reviews |
|---|---:|
| 1 MB | 1,338 |
| 10 MB | 12,585 |
| 100 MB | 122,252 |
| 1 GB | 1,225,518 |

### Sentiment Labels

Amazon ratings are mapped to three sentiment classes:

| Rating | Sentiment | Label |
|---|---|---:|
| 1–2 stars | Negative | 0 |
| 3 stars | Neutral | 1 |
| 4–5 stars | Positive | 2 |

## Data Preprocessing

Review text is cleaned before being passed to BERT.

The preprocessing pipeline includes:

- Normalizing whitespace
- Removing URLs
- Removing unsupported special characters
- Converting text to lowercase
- Removing leading and trailing whitespace

The dataset is split using stratified sampling to preserve the sentiment distribution:

- **70% Training**
- **15% Validation**
- **15% Testing**

## BERT Tokenization

The project uses the Hugging Face tokenizer for `bert-base-uncased`.

Reviews are converted into the input representation expected by BERT using:

- WordPiece tokenization
- `[CLS]` and `[SEP]` special tokens
- Attention masks
- Padding
- Truncation

The maximum input sequence length is **512 tokens**.

## Model Architecture

The sentiment classifier is implemented using `BertForSequenceClassification` with the pretrained `bert-base-uncased` model.

The classification problem contains three output classes:

```text
Amazon Review
     |
     v
BERT Tokenizer
     |
     v
bert-base-uncased
     |
     v
Classification Head
     |
     v
Negative / Neutral / Positive
```

## Training

The model is fine-tuned using **PyTorch** and the **AdamW optimizer**.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Model | `bert-base-uncased` |
| Optimizer | AdamW |
| Learning Rate | `2e-5` |
| Batch Size | 16 |
| Epochs | 4 |
| Maximum Sequence Length | 512 |
| Gradient Clipping | 1.0 |
| Number of Classes | 3 |

### Training Results

A representative training run produced:

| Epoch | Training Loss | Validation Accuracy | Validation Loss |
|---:|---:|---:|---:|
| 1 | 0.63 | 82% | 0.51 |
| 2 | 0.39 | 88% | 0.33 |
| 3 | 0.24 | 90% | 0.30 |
| 4 | 0.13 | 90% | 0.32 |

The results show decreasing training loss and improving validation accuracy as BERT adapts to the sentiment classification task.

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

A test run achieved an overall accuracy of approximately **87.1%**.

| Sentiment | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.72 | 0.54 | 0.62 |
| Neutral | 0.40 | 0.46 | 0.43 |
| Positive | 0.93 | 0.95 | 0.94 |

The model performs particularly well on positive reviews, while neutral and negative reviews are more challenging.

## Optimizer and Hyperparameter Analysis

The project also investigates optimization choices for BERT fine-tuning.

Experiments include:

- Adam optimizer
- AdamW optimizer
- Learning-rate tuning
- Training convergence analysis
- Performance comparison across configurations

The effect of different learning rates is evaluated using convergence and performance plots.

## Tech Stack

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **BERT**
- **scikit-learn**
- **pandas**
- **NumPy**
- **Google Colab / Jupyter Notebook**

## Installation

Install the required dependencies:

```bash
pip install torch transformers pandas numpy scikit-learn gdown tabulate
```

## Running the Project

The complete implementation is available in:

```text
Finetune_BERT.ipynb
```

Run the notebook sequentially to:

1. Download and process the Amazon Reviews dataset
2. Generate smaller dataset subsets
3. Clean and label review text
4. Split the dataset into training, validation, and test sets
5. Tokenize reviews using BERT
6. Initialize the pretrained BERT model
7. Fine-tune the model
8. Evaluate the model
9. Analyze convergence and performance

GPU acceleration is recommended for BERT fine-tuning.

## Project Structure

```text
.
├── Finetune_BERT.ipynb
├── Midterm Project 2.pptx
├── README.md
└── data/
```

- `Finetune_BERT.ipynb` — dataset processing, BERT fine-tuning, and evaluation
- `Midterm Project 2.pptx` — project methodology, experiments, and results
- `data/` — generated Amazon review subsets

> **Note:** The original Amazon Reviews dataset and generated CSV subsets are large and are not intended to be committed to the repository.

## References

- Devlin et al., **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
- Vaswani et al., **Attention Is All You Need**
- Stanford Amazon Reviews Dataset
- Hugging Face Transformers
