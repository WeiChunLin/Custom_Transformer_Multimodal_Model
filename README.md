# PyTorch Multimodal Classification Model with Transformer Encoder

This repository provides the codebase for the multi-modal predictive model for glaucoma surgical outcomes. We utilize custom pre-trained word2vec model and transofrmer encoder to extract information from operative notes and combine it with static features. The code is written in Python and uses PyTorch for the deep learning components.

## Table of Contents
- [Data Structure](#data-structure)
- [Functionalities](#functionalities)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Code structure](#code-structure)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Training](#training)
- [Evaluation](#evaluation)
- [License](#license)

## Data Structure

The dataset should include:

- A target column named `output`, which contains three classes for prediction.
- A feature column named `clean-up operative notes` that includes the operative notes.
- 75 columns of structured EHR data.

## Functionalities

This codebase is designed to:

1. Import an Excel file into a Pandas DataFrame.
2. Split the feature and target variables into training, validation, and test sets (70%/10%/20%).
3. Initialize a ClinicalBERT tokenizer for text processing.
4. Tokenize the clean-up operative notes and create data loaders for training, validation, and testing.
5. Utilize an input dimension of [512, 75, 1], where:
   - The first dimension is the encoded input IDs from the custom word2vec.
   - The second dimension contains structured EHR (Electronic Health Records) data.
   - The third dimension holds the outcome labels.
6. Set up a PyTorch model class that combines a pre-trained word2vec and transformer encoder with static data.
7. Initialize and configure the `Multi_TF_Class` and move it to the specified device (CPU or GPU).
8. Set three functions: get_accuracy, train, and evaluate.
9. Start to train the PyTorch model using a specified set of parameters, optimizer, and loss function.
10. Evaluate the model on a test set to generate predictions with ROCs, P-R curves, and classification reports.

## Getting Started

### Prerequisites

Ensure that you have:
- Python 
- PyTorch
- Transformers library from Hugging Face
- Other requirements 

### Installation

To install the necessary packages, run the following command:

```bash
pip install pandas
```
## Code structure

- `Word2Vec.load()`: For loading pre-trained Word2Vec models.
- `PositionalEncoding`: Class for generating positional encoding.
- `Multi_TF_Class`: The main classification model class with transformer encoder.
- `train()`: Function to train the model.
- `evaluate()`: Function to evaluate the model.
- `get_accuracy()`: Function to compute accuracy during training and validation.
  
## Model Architecture

Our model leverages a pre-trained ClinicalBERT and adds custom layers for dimensionality reduction and combining with static data for classification tasks. The architecture is defined in the `Multi_BERT` class.

## Features

- Transformer Encoder Layer
- Positional Encoding
- Pretrained Word2Vec model for embeddings

## Training

The model is trained using a defined set of hyperparameters, a specified loss function, and an optimizer. 
- **Batch Size**: 16
- **Epochs**: 200
- **Learning Rate**: 4e-5
- **Weight Decay for L2 Regularization**: 1e-5
- **Class Weights**: [0.2584, 0.8678, 0.8737]

## Evaluation
The model is evaluated using a separate test dataset. Evaluation metrics include AUC (Area Under the Curve), Precision-Recall Curve, and Classification reports.

## License
This project is licensed under the MIT License.
