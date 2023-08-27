#Import packages
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
from sklearn import metrics
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC
import time
import copy
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import math 
import gensim
from gensim.models import Word2Vec,KeyedVectors
from gensim.models.phrases import Phraser, Phrases, ENGLISH_CONNECTOR_WORDS
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.metrics import roc_auc_score, make_scorer, precision_score, recall_score, f1_score

'''
Sets random seeds to ensure reproducibility across different runs.
'''
random_seed = 101 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print(device)

"""
Script for Data Preprocessing and Tokenizer Initialization

This script reads an Excel file, splits the dataset into training, validation, and test sets, and initializes custom Word2Vec model.

Data Structure:
    - The target variable (label) for prediction is stored in a column named 'output' with 3 classes.
    - The first feature column is clean-up operative notes.

Functionalities:
    1. Reads the Excel file into a Pandas DataFrame.
    2. Splits the DataFrame into feature and label sets.
    3. Further splits these into training, validation, and test sets.
    4. Outputs the shapes of these datasets for verification.
    5. Separates dynamic features from static features.
    6. Initializes a custom tokenizer for further processing.
    7. Tokenize the notes and create training, validation, and testing dataloaders.
    8. Input dimensions are [512, 75, 1].
    9. The first dimension represents the encoded input IDs for the custom Word2Vec model.
    10. The third dimension represents the structured EHR data.
    11. The fourth dimension represents the outcome labels.
"""

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel('glaucoma_surgery_dataset.xlsx', sheet_name='Sheet1')

# Isolate the target variable (label) which we want to predict
labels = df['output']
# Remove the target variable from the feature set; axis=1 means we drop a column not a row
features = df.drop('output', axis=1)

# Split the data into a training set and a temporary validation/test set
# We're using 30% of the data for the temporary validation/test set, stratified by the label
train_features, val_test_features, train_labels, val_test_labels = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

# Further split the temporary validation/test set into validation and test sets
# 2/3 of the data goes to the test set, stratified by the label
val_features, test_features, val_labels, test_labels = train_test_split(val_test_features, val_test_labels, test_size=(2/3), random_state=42, stratify=val_test_labels)

# Display shapes to ensure everything is as expected
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Val Features Shape:', val_features.shape)
print('Val Labels Shape:', val_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Select static features by dropping the first column from the train, validation and test feature sets
train_static = train_features.iloc[:, 1:]
val_static = val_features.iloc[:, 1:]
test_static = test_features.iloc[:, 1:]

# Load the pre-trained Word2Vec model
word2vec = Word2Vec.load("word2vec_D50_May.model")

# Set the size of the word embeddings
n_embedding = 50

#Tokenize the operative notes using custom Word2Vec model.

#Generate the dataset and dataloader using TensorDataset and DataLoader from PyTorch.

"""Start to train the model"""

# Setting up hyperparameters and initial configurations
batchSize = 16  # Batch size for training
epochs = 200 # Number of epochs to train for
LEARNING_RATE = 4e-5  # Learning rate for the optimizer
WEIGHT_DECAY = 1e-5  # Weight decay factor for L2 regularization

# Initialize class weights and move to GPU
# These weights are useful if the dataset is imbalanced.
class_weights = torch.tensor([0.2584, 0.8678, 0.8737]).to(device)  

# Loss Function: Cross Entropy Loss with class weights
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer: Adam optimizer with learning rate and weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Ground truth labels for the validation set
y_val = val_labels.tolist()

# Start training the model
train_losses, val_loss = train(model, optimizer, loss_fn, train_dataloader, val_dataloader, y_val, epochs, batchSize)

# Plotting the training and validation loss
plt.plot(train_losses, label='train loss')
plt.plot(val_loss, label='test loss')
plt.legend()
plt.show()

# Saving the trained model weights to disk
PATH = 'custom-Transformer-Mulimodal.pth'
torch.save(model.state_dict(), PATH)

class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding used in the Transformer model.
    The code is inspired from PyTorch's tutorial on Transformers.
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        """
        Initialize the PositionalEncoding module.

        Parameters:
            d_model (int): Dimension of the model
            vocab_size (int): Size of the vocabulary
            dropout (float): Dropout rate
        """
        super().__init__()
        
        # Initialize dropout layer with given dropout rate
        self.dropout = nn.Dropout(p=dropout)

        # Initialize positional encoding matrix with zeros
        pe = torch.zeros(vocab_size, d_model)

        # Generate positions from 0 to vocab_size - 1
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)

        # Calculate the terms to be divided for sine and cosine
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # Generate sine and cosine positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension for batch size
        pe = pe.unsqueeze(0)

        # Register the tensor 'pe' as a buffer to be part of the model
        # It will not be updated during backpropagation
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass for positional encoding.
        Parameters:
            x (Tensor): The input sequence of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor: Positionally-encoded input
        """
        # Add the positional encoding to the input tensor 'x'
        # Note: ': x.size(1)' is used to match the sequence length of 'x'
        x = x + self.pe[:, : x.size(1), :]

        # Apply dropout for regularization
        return self.dropout(x)

class Multi_TF_Class (nn.Module):
    """
    Text classifier that uses a Transformer encoder along with static features.
    """

    def __init__(self,
                 embedding,
                 nhead=10,
                 dim_feedforward=512,
                 num_layers=6,
                 dropout=0.1,
                 classifier_dropout=0.1,
                 n_node_layer1=160,
                 n_node_layer2=32,
                 static_size=75):
        """
        Initialize the model.

        Parameters:
            embedding (Tensor): Pre-trained word embeddings
            nhead (int): Number of attention heads
            dim_feedforward (int): Dimension of the feedforward network
            num_layers (int): Number of transformer layers
            dropout (float): Dropout rate for transformer
            classifier_dropout (float): Dropout rate for classifier layers
            n_node_layer1 (int): Number of nodes in the first hidden layer
            n_node_layer2 (int): Number of nodes in the second hidden layer
            static_size (int): Size of the static feature vector
        """
        
        super(Multi_TF_Class, self).__init__()

        vocab_size, d_model = embedding.shape
        
        # Ensure that the number of heads divides evenly into the model dimension
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        # Embedding Layer
        self.emb = nn.Embedding.from_pretrained(embedding, freeze=False, padding_idx=0)
        
        # Positional Encoding Layer
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=vocab_size,
        )
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Static feature settings
        self.static_size = static_size
        self.d_model = d_model
        
        # Batch Normalization Layers
        self.batchnorm1 = nn.BatchNorm1d(n_node_layer1, momentum=0.1)
        self.batchnorm2 = nn.BatchNorm1d(n_node_layer2, momentum=0.1)
        
        # Fully Connected Layers
        self.linear1 = nn.Linear(d_model + static_size, n_node_layer1)
        self.linear2 = nn.Linear(n_node_layer1, n_node_layer2)
        
        # Activation and Dropout Layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=classifier_dropout)
        
        # Final Classifier Layer
        self.classifier = nn.Linear(n_node_layer2, 3)

    def forward(self, x, x_static):
        """
        Forward pass of the model.

        Parameters:
            x (Tensor): Textual input sequences
            x_static (Tensor): Static features

        Returns:
            Tensor: Classifier output
        """

        # Embedding and Positional Encoding
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Take the mean over the sequence dimension
        TF_out = x.mean(dim=1)
        
        # Concatenate the static features
        inputs = torch.cat([TF_out, x_static], dim=1)
        
        # Fully Connected Layers with activations, batch normalization, and dropout
        out = self.linear1(inputs)
        out = self.relu(out)
        out = self.batchnorm1(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.relu(out)
        out = self.batchnorm2(out)
        out = self.dropout(out)
        
        # Classifier
        out = self.classifier(out)

        return out

# Convert the pretrained weight array to a PyTorch tensor
# Dimension of pretrained_weight = 50
pretrained_weight = torch.FloatTensor(pretrained_weight)

# Instantiate the model with the various hyperparameters and the pretrained weights
model = Multi_TF_Class(
    pretrained_weight,       # pretrained embeddings
    nhead=10,                # number of heads in multihead attention
    dim_feedforward=768,     # dimensions of the feedforward layers in the transformer encoder
    num_layers=12,           # number of transformer encoder layers
    dropout=0.1,             # dropout rate for the transformer encoder
    classifier_dropout=0.5,  # dropout rate for the classifier
    n_node_layer1=256,       # number of nodes in the first hidden layer
    n_node_layer2=48,        # number of nodes in the second hidden layer
    static_size=75,          # number of static features
)

# Move the model to the selected device (GPU or CPU)
# This ensures that all the computations will be carried out on the same device as the model
model.to(device)

def get_accuracy(out, actual_labels, batchSize):
    '''
    Computes the accuracy of a model's predictions for a given batch.
    
    Parameters:
    - out (Tensor): The log probabilities or logits returned by the model.
    - actual_labels (Tensor): The actual labels for the batch.
    - batchSize (int): The size of the batch.
    
    Returns:
    float: The accuracy for the batch.
    '''
    # Get the predicted labels from the maximum value of log probabilities
    predictions = out.max(dim=1)[1]
    # Count the number of correct predictions
    correct = (predictions == actual_labels).sum().item()
    # Compute the accuracy for the batch
    accuracy = correct / batchSize
    
    return accuracy

# Define the training function
def train(model, optimizer, loss_fn, train_dataloader, val_dataloader, y_val, epochs=20, batchSize=16):
    """
    Train a PyTorch model using the given parameters.

    Parameters:
        model (nn.Module): The PyTorch model to train.
        optimizer (Optimizer): The optimizer to use during training.
        loss_fn (function): The loss function.
        train_dataloader (DataLoader): DataLoader for the training set.
        val_dataloader (DataLoader): DataLoader for the validation set.
        y_val (array-like): Ground truth labels for the validation set.
        epochs (int, optional): Number of epochs to train for. Default is 20.
        batchSize (int, optional): Size of batches. Default is 16.

    Returns:
        train_losses (array): Array of training losses for each epoch.
        val_losses (array): Array of validation losses for each epoch.
    """
    # Initialize device based on GPU/CPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize variables to keep track of best metrics
    best_val_loss = 2
    best_accuracy = 0
    best_AUC = 0
    best_p = []
    
    # Initialize arrays to store losses for plotting later
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)

    # Start of the training loop
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val AUC':^9} | {'Val F1':^9} |{'Elapsed':^9}")
    print("-"*60)

    # Loop through each epoch
    for epoch_i in tqdm(range(epochs)):
        # Keep track of time taken for each epoch
        t0_epoch = time.time()
        total_loss = 0
        epoc_acc = 0
        
        # Set model to train mode
        model.train()

        # Loop through each batch in the training dataloader
        for step, batch in enumerate(train_dataloader):
            # Move the batch tensors to the device
            b_input_ids, b_input_tbl, b_labels = batch
            b_input_ids, b_input_tbl, b_labels = b_input_ids.to(device), b_input_tbl.to(device), b_labels.long().to(device)
            
            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass
            logits = model(b_input_ids, b_input_tbl)
            logits = logits.float().to(device)
            
            # Compute the loss and accumulate it
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()
            
            # Compute the accuracy for the batch and accumulate it
            epoc_acc += get_accuracy(logits, b_labels, batchSize)
            
            # Perform a backward pass to update the weights
            loss.backward()
            
            # Update the optimizer parameters
            optimizer.step()
            
        # Compute the average training loss and accuracy for the epoch
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_acc = epoc_acc / len(train_dataloader)
        
        # Store the average training loss
        train_losses[epoch_i] = avg_train_loss

        # If a validation dataloader is provided, perform evaluation
        if val_dataloader is not None:
            val_loss, val_accuracy, val_AUC, val_f1, p = evaluate(model, val_dataloader, y_val, batchSize=16)
            
            # Store the validation loss
            val_losses[epoch_i] = val_loss

            # Update best metrics if necessary
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
            if val_AUC > best_AUC:
                best_AUC = val_AUC
                best_p = p
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts_BERTv1 = copy.deepcopy(model.state_dict())
                
            # Calculate time taken for the epoch
            time_elapsed = time.time() - t0_epoch
            
            # Print all metrics
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {avg_train_acc:^9.2f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {val_AUC:^9.4f} | {val_f1:^9.4f} | {time_elapsed:^9.2f}")
            
    # Training complete, print the best metrics
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%.")
    print(f"Training complete! Best AUC: {best_AUC:.4f}.")
    
    # Return training and validation losses and best model
    return train_losses, val_losses


def evaluate(model, val_dataloader, y_val, batchSize=16):
    """
    Evaluate a PyTorch model on a validation set.

    Parameters:
        model (nn.Module): The PyTorch model to evaluate.
        val_dataloader (DataLoader): DataLoader for the validation set.
        y_val (array-like): Ground truth labels for the validation set.
        batchSize (int, optional): Size of batches. Default is 16.

    Returns:
        val_loss (float): Average validation loss.
        val_accuracy (float): Average validation accuracy.
        val_AUC (float): Area under the ROC curve for the validation set.
        val_f1 (float): F1 score for the validation set.
        p (array): Array of prediction probabilities.

    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to store evaluation metrics and predictions
    val_accuracy = []
    val_loss = []
    outputs_list = []
    y_pred_list = []
    probs_list = []
    
    preds_list = torch.tensor([], dtype=torch.long, device=device)
    labels_list = torch.tensor([], dtype=torch.long, device=device)

    # Loop over batches in the validation DataLoader
    for batch in val_dataloader:
        # Extract input features and labels from the batch
        b_input_ids, b_input_tbl, b_labels = batch
        b_input_ids, b_input_tbl, b_labels = b_input_ids.to(device),  b_input_tbl.to(device), b_labels.long().to(device)

        # Perform forward pass and compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_input_tbl)
            logits = logits.float().to(device)
            
        # Calculate probabilities using softmax
        y_val_probs = torch.nn.functional.softmax(logits, dim=1)
        outputs_list.append(logits)
        probs_list.append(y_val_probs)
        
        # Get the predicted labels
        predictions = logits.max(dim=1)[1]
        preds_list = torch.cat([preds_list, predictions])
        labels_list = torch.cat([labels_list, b_labels])

        # Calculate loss for the batch
        loss = loss_fn(logits, b_labels).to(device)
        val_loss.append(loss.item())

        # Calculate accuracy for the batch
        val_accuracy = get_accuracy(logits, b_labels, batchSize)

    # Compute average loss and accuracy
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    
    # Convert tensors to NumPy arrays for scikit-learn metrics
    p = torch.cat(probs_list).detach().cpu().numpy()
    preds_list = preds_list.cpu().numpy()
    labels_list = labels_list.cpu().numpy()

    # Calculate F1 score and AUC
    val_f1 = f1_score(labels_list, preds_list, average='weighted')
    val_AUC = roc_auc_score(y_val, p, multi_class='ovr')
    
    return val_loss, val_accuracy, val_AUC, val_f1, p

"""
This following code performs the following operations:

1. Load the pre-trained model from the specified file.
2. Evaluate the model on a test set to generate predictions.
3. Compute and plot Receiver Operating Characteristic (ROC) curves for each class and their macro-average.
4. Compute and plot Precision-Recall (P-R) curves for each class and their macro-average.
5. Print out the classification report and the confusion matrix for model evaluation.

Outputs:
- Plots of ROC and P-R curves.
- Printed classification report and confusion matrix.
"""

# Define the path where the model's state dictionary will be saved
PATH = 'custom-Transformer-Mulimodal.pth'
# Save the model's state dictionary to the specified path
torch.save(model.state_dict(), PATH)

# Load the best model
model.load_state_dict(torch.load('Multi_CustomT.pth'))

# Get predictions for test set
model.eval()
test_probabilities = []
test_true_labels = []

for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_tbl = batch[1].to(device)
    b_labels = batch[2].to(device)
    
    with torch.no_grad():
        logits = model(b_input_ids, b_input_tbl)
        logits = logits.float().to(device)
    
    #logits = outputs[0]
    probs = torch.nn.functional.softmax(logits, dim=1)
    test_probabilities.extend(probs.detach().cpu().numpy())
    test_true_labels.extend(b_labels.detach().cpu().numpy())

test_probabilities = np.array(test_probabilities)
test_true_labels = np.array(test_true_labels)

# Compute macro-average ROC curve and ROC area
fpr = dict()
tpr = dict()
roc_auc = dict()
all_fpr = np.linspace(0, 1, 100)

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(test_true_labels == i, test_probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.4f})'.format(i, roc_auc[i]))

# Compute macro-average ROC curve and ROC area
mean_tpr = np.zeros_like(all_fpr)
for i in range(3):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= 3
mean_auc = auc(all_fpr, mean_tpr)
plt.plot(all_fpr, mean_tpr, color='b', linestyle='--', lw=2, label='Macro-average ROC (area = {0:0.4f})'.format(mean_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.show()

# Compute macro-average P-R curve and P-R area
precision = dict()
recall = dict()
average_precision = dict()

for i in range(3):
    precision[i], recall[i], _ = precision_recall_curve(test_true_labels == i, test_probabilities[:, i])
    average_precision[i] = average_precision_score(test_true_labels == i, test_probabilities[:, i])
    plt.step(recall[i], precision[i], lw=2, where='post', label='P-R curve of class {0} (area = {1:0.4f})'.format(i, average_precision[i]))

# Macro-average P-R curve and P-R area
mean_precision = sum(average_precision.values()) / 3
plt.plot([0, 1], [mean_precision, mean_precision], linestyle='--', lw=2, color='b', label='Macro-average P-R (area = {0:0.4f})'.format(mean_precision))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='upper right')
plt.title('Precision-Recall (P-R) Curves')
plt.show()

# Print classification report and confusion matrix
predicted_classes = np.argmax(test_probabilities, axis=1)
print(classification_report(test_true_labels, predicted_classes, target_names=['Success', 'Low IOP', 'High IOP']))
print("\nConfusion Matrix:\n", confusion_matrix(test_true_labels, predicted_classes))
