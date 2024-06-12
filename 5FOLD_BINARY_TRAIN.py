'''
import os
import sys
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction
from datasets import load_dataset, DatasetDict
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import AutoModel
'''

import os
import sys
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction
from datasets import load_dataset, DatasetDict
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import AutoModel
from datasets import DatasetDict, load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.special import expit as sigmoid


# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')  # Use 'binary' for binary classification
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }



# Set the environment variables to (hopefully) use more GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

###

'''
def monitor_memory_usage(device=None):
  """
  Monitors memory usage on the specified device (default: GPU if available).

  Args:
      device: The device to monitor memory usage on.

  Returns:
      A dictionary containing the following keys:
          allocated: Current allocated memory in bytes.
          max_allocated: Maximum memory ever allocated in bytes.
          reserved: Total amount of reserved memory in bytes.
          max_reserved: Maximum reserved memory ever allocated in bytes.
  """
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

  if device == "cuda":
    allocated = torch.cuda.memory_allocated()
    max_allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_reserved = torch.cuda.max_memory_reserved()
  else:
    allocated = torch.get_allocated_memory()
    max_allocated = torch.max_allocated_memory()
    reserved = 0  # Not available on CPU
    max_reserved = 0  # Not available on CPU

  return {
      "allocated": allocated,
      "max_allocated": max_allocated,
      "reserved": reserved,
      "max_reserved": max_reserved,
  }


###
'''

os.chdir('/home/tac0225/Documents/plm')
print(os.getcwd())

# GPU
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available.")

# Define paths to your datasets
data_files = {
    "train": "5fold/X_train_fold1.csv",
    "validation": "5fold/X_val_fold1.csv",
    "test": "5fold/X_test_fold1.csv"
}

# Load your datasets
datasets = load_dataset('csv', data_files=data_files)

# Provide labels
labels_files = {
    "train": "5fold/y_train_binary_fold1.csv",
    "validation": "5fold/y_val_binary_fold1.csv",
    "test": "5fold/y_test_binary_fold1.csv"
}

# Load your labels (assuming binary classification for this example)
labels_datasets = load_dataset('csv', data_files=labels_files)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

# Dynamically pad each batch individually to the longest sequence in that batch
#data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)


# Function to tokenize sequences
def tokenize_function(examples):
    return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=200) #May need to change the max_length (PLM-ARG = 200, originally I did 512)

# Apply tokenization to all splits
tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Assuming your datasets and labels are aligned and correctly tokenized
final_datasets = DatasetDict({
    "train": tokenized_datasets["train"].add_column("label", labels_datasets["train"]["binary_label"]),
    "validation": tokenized_datasets["validation"].add_column("label", labels_datasets["validation"]["binary_label"]),
    "test": tokenized_datasets["test"].add_column("label", labels_datasets["test"]["binary_label"])
})

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D", num_labels=2)

# Configure GPU as the device
device = torch.device(f'cuda:{torch.cuda.current_device()}')

# Running on GPU??
model.to(device)  # Move model to the appropriate device
# Print the device where the first parameter tensor of the model is located
print(next(model.parameters()).device)

# Empty the cache for memory
#torch.cuda.empty_cache()

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results_fold1_binary",
    num_train_epochs=20,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
    warmup_steps=500,
    save_strategy="epoch",  # Save the model at the end of each epoch to match the evaluation strategy
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)



# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_datasets["train"],
    eval_dataset=final_datasets["validation"],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

#Evaluate the model
results = trainer.evaluate(final_datasets["test"])
print(results)

output_dir = './'

metrics_path = os.path.join(output_dir, "results_binary_fold1.txt")
with open(metrics_path, "w") as file:
    for key, value in results.items():
        file.write(f"{key}: {value}\n")

