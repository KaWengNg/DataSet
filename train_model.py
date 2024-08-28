from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup  # Import the scheduler
)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the training and validation datasets (same as before)
df_train = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/train_OutOfDomain.csv', usecols=['idx', 'sentence', 'label'], encoding='ISO-8859-1')
df_train_ec = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/train_EC.csv', usecols=['idx', 'sentence', 'label'], encoding='ISO-8859-1')
df_train_lms = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/train_LMS.csv', usecols=['idx', 'sentence', 'label'], encoding='ISO-8859-1')
df_train = pd.concat([df_train, df_train_ec], ignore_index=True)
df_train = pd.concat([df_train, df_train_lms], ignore_index=True)

df_validate = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/validate_OutOfDomain.csv', usecols=['idx', 'sentence', 'label'], encoding='ISO-8859-1')
df_validate_ec = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/validate_EC.csv', usecols=['idx', 'sentence', 'label'], encoding='ISO-8859-1')
df_validate_lms = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/validate_LMS.csv', usecols=['idx', 'sentence', 'label'], encoding='ISO-8859-1')
df_validate = pd.concat([df_validate, df_validate_ec], ignore_index=True)
df_validate = pd.concat([df_validate, df_validate_lms], ignore_index=True)

# Clean dataset, remove duplicates based on the 'sentence' column
df_train = df_train.drop_duplicates(subset='sentence')
df_validate = df_validate.drop_duplicates(subset='sentence')

# Create a continuous label mapping
unique_labels = pd.concat([df_train['label'], df_validate['label']]).unique()
label2id = {label: i for i, label in enumerate(sorted(unique_labels))}
id2label = {i: label for label, i in label2id.items()}

# Map the labels in the datasets
df_train['label'] = df_train['label'].map(label2id)
df_validate['label'] = df_validate['label'].map(label2id)

# Convert the dataframes to Hugging Face datasets
train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_validate)
shuffled_train_dataset = train_dataset.shuffle(seed=42)
shuffled_val_dataset = val_dataset.shuffle(seed=42)

dataset = DatasetDict({
    "train": shuffled_train_dataset,
    "validation": shuffled_val_dataset
})

# Ensuring they contain standard Python types
id2label = {int(k): str(v) for k, v in id2label.items()}
label2id = {str(k): int(v) for k, v in label2id.items()}

# Correct model checkpoint
model_checkpoint = 'sentence-transformers/paraphrase-distilroberta-base-v1'

# Generate classification model from model_checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=len(label2id), id2label=id2label, label2id=label2id)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# Add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    # Extract text
    text = examples["sentence"]

    # Tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p.predictions, p.label_ids
    
    # If predictions is a tuple, get the first element (logits)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # If the predictions have more than 2 dimensions, reduce them accordingly
    if predictions.ndim > 2:
        predictions = predictions.squeeze()
    
    # Now apply argmax
    predictions = np.argmax(predictions, axis=1)
    
    precision_score_value = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall_score_value = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1_score_value = f1_score(labels, predictions, average='weighted', zero_division=0)
    accuracy_score_value = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy_score_value,
        'precision': precision_score_value,
        'recall': recall_score_value,
        'f1': f1_score_value
    }

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df_train['label']),
    y=df_train['label']
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Subclass the Trainer class to use custom loss function
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=16,  # Increased capacity
                        lora_alpha=32,  # Increased capacity
                        bias="none",
                        lora_dropout=0.15,  # Adjusted dropout
                        target_modules=['query', "key"])

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

batch_size = 8
num_train_epochs = 11

# Total number of training steps
total_steps = len(train_dataset) // batch_size * num_train_epochs

# Define warmup steps (10% of total steps)
warmup_steps = int(0.1 * total_steps)

# Define training arguments
training_args = TrainingArguments(
    output_dir=model_checkpoint + "-lora-finetune",
    learning_rate=5e-5,  # Adjusted learning rate
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,  # More epochs to allow overfitting
    warmup_steps=warmup_steps,  # Set warmup steps
    weight_decay=0.02,  # Adjusted weight decay
    evaluation_strategy="epoch",
    save_strategy="epoch",   
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    lr_scheduler_type="linear"  # Use a linear learning rate decay
)

# Create trainer object
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()
