import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from peft import get_peft_model, LoraConfig

# Load the training and validation datasets
df_train = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/train_OutOfDomain.csv', usecols=['sentence', 'label'], encoding='ISO-8859-1')
df_train_ec = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/train_EC.csv', usecols=['sentence', 'label'], encoding='ISO-8859-1')
df_train_lms = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/train_LMS.csv', usecols=['sentence', 'label'], encoding='ISO-8859-1')
df_train = pd.concat([df_train, df_train_ec], ignore_index=True)
df_train = pd.concat([df_train, df_train_lms], ignore_index=True)

df_validate = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/validate_OutOfDomain.csv', usecols=['sentence', 'label'], encoding='ISO-8859-1')
df_validate_ec = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/validate_EC.csv', usecols=['sentence', 'label'], encoding='ISO-8859-1')
df_validate_lms = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/validate_LMS.csv', usecols=['sentence', 'label'], encoding='ISO-8859-1')
df_validate = pd.concat([df_validate, df_validate_ec], ignore_index=True)
df_validate = pd.concat([df_validate, df_validate_lms], ignore_index=True)

# Clean dataset, remove duplicates based on the 'sentence' column
df_train = df_train.drop_duplicates(subset='sentence')
df_validate = df_validate.drop_duplicates(subset='sentence')

# Create a continuous label mapping
unique_labels = pd.concat([df_train['label'], df_validate['label']]).unique()
label2id = {int(label): i for i, label in enumerate(sorted(unique_labels))}
id2label = {i: int(label) for label, i in label2id.items()}  # Convert int64 to int

# Map the labels in the datasets
df_train['label'] = df_train['label'].map(label2id)
df_validate['label'] = df_validate['label'].map(label2id)

# Tokenizer
model_checkpoint = 'sentence-transformers/all-MiniLM-L12-v2'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenize the training and validation data
tokenized_train = tokenizer(df_train['sentence'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
tokenized_val = tokenizer(df_validate['sentence'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

# Create a Hugging Face Dataset from the tokenized data
train_dataset = Dataset.from_dict({
    'input_ids': tokenized_train['input_ids'],
    'attention_mask': tokenized_train['attention_mask'],  # Include attention masks
    'label': torch.tensor(df_train['label'].values, dtype=torch.int)
})

val_dataset = Dataset.from_dict({
    'input_ids': tokenized_val['input_ids'],
    'attention_mask': tokenized_val['attention_mask'],  # Include attention masks
    'label': torch.tensor(df_validate['label'].values, dtype=torch.int)
})

# Combine the datasets into a DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define evaluation function
def compute_metrics(p):
    predictions, labels = p.predictions, p.label_ids
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
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

# Load model with LoRA
peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=16,
                        lora_alpha=32,
                        bias="none",
                        lora_dropout=0.5,
                        target_modules=['query', "key"])

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(label2id), id2label=id2label, label2id=label2id)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Training arguments
batch_size = 8
num_train_epochs = 70

total_steps = len(train_dataset) // batch_size * num_train_epochs
warmup_steps = int(0.50 * total_steps)
#print(warmup_steps)

training_args = TrainingArguments(
    output_dir=model_checkpoint + "-lora-finetune",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_steps=warmup_steps,
    weight_decay=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",   
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    lr_scheduler_type="linear"
)

# Create Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()
