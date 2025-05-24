from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, AutoTokenizer
from datasets import Dataset
from src.config import label_list, model_name
from src.preprocess import tokenize_and_align_labels
import json

# Load raw data
with open("data/raw/synthetic_data.json") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use batched=True for efficiency
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

# Setup label mappings explicitly in model config
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}
model.config.id2label = id2label
model.config.label2id = label2id

collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="./models/ner_model",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
)

trainer.train()

# Save model and tokenizer with config (including labels)
model.save_pretrained("./models/ner_model")
tokenizer.save_pretrained("./models/ner_model")
