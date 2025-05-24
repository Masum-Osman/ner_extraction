from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset
from src.config import label_list, model_name
from src.preprocess import tokenize_and_align_labels
from transformers import AutoTokenizer
import json

with open("data/raw/synthetic.json") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
tokenized = dataset.map(tokenize_and_align_labels)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))
collator = DataCollatorForTokenClassification(tokenizer)


# Training setup
args = TrainingArguments(
    output_dir="./models/ner_model",
    evaluation_strategy="no",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy='no'
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized,
    data_collator=collator,
)

trainer.train()