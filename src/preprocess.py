from src.config import label_to_id
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_and_align_labels(examples):
    text = examples["text"]
    entities = examples["entities"]

    tokens = tokenizer(text, truncation=True, return_offsets_mapping=True)
    labels = ['O'] * len(tokens["input_ids"])

    for entity_text, entitiy_label in entities:
        start_char = text.find(entity_text)
        end_char = start_char + len(entity_text)

        for i, (start, end) in enumerate(tokens["offset_mapping"]):
            if start >= start_char and end <= end_char and start != end:
                prefix = "B-" if start == start_char else "I-"
                labels[i] = f"{prefix}{entitiy_label}"
        
    tokens["labels"] = [label_to_id[label] for label in labels]
    return tokens