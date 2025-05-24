from src.config import label_to_id
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_and_align_labels(examples):
    texts = examples["text"]
    entities_list = examples["entities"]

    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_offsets_mapping=True,
    )

    all_labels = []

    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        labels = ["O"] * len(offsets)
        text = texts[i]
        entities = entities_list[i]

        for entity_text, entity_label in entities:
            start_char = text.find(entity_text)
            if start_char == -1:
                continue
            end_char = start_char + len(entity_text)

            for idx, (start, end) in enumerate(offsets):
                if start >= start_char and end <= end_char and start != end:
                    prefix = "B-" if start == start_char else "I-"
                    labels[idx] = prefix + entity_label

        label_ids = [label_to_id[label] for label in labels]
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    # Remove offset_mapping so dataset is clean
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs
