label_list = [
    "O",
    "B-PERSON",
    "I-PERSON",
    "B-ORG",
    "I-ORG",
    "B-LOCATION",
    "I-LOCATION",
    "B-DATE",
    "I-DATE"
]

label_to_id = {label: i for i, label in enumerate(label_list)}

model_name = "bert-base-cased"
