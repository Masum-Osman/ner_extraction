label_list = ['O', 'B-PERSON', 'I-PERSON', 'B-ORG', 'I-ORG', 'B-LOCATION', 'I-LOCATION', 'B-DATE', 'I-DATE']
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

model_name = "bert-base-cased"