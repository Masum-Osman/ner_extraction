from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import torch

device = 0 if torch.backends.mps.is_available() else -1
print(f"Device set to use {'mps:0' if device == 0 else 'CPU'}")

model_path = "./models/ner_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

print("Loaded model labels:", model.config.id2label)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",  # ensure this is set
    device=device
)

def extract_entities(text):
    results = ner_pipeline(text)
    print(f"\nInput: {text}")
    print("Raw NER output:")
    for entity in results:
        print(f"{entity['word']} → {entity['entity_group']} (score: {entity['score']:.3f})")

if __name__ == "__main__":
    extract_entities("Alice went to Paris in 2024.")
    extract_entities("Barak Obama was a president of the USA.")
    extract_entities("The meeting is scheduled for 2023-10-01.")
    extract_entities("The Eiffel Tower is located in Paris.")
    extract_entities("Masum is a software engineer from Dhaka.")
