import json

def generate_synthetic_data():
    return [
        {"text": "Alice went to Paris in 2024.", "entities": [("Alice", "PERSON"), ("Paris", "LOCATION"), ("2024", "DATE")]},
        {"text": "Bob joined Google in 2021.", "entities": [("Bob", "PERSON"), ("Google", "ORG"), ("2021", "DATE")]},
        {"text": "Eva moved to New York last June.", "entities": [("Eva", "PERSON"), ("New York", "LOCATION"), ("last June", "DATE")]},
        {"text": "OpenAI hired John in San Francisco.", "entities": [("OpenAI", "ORG"), ("John", "PERSON"), ("San Francisco", "LOCATION")]},
        {"text": "Michael met Sarah in Berlin on Monday.", "entities": [("Michael", "PERSON"), ("Sarah", "PERSON"), ("Berlin", "LOCATION"), ("Monday", "DATE")]}
    ]

if __name__ == "__main__":
    data = generate_synthetic_data()
    with open("data/raw/synthetic_data.json", "w") as f:
        json.dump(data, f, indent=2)