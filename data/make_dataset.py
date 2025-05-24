import json
import random
import os

# Entities to randomly select from
people = ["Alice", "Bob", "Eva", "John", "Michael", "Sarah", "David", "Emma", "Olivia", "Liam"]
organizations = ["Google", "OpenAI", "Microsoft", "Apple", "Amazon", "Facebook"]
locations = ["Paris", "New York", "San Francisco", "Berlin", "London", "Tokyo"]
dates = ["2024", "2021", "last June", "on Monday", "yesterday", "2023-10-01", "next week", "today"]

# Sentence templates
templates = [
    "{person} went to {location} in {date}.",
    "{person} joined {organization} in {date}.",
    "{person} moved to {location} {date}.",
    "{organization} hired {person} in {location}.",
    "{person1} met {person2} in {location} {date}.",
    "{organization} was founded in {location} in {date}.",
    "{person} started working at {organization} {date}.",
    "{person} visited {location} on {date}.",
    "{person} had a meeting with {organization} in {location} on {date}.",
    "{organization} hosted an event in {location} {date}."
]

def generate_synthetic_data_entry():
    template = random.choice(templates)
    person = random.choice(people)
    person2 = random.choice([p for p in people if p != person])
    organization = random.choice(organizations)
    location = random.choice(locations)
    date = random.choice(dates)

    text = template.format(
        person=person,
        person1=person,
        person2=person2,
        organization=organization,
        location=location,
        date=date
    )

    entities = []
    if "{person}" in template or "{person1}" in template:
        entities.append((person, "PERSON"))
    if "{person2}" in template:
        entities.append((person2, "PERSON"))
    if "{organization}" in template:
        entities.append((organization, "ORG"))
    if "{location}" in template:
        entities.append((location, "LOCATION"))
    if "{date}" in template:
        entities.append((date, "DATE"))

    return {"text": text, "entities": entities}

# Generate data
synthetic_data = [generate_synthetic_data_entry() for _ in range(10000)]

# Ensure directory exists
os.makedirs("data/raw", exist_ok=True)

# Save to file
output_path = "data/raw/synthetic_data.json"
with open(output_path, "w") as f:
    json.dump(synthetic_data, f, indent=2)

print(f"✅ Generated 10,000 samples and saved to {output_path}")
