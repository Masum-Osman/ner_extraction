# NER Extraction Project

A Named Entity Recognition (NER) project using the BERT model to extract entities like persons, organizations, locations, and dates from text.

## Overview

This project implements a fine-tuned BERT model for Named Entity Recognition using the `bert-base-cased` architecture. It can identify the following entity types:
- Person names
- Organizations
- Locations
- Dates

## Project Structure
```
.
├── data/
│   ├── make_dataset.py    # Synthetic data generation script
│   └── raw/               # Raw data storage
├── src/
│   ├── config.py         # Model configuration and label definitions
│   ├── infer.py          # Inference script for entity extraction
│   ├── preprocess.py     # Data preprocessing utilities
│   └── train.py          # Model training script
├── run.sh                # Main execution script
└── requirements.txt      # Project dependencies
```

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project workflow is managed through `run.sh`, which executes the following steps:

1. Generate synthetic training data:
```bash
python -m data.make_dataset
```

2. Train the NER model:
```bash
python -m src.train
```

3. Run inference:
```bash
python -m src.infer
```

## Model Details

- Base Model: `bert-base-cased`
- Entity Labels:
  - PERSON: Person names
  - ORG: Organization names
  - LOCATION: Location names
  - DATE: Date expressions

## Training

The model is trained with the following configuration:
- Batch size: 8
- Number of epochs: 5
- Weight decay: 0.01
- Training logs are saved in the `./logs` directory
- Model checkpoints are saved in `./models/ner_model`

## Inference

The inference script (`src/infer.py`) provides a simple interface to extract entities from text. It automatically uses MPS (Metal Performance Shaders) acceleration on compatible Apple Silicon devices, falling back to CPU when unavailable.

Example usage:
```python
from src.infer import extract_entities

extract_entities("Alice went to Paris in 2024.")
```

Output format includes the entity text, type, and confidence score for each detected entity.