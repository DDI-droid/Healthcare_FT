# gspo_readmit/data.py
from datasets import Dataset
import json
import os

def load_jsonl(path):
    """Load JSONL file with error handling and validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                ex = json.loads(line)
                # Validate required fields
                if "text" not in ex:
                    raise ValueError(f"Missing 'text' field in line {line_num}")
                if "label" not in ex:
                    raise ValueError(f"Missing 'label' field in line {line_num}")
                
                # Validate text is not empty
                text = ex["text"]
                if not text or not text.strip():
                    raise ValueError(f"Empty 'text' field in line {line_num}")
                
                # Validate label is 0 or 1
                label = int(ex["label"])
                if label not in [0, 1]:
                    raise ValueError(f"Invalid label {label} in line {line_num}. Must be 0 or 1.")
                
                yield {"text": text.strip(), "label": label}
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")
            except ValueError as e:
                raise ValueError(f"Validation error on line {line_num}: {e}")

def get_dataset(jsonl_path: str):
    """Load and validate dataset from JSONL file."""
    rows = list(load_jsonl(jsonl_path))
    
    if len(rows) == 0:
        raise ValueError(f"Dataset is empty: {jsonl_path}")
    
    # Validate label distribution
    labels = [row["label"] for row in rows]
    label_counts = {0: labels.count(0), 1: labels.count(1)}
    
    if label_counts[0] == 0 or label_counts[1] == 0:
        raise ValueError(f"Dataset has only one class. Label distribution: {label_counts}")
    
    return Dataset.from_list(rows)
