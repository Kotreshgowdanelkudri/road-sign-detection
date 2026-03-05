"""
Label mapping utilities for traffic sign classification
"""
import json
import os
from config import LABEL_MAPPING_PATH, DATASET_PATH


def load_label_mapping():
    """
    Load label mapping from JSON file
    Returns: dict mapping numeric labels to sign names
    """
    if not os.path.exists(LABEL_MAPPING_PATH):
        print(f"Warning: Label mapping file not found at {LABEL_MAPPING_PATH}")
        return {}
    try:
        with open(LABEL_MAPPING_PATH, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        if not mapping:
            return {}
        return {int(k): v for k, v in mapping.items()}
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Could not load label mapping: {e}")
        return {}


def save_label_mapping(mapping):
    """
    Save label mapping to JSON file
    Args:
        mapping: dict mapping numeric labels to sign names
    """
    try:
        with open(LABEL_MAPPING_PATH, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2)
        print(f"Label mapping saved to {LABEL_MAPPING_PATH}")
    except (IOError, OSError) as e:
        print(f"Error saving label mapping: {e}")
        raise


def get_label_name(label_id, mapping=None):
    """
    Get the name of a traffic sign from its numeric label
    Args:
        label_id: numeric label (int)
        mapping: optional label mapping dict
    Returns: sign name (str)
    """
    if mapping is None:
        mapping = load_label_mapping()
    
    return mapping.get(int(label_id), f"Unknown Sign {label_id}")


def discover_dataset_structure(dataset_path=None):
    """
    Analyze dataset structure and discover classes
    Args:
        dataset_path: path to dataset directory
    Returns: dict with dataset information
    """
    if dataset_path is None:
        dataset_path = DATASET_PATH
    
    if not os.path.exists(dataset_path):
        return {
            'exists': False,
            'structure': None,
            'num_classes': 0,
            'classes': []
        }
    
    # Check if dataset has subdirectories (folder-based labeling)
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Filter out non-numeric directories
    numeric_subdirs = []
    for d in subdirs:
        try:
            int(d)
            numeric_subdirs.append(d)
        except ValueError:
            pass
    
    if numeric_subdirs:
        # Folder-based structure detected
        classes = sorted([int(d) for d in numeric_subdirs])
        return {
            'exists': True,
            'structure': 'folder-based',
            'num_classes': len(classes),
            'classes': classes
        }
    else:
        # All images in one folder
        return {
            'exists': True,
            'structure': 'flat',
            'num_classes': 0,
            'classes': []
        }


def create_default_mapping(num_classes):
    """
    Create a default label mapping for numeric classes
    Args:
        num_classes: number of classes
    Returns: dict mapping
    """
    mapping = {}
    for i in range(num_classes):
        mapping[i] = f"Traffic Sign Class {i}"
    return mapping


def validate_mapping(mapping, num_classes):
    """
    Validate that mapping covers all classes
    Args:
        mapping: label mapping dict
        num_classes: expected number of classes
    Returns: bool
    """
    if len(mapping) != num_classes:
        print(f"Warning: Mapping has {len(mapping)} entries but dataset has {num_classes} classes")
        return False
    
    for i in range(num_classes):
        if i not in mapping:
            print(f"Warning: Class {i} not found in mapping")
            return False
    
    return True


if __name__ == "__main__":
    # Test the label mapping utilities
    print("Discovering dataset structure...")
    info = discover_dataset_structure()
    print(f"Dataset exists: {info['exists']}")
    print(f"Structure: {info['structure']}")
    print(f"Number of classes: {info['num_classes']}")
    
    if info['num_classes'] > 0:
        print(f"Classes: {info['classes'][:10]}...")  # Show first 10
        
        # Load or create mapping
        mapping = load_label_mapping()
        if not mapping or len(mapping) != info['num_classes']:
            print("Creating default mapping...")
            mapping = create_default_mapping(info['num_classes'])
            save_label_mapping(mapping)
        
        # Validate
        is_valid = validate_mapping(mapping, info['num_classes'])
        print(f"Mapping valid: {is_valid}")
        
        # Show some examples
        print("\nSample mappings:")
        for i in list(mapping.keys())[:5]:
            print(f"  {i}: {mapping[i]}")
