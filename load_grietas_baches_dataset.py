"""
Custom Dataset Loader for GRIETAS (cracks) and BACHES (potholes) Image Classification
This script loads images from the dataset_grietas_baches folder and prepares them for AutoML training.
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import glob

def load_grietas_baches_dataset(
    dataset_path='./dataset_grietas_baches',
    target_size=(224, 224),
    validation_split=0.2,
    test_split=0.1,
    random_state=42
):
    """
    Load GRIETAS and BACHES dataset from folder structure.
    
    Args:
        dataset_path: Path to the dataset folder containing GRIETAS and BACHES subfolders
        target_size: Target image size (height, width) for resizing
        validation_split: Proportion of training data to use for validation
        test_split: Proportion of total data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of ((train_data, train_labels), (val_data, val_labels), (test_data, test_labels))
        Images are in NCHW format (batch, channels, height, width) as float32 in range [0, 1]
    """
    
    print(f"Loading dataset from {dataset_path}...")
    
    # Class mapping: BACHES=0, GRIETAS=1
    class_names = ['BACHES', 'GRIETAS']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    all_images = []
    all_labels = []
    
    # Load images from each class folder
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        image_files = glob.glob(os.path.join(class_path, '*.png'))
        
        print(f"Loading {len(image_files)} images from {class_name}...")
        
        for img_path in image_files:
            try:
                # Load and resize image
                img = Image.open(img_path).convert('RGB')
                img = img.resize(target_size, Image.BILINEAR)
                
                # Convert to numpy array and normalize
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Convert to CHW format (channels, height, width)
                img_array = np.transpose(img_array, (2, 0, 1))
                
                all_images.append(img_array)
                all_labels.append(class_to_idx[class_name])
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    # Convert to numpy arrays
    all_images = np.array(all_images, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    
    print(f"Total images loaded: {len(all_images)}")
    print(f"Image shape: {all_images.shape}")
    print(f"Class distribution: BACHES={np.sum(all_labels == 0)}, GRIETAS={np.sum(all_labels == 1)}")
    
    # Split into train+val and test sets
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        all_images, all_labels,
        test_size=test_split,
        random_state=random_state,
        stratify=all_labels
    )
    
    # Split train_val into train and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images, train_val_labels,
        test_size=validation_split / (1 - test_split),  # Adjust for already split test set
        random_state=random_state,
        stratify=train_val_labels
    )
    
    print(f"\nDataset splits:")
    print(f"  Training:   {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    print(f"  Test:       {len(test_images)} images")
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def save_grietas_baches_dataset(
    output_dir='~/.tinygrad_datasets/grietas_baches',
    dataset_path='./dataset_grietas_baches',
    target_size=(224, 224),
    validation_split=0.2,
    test_split=0.1
):
    """
    Load and save the GRIETAS and BACHES dataset in numpy format for faster loading.
    
    Args:
        output_dir: Directory to save the processed dataset
        dataset_path: Path to the original dataset
        target_size: Target image size for resizing
        validation_split: Proportion for validation split
        test_split: Proportion for test split
    """
    # Create output directory
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = \
        load_grietas_baches_dataset(dataset_path, target_size, validation_split, test_split)
    
    # Save to disk
    print(f"\nSaving dataset to {output_dir}...")
    np.save(os.path.join(output_dir, 'train_data.npy'), train_data)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'val_data.npy'), val_data)
    np.save(os.path.join(output_dir, 'val_labels.npy'), val_labels)
    np.save(os.path.join(output_dir, 'test_data.npy'), test_data)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)
    
    # Save metadata
    metadata = {
        'num_classes': 2,
        'class_names': ['BACHES', 'GRIETAS'],
        'image_shape': list(train_data.shape[1:]),  # (C, H, W)
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data)
    }
    
    import json
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Dataset saved successfully!")
    print(f"Metadata: {metadata}")
    
    return output_dir


if __name__ == '__main__':
    # When run directly, process and save the dataset
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and save GRIETAS and BACHES dataset')
    parser.add_argument('--dataset-path', default='./dataset_grietas_baches',
                        help='Path to dataset folder')
    parser.add_argument('--output-dir', default='~/.tinygrad_datasets/grietas_baches',
                        help='Output directory for processed dataset')
    parser.add_argument('--target-size', nargs=2, type=int, default=[224, 224],
                        help='Target image size (height width)')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Test split ratio')
    
    args = parser.parse_args()
    
    save_grietas_baches_dataset(
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        target_size=tuple(args.target_size),
        validation_split=args.validation_split,
        test_split=args.test_split
    )
