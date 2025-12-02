#!/usr/bin/env python
"""
Test script to verify GRIETAS and BACHES dataset setup
Run this to check if everything is configured correctly before starting the full training
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ Pillow (PIL)")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    try:
        from sklearn.model_selection import train_test_split
        print(f"✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    return True


def test_system_parameters():
    """Test if system parameters are configured correctly"""
    print("\n" + "=" * 60)
    print("Testing system parameters...")
    print("=" * 60)
    
    try:
        from system_parameters import SystemParameters as SP
        
        print(f"Dataset name: {SP.DATASET_NAME}")
        print(f"Dataset path: {SP.DATASET_PATH}")
        print(f"Dataset type: {SP.DATASET_TYPE} (1=Image, 2=Regression, 3=TimeSeries)")
        print(f"Dataset shape: {SP.DATASET_SHAPE}")
        print(f"Dataset classes: {SP.DATASET_CLASSES}")
        print(f"Batch size: {SP.DATASET_BATCH_SIZE}")
        print(f"Validation split: {SP.DATASET_VALIDATION_SPLIT}")
        
        if SP.DATASET_NAME != 'grietas_baches':
            print(f"⚠ Warning: DATASET_NAME is '{SP.DATASET_NAME}', expected 'grietas_baches'")
            print("  Update DATASET_NAME in system_parameters.py")
            return False
        
        if SP.DATASET_TYPE != 1:
            print(f"✗ Error: DATASET_TYPE is {SP.DATASET_TYPE}, expected 1 (Image)")
            return False
        
        if SP.DATASET_CLASSES != 2:
            print(f"⚠ Warning: DATASET_CLASSES is {SP.DATASET_CLASSES}, expected 2")
        
        print("✓ System parameters configured correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error loading system parameters: {e}")
        return False


def test_dataset_folder():
    """Test if dataset folder structure is correct"""
    print("\n" + "=" * 60)
    print("Testing dataset folder structure...")
    print("=" * 60)
    
    from system_parameters import SystemParameters as SP
    dataset_path = SP.DATASET_PATH
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset folder not found: {dataset_path}")
        print("\nCreate the folder structure:")
        print(f"  mkdir -p {dataset_path}/BACHES")
        print(f"  mkdir -p {dataset_path}/GRIETAS")
        return False
    
    print(f"✓ Dataset folder exists: {dataset_path}")
    
    baches_path = os.path.join(dataset_path, 'BACHES')
    grietas_path = os.path.join(dataset_path, 'GRIETAS')
    
    if not os.path.exists(baches_path):
        print(f"✗ BACHES subfolder not found: {baches_path}")
        return False
    print(f"✓ BACHES folder exists")
    
    if not os.path.exists(grietas_path):
        print(f"✗ GRIETAS subfolder not found: {grietas_path}")
        return False
    print(f"✓ GRIETAS folder exists")
    
    # Count images
    import glob
    baches_images = glob.glob(os.path.join(baches_path, '*.png')) + \
                    glob.glob(os.path.join(baches_path, '*.jpg')) + \
                    glob.glob(os.path.join(baches_path, '*.jpeg'))
    grietas_images = glob.glob(os.path.join(grietas_path, '*.png')) + \
                     glob.glob(os.path.join(grietas_path, '*.jpg')) + \
                     glob.glob(os.path.join(grietas_path, '*.jpeg'))
    
    print(f"  BACHES images: {len(baches_images)}")
    print(f"  GRIETAS images: {len(grietas_images)}")
    
    if len(baches_images) == 0:
        print("✗ No images found in BACHES folder")
        return False
    
    if len(grietas_images) == 0:
        print("✗ No images found in GRIETAS folder")
        return False
    
    total_images = len(baches_images) + len(grietas_images)
    print(f"✓ Total images: {total_images}")
    
    if total_images < 10:
        print("⚠ Warning: Very few images. Consider adding more for better training.")
    
    return True


def test_dataset_loader():
    """Test if the dataset loader works"""
    print("\n" + "=" * 60)
    print("Testing dataset loader...")
    print("=" * 60)
    
    try:
        from load_grietas_baches_dataset import load_grietas_baches_dataset
        print("✓ Dataset loader module imported")
    except ImportError as e:
        print(f"✗ Failed to import dataset loader: {e}")
        return False
    
    try:
        from system_parameters import SystemParameters as SP
        
        print(f"Loading dataset from {SP.DATASET_PATH}...")
        (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = \
            load_grietas_baches_dataset(
                dataset_path=SP.DATASET_PATH,
                target_size=(224, 224),
                validation_split=0.2,
                test_split=0.1
            )
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  Training samples: {len(train_data)}")
        print(f"  Validation samples: {len(val_data)}")
        print(f"  Test samples: {len(test_data)}")
        print(f"  Image shape: {train_data.shape[1:]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_dataset_class():
    """Test if the custom dataset class works"""
    print("\n" + "=" * 60)
    print("Testing custom dataset class...")
    print("=" * 60)
    
    try:
        from app.common.dataset import GrietasBachesDataset
        from system_parameters import SystemParameters as SP
        
        print("✓ GrietasBachesDataset class imported")
        
        dataset = GrietasBachesDataset(
            dataset_name=SP.DATASET_NAME,
            shape=SP.DATASET_SHAPE,
            class_count=SP.DATASET_CLASSES,
            batch_size=SP.DATASET_BATCH_SIZE,
            validation_split=SP.DATASET_VALIDATION_SPLIT,
            dataset_path=SP.DATASET_PATH
        )
        
        print("✓ Dataset object created")
        print("Loading data...")
        
        dataset.load()
        
        print("✓ Dataset loaded successfully!")
        print(f"  Training steps: {dataset.get_training_steps()}")
        print(f"  Validation steps: {dataset.get_validation_steps()}")
        print(f"  Input shape: {dataset.get_input_shape()}")
        print(f"  Number of classes: {dataset.get_classes_count()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test custom dataset class: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_init_nodes():
    """Test if InitNodes can create the dataset"""
    print("\n" + "=" * 60)
    print("Testing InitNodes dataset creation...")
    print("=" * 60)
    
    try:
        from app.init_nodes import InitNodes
        from system_parameters import SystemParameters as SP
        
        print("✓ InitNodes imported")
        
        nodes = InitNodes()
        dataset = nodes.get_dataset()
        
        print(f"✓ Dataset created: {type(dataset).__name__}")
        
        if type(dataset).__name__ != 'GrietasBachesDataset':
            print(f"⚠ Warning: Expected GrietasBachesDataset, got {type(dataset).__name__}")
            print("  Check if DATASET_NAME is set to 'grietas_baches'")
            return False
        
        print("Loading dataset via InitNodes...")
        dataset.load()
        print("✓ Dataset loaded successfully via InitNodes!")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test InitNodes: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("GRIETAS and BACHES Dataset Setup Test")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("System Parameters", test_system_parameters),
        ("Dataset Folder", test_dataset_folder),
        ("Dataset Loader", test_dataset_loader),
        ("Custom Dataset Class", test_custom_dataset_class),
        ("InitNodes Integration", test_init_nodes),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\nYou're ready to run the architecture search:")
        print("  python run_master.py  # Start master node")
        print("  python run_slave.py   # Start slave node (in another terminal)")
        print("\nOr use the standalone benchmark:")
        print("  python benchmark_grietas_baches.py --mode simple")
        return 0
    else:
        print("\n" + "=" * 70)
        print("⚠ SOME TESTS FAILED")
        print("=" * 70)
        print("\nPlease fix the issues above before running the training.")
        print("See GRIETAS_BACHES_SETUP.md for detailed setup instructions.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
