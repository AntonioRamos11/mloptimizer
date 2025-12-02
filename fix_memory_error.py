#!/usr/bin/env python
"""
Quick fix for BrokenProcessPool / Out of Memory errors

This script:
1. Checks current settings
2. Suggests safer values
3. Optionally updates system_parameters.py
"""

import sys
import re

def read_current_settings():
    """Read current system parameters"""
    try:
        with open('system_parameters.py', 'r') as f:
            content = f.read()
        
        # Extract current values
        batch_size = None
        dataset_shape = None
        
        batch_match = re.search(r'DATASET_BATCH_SIZE:\s*int\s*=\s*(\d+)', content)
        if batch_match:
            batch_size = int(batch_match.group(1))
        
        shape_match = re.search(r'DATASET_SHAPE\s*=\s*dataset_config\[DATASET_NAME\]\[\'shape\'\]', content)
        if shape_match:
            # Try to find the grietas_baches config
            config_match = re.search(r"'grietas_baches':\s*\{\s*'shape':\s*\((\d+),\s*(\d+),\s*(\d+)\)", content)
            if config_match:
                dataset_shape = tuple(map(int, config_match.groups()))
        
        return batch_size, dataset_shape, content
        
    except FileNotFoundError:
        print("ERROR: system_parameters.py not found!")
        return None, None, None


def calculate_memory_usage(batch_size, shape):
    """Estimate GPU memory usage"""
    if not shape or not batch_size:
        return None
    
    # Rough calculation: batch_size * H * W * C * 4 bytes (float32)
    # Plus model weights, activations, etc. (multiply by ~10 for safety)
    image_memory = batch_size * shape[0] * shape[1] * shape[2] * 4
    estimated_total = image_memory * 10  # Very rough estimate
    
    return estimated_total / (1024**3)  # Convert to GB


def suggest_settings(batch_size, shape):
    """Suggest safer settings"""
    suggestions = []
    
    if batch_size and batch_size > 16:
        suggestions.append({
            'param': 'DATASET_BATCH_SIZE',
            'current': batch_size,
            'suggested': 16,
            'reason': 'Reduce memory usage for large images (224x224)'
        })
    
    if shape and shape[0] > 128:
        suggestions.append({
            'param': 'DATASET_SHAPE (grietas_baches)',
            'current': f"{shape}",
            'suggested': "(128, 128, 3)",
            'reason': 'Smaller images = much less memory, still good accuracy'
        })
    
    return suggestions


def apply_fixes(content, suggestions, auto_apply=False):
    """Apply suggested fixes to content"""
    modified = content
    changes_made = []
    
    for suggestion in suggestions:
        param = suggestion['param']
        
        if param == 'DATASET_BATCH_SIZE':
            pattern = r'(DATASET_BATCH_SIZE:\s*int\s*=\s*)\d+'
            replacement = f'\\g<1>{suggestion["suggested"]}'
            if re.search(pattern, modified):
                modified = re.sub(pattern, replacement, modified)
                changes_made.append(param)
        
        elif param.startswith('DATASET_SHAPE'):
            # Update the grietas_baches config
            pattern = r"('grietas_baches':\s*\{\s*'shape':\s*)\(\d+,\s*\d+,\s*\d+\)"
            replacement = f'\\g<1>{suggestion["suggested"]}'
            if re.search(pattern, modified):
                modified = re.sub(pattern, replacement, modified)
                changes_made.append(param)
    
    return modified, changes_made


def main():
    print("=" * 70)
    print("MEMORY ERROR FIX TOOL")
    print("=" * 70)
    
    # Read current settings
    print("\nReading system_parameters.py...")
    batch_size, shape, content = read_current_settings()
    
    if content is None:
        sys.exit(1)
    
    print("\nCurrent Settings:")
    print(f"  DATASET_BATCH_SIZE: {batch_size}")
    print(f"  DATASET_SHAPE (grietas_baches): {shape}")
    
    # Estimate memory
    if batch_size and shape:
        estimated_mem = calculate_memory_usage(batch_size, shape)
        print(f"\n  Estimated GPU memory needed: ~{estimated_mem:.1f} GB")
        print("  (Very rough estimate, actual may vary)")
    
    # Get suggestions
    suggestions = suggest_settings(batch_size, shape)
    
    if not suggestions:
        print("\n✓ Settings look reasonable for memory usage!")
        print("\nIf you're still getting errors, try:")
        print("  • Close other GPU applications")
        print("  • Check GPU memory: nvidia-smi")
        print("  • Reduce batch size further (try 8 or 4)")
        return
    
    print("\n" + "=" * 70)
    print("RECOMMENDED CHANGES")
    print("=" * 70)
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion['param']}")
        print(f"   Current:   {suggestion['current']}")
        print(f"   Suggested: {suggestion['suggested']}")
        print(f"   Reason:    {suggestion['reason']}")
    
    # Ask user
    print("\n" + "=" * 70)
    response = input("\nApply these changes to system_parameters.py? (y/n): ")
    
    if response.lower() == 'y':
        # Create backup
        backup_file = 'system_parameters.py.backup'
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"\n✓ Backup created: {backup_file}")
        
        # Apply fixes
        modified, changes = apply_fixes(content, suggestions)
        
        if changes:
            with open('system_parameters.py', 'w') as f:
                f.write(modified)
            
            print(f"✓ Updated system_parameters.py")
            print(f"  Changed: {', '.join(changes)}")
            
            print("\n" + "=" * 70)
            print("NEXT STEPS")
            print("=" * 70)
            print("\n1. Verify changes:")
            print("     grep BATCH_SIZE system_parameters.py")
            print("     grep -A5 grietas_baches system_parameters.py")
            
            print("\n2. Restart slave node:")
            print("     python run_slave.py")
            
            print("\n3. Monitor memory usage:")
            print("     watch -n 1 nvidia-smi")
            
            print("\n4. If still crashing, reduce further:")
            print("     DATASET_BATCH_SIZE = 8 (or even 4)")
            
        else:
            print("⚠️  No changes could be applied automatically")
            print("   You may need to edit system_parameters.py manually")
    else:
        print("\nNo changes made.")
        print("\nTo apply manually, edit system_parameters.py:")
        for suggestion in suggestions:
            print(f"\n  {suggestion['param']}: {suggestion['suggested']}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
