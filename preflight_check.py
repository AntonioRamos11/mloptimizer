#!/usr/bin/env python
"""
Pre-flight check - verify settings won't freeze your PC
Run this BEFORE starting training
"""

import sys
import psutil

def check_system_resources():
    """Check available system resources"""
    print("="*70)
    print("SYSTEM RESOURCES CHECK")
    print("="*70)
    
    # Check RAM
    mem = psutil.virtual_memory()
    print(f"\n📊 RAM:")
    print(f"   Total:     {mem.total / (1024**3):.2f} GB")
    print(f"   Available: {mem.available / (1024**3):.2f} GB")
    print(f"   Used:      {mem.percent}%")
    
    if mem.total < 8 * (1024**3):  # Less than 8GB
        print(f"   ⚠️  WARNING: Low RAM ({mem.total / (1024**3):.1f} GB)")
        print(f"      Recommended: 8GB+ for training")
    elif mem.percent > 70:
        print(f"   ⚠️  WARNING: High memory usage already ({mem.percent}%)")
        print(f"      Close other applications before training")
    else:
        print(f"   ✓ RAM OK")
    
    # Check GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', 
                                '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')[0].split(',')
            gpu_total = int(gpu_info[0])
            gpu_used = int(gpu_info[1])
            gpu_percent = (gpu_used / gpu_total) * 100
            
            print(f"\n🎮 GPU:")
            print(f"   Total:     {gpu_total} MB")
            print(f"   Used:      {gpu_used} MB ({gpu_percent:.1f}%)")
            print(f"   Available: {gpu_total - gpu_used} MB")
            
            if gpu_total < 4000:  # Less than 4GB
                print(f"   ⚠️  WARNING: Low GPU memory ({gpu_total}MB)")
            else:
                print(f"   ✓ GPU OK")
    except:
        print(f"\n🎮 GPU: Not available or nvidia-smi not found")


def check_training_settings():
    """Check training settings in system_parameters.py"""
    print("\n" + "="*70)
    print("TRAINING SETTINGS CHECK")
    print("="*70)
    
    try:
        from system_parameters import SystemParameters as SP
        
        # Check image size
        shape = SP.DATASET_SHAPE
        print(f"\n📐 Image Size: {shape}")
        
        pixels = shape[0] * shape[1] if len(shape) >= 2 else 0
        
        if pixels > 150 * 150:
            print(f"   ⚠️  WARNING: Large images ({shape[0]}x{shape[1]})")
            print(f"      This uses LOTS of memory!")
            print(f"      Recommended: 96x96 or 128x128")
            is_safe = False
        elif pixels > 100 * 100:
            print(f"   ⚠️  CAUTION: Medium-large images ({shape[0]}x{shape[1]})")
            print(f"      Should be OK with small batch size")
            is_safe = True
        else:
            print(f"   ✓ Image size is safe")
            is_safe = True
        
        # Check batch size
        batch_size = SP.DATASET_BATCH_SIZE
        print(f"\n📦 Batch Size: {batch_size}")
        
        if pixels > 150 * 150 and batch_size > 4:
            print(f"   ⚠️  WARNING: Batch size too large for image size!")
            print(f"      With {shape[0]}x{shape[1]} images, use batch size 2-4")
            is_safe = False
        elif pixels > 100 * 100 and batch_size > 8:
            print(f"   ⚠️  CAUTION: Batch size might be too large")
            print(f"      Recommended: 4-8 for this image size")
            is_safe = True
        else:
            print(f"   ✓ Batch size is safe")
        
        # Estimate memory usage
        bytes_per_image = pixels * 3 * 4  # RGB, float32
        batch_memory_mb = (bytes_per_image * batch_size) / (1024**2)
        estimated_total_mb = batch_memory_mb * 20  # Rough estimate including model
        
        print(f"\n💾 Estimated Memory per Batch:")
        print(f"   Images:    ~{batch_memory_mb:.1f} MB")
        print(f"   Total:     ~{estimated_total_mb:.1f} MB (very rough estimate)")
        
        if estimated_total_mb > 10000:  # 10GB
            print(f"   ⚠️  WARNING: High memory usage expected!")
            print(f"      Your PC might freeze!")
            is_safe = False
        elif estimated_total_mb > 5000:  # 5GB
            print(f"   ⚠️  CAUTION: Moderate memory usage")
            print(f"      Monitor memory during training")
            is_safe = True
        else:
            print(f"   ✓ Memory usage should be OK")
        
        # Check epochs
        exploration_epochs = SP.EXPLORATION_EPOCHS
        hof_epochs = SP.HALL_OF_FAME_EPOCHS
        
        print(f"\n⏱️  Training Duration:")
        print(f"   Exploration epochs: {exploration_epochs}")
        print(f"   Hall of Fame epochs: {hof_epochs}")
        
        if hof_epochs > 50:
            print(f"   ⚠️  CAUTION: Long training runs ({hof_epochs} epochs)")
            print(f"      More time for memory leaks to accumulate")
            print(f"      Consider reducing to 30-50 epochs")
        else:
            print(f"   ✓ Training duration is reasonable")
        
        return is_safe
        
    except Exception as e:
        print(f"\n❌ Error checking settings: {e}")
        return False


def show_recommendations(is_safe):
    """Show final recommendations"""
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if not is_safe:
        print("\n⚠️  SETTINGS ARE NOT SAFE - PC WILL LIKELY FREEZE!")
        print("\n🔧 IMMEDIATE ACTIONS REQUIRED:")
        print("\n1. REDUCE IMAGE SIZE (CRITICAL):")
        print("   Edit system_parameters.py, line ~56:")
        print("   'grietas_baches': {")
        print("       'shape': (96, 96, 3),  # Change to 96x96")
        print("       'classes': 2")
        print("   }")
        
        print("\n2. REDUCE BATCH SIZE:")
        print("   Edit system_parameters.py, line ~70:")
        print("   DATASET_BATCH_SIZE: int = 4  # Change to 4 or 2")
        
        print("\n3. REDUCE EPOCHS:")
        print("   Edit system_parameters.py, line ~85:")
        print("   HALL_OF_FAME_EPOCHS: int = 30  # Change to 30")
        
        print("\n❌ DO NOT START TRAINING UNTIL YOU FIX THESE!")
        
    else:
        print("\n✓ Settings look SAFE to run!")
        print("\n📝 Optional optimizations:")
        print("   • Monitor memory: watch -n 1 'free -h'")
        print("   • Close other applications")
        print("   • Have emergency cleanup ready: python emergency_memory_cleanup.py --kill")
    
    print("\n" + "="*70)


def main():
    """Main function"""
    print("\n" + "="*70)
    print("PRE-FLIGHT CHECK - MEMORY SAFETY")
    print("="*70)
    print("\nChecking if your system can handle training without freezing...\n")
    
    # Check system resources
    check_system_resources()
    
    # Check training settings
    is_safe = check_training_settings()
    
    # Show recommendations
    show_recommendations(is_safe)
    
    # Ask user
    if not is_safe:
        print("\n⚠️  WARNING: Training with these settings will likely FREEZE your PC!")
        print("Please fix the settings above before running.")
        return 1
    else:
        response = input("\nSettings look safe. Ready to start training? (y/n): ")
        if response.lower() == 'y':
            print("\n✓ Good luck! Monitor memory usage while training.")
            print("  If PC starts freezing, press Ctrl+C to stop.")
            return 0
        else:
            print("\n✓ OK, make any adjustments needed and run this check again.")
            return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
