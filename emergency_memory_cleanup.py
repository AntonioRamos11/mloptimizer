#!/usr/bin/env python
"""
Emergency memory cleanup and monitoring script

Use this if your PC is freezing due to memory issues during training
"""

import os
import sys
import psutil
import time

def check_memory():
    """Check current memory usage"""
    mem = psutil.virtual_memory()
    print(f"\n{'='*60}")
    print(f"SYSTEM MEMORY STATUS")
    print(f"{'='*60}")
    print(f"Total RAM:     {mem.total / (1024**3):.2f} GB")
    print(f"Available:     {mem.available / (1024**3):.2f} GB")
    print(f"Used:          {mem.used / (1024**3):.2f} GB ({mem.percent}%)")
    print(f"Free:          {mem.free / (1024**3):.2f} GB")
    
    if mem.percent > 90:
        print(f"\n⚠️  WARNING: Memory usage is CRITICAL ({mem.percent}%)")
        return True
    elif mem.percent > 75:
        print(f"\n⚠️  WARNING: Memory usage is HIGH ({mem.percent}%)")
        return False
    else:
        print(f"\n✓ Memory usage is OK ({mem.percent}%)")
        return False
    return False


def find_python_processes():
    """Find Python processes and their memory usage"""
    print(f"\n{'='*60}")
    print(f"PYTHON PROCESSES")
    print(f"{'='*60}")
    
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                mem_mb = proc.info['memory_info'].rss / (1024**2)
                python_procs.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'memory_mb': mem_mb,
                    'cmdline': ' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else ''
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Sort by memory usage
    python_procs.sort(key=lambda x: x['memory_mb'], reverse=True)
    
    if python_procs:
        print(f"\n{'PID':<10} {'Memory':<12} {'Command'}")
        print("-" * 60)
        for proc in python_procs[:10]:  # Top 10
            print(f"{proc['pid']:<10} {proc['memory_mb']:>8.1f} MB   {proc['cmdline'][:50]}")
        
        return python_procs
    else:
        print("No Python processes found")
        return []


def kill_training_processes():
    """Kill training processes to free memory"""
    print(f"\n{'='*60}")
    print(f"EMERGENCY: KILLING TRAINING PROCESSES")
    print(f"{'='*60}")
    
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            
            # Look for training processes
            if any(keyword in cmdline.lower() for keyword in ['run_slave', 'run_master', 'training_slave']):
                print(f"Killing process {proc.info['pid']}: {cmdline[:60]}")
                proc.kill()
                killed += 1
                time.sleep(0.5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            pass
    
    if killed > 0:
        print(f"\n✓ Killed {killed} training process(es)")
        time.sleep(2)  # Wait for cleanup
    else:
        print("\n✓ No training processes found")
    
    return killed


def show_recommendations():
    """Show recommendations to prevent memory issues"""
    print(f"\n{'='*60}")
    print(f"PREVENTION RECOMMENDATIONS")
    print(f"{'='*60}")
    print("\n1. REDUCE IMAGE SIZE (CRITICAL):")
    print("   Edit system_parameters.py:")
    print("   'grietas_baches': {'shape': (96, 96, 3), 'classes': 2}")
    print("   ↑ Currently set to 96x96 (good!)")
    
    print("\n2. REDUCE BATCH SIZE:")
    print("   DATASET_BATCH_SIZE: int = 4  # Or even 2")
    
    print("\n3. LIMIT EPOCHS:")
    print("   EXPLORATION_EPOCHS: int = 5")
    print("   HALL_OF_FAME_EPOCHS: int = 30")
    
    print("\n4. MONITOR WHILE TRAINING:")
    print("   watch -n 1 'free -h && nvidia-smi'")
    
    print("\n5. IF PC FREEZES:")
    print("   • Press Ctrl+Alt+F2 (switch to terminal)")
    print("   • Login")
    print("   • Run: python emergency_memory_cleanup.py --kill")
    print("   • Press Ctrl+Alt+F7 (back to GUI)")
    
    print(f"\n{'='*60}")


def main():
    """Main function"""
    print("\n" + "="*60)
    print("EMERGENCY MEMORY CLEANUP TOOL")
    print("="*60)
    
    # Check if kill flag is set
    kill_flag = '--kill' in sys.argv
    
    # Check memory
    is_critical = check_memory()
    
    # Show Python processes
    procs = find_python_processes()
    
    # If memory is critical or kill flag, offer to kill processes
    if kill_flag or (is_critical and procs):
        if kill_flag:
            print("\n⚠️  Kill flag detected, terminating training processes...")
            kill_training_processes()
        else:
            response = input("\n⚠️  Memory is critical! Kill training processes? (y/n): ")
            if response.lower() == 'y':
                kill_training_processes()
    
    # Recheck memory after cleanup
    if kill_flag or is_critical:
        print("\nRechecking memory after cleanup...")
        time.sleep(2)
        check_memory()
    
    # Show recommendations
    show_recommendations()
    
    print("\n" + "="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
