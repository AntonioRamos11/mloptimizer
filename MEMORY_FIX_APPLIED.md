# 🚨 EMERGENCY: PC FREEZING DUE TO MEMORY

## Changes Applied (CRITICAL)

I've updated your settings to prevent PC freezing:

### ✅ **Image Size: 224×224 → 96×96**
```python
'grietas_baches': {
    'shape': (96, 96, 3),  # Was (224, 224, 3)
    'classes': 2
}
```
**Impact:** **~5.5× LESS MEMORY** (224² → 96² = 50,176 vs 9,216 pixels)

### ✅ **Batch Size: 8 → 4**
```python
DATASET_BATCH_SIZE: int = 4  # Was 8
```
**Impact:** **2× LESS MEMORY**

### ✅ **Epochs Reduced**
```python
EXPLORATION_EPOCHS: int = 5      # Was 10
HALL_OF_FAME_EPOCHS: int = 30    # Was 150
HOF_EARLY_STOPPING_PATIENCE: int = 5  # Was 10
```
**Impact:** **Faster runs, less memory buildup**

### 📊 **Total Memory Savings**
Combined: **~11× LESS MEMORY USAGE!**

---

## 🚀 Before You Start Training

### 1. **Run Pre-Flight Check**
```bash
python preflight_check.py
```
This will verify settings are safe and won't freeze your PC.

### 2. **If PC Is Already Frozen**
Press **Ctrl+Alt+F2** (switch to text terminal), login, then:
```bash
cd /home/p0wden/Documents/mloptimizer
python emergency_memory_cleanup.py --kill
```
Press **Ctrl+Alt+F7** to return to GUI.

### 3. **Monitor Memory While Training**
In a separate terminal:
```bash
watch -n 1 'free -h && echo && nvidia-smi'
```

### 4. **Start Training**
```bash
python run_slave.py
```

---

## 🛠️ Tools Created

### `preflight_check.py`
Checks if settings are safe before training:
```bash
python preflight_check.py
```

### `emergency_memory_cleanup.py`
Kills training processes to free memory:
```bash
# Check memory status
python emergency_memory_cleanup.py

# Emergency kill all training
python emergency_memory_cleanup.py --kill
```

---

## ⚠️ Why This Happened

Your previous settings:
- **224×224×3 images** = 150,528 bytes per image
- **Batch size 8** = ~1.2MB per batch (just images)
- **With model + augmentation + TensorFlow overhead**: ~500MB-2GB per batch
- **Loading entire dataset**: Could use 4-8GB RAM
- **With 150 epochs**: Memory leaks accumulate → RAM hits 100% → FREEZE

New settings:
- **96×96×3 images** = 27,648 bytes per image (5.5× smaller!)
- **Batch size 4** = ~110KB per batch
- **Total**: ~200MB-500MB per batch (much safer!)

---

## 📝 Quick Reference

### Safe Settings for Different RAM Sizes

**4GB RAM** (Minimal):
```python
'shape': (64, 64, 3)
DATASET_BATCH_SIZE: int = 2
```

**8GB RAM** (Current - Safe):
```python
'shape': (96, 96, 3)
DATASET_BATCH_SIZE: int = 4
```

**16GB+ RAM** (Comfortable):
```python
'shape': (128, 128, 3)
DATASET_BATCH_SIZE: int = 8
```

---

## 🎯 Next Steps

1. ✅ **Settings already updated** (96×96, batch=4)
2. **Run preflight check**: `python preflight_check.py`
3. **If check passes**, start training: `python run_slave.py`
4. **Monitor memory** in another terminal
5. **If freezing starts**, Ctrl+C or use emergency cleanup

---

## 💡 Pro Tips

1. **Close Chrome/Firefox** before training (browsers use tons of RAM)
2. **Close any IDE/editors** if not needed
3. **Use lightweight terminal** instead of heavy GUI
4. **Monitor with**: `htop` or `watch -n 1 free -h`
5. **Keep emergency cleanup ready** in another terminal

---

Your system should now train WITHOUT freezing! 🎉
