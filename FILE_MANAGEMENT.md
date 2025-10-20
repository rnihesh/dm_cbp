# File Management Guide

## 📦 Pickle Files Explanation

After running the notebook and saving, you'll have multiple pickle files. Here's what each one does:

### ⭐ Essential Files (KEEP)

| File | Size | Purpose | When to Use |
|------|------|---------|-------------|
| `complete_checkpoint_*.pkl` | ~1.1GB | **Complete state** - All data, models, results | Resume full analysis |
| `results_only_*.pkl` | ~2KB | Lightweight metrics only | Quick result checks |
| `kmeans_final.joblib` | ~6MB | Standalone K-Means model | Production deployment |
| `scaler_final.joblib` | ~2KB | Standalone scaler | Production deployment |
| `feature_scores_*.csv` | ~1KB | Feature importance (readable) | Manual inspection |
| `performance_metrics_*.json` | ~1KB | Metrics (readable) | Manual inspection |

### ❌ Redundant Files (SAFE TO DELETE)

| File | Size | Why Redundant |
|------|------|---------------|
| `datasets_*.pkl` | ~1.1GB | Already in `complete_checkpoint` |
| `clustering_results_*.pkl` | ~12MB | Already in `complete_checkpoint` |
| `kmeans_model_*.joblib` | ~6MB | Duplicate of `kmeans_final.joblib` |
| `scaler_*.joblib` | ~2KB | Duplicate of `scaler_final.joblib` |
| `feature_info_*.pkl` | ~2KB | Already in `complete_checkpoint` |
| `performance_metrics_*.pkl` | ~1KB | Already in `complete_checkpoint` + have JSON |

**Total redundancy:** ~1.1GB

---

## 🗑️ Cleanup Instructions

### Option 1: Automatic (Recommended)

```bash
# Run the cleanup script
./cleanup_duplicates.sh
```

This safely removes all duplicate files while keeping essentials.

### Option 2: Manual

```bash
cd saved_models/

# Delete duplicate datasets
rm datasets_*.pkl

# Delete duplicate clustering results
rm clustering_results_*.pkl

# Delete timestamped models (keep *_final.joblib)
rm kmeans_model_*.joblib
rm scaler_*.joblib

# Delete duplicate feature info
rm feature_info_*.pkl

# Delete pickle metrics (keep JSON)
rm performance_metrics_*.pkl
```

---

## 💾 What You Need to Resume

**After cleanup, to resume your work after restart:**

1. **Open notebook** → `test.ipynb`
2. **Load checkpoint** → Use `complete_checkpoint_*.pkl`
3. **That's it!** Everything restored in 10 seconds

The complete checkpoint contains:
- ✅ All train/val/test datasets
- ✅ Scaled data
- ✅ Trained models (K-Means, Scaler)
- ✅ All cluster assignments
- ✅ All performance metrics
- ✅ Feature information
- ✅ Configuration parameters

---

## 📊 Disk Space Savings

| Before Cleanup | After Cleanup | Saved |
|----------------|---------------|-------|
| ~2.3 GB | ~1.1 GB | ~1.1 GB |

---

## 🚀 Production Deployment

If you want to deploy just the models (not for research):

**Keep only:**
- `kmeans_final.joblib` (~6MB)
- `scaler_final.joblib` (~2KB)
- `feature_scores_*.csv` (for reference)

**Total:** ~6MB for production!

Then load in your Python app:
```python
import joblib
kmeans = joblib.load('saved_models/kmeans_final.joblib')
scaler = joblib.load('saved_models/scaler_final.joblib')

# Use for predictions
new_data_scaled = scaler.transform(new_data)
predictions = kmeans.predict(new_data_scaled)
```

---

## 📝 Summary

**For research/resume work:**
- Keep: `complete_checkpoint_*.pkl` (1.1GB)
- Delete: Everything else except CSV/JSON files

**For production deployment:**
- Keep: `*_final.joblib` files (6MB total)
- Delete: Everything else

**For quick metrics check:**
- Keep: `results_only_*.pkl` (2KB)
- Or: `performance_metrics_*.json` (human-readable)
