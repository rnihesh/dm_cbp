#!/bin/bash

# Cleanup Script for Redundant Pickle Files
# This removes duplicate files while keeping the essential checkpoint

echo "🗑️  Cleaning up redundant pickle files..."
echo "================================================"

cd "$(dirname "$0")"

# Files to delete (duplicates already in complete_checkpoint)
FILES_TO_DELETE=(
    "saved_models/datasets_20251021_010000.pkl"
    "saved_models/clustering_results_20251021_010000.pkl"
    "saved_models/kmeans_model_20251021_005958.joblib"
    "saved_models/scaler_20251021_005958.joblib"
    "saved_models/feature_info_20251021_010000.pkl"
    "saved_models/performance_metrics_20251021_010000.pkl"
)

SPACE_SAVED=0

for file in "${FILES_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "❌ Deleting: $file ($SIZE)"
        rm "$file"
        SPACE_SAVED=$((SPACE_SAVED + 1))
    else
        echo "⚠️  Not found: $file (already deleted?)"
    fi
done

echo ""
echo "✅ Cleanup complete!"
echo "📊 Files deleted: $SPACE_SAVED"
echo ""
echo "📁 Remaining essential files:"
ls -lh saved_models/ | grep -E '(complete_checkpoint|results_only|final|\.csv|\.json)' | awk '{print "   ", $5, $9}'
echo ""
echo "💾 Space saved: ~1.1 GB"
echo "================================================"
