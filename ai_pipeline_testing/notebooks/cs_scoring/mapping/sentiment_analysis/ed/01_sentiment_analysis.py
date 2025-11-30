"""
Simple Sentiment Analysis - Just Run This!
No setup needed, everything automatic
"""

import os
from pathlib import Path

# Set your D: drive path
BASE_DIR = Path(__file__).parent  # Use the script's directory

# Create directories and set paths
for folder in ['models', 'datasets', 'cache', 'results']:
    (BASE_DIR / folder).mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(BASE_DIR / 'cache')
os.environ['TRANSFORMERS_CACHE'] = str(BASE_DIR / 'models')
os.environ['HF_DATASETS_CACHE'] = str(BASE_DIR / 'datasets')

# Now import everything
import torch
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

print(f"✓ All files will be saved to: {BASE_DIR}")
print()

# Configuration
MODEL_ID = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'  # Change if needed
BATCH_SIZE = 32

# Start timing
start_time = time.time()

# Load dataset
print("Loading Amazon Polarity dataset...")
dataset_start = time.time()
dataset = load_dataset("fancyzhx/amazon_polarity", trust_remote_code=True)
test_data = dataset['test'].select(range(5000))  # Use first 5000 samples

texts = [f"{item['title']} {item['content']}" for item in test_data]
labels = [item['label'] for item in test_data]
dataset_time = time.time() - dataset_start
print(f"✓ Loaded {len(texts)} samples in {dataset_time:.2f}s\n")

# Load model
print(f"Loading model: {MODEL_ID}")
model_start = time.time()
device = 0 if torch.cuda.is_available() else -1
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
classifier = pipeline("text-classification", model=MODEL_ID, tokenizer=tokenizer, device=device, truncation=True, max_length=128)
model_time = time.time() - model_start
print(f"✓ Model loaded on {'GPU' if device == 0 else 'CPU'} in {model_time:.2f}s\n")

# Predict
print("Running predictions...")
predict_start = time.time()
predictions = []
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i:i + BATCH_SIZE]
    predictions.extend(classifier(batch))
predict_time = time.time() - predict_start

# Map to binary
binary_preds = []
for pred in predictions:
    label = pred['label'].lower()
    if 'negative' in label:
        binary_preds.append(0)
    elif 'positive' in label:
        binary_preds.append(1)
    else:
        binary_preds.append(1)

# Calculate total time
total_time = time.time() - start_time

# Calculate metrics
print("\n" + "="*60)
print("RESULTS")
print("="*60)

accuracy = accuracy_score(labels, binary_preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, binary_preds, average='binary')
cm = confusion_matrix(labels, binary_preds)

print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")

print(f"\nConfusion Matrix:")
print(f"  TN: {cm[0][0]:6d}  |  FP: {cm[0][1]:6d}")
print(f"  FN: {cm[1][0]:6d}  |  TP: {cm[1][1]:6d}")

print(f"\nTiming:")
print(f"  Dataset loading:  {dataset_time:8.2f}s")
print(f"  Model loading:    {model_time:8.2f}s")
print(f"  Prediction:       {predict_time:8.2f}s")
print(f"  Total time:       {total_time:8.2f}s")
print(f"  Avg per sample:   {predict_time/len(texts)*1000:8.2f}ms")

# Save results
results_dir = BASE_DIR / 'results'
results_dir.mkdir(exist_ok=True)

# Save CSV
df = pd.DataFrame({
    'text': texts,
    'true_label': labels,
    'predicted_label': binary_preds,
    'correct': [t == p for t, p in zip(labels, binary_preds)]
})
csv_file = results_dir / 'predictions.csv'
df.to_csv(csv_file, index=False)
print(f"\n✓ Predictions saved: {csv_file}")

# Save summary metrics
summary_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score',
               'True Negatives (TN)', 'False Positives (FP)',
               'False Negatives (FN)', 'True Positives (TP)',
               'Dataset Load Time (s)', 'Model Load Time (s)',
               'Prediction Time (s)', 'Total Time (s)',
               'Avg Time per Sample (ms)', 'Total Samples', 'Batch Size'],
    'Value': [accuracy, precision, recall, f1,
              cm[0][0], cm[0][1], cm[1][0], cm[1][1],
              dataset_time, model_time, predict_time, total_time,
              predict_time/len(texts)*1000, len(texts), BATCH_SIZE]
})
summary_file = results_dir / 'summary_metrics.csv'
summary_df.to_csv(summary_file, index=False)
print(f"✓ Summary metrics saved: {summary_file}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            annot_kws={'size': 16, 'weight': 'bold', 'color': 'black'},
            cbar_kws={'label': 'Count'},
            vmin=0)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
img_file = results_dir / 'confusion_matrix.png'
plt.savefig(img_file, dpi=300, bbox_inches='tight')
print(f"✓ Chart saved: {img_file}")
plt.close()

# Plot metrics
fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]
colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
bars = ax.bar(metrics_names, values, color=colors)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Performance Metrics')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
img_file2 = results_dir / 'metrics.png'
plt.savefig(img_file2, dpi=300, bbox_inches='tight')
print(f"✓ Chart saved: {img_file2}")
plt.close()

print("\n" + "="*60)
print("✓ COMPLETE!")
print("="*60)
print(f"\nAll files saved to: {results_dir}")