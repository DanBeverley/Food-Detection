import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import logging
from PIL import Image
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_dataset_distribution(train_metadata_path, val_metadata_path, label_map_path):
    """Analyze the distribution of classes and instances in train/val splits."""
    
    with open(train_metadata_path, 'r') as f:
        train_data = json.load(f)
    with open(val_metadata_path, 'r') as f:
        val_data = json.load(f)
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    train_classes = Counter([item['class_name'] for item in train_data])
    val_classes = Counter([item['class_name'] for item in val_data])
    
    train_instances = defaultdict(set)
    val_instances = defaultdict(set)
    
    for item in train_data:
        train_instances[item['class_name']].add(item['instance_name'])
    for item in val_data:
        val_instances[item['class_name']].add(item['instance_name'])
    
    issues = []
    
    min_samples_threshold = 50
    for class_name, count in train_classes.items():
        if count < min_samples_threshold:
            issues.append(f"Class '{class_name}' has only {count} training samples")
    
    max_count = max(train_classes.values())
    min_count = min(train_classes.values())
    imbalance_ratio = max_count / min_count
    if imbalance_ratio > 10:
        issues.append(f"Severe class imbalance detected: ratio {imbalance_ratio:.2f}")
    
    train_only_classes = set(train_classes.keys()) - set(val_classes.keys())
    val_only_classes = set(val_classes.keys()) - set(train_classes.keys())
    
    if train_only_classes:
        issues.append(f"Classes only in training: {train_only_classes}")
    if val_only_classes:
        issues.append(f"Classes only in validation: {val_only_classes}")
    
    report = {
        'total_classes': len(label_map),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'train_classes': len(train_classes),
        'val_classes': len(val_classes),
        'imbalance_ratio': imbalance_ratio,
        'avg_samples_per_class_train': np.mean(list(train_classes.values())),
        'std_samples_per_class_train': np.std(list(train_classes.values())),
        'min_samples_per_class': min_count,
        'max_samples_per_class': max_count,
        'issues': issues
    }
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    top_20_classes = dict(train_classes.most_common(20))
    plt.bar(range(len(top_20_classes)), list(top_20_classes.values()))
    plt.xticks(range(len(top_20_classes)), list(top_20_classes.keys()), rotation=45, ha='right')
    plt.title('Top 20 Classes by Training Samples')
    plt.ylabel('Number of Samples')
    
    plt.subplot(2, 2, 2)
    plt.hist(list(train_classes.values()), bins=30, edgecolor='black')
    plt.xlabel('Number of Samples per Class')
    plt.ylabel('Number of Classes')
    plt.title('Distribution of Sample Counts')
    
    plt.subplot(2, 2, 3)
    common_classes = sorted(set(train_classes.keys()) & set(val_classes.keys()))[:20]
    train_counts = [train_classes[c] for c in common_classes]
    val_counts = [val_classes.get(c, 0) for c in common_classes]
    
    x = np.arange(len(common_classes))
    width = 0.35
    plt.bar(x - width/2, train_counts, width, label='Train')
    plt.bar(x + width/2, val_counts, width, label='Val')
    plt.xticks(x, common_classes, rotation=45, ha='right')
    plt.legend()
    plt.title('Train vs Validation Samples (Top 20 Classes)')
    
    plt.subplot(2, 2, 4)
    instances_per_class = [len(instances) for instances in train_instances.values()]
    plt.hist(instances_per_class, bins=20, edgecolor='black')
    plt.xlabel('Number of Instances per Class')
    plt.ylabel('Number of Classes')
    plt.title('Distribution of Instances per Class')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return report


def analyze_prediction_errors(model_path, val_metadata_path, label_map_path, num_samples=100):
    """Analyze where the model is making mistakes."""
    
    model = tf.keras.models.load_model(model_path)
    with open(val_metadata_path, 'r') as f:
        val_data = json.load(f)
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    label_to_idx = {v: int(k) for k, v in label_map.items()}
    
    sampled_data = np.random.choice(val_data, min(num_samples, len(val_data)), replace=False)
    
    confusion_data = defaultdict(lambda: defaultdict(int))
    confidence_by_correctness = {'correct': [], 'incorrect': []}
    
    for item in sampled_data:
        img = Image.open(item['image_path']).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)
        
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = label_map[str(predicted_idx)]
        confidence = predictions[predicted_idx]
        
        true_class = item['class_name']
        true_idx = label_to_idx[true_class]
        
        confusion_data[true_class][predicted_class] += 1
        
        if predicted_idx == true_idx:
            confidence_by_correctness['correct'].append(confidence)
        else:
            confidence_by_correctness['incorrect'].append(confidence)
    
    error_patterns = []
    for true_class, predictions in confusion_data.items():
        total = sum(predictions.values())
        correct = predictions.get(true_class, 0)
        accuracy = correct / total if total > 0 else 0
        
        if accuracy < 0.5:
            top_confused = sorted(
                [(pred, count) for pred, count in predictions.items() if pred != true_class],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            error_patterns.append({
                'class': true_class,
                'accuracy': accuracy,
                'confused_with': top_confused
            })
    
    return {
        'error_patterns': sorted(error_patterns, key=lambda x: x['accuracy']),
        'avg_confidence_correct': np.mean(confidence_by_correctness['correct']) if confidence_by_correctness['correct'] else 0,
        'avg_confidence_incorrect': np.mean(confidence_by_correctness['incorrect']) if confidence_by_correctness['incorrect'] else 0
    }


if __name__ == "__main__":
    report = analyze_dataset_distribution(
        'data/classification/train_metadata.json',
        'data/classification/val_metadata.json',
        'data/classification/label_map.json'
    )
    
    print("\n=== Dataset Analysis Report ===")
    for key, value in report.items():
        if key != 'issues':
            print(f"{key}: {value}")
    
    print("\n=== Identified Issues ===")
    for issue in report['issues']:
        print(f"- {issue}")
    
    model_path = '/kaggle/working/best_classification_model.keras'
    if Path(model_path).exists():
        error_analysis = analyze_prediction_errors(
            model_path,
            'data/classification/val_metadata.json',
            'data/classification/label_map.json'
        )
        
        print("\n=== Error Analysis ===")
        print(f"Avg confidence when correct: {error_analysis['avg_confidence_correct']:.3f}")
        print(f"Avg confidence when incorrect: {error_analysis['avg_confidence_incorrect']:.3f}")
        print("\nMost problematic classes:")
        for pattern in error_analysis['error_patterns'][:10]:
            print(f"- {pattern['class']} (acc: {pattern['accuracy']:.2f})")
            for confused_class, count in pattern['confused_with']:
                print(f"  → confused with {confused_class} ({count} times)")