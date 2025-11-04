"""
Evaluation utilities for RoadBuddy challenge
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns


class RoadBuddyEvaluator:
    """Evaluate predictions on training set"""
    
    def __init__(self, ground_truth_path: str, predictions_path: str):
        """
        Initialize evaluator
        
        Args:
            ground_truth_path: Path to ground truth JSON (train.json)
            predictions_path: Path to predictions CSV
        """
        self.ground_truth = self._load_ground_truth(ground_truth_path)
        self.predictions = self._load_predictions(predictions_path)
        
        print(f"📊 Loaded {len(self.ground_truth)} ground truth samples")
        print(f"📊 Loaded {len(self.predictions)} predictions")
    
    def _load_ground_truth(self, path: str) -> Dict[str, str]:
        """Load ground truth answers"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)['data']
        
        # Map ID to answer letter
        gt_dict = {}
        for sample in data:
            answer_full = sample['answer']  # e.g., "A. Đúng"
            answer_letter = answer_full.split('.')[0].strip()
            gt_dict[sample['id']] = answer_letter
        
        return gt_dict
    
    def _load_predictions(self, path: str) -> Dict[str, str]:
        """Load predictions"""
        df = pd.read_csv(path)
        return dict(zip(df['id'], df['answer']))
    
    def calculate_accuracy(self) -> float:
        """Calculate overall accuracy"""
        correct = 0
        total = 0
        
        for sample_id, gt_answer in self.ground_truth.items():
            if sample_id in self.predictions:
                pred_answer = self.predictions[sample_id]
                if pred_answer == gt_answer:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def calculate_per_choice_accuracy(self) -> Dict[str, Dict]:
        """Calculate accuracy per number of choices"""
        # Load full data to get choice counts
        json_path = Path(list(self.ground_truth.keys())[0]).parent.parent / "train.json"
        
        stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        # Simple analysis by looking at answer distribution
        for sample_id, gt_answer in self.ground_truth.items():
            # Count available choices based on answer (rough estimate)
            if sample_id in self.predictions:
                pred = self.predictions[sample_id]
                
                # Determine num choices (rough heuristic)
                if gt_answer in ['A', 'B']:
                    num_choices = 2  # Likely binary
                else:
                    num_choices = 4  # Likely 4 choices
                
                stats[num_choices]['total'] += 1
                if pred == gt_answer:
                    stats[num_choices]['correct'] += 1
        
        # Calculate accuracy
        results = {}
        for num_choices, counts in stats.items():
            acc = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            results[num_choices] = {
                'accuracy': acc,
                'correct': counts['correct'],
                'total': counts['total']
            }
        
        return results
    
    def confusion_matrix(self) -> np.ndarray:
        """Generate confusion matrix"""
        # Determine unique labels
        all_labels = set(self.ground_truth.values()) | set(self.predictions.values())
        labels = sorted(all_labels)
        
        # Create matrix
        n = len(labels)
        matrix = np.zeros((n, n), dtype=int)
        label_to_idx = {label: i for i, label in enumerate(labels)}
        
        for sample_id, gt_answer in self.ground_truth.items():
            if sample_id in self.predictions:
                pred_answer = self.predictions[sample_id]
                i = label_to_idx[gt_answer]
                j = label_to_idx[pred_answer]
                matrix[i, j] += 1
        
        return matrix, labels
    
    def plot_confusion_matrix(self, save_path: str = "confusion_matrix.png"):
        """Plot and save confusion matrix"""
        matrix, labels = self.confusion_matrix()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📊 Confusion matrix saved to: {save_path}")
        plt.close()
    
    def analyze_errors(self, top_n: int = 10) -> List[Tuple[str, str, str]]:
        """Analyze most common errors"""
        errors = []
        
        for sample_id, gt_answer in self.ground_truth.items():
            if sample_id in self.predictions:
                pred_answer = self.predictions[sample_id]
                if pred_answer != gt_answer:
                    errors.append((sample_id, gt_answer, pred_answer))
        
        # Group by error type
        error_types = Counter([(gt, pred) for _, gt, pred in errors])
        
        print(f"\n❌ Total errors: {len(errors)} / {len(self.ground_truth)}")
        print(f"\n🔍 Top {top_n} error patterns:")
        for (gt, pred), count in error_types.most_common(top_n):
            print(f"   {gt} → {pred}: {count} times ({count/len(errors)*100:.1f}%)")
        
        return errors[:top_n]
    
    def generate_report(self, save_path: str = "evaluation_report.txt"):
        """Generate comprehensive evaluation report"""
        report = []
        report.append("=" * 70)
        report.append("RoadBuddy Evaluation Report")
        report.append("=" * 70)
        report.append("")
        
        # Overall accuracy
        accuracy = self.calculate_accuracy()
        report.append(f"Overall Accuracy: {accuracy*100:.2f}%")
        report.append("")
        
        # Per-choice accuracy
        report.append("Accuracy by Question Type:")
        per_choice = self.calculate_per_choice_accuracy()
        for num_choices, stats in sorted(per_choice.items()):
            report.append(f"  {num_choices} choices: {stats['accuracy']*100:.2f}% "
                         f"({stats['correct']}/{stats['total']})")
        report.append("")
        
        # Answer distribution
        report.append("Answer Distribution:")
        gt_dist = Counter(self.ground_truth.values())
        pred_dist = Counter(self.predictions.values())
        
        report.append("  Ground Truth:")
        for answer, count in sorted(gt_dist.items()):
            pct = count / len(self.ground_truth) * 100
            report.append(f"    {answer}: {count} ({pct:.1f}%)")
        
        report.append("  Predictions:")
        for answer, count in sorted(pred_dist.items()):
            pct = count / len(self.predictions) * 100
            report.append(f"    {answer}: {count} ({pct:.1f}%)")
        report.append("")
        
        # Confusion matrix summary
        matrix, labels = self.confusion_matrix()
        report.append("Confusion Matrix:")
        report.append("  " + "  ".join(f"{l:>4}" for l in labels))
        for i, label in enumerate(labels):
            row = "  ".join(f"{matrix[i, j]:>4}" for j in range(len(labels)))
            report.append(f"{label} {row}")
        report.append("")
        
        report.append("=" * 70)
        
        # Save report
        report_text = "\n".join(report)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n📄 Report saved to: {save_path}")
        
        return report_text


def main():
    """Evaluate predictions"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RoadBuddy predictions")
    parser.add_argument('--ground-truth', type=str,
                       default='traffic_buddy_train+public_test/train/train.json',
                       help='Path to ground truth JSON')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions CSV')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Directory to save evaluation outputs')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = RoadBuddyEvaluator(args.ground_truth, args.predictions)
    
    # Generate report
    report_path = Path(args.output_dir) / "evaluation_report.txt"
    evaluator.generate_report(str(report_path))
    
    # Plot confusion matrix
    cm_path = Path(args.output_dir) / "confusion_matrix.png"
    evaluator.plot_confusion_matrix(str(cm_path))
    
    # Analyze errors
    evaluator.analyze_errors(top_n=20)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

