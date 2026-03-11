# A/B Testing Framework for model comparison
from typing import Dict, Optional, List
from datetime import datetime
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ABTestFramework:
    """
    A/B testing framework for comparing model versions in production.
    """
    
    def __init__(
        self,
        experiment_name: str,
        control_model_id: str,
        treatment_model_id: str,
        traffic_split: float = 0.1
    ):
        """
        Initialize A/B test.
        
        Args:
            experiment_name: Name of the experiment
            control_model_id: Current production model
            treatment_model_id: Candidate model to test
            traffic_split: Percentage of traffic to treatment (0.0-1.0)
        """
        self.experiment_name = experiment_name
        self.control_model_id = control_model_id
        self.treatment_model_id = treatment_model_id
        self.traffic_split = traffic_split
        
        # Metrics storage
        self.control_metrics = {
            'predictions': [],
            'latencies': [],
            'errors': [],
            'true_labels': []
        }
        self.treatment_metrics = {
            'predictions': [],
            'latencies': [],
            'errors': [],
            'true_labels': []
        }
        
        self.start_time = datetime.now()
        
        logger.info(
            f"Started A/B test: {experiment_name} "
            f"(control: {control_model_id}, treatment: {treatment_model_id}, "
            f"split: {traffic_split*100:.1f}%)"
        )
    
    def assign_variant(self, user_id: str) -> str:
        """
        Assign user to control or treatment group.
        Uses consistent hashing for stable assignments.
        
        Args:
            user_id: User identifier
        
        Returns:
            'control' or 'treatment'
        """
        # Hash user_id to get consistent assignment
        hash_value = hash(user_id) % 100 / 100.0
        
        if hash_value < self.traffic_split:
            return 'treatment'
        return 'control'
    
    def log_prediction(
        self,
        variant: str,
        prediction: float,
        latency_ms: float,
        true_label: Optional[int] = None,
        error: Optional[str] = None
    ):
        """
        Log prediction for either control or treatment.
        
        Args:
            variant: 'control' or 'treatment'
            prediction: Model prediction
            latency_ms: Inference latency
            true_label: Ground truth label (when available)
            error: Error message if prediction failed
        """
        metrics = self.control_metrics if variant == 'control' else self.treatment_metrics
        
        metrics['predictions'].append(prediction)
        metrics['latencies'].append(latency_ms)
        metrics['errors'].append(error is not None)
        
        if true_label is not None:
            metrics['true_labels'].append(true_label)
    
    def calculate_metrics(self, variant: str) -> Dict:
        """
        Calculate performance metrics for a variant.
        
        Args:
            variant: 'control' or 'treatment'
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.control_metrics if variant == 'control' else self.treatment_metrics
        
        if not metrics['predictions']:
            return {}
        
        result = {
            'variant': variant,
            'sample_size': len(metrics['predictions']),
            
            # Prediction metrics
            'mean_prediction': float(np.mean(metrics['predictions'])),
            'prediction_std': float(np.std(metrics['predictions'])),
            
            # Latency metrics
            'mean_latency_ms': float(np.mean(metrics['latencies'])),
            'p95_latency_ms': float(np.percentile(metrics['latencies'], 95)),
            'p99_latency_ms': float(np.percentile(metrics['latencies'], 99)),
            
            # Error metrics
            'error_rate': float(np.mean(metrics['errors']))
        }
        
        # Calculate accuracy if labels available
        if metrics['true_labels']:
            predictions_binary = np.array(metrics['predictions']) > 0.5
            accuracy = np.mean(predictions_binary == np.array(metrics['true_labels']))
            result['accuracy'] = float(accuracy)
            
            # Calculate precision and recall
            tp = np.sum((predictions_binary == 1) & (np.array(metrics['true_labels']) == 1))
            fp = np.sum((predictions_binary == 1) & (np.array(metrics['true_labels']) == 0))
            fn = np.sum((predictions_binary == 0) & (np.array(metrics['true_labels']) == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            result['precision'] = float(precision)
            result['recall'] = float(recall)
            result['f1_score'] = float(f1)
        
        return result
    
    def compare_variants(self) -> Dict:
        """
        Statistical comparison of control vs treatment.
        
        Returns:
            Comparison results with statistical significance
        """
        control = self.calculate_metrics('control')
        treatment = self.calculate_metrics('treatment')
        
        if not control or not treatment:
            logger.warning("Insufficient data for comparison")
            return {}
        
        comparison = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'control': control,
            'treatment': treatment,
            'tests': {}
        }
        
        # Statistical tests
        
        # 1. Latency comparison (t-test)
        t_stat, p_value = stats.ttest_ind(
            self.control_metrics['latencies'],
            self.treatment_metrics['latencies']
        )
        comparison['tests']['latency'] = {
            'test': 't-test',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'winner': 'treatment' if treatment['mean_latency_ms'] < control['mean_latency_ms'] else 'control'
        }
        
        # 2. Error rate comparison (chi-square test)
        control_errors = sum(self.control_metrics['errors'])
        control_success = len(self.control_metrics['errors']) - control_errors
        treatment_errors = sum(self.treatment_metrics['errors'])
        treatment_success = len(self.treatment_metrics['errors']) - treatment_errors
        
        contingency_table = [[control_errors, control_success], [treatment_errors, treatment_success]]
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        
        comparison['tests']['error_rate'] = {
            'test': 'chi-square',
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'winner': 'treatment' if treatment['error_rate'] < control['error_rate'] else 'control'
        }
        
        # 3. Accuracy comparison (if labels available)
        if 'accuracy' in control and 'accuracy' in treatment:
            # Proportion test
            z_stat, p_value = self._proportion_test(
                control['accuracy'],
                treatment['accuracy'],
                control['sample_size'],
                treatment['sample_size']
            )
            
            comparison['tests']['accuracy'] = {
                'test': 'proportion-test',
                'z_statistic': float(z_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'winner': 'treatment' if treatment['accuracy'] > control['accuracy'] else 'control'
            }
        
        # Overall recommendation
        comparison['recommendation'] = self._make_recommendation(comparison)
        
        return comparison
    
    def _proportion_test(
        self,
        p1: float,
        p2: float,
        n1: int,
        n2: int
    ) -> tuple:
        """
        Two-proportion z-test.
        
        Returns:
            (z_statistic, p_value)
        """
        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        z = (p2 - p1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return z, p_value
    
    def _make_recommendation(self, comparison: Dict) -> Dict:
        """
        Make deployment recommendation based on test results.
        """
        tests = comparison['tests']
        treatment = comparison['treatment']
        control = comparison['control']
        
        # Count wins
        treatment_wins = sum(1 for test in tests.values() if test['winner'] == 'treatment')
        control_wins = sum(1 for test in tests.values() if test['winner'] == 'control')
        
        # Check for statistically significant improvements
        significant_improvements = sum(
            1 for test in tests.values()
            if test['winner'] == 'treatment' and test['significant']
        )
        
        # Check for degradations
        significant_degradations = sum(
            1 for test in tests.values()
            if test['winner'] == 'control' and test['significant']
        )
        
        # Make recommendation
        if significant_degradations > 0:
            decision = 'reject'
            reason = f"Treatment shows {significant_degradations} significant degradation(s)"
        elif significant_improvements >= 2:
            decision = 'promote'
            reason = f"Treatment shows {significant_improvements} significant improvement(s)"
        elif treatment_wins > control_wins and treatment['error_rate'] <= control['error_rate']:
            decision = 'promote_with_caution'
            reason = "Treatment performs better but lacks strong statistical significance"
        else:
            decision = 'continue_testing'
            reason = "Insufficient evidence to make decision. Continue collecting data."
        
        return {
            'decision': decision,
            'reason': reason,
            'treatment_wins': treatment_wins,
            'control_wins': control_wins,
            'significant_improvements': significant_improvements,
            'significant_degradations': significant_degradations,
            'confidence': 'high' if significant_improvements >= 2 else 'medium' if treatment_wins > control_wins else 'low'
        }
    
    def export_report(self) -> str:
        """
        Export human-readable test report.
        
        Returns:
            Markdown-formatted report
        """
        comparison = self.compare_variants()
        
        if not comparison:
            return "Insufficient data for report"
        
        report = f"""
# A/B Test Report: {self.experiment_name}

## Experiment Details
- **Start Time:** {comparison['start_time']}
- **Duration:** {comparison['duration_hours']:.2f} hours
- **Control Model:** {self.control_model_id}
- **Treatment Model:** {self.treatment_model_id}
- **Traffic Split:** {self.traffic_split*100:.1f}% to treatment

## Performance Metrics

### Control (Current Production)
- Sample Size: {comparison['control']['sample_size']}
- Mean Latency: {comparison['control']['mean_latency_ms']:.2f}ms
- P95 Latency: {comparison['control']['p95_latency_ms']:.2f}ms
- Error Rate: {comparison['control']['error_rate']:.2%}
"""
        
        if 'accuracy' in comparison['control']:
            report += f"- Accuracy: {comparison['control']['accuracy']:.2%}\n"
            report += f"- Precision: {comparison['control']['precision']:.2%}\n"
            report += f"- Recall: {comparison['control']['recall']:.2%}\n"
            report += f"- F1 Score: {comparison['control']['f1_score']:.3f}\n"
        
        report += f"""
### Treatment (Candidate Model)
- Sample Size: {comparison['treatment']['sample_size']}
- Mean Latency: {comparison['treatment']['mean_latency_ms']:.2f}ms
- P95 Latency: {comparison['treatment']['p95_latency_ms']:.2f}ms
- Error Rate: {comparison['treatment']['error_rate']:.2%}
"""
        
        if 'accuracy' in comparison['treatment']:
            report += f"- Accuracy: {comparison['treatment']['accuracy']:.2%}\n"
            report += f"- Precision: {comparison['treatment']['precision']:.2%}\n"
            report += f"- Recall: {comparison['treatment']['recall']:.2%}\n"
            report += f"- F1 Score: {comparison['treatment']['f1_score']:.3f}\n"
        
        report += "\n## Statistical Tests\n\n"
        
        for test_name, test_result in comparison['tests'].items():
            report += f"### {test_name.replace('_', ' ').title()}\n"
            report += f"- Test: {test_result['test']}\n"
            report += f"- P-value: {test_result['p_value']:.4f}\n"
            report += f"- Significant: {'✅ Yes' if test_result['significant'] else '❌ No'}\n"
            report += f"- Winner: **{test_result['winner'].upper()}**\n\n"
        
        report += "## Recommendation\n\n"
        rec = comparison['recommendation']
        report += f"**Decision:** {rec['decision'].upper()}\n\n"
        report += f"**Reason:** {rec['reason']}\n\n"
        report += f"**Confidence:** {rec['confidence'].upper()}\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Initialize A/B test
    ab_test = ABTestFramework(
        experiment_name="fraud_detector_v2_test",
        control_model_id="fraud_detector_v1.0",
        treatment_model_id="fraud_detector_v2.0",
        traffic_split=0.1
    )
    
    # Simulate predictions
    for i in range(1000):
        user_id = f"user_{i}"
        variant = ab_test.assign_variant(user_id)
        
        # Simulate prediction (treatment slightly better)
        if variant == 'treatment':
            prediction = np.random.beta(2, 3)
            latency = np.random.normal(80, 10)
        else:
            prediction = np.random.beta(2, 3.5)
            latency = np.random.normal(95, 15)
        
        true_label = int(prediction > 0.6)
        
        ab_test.log_prediction(
            variant=variant,
            prediction=prediction,
            latency_ms=latency,
            true_label=true_label
        )
    
    # Generate report
    report = ab_test.export_report()
    print(report)
