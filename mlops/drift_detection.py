# Drift Detection - Monitor for data and concept drift
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detect feature drift and concept drift in production models.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            significance_level: P-value threshold for statistical tests
        """
        self.significance_level = significance_level
        self.baseline_stats = {}
    
    def set_baseline(self, name: str, data: np.ndarray):
        """
        Set baseline distribution for a feature.
        
        Args:
            name: Feature name
            data: Training data distribution
        """
        self.baseline_stats[name] = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'quantiles': np.percentile(data, [25, 50, 75]),
            'sample': data[:1000] if len(data) > 1000 else data
        }
        
        logger.info(f"Set baseline for {name}: mean={self.baseline_stats[name]['mean']:.2f}")
    
    def detect_feature_drift(
        self,
        feature_name: str,
        current_data: np.ndarray,
        method: str = 'ks'
    ) -> Tuple[bool, float, Dict]:
        """
        Detect drift in a single feature.
        
        Args:
            feature_name: Name of the feature
            current_data: Current production data
            method: Statistical test ('ks', 'chi2', 'psi')
        
        Returns:
            (drift_detected, p_value, statistics)
        """
        if feature_name not in self.baseline_stats:
            raise ValueError(f"No baseline set for {feature_name}")
        
        baseline = self.baseline_stats[feature_name]['sample']
        
        if method == 'ks':
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(baseline, current_data)
            drift_detected = p_value < self.significance_level
            
        elif method == 'chi2':
            # Chi-square test for categorical features
            # Bin continuous features first
            baseline_hist, bins = np.histogram(baseline, bins=10)
            current_hist, _ = np.histogram(current_data, bins=bins)
            
            statistic, p_value = stats.chisquare(current_hist, baseline_hist)
            drift_detected = p_value < self.significance_level
            
        elif method == 'psi':
            # Population Stability Index
            statistic = self._calculate_psi(baseline, current_data)
            p_value = None  # PSI doesn't have p-value
            drift_detected = statistic > 0.2  # PSI > 0.2 indicates significant drift
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        stats_dict = {
            'method': method,
            'statistic': float(statistic),
            'p_value': float(p_value) if p_value is not None else None,
            'drift_detected': drift_detected,
            'baseline_mean': float(self.baseline_stats[feature_name]['mean']),
            'current_mean': float(np.mean(current_data)),
            'baseline_std': float(self.baseline_stats[feature_name]['std']),
            'current_std': float(np.std(current_data)),
            'timestamp': datetime.now().isoformat()
        }
        
        if drift_detected:
            logger.warning(f"Drift detected in {feature_name}: {method}={statistic:.4f}, p={p_value}")
        
        return drift_detected, p_value or statistic, stats_dict
    
    def detect_prediction_drift(
        self,
        baseline_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Tuple[bool, Dict]:
        """
        Detect drift in model predictions.
        
        Args:
            baseline_predictions: Historical predictions
            current_predictions: Recent predictions
        
        Returns:
            (drift_detected, statistics)
        """
        # Test for distribution shift
        statistic, p_value = stats.ks_2samp(baseline_predictions, current_predictions)
        drift_detected = p_value < self.significance_level
        
        # Calculate additional metrics
        baseline_mean = np.mean(baseline_predictions)
        current_mean = np.mean(current_predictions)
        mean_shift = abs(current_mean - baseline_mean)
        
        # Calculate prediction concentration (fraud rate proxy)
        baseline_fraud_rate = np.mean(baseline_predictions > 0.5)
        current_fraud_rate = np.mean(current_predictions > 0.5)
        
        stats_dict = {
            'ks_statistic': float(statistic),
            'p_value': float(p_value),
            'drift_detected': drift_detected,
            'baseline_mean': float(baseline_mean),
            'current_mean': float(current_mean),
            'mean_shift': float(mean_shift),
            'baseline_fraud_rate': float(baseline_fraud_rate),
            'current_fraud_rate': float(current_fraud_rate),
            'fraud_rate_change': float(current_fraud_rate - baseline_fraud_rate),
            'timestamp': datetime.now().isoformat()
        }
        
        if drift_detected:
            logger.warning(
                f"Prediction drift detected: mean shift={mean_shift:.4f}, "
                f"fraud rate change={stats_dict['fraud_rate_change']:.4f}"
            )
        
        return drift_detected, stats_dict
    
    def detect_concept_drift(
        self,
        predictions: np.ndarray,
        actual_labels: np.ndarray,
        window_size: int = 100
    ) -> Tuple[bool, Dict]:
        """
        Detect concept drift by monitoring accuracy over time.
        
        Args:
            predictions: Model predictions
            actual_labels: True labels (when available)
            window_size: Size of sliding window
        
        Returns:
            (drift_detected, statistics)
        """
        if len(predictions) < window_size * 2:
            logger.warning("Insufficient data for concept drift detection")
            return False, {}
        
        # Split into windows
        mid_point = len(predictions) // 2
        window_1 = predictions[:mid_point]
        window_2 = predictions[mid_point:]
        labels_1 = actual_labels[:mid_point]
        labels_2 = actual_labels[mid_point:]
        
        # Calculate accuracy for each window
        acc_1 = np.mean((window_1 > 0.5) == labels_1)
        acc_2 = np.mean((window_2 > 0.5) == labels_2)
        
        # Significant accuracy drop indicates concept drift
        accuracy_drop = acc_1 - acc_2
        drift_detected = accuracy_drop > 0.05  # 5% drop threshold
        
        stats_dict = {
            'early_accuracy': float(acc_1),
            'recent_accuracy': float(acc_2),
            'accuracy_drop': float(accuracy_drop),
            'drift_detected': drift_detected,
            'timestamp': datetime.now().isoformat()
        }
        
        if drift_detected:
            logger.warning(f"Concept drift detected: accuracy drop={accuracy_drop:.4f}")
        
        return drift_detected, stats_dict
    
    def _calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI > 0.2 indicates significant drift
        """
        # Create bins based on baseline distribution
        bin_edges = np.percentile(baseline, np.linspace(0, 100, bins + 1))
        
        # Calculate distributions
        baseline_dist, _ = np.histogram(baseline, bins=bin_edges)
        current_dist, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize
        baseline_pct = baseline_dist / len(baseline)
        current_pct = current_dist / len(current)
        
        # Avoid division by zero
        baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
        current_pct = np.where(current_pct == 0, 0.0001, current_pct)
        
        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        
        return psi
    
    def generate_drift_report(self, features_data: Dict[str, np.ndarray]) -> Dict:
        """
        Generate comprehensive drift report for all features.
        
        Args:
            features_data: Dictionary mapping feature names to current data
        
        Returns:
            Drift report with statistics for all features
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'features': {},
            'overall_drift_detected': False
        }
        
        for feature_name, data in features_data.items():
            if feature_name not in self.baseline_stats:
                logger.warning(f"Skipping {feature_name}: no baseline set")
                continue
            
            drift_detected, _, stats = self.detect_feature_drift(feature_name, data)
            report['features'][feature_name] = stats
            
            if drift_detected:
                report['overall_drift_detected'] = True
        
        return report


class DriftMonitor:
    """
    Continuous drift monitoring with alerting.
    """
    
    def __init__(self, detector: DriftDetector, alert_threshold: int = 3):
        self.detector = detector
        self.alert_threshold = alert_threshold
        self.drift_counts = {}
    
    def check_and_alert(
        self,
        feature_name: str,
        current_data: np.ndarray
    ) -> Optional[Dict]:
        """
        Check for drift and trigger alert if threshold exceeded.
        
        Returns:
            Alert details if threshold exceeded, None otherwise
        """
        drift_detected, _, stats = self.detector.detect_feature_drift(
            feature_name,
            current_data
        )
        
        if drift_detected:
            self.drift_counts[feature_name] = self.drift_counts.get(feature_name, 0) + 1
            
            if self.drift_counts[feature_name] >= self.alert_threshold:
                alert = {
                    'feature': feature_name,
                    'consecutive_drift_detections': self.drift_counts[feature_name],
                    'stats': stats,
                    'severity': 'high',
                    'action_required': 'Model retraining recommended'
                }
                
                logger.error(f"DRIFT ALERT: {feature_name} - {alert}")
                return alert
        else:
            self.drift_counts[feature_name] = 0
        
        return None


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = DriftDetector()
    
    # Set baseline from training data
    baseline_amounts = np.random.normal(100, 50, 10000)
    detector.set_baseline('transaction_amount', baseline_amounts)
    
    # Simulate production data with drift
    drifted_amounts = np.random.normal(150, 60, 1000)  # Distribution shifted
    
    # Detect drift
    drift_detected, p_value, stats = detector.detect_feature_drift(
        'transaction_amount',
        drifted_amounts
    )
    
    print(f"Drift detected: {drift_detected}")
    print(f"Statistics: {stats}")
    
    # Monitor continuously
    monitor = DriftMonitor(detector)
    alert = monitor.check_and_alert('transaction_amount', drifted_amounts)
    
    if alert:
        print(f"Alert triggered: {alert}")
