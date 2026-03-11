# Model Monitoring - Track production model performance
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Monitor model performance, latency, and health in production.
    """
    
    def __init__(
        self,
        model_name: str,
        window_size: int = 1000,
        latency_threshold_ms: float = 100.0,
        error_rate_threshold: float = 0.01
    ):
        """
        Initialize model monitor.
        
        Args:
            model_name: Name of the model being monitored
            window_size: Number of predictions to keep in memory
            latency_threshold_ms: Alert if latency exceeds this
            error_rate_threshold: Alert if error rate exceeds this
        """
        self.model_name = model_name
        self.window_size = window_size
        self.latency_threshold_ms = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        
        # Metrics storage (in-memory, use Redis/Prometheus in prod)
        self.predictions = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Counters
        self.total_predictions = 0
        self.total_errors = 0
        
        logger.info(f"Initialized monitor for {model_name}")
    
    def log_prediction(
        self,
        prediction: float,
        latency_ms: float,
        features: Dict,
        error: Optional[str] = None
    ):
        """
        Log a single prediction event.
        
        Args:
            prediction: Model prediction value
            latency_ms: Inference latency in milliseconds
            features: Input features used
            error: Error message if prediction failed
        """
        self.predictions.append(prediction)
        self.latencies.append(latency_ms)
        self.errors.append(error is not None)
        self.timestamps.append(datetime.now())
        
        self.total_predictions += 1
        if error:
            self.total_errors += 1
            logger.error(f"Prediction error: {error}")
        
        # Check for threshold violations
        if latency_ms > self.latency_threshold_ms:
            logger.warning(
                f"High latency: {latency_ms:.2f}ms (threshold: {self.latency_threshold_ms}ms)"
            )
    
    def get_metrics(self, window_minutes: Optional[int] = None) -> Dict:
        """
        Get current monitoring metrics.
        
        Args:
            window_minutes: Time window to analyze (None = all data)
        
        Returns:
            Dictionary of metrics
        """
        # Filter by time window if specified
        if window_minutes:
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            indices = [i for i, ts in enumerate(self.timestamps) if ts >= cutoff]
            predictions = [self.predictions[i] for i in indices]
            latencies = [self.latencies[i] for i in indices]
            errors = [self.errors[i] for i in indices]
        else:
            predictions = list(self.predictions)
            latencies = list(self.latencies)
            errors = list(self.errors)
        
        if not predictions:
            return {}
        
        # Calculate metrics
        metrics = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'window_size': len(predictions),
            
            # Prediction metrics
            'prediction_mean': float(np.mean(predictions)),
            'prediction_std': float(np.std(predictions)),
            'prediction_min': float(np.min(predictions)),
            'prediction_max': float(np.max(predictions)),
            'prediction_percentiles': {
                'p50': float(np.percentile(predictions, 50)),
                'p95': float(np.percentile(predictions, 95)),
                'p99': float(np.percentile(predictions, 99))
            },
            
            # Latency metrics
            'latency_mean_ms': float(np.mean(latencies)),
            'latency_p50_ms': float(np.percentile(latencies, 50)),
            'latency_p95_ms': float(np.percentile(latencies, 95)),
            'latency_p99_ms': float(np.percentile(latencies, 99)),
            'latency_max_ms': float(np.max(latencies)),
            
            # Error metrics
            'error_count': sum(errors),
            'error_rate': float(sum(errors) / len(errors)) if errors else 0.0,
            
            # Throughput
            'requests_per_second': self._calculate_rps(window_minutes),
            
            # Lifetime metrics
            'total_predictions': self.total_predictions,
            'total_errors': self.total_errors,
            'lifetime_error_rate': float(self.total_errors / self.total_predictions) if self.total_predictions > 0 else 0.0
        }
        
        # Health status
        metrics['health_status'] = self._determine_health_status(metrics)
        
        return metrics
    
    def get_prediction_distribution(self, bins: int = 20) -> Dict:
        """
        Get distribution of predictions for drift detection.
        
        Returns:
            Histogram of predictions
        """
        if not self.predictions:
            return {}
        
        hist, bin_edges = np.histogram(list(self.predictions), bins=bins)
        
        return {
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'total_predictions': len(self.predictions)
        }
    
    def check_anomalies(self) -> List[Dict]:
        """
        Check for anomalous patterns in monitoring data.
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        metrics = self.get_metrics()
        
        if not metrics:
            return anomalies
        
        # High latency
        if metrics['latency_p95_ms'] > self.latency_threshold_ms:
            anomalies.append({
                'type': 'high_latency',
                'severity': 'warning',
                'value': metrics['latency_p95_ms'],
                'threshold': self.latency_threshold_ms,
                'message': f"P95 latency ({metrics['latency_p95_ms']:.2f}ms) exceeds threshold"
            })
        
        # High error rate
        if metrics['error_rate'] > self.error_rate_threshold:
            anomalies.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'value': metrics['error_rate'],
                'threshold': self.error_rate_threshold,
                'message': f"Error rate ({metrics['error_rate']:.2%}) exceeds threshold"
            })
        
        # Unusual prediction distribution
        if metrics['prediction_std'] > 2 * metrics['prediction_mean']:
            anomalies.append({
                'type': 'high_variance',
                'severity': 'warning',
                'value': metrics['prediction_std'],
                'message': "Prediction variance is unusually high"
            })
        
        return anomalies
    
    def _calculate_rps(self, window_minutes: Optional[int] = None) -> float:
        """Calculate requests per second."""
        if not self.timestamps:
            return 0.0
        
        if window_minutes:
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            recent_timestamps = [ts for ts in self.timestamps if ts >= cutoff]
            count = len(recent_timestamps)
            duration = window_minutes * 60
        else:
            count = len(self.timestamps)
            duration = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
        
        return count / duration if duration > 0 else 0.0
    
    def _determine_health_status(self, metrics: Dict) -> str:
        """Determine overall health status."""
        if metrics['error_rate'] > self.error_rate_threshold:
            return 'unhealthy'
        
        if metrics['latency_p95_ms'] > self.latency_threshold_ms * 1.5:
            return 'degraded'
        
        if metrics['latency_p95_ms'] > self.latency_threshold_ms:
            return 'warning'
        
        return 'healthy'
    
    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        metrics = self.get_metrics()
        
        if not metrics:
            return ""
        
        lines = [
            f"# HELP model_predictions_total Total number of predictions",
            f"# TYPE model_predictions_total counter",
            f'model_predictions_total{{model="{self.model_name}"}} {self.total_predictions}',
            f"",
            f"# HELP model_errors_total Total number of errors",
            f"# TYPE model_errors_total counter",
            f'model_errors_total{{model="{self.model_name}"}} {self.total_errors}',
            f"",
            f"# HELP model_latency_ms Prediction latency in milliseconds",
            f"# TYPE model_latency_ms histogram",
            f'model_latency_ms{{model="{self.model_name}",quantile="0.5"}} {metrics["latency_p50_ms"]}',
            f'model_latency_ms{{model="{self.model_name}",quantile="0.95"}} {metrics["latency_p95_ms"]}',
            f'model_latency_ms{{model="{self.model_name}",quantile="0.99"}} {metrics["latency_p99_ms"]}',
            f"",
            f"# HELP model_prediction_mean Mean prediction value",
            f"# TYPE model_prediction_mean gauge",
            f'model_prediction_mean{{model="{self.model_name}"}} {metrics["prediction_mean"]}',
        ]
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    import time
    import random
    
    # Initialize monitor
    monitor = ModelMonitor(
        model_name="fraud_detector_v1",
        window_size=1000,
        latency_threshold_ms=100.0
    )
    
    # Simulate predictions
    for i in range(100):
        prediction = random.random()
        latency = random.uniform(10, 150)
        features = {'amount': random.uniform(10, 1000)}
        
        monitor.log_prediction(
            prediction=prediction,
            latency_ms=latency,
            features=features
        )
        
        time.sleep(0.01)
    
    # Get metrics
    metrics = monitor.get_metrics()
    print(f"Metrics: {metrics}")
    
    # Check for anomalies
    anomalies = monitor.check_anomalies()
    if anomalies:
        print(f"Anomalies detected: {anomalies}")
    
    # Export Prometheus metrics
    prom_metrics = monitor.export_prometheus_metrics()
    print(f"\nPrometheus metrics:\n{prom_metrics}")
