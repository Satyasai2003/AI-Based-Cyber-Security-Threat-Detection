import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

class CyberThreatDetection:
    """
    A comprehensive cyber threat detection system using machine learning.
    This implementation uses a Random Forest classifier as the base model but
    can be extended to use other algorithms.
    """
    
    def __init__(self, model_dir="models"):
        """
        Initialize the Cyber Threat Detection system.
        
        Args:
            model_dir (str): Directory to save trained models
        """
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def preprocess_data(self, data, target_col='is_threat', drop_cols=None):
        """
        Preprocess the input data for training or prediction.
        
        Args:
            data (DataFrame): Input data
            target_col (str): Name of the target column
            drop_cols (list): Columns to drop from the dataset
            
        Returns:
            X (DataFrame): Features
            y (Series): Target variable (if present in data)
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Drop unnecessary columns
        if drop_cols:
            df = df.drop(columns=drop_cols, errors='ignore')
        
        # Handle missing values
        df = df.fillna(0)
        
        # Extract target if present
        y = None
        if target_col in df.columns:
            y = df[target_col]
            df = df.drop(columns=[target_col])
        
        # Convert categorical features to numeric
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = pd.factorize(df[col])[0]
        
        return df, y
    
    def train(self, data, target_col='is_threat', drop_cols=None, test_size=0.2, random_state=42):
        """
        Train the threat detection model.
        
        Args:
            data (DataFrame): Training data
            target_col (str): Name of the target column
            drop_cols (list): Columns to drop from the dataset
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Training metrics
        """
        # Preprocess data
        X, y = self.preprocess_data(data, target_col, drop_cols)
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        print("Training base model for feature selection...")
        base_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        base_model.fit(X_train_scaled, y_train)
        
        # Select important features
        self.feature_selector = SelectFromModel(base_model, threshold="mean")
        self.feature_selector.fit(X_train_scaled, y_train)
        
        # Transform data using selected features
        X_train_selected = self.feature_selector.transform(X_train_scaled)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Get selected feature names
        selected_features = X_train.columns[self.feature_selector.get_support()]
        print(f"Selected {len(selected_features)} features: {selected_features.tolist()}")
        
        # Train final model
        print("Training final model...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=random_state
        )
        
        self.model.fit(X_train_selected, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_selected)
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Plot feature importance
        self._plot_feature_importance(self.model, selected_features)
        
        # Save model
        self._save_model()
        
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'selected_features': selected_features.tolist()
        }
    
    def predict(self, data, drop_cols=None):
        """
        Make predictions on new data.
        
        Args:
            data (DataFrame): Data to make predictions on
            drop_cols (list): Columns to drop from the dataset
            
        Returns:
            ndarray: Predicted classes
            ndarray: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Preprocess data
        X, _ = self.preprocess_data(data, target_col=None, drop_cols=drop_cols)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Select features
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Make predictions
        predictions = self.model.predict(X_selected)
        probabilities = self.model.predict_proba(X_selected)
        
        return predictions, probabilities
    
    def _plot_feature_importance(self, model, feature_names):
        """
        Plot feature importance.
        
        Args:
            model: Trained model
            feature_names: Names of features
        """
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance for Cyber Threat Detection')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f"{self.model_dir}/feature_importance.png")
        plt.close()
    
    def _save_model(self):
        """Save the trained model to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{self.model_dir}/threat_model_{timestamp}.joblib"
        scaler_path = f"{self.model_dir}/scaler_{timestamp}.joblib"
        selector_path = f"{self.model_dir}/feature_selector_{timestamp}.joblib"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_selector, selector_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path, scaler_path, selector_path):
        """
        Load a previously trained model.
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str): Path to the saved scaler
            selector_path (str): Path to the saved feature selector
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_selector = joblib.load(selector_path)
        print("Model loaded successfully")
    
    def evaluate_live_data(self, data_stream, batch_size=100, threshold=0.75):
        """
        Evaluate a live data stream for threats.
        
        Args:
            data_stream (iterable): Stream of data records
            batch_size (int): Number of records to process at once
            threshold (float): Probability threshold for threat classification
            
        Returns:
            Generator: Yields batches of detection results
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        batch = []
        for i, record in enumerate(data_stream):
            batch.append(record)
            
            if len(batch) >= batch_size:
                # Convert batch to DataFrame
                batch_df = pd.DataFrame(batch)
                
                # Make predictions
                _, probs = self.predict(batch_df, drop_cols=['timestamp', 'is_threat'])
                
                # Apply threshold
                threat_indices = np.where(probs[:, 1] > threshold)[0]
                threat_records = [batch[i] for i in threat_indices]
                
                # Reset batch
                batch = []
                
                yield {
                    'total_records': batch_size,
                    'threats_detected': len(threat_records),
                    'threat_indices': threat_indices.tolist(),
                    'threat_records': threat_records,
                    'threat_probabilities': probs[threat_indices, 1].tolist()
                }


# Example usage
def generate_sample_data(n_samples=10000):
    """Generate synthetic data for demonstration"""
    np.random.seed(42)
    
    # Create features
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='5min'),
        'src_ip': np.random.choice(['192.168.1.' + str(i) for i in range(1, 255)], n_samples),
        'dst_ip': np.random.choice(['10.0.0.' + str(i) for i in range(1, 255)], n_samples),
        'protocol': np.random.choice(['TCP', 'UDP', 'HTTP', 'HTTPS'], n_samples),
        'src_port': np.random.randint(1024, 65535, n_samples),
        'dst_port': np.random.choice([80, 443, 22, 53, 8080, 3389], n_samples),
        'packet_length': np.random.normal(500, 200, n_samples).astype(int),
        'packet_count': np.random.poisson(50, n_samples),
        'duration_ms': np.random.exponential(1000, n_samples).astype(int),
        'bytes_sent': np.random.poisson(1000, n_samples),
        'bytes_received': np.random.poisson(2000, n_samples),
        'tcp_flags': np.random.choice(['ACK', 'SYN', 'FIN', 'RST', 'SYN-ACK'], n_samples),
        'login_attempts': np.random.poisson(0.1, n_samples),
        'error_rate': np.random.beta(0.5, 10, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate synthetic attack patterns (about 5% of the data)
    attack_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    
    # Normal traffic
    df['is_threat'] = 0
    
    # SQL injection attempts
    sql_attacks = attack_indices[:len(attack_indices)//3]
    df.loc[sql_attacks, 'protocol'] = 'HTTP'
    df.loc[sql_attacks, 'dst_port'] = 80
    df.loc[sql_attacks, 'packet_length'] = np.random.normal(800, 100, len(sql_attacks)).astype(int)
    df.loc[sql_attacks, 'error_rate'] = np.random.beta(5, 1, len(sql_attacks))
    df.loc[sql_attacks, 'is_threat'] = 1
    
    # Brute force login attempts
    bf_attacks = attack_indices[len(attack_indices)//3:2*len(attack_indices)//3]
    df.loc[bf_attacks, 'login_attempts'] = np.random.poisson(10, len(bf_attacks))
    df.loc[bf_attacks, 'protocol'] = np.random.choice(['SSH', 'HTTP'], len(bf_attacks))
    df.loc[bf_attacks, 'dst_port'] = np.random.choice([22, 80, 443], len(bf_attacks))
    df.loc[bf_attacks, 'is_threat'] = 1
    
    # DDoS attacks
    ddos_attacks = attack_indices[2*len(attack_indices)//3:]
    df.loc[ddos_attacks, 'packet_count'] = np.random.poisson(500, len(ddos_attacks))
    df.loc[ddos_attacks, 'bytes_sent'] = np.random.poisson(5000, len(ddos_attacks))
    df.loc[ddos_attacks, 'protocol'] = 'TCP'
    df.loc[ddos_attacks, 'tcp_flags'] = 'SYN'
    df.loc[ddos_attacks, 'is_threat'] = 1
    
    return df


# Function for real-time threat monitoring
def monitor_network_traffic(detector, interval_seconds=1, max_iterations=10):
    """
    Simulate real-time network traffic monitoring.
    
    Args:
        detector: Trained CyberThreatDetection model
        interval_seconds: Monitoring interval in seconds
        max_iterations: Maximum number of monitoring iterations
    """
    import time
    
    print(f"Starting real-time network traffic monitoring (interval: {interval_seconds}s)")
    
    for i in range(max_iterations):
        # Generate a small batch of traffic data
        traffic_data = generate_sample_data(n_samples=50)
        
        # Make predictions
        predictions, probabilities = detector.predict(traffic_data, drop_cols=['timestamp', 'is_threat'])
        
        # Identify threats
        threats = traffic_data[predictions == 1]
        num_threats = len(threats)
        
        print(f"Iteration {i+1}: Analyzed 50 connections, detected {num_threats} potential threats")
        
        if num_threats > 0:
            # Print details of the first few threats
            max_display = min(3, num_threats)
            for j in range(max_display):
                threat = threats.iloc[j]
                print(f"  - Threat {j+1}: {threat['src_ip']} -> {threat['dst_ip']} "
                      f"({threat['protocol']}, port {threat['dst_port']})")
        
        time.sleep(interval_seconds)
    
    print("Monitoring complete")


# Main execution
if __name__ == "__main__":
    print("Cyber Threat Detection System")
    print("-----------------------------")
    
    # Generate sample data
    print("Generating synthetic network traffic data...")
    data = generate_sample_data(n_samples=50000)
    print(f"Generated {len(data)} traffic records with {data['is_threat'].sum()} threats")
    
    # Initialize and train the model
    detector = CyberThreatDetection(model_dir="threat_models")
    
    print("\nTraining threat detection model...")
    metrics = detector.train(
        data, 
        target_col='is_threat',
        drop_cols=['timestamp'],  # Drop non-predictive columns
        test_size=0.25
    )
    
    # Display training results
    print("\nModel Training Results:")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    clf_report = metrics['classification_report']
    print(f"\nClassification Report:")
    print(f"  Accuracy: {clf_report['accuracy']:.4f}")
    print(f"  Precision (threat class): {clf_report['1']['precision']:.4f}")
    print(f"  Recall (threat class): {clf_report['1']['recall']:.4f}")
    print(f"  F1 Score (threat class): {clf_report['1']['f1-score']:.4f}")
    
    # Simulate real-time monitoring
    print("\nStarting real-time threat monitoring simulation...")
    monitor_network_traffic(detector, interval_seconds=0.5, max_iterations=5)
    
    print("\nThreat detection model training and simulation complete.")
