import pandas as pd
import numpy as np
import ipaddress
import random
import joblib
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

# Reusing your CyberThreatDataGenerator class
class CyberThreatDataGenerator:
    """
    Advanced data generator for cybersecurity threat detection models.
    Creates realistic network traffic data with various attack patterns.
    """
    
    def __init__(self, seed=42):
        """
        Initialize the cyber threat data generator.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Define common network parameters
        self.internal_networks = ['192.168.0.0/16', '10.0.0.0/8', '172.16.0.0/12']
        self.external_networks = ['34.56.0.0/16', '45.67.0.0/16', '23.45.0.0/16', '209.85.0.0/16']
        self.common_ports = {
            'HTTP': 80,
            'HTTPS': 443,
            'SSH': 22,
            'FTP': 21,
            'SMTP': 25,
            'DNS': 53,
            'RDP': 3389,
            'SMB': 445,
            'TELNET': 23,
            'MYSQL': 3306,
            'MSSQL': 1433,
            'POSTGRESQL': 5432,
            'REDIS': 6379,
            'MONGODB': 27017
        }
        
        # Define attack patterns
        self.attack_types = [
            'SQL_INJECTION',
            'BRUTE_FORCE',
            'DDOS',
            'PORT_SCAN',
            'MALWARE_COMMUNICATION',
            'DATA_EXFILTRATION',
            'LATERAL_MOVEMENT',
            'PHISHING',
            'XSS',
            'UNAUTHORIZED_ACCESS'
        ]
    
    def _generate_ip(self, network_range):
        """Generate a random IP address within a CIDR range."""
        network = ipaddress.ip_network(network_range)
        # Get the size of the network
        network_size = network.num_addresses
        # Generate a random integer between 0 and network_size - 1
        host_index = random.randint(0, network_size - 1)
        # Return the IP address as string
        return str(network[host_index])
    
    def _generate_internal_ip(self):
        """Generate a random internal (private) IP address."""
        network = random.choice(self.internal_networks)
        return self._generate_ip(network)
    
    def _generate_external_ip(self):
        """Generate a random external (public) IP address."""
        network = random.choice(self.external_networks)
        return self._generate_ip(network)
    
    def _generate_timestamp_series(self, start_date, n_samples, interval_min=5, random_offset=True):
        """Generate a series of timestamps with an optional random offset."""
        # Create base timestamp series
        if random_offset:
            # Add random noise to the intervals (between 0 and interval_min minutes)
            offsets = np.cumsum(np.random.uniform(0, interval_min * 60, n_samples))
            timestamps = [start_date + timedelta(seconds=offset) for offset in offsets]
        else:
            # Regular intervals
            timestamps = pd.date_range(start=start_date, periods=n_samples, freq=f'{interval_min}min')
        
        return timestamps
    
    def _generate_normal_traffic(self, n_samples, start_date):
        """Generate normal network traffic data."""
        # Create base data dictionary
        data = {
            'timestamp': self._generate_timestamp_series(start_date, n_samples),
            'src_ip': [self._generate_internal_ip() for _ in range(n_samples)],
            'dst_ip': [],
            'protocol': [],
            'src_port': np.random.randint(1024, 65535, n_samples),
            'dst_port': [],
            'packet_length': np.random.normal(500, 200, n_samples).astype(int),
            'packet_count': np.random.poisson(50, n_samples),
            'duration_ms': np.random.exponential(1000, n_samples).astype(int),
            'bytes_sent': np.random.poisson(1000, n_samples),
            'bytes_received': np.random.poisson(2000, n_samples),
            'tcp_flags': np.random.choice(['ACK', 'SYN', 'FIN', 'RST', 'SYN-ACK'], n_samples),
            'login_attempts': np.random.poisson(0.1, n_samples),
            'error_rate': np.random.beta(0.5, 10, n_samples),
            'http_method': [],
            'http_endpoint': [],
            'http_status': [],
            'dns_query': [],
            'tls_version': [],
            'user_agent': [],
            'is_threat': np.zeros(n_samples, dtype=int),
            'threat_type': ['NONE'] * n_samples
        }
        
        # Fill in protocol-specific details
        protocols = np.random.choice(['HTTP', 'HTTPS', 'DNS', 'SSH', 'FTP', 'SMB', 'RDP'], 
                                    n_samples, 
                                    p=[0.4, 0.3, 0.1, 0.05, 0.05, 0.05, 0.05])
        data['protocol'] = protocols.tolist()  # Initialize protocol list
        
        for i, protocol in enumerate(protocols):
            # Set destination IP (80% external for HTTP/HTTPS/DNS, 80% internal for others)
            if protocol in ['HTTP', 'HTTPS', 'DNS']:
                data['dst_ip'].append(self._generate_external_ip() if random.random() < 0.8 else self._generate_internal_ip())
            else:
                data['dst_ip'].append(self._generate_internal_ip() if random.random() < 0.8 else self._generate_external_ip())
            
            # Set destination port based on protocol
            if protocol in self.common_ports:
                data['dst_port'].append(self.common_ports[protocol])
            else:
                data['dst_port'].append(random.randint(1, 65535))
            
            # HTTP and HTTPS specific fields
            if protocol in ['HTTP', 'HTTPS']:
                data['http_method'].append(np.random.choice(['GET', 'POST', 'PUT', 'DELETE'], p=[0.7, 0.2, 0.05, 0.05]))
                endpoints = ['/index.html', '/api/v1/users', '/login', '/images/logo.png', '/css/style.css', '/js/main.js']
                data['http_endpoint'].append(random.choice(endpoints))
                data['http_status'].append(np.random.choice([200, 301, 302, 304, 400, 404, 500], p=[0.8, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01]))
                user_agents = [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
                ]
                data['user_agent'].append(random.choice(user_agents))
                data['tls_version'].append('TLSv1.3' if protocol == 'HTTPS' else None)
            else:
                data['http_method'].append(None)
                data['http_endpoint'].append(None)
                data['http_status'].append(None)
                data['user_agent'].append(None)
                data['tls_version'].append(None)
                
            # DNS specific fields
            if protocol == 'DNS':
                domains = ['example.com', 'google.com', 'github.com', 'microsoft.com', 'amazon.com']
                data['dns_query'].append(random.choice(domains))
            else:
                data['dns_query'].append(None)
        
        return data
    
    def _inject_sql_injection_attacks(self, data, attack_ratio=0.05):
        """Inject SQL injection attack patterns into the dataset."""
        n_samples = len(data['timestamp'])
        n_attacks = int(n_samples * attack_ratio)
        attack_indices = np.random.choice(n_samples, size=n_attacks, replace=False)
        
        sql_injection_patterns = [
            "' OR '1'='1", 
            "admin' --", 
            "'; DROP TABLE users; --",
            "1 UNION SELECT username, password FROM users",
            "1; SELECT * FROM information_schema.tables"
        ]
        
        for idx in attack_indices:
            # Set attack signature
            data['protocol'][idx] = 'HTTP'
            data['dst_port'][idx] = self.common_ports['HTTP']
            data['http_method'][idx] = 'POST'
            data['http_endpoint'][idx] = '/login' if random.random() < 0.7 else '/api/v1/users'
            data['packet_length'][idx] = np.random.normal(800, 100)
            data['error_rate'][idx] = np.random.beta(5, 1)
            data['http_status'][idx] = np.random.choice([200, 500], p=[0.3, 0.7])
            
            # Add SQL injection payload to user agent
            if data['user_agent'][idx]:
                injection = random.choice(sql_injection_patterns)
                data['user_agent'][idx] = f"Malicious/1.0 (SQL Injection: {injection})"
            
            # Mark as threat
            data['is_threat'][idx] = 1
            data['threat_type'][idx] = 'SQL_INJECTION'
        
        return data, attack_indices
    
    def _inject_brute_force_attacks(self, data, attack_ratio=0.05):
        """Inject brute force attack patterns into the dataset."""
        n_samples = len(data['timestamp'])
        n_attacks = int(n_samples * attack_ratio)
        attack_indices = np.random.choice(n_samples, size=n_attacks, replace=False)
        
        for idx in attack_indices:
            # Set attack signature
            protocol = np.random.choice(['SSH', 'HTTP', 'FTP'], p=[0.5, 0.3, 0.2])
            data['protocol'][idx] = protocol
            data['dst_port'][idx] = self.common_ports[protocol]
            data['login_attempts'][idx] = np.random.poisson(10)
            data['duration_ms'][idx] = np.random.uniform(100, 300)  # Fast attempts
            
            if protocol == 'HTTP':
                data['http_method'][idx] = 'POST'
                data['http_endpoint'][idx] = '/login'
                data['http_status'][idx] = 401  # Unauthorized
            
            # Mark as threat
            data['is_threat'][idx] = 1
            data['threat_type'][idx] = 'BRUTE_FORCE'
        
        return data, attack_indices
    
    def _inject_ddos_attacks(self, data, attack_ratio=0.05):
        """Inject DDoS attack patterns into the dataset."""
        n_samples = len(data['timestamp'])
        n_attacks = int(n_samples * attack_ratio)
        attack_indices = np.random.choice(n_samples, size=n_attacks, replace=False)
        
        # Group attacks to come from similar source IPs
        attack_src_ips = [self._generate_external_ip() for _ in range(10)]
        
        for idx in attack_indices:
            # Set attack signature
            data['protocol'][idx] = 'TCP'
            data['src_ip'][idx] = random.choice(attack_src_ips)  # Same botnet sources
            data['dst_ip'][idx] = self._generate_internal_ip()  # Target internal servers
            data['dst_port'][idx] = np.random.choice([80, 443])  # Target web servers
            data['packet_count'][idx] = np.random.poisson(500)
            data['bytes_sent'][idx] = np.random.poisson(5000)
            data['tcp_flags'][idx] = 'SYN'  # SYN flood
            data['duration_ms'][idx] = np.random.uniform(10, 100)  # Very short connections
            
            # Mark as threat
            data['is_threat'][idx] = 1
            data['threat_type'][idx] = 'DDOS'
        
        return data, attack_indices
    
    def _inject_port_scan_attacks(self, data, attack_ratio=0.05):
        """Inject port scanning attack patterns into the dataset."""
        n_samples = len(data['timestamp'])
        n_attacks = int(n_samples * attack_ratio)
        attack_indices = np.random.choice(n_samples, size=n_attacks, replace=False)
        
        # Use same source for all port scans
        scan_src_ip = self._generate_external_ip()
        
        for idx in attack_indices:
            # Set attack signature
            data['src_ip'][idx] = scan_src_ip
            data['dst_ip'][idx] = self._generate_internal_ip()
            data['dst_port'][idx] = np.random.randint(1, 10000)  # Random ports
            data['packet_count'][idx] = np.random.randint(1, 5)  # Few packets
            data['bytes_sent'][idx] = np.random.randint(40, 100)  # Small packets
            data['bytes_received'][idx] = np.random.randint(0, 100)  # Often no response
            data['duration_ms'][idx] = np.random.randint(1, 100)  # Very quick
            data['tcp_flags'][idx] = np.random.choice(['SYN', 'FIN', 'ACK'])
            
            # Mark as threat
            data['is_threat'][idx] = 1
            data['threat_type'][idx] = 'PORT_SCAN'
        
        return data, attack_indices
    
    def _inject_data_exfiltration(self, data, attack_ratio=0.05):
        """Inject data exfiltration attack patterns into the dataset."""
        n_samples = len(data['timestamp'])
        n_attacks = int(n_samples * attack_ratio)
        attack_indices = np.random.choice(n_samples, size=n_attacks, replace=False)
        
        for idx in attack_indices:
            # Set attack signature
            data['protocol'][idx] = np.random.choice(['HTTPS', 'DNS'])
            data['dst_ip'][idx] = self._generate_external_ip()  # External destination
            
            if data['protocol'][idx] == 'HTTPS':
                data['dst_port'][idx] = 443
                data['bytes_sent'][idx] = np.random.randint(100000, 5000000)  # Large data transfer
                data['tls_version'][idx] = 'TLSv1.3'
            else:  # DNS tunneling
                data['dst_port'][idx] = 53
                data['dns_query'][idx] = f"data-{np.random.randint(10000)}.exfil.example.com"
                data['packet_count'][idx] = np.random.randint(50, 200)  # Many small queries
            
            # Mark as threat
            data['is_threat'][idx] = 1
            data['threat_type'][idx] = 'DATA_EXFILTRATION'
        
        return data, attack_indices
    
    def _inject_malware_communication(self, data, attack_ratio=0.05):
        """Inject malware C2 communication patterns into the dataset."""
        n_samples = len(data['timestamp'])
        n_attacks = int(n_samples * attack_ratio)
        attack_indices = np.random.choice(n_samples, size=n_attacks, replace=False)
        
        # Known malicious domains and IPs
        malicious_domains = [
            'evil-malware.ru',
            'ransomware-c2.cn',
            'trojan-updates.biz',
            'botnet-master.io'
        ]
        
        for idx in attack_indices:
            # Set attack signature
            data['protocol'][idx] = np.random.choice(['HTTP', 'HTTPS', 'DNS'])
            data['dst_ip'][idx] = self._generate_external_ip()
            
            if data['protocol'][idx] == 'DNS':
                data['dst_port'][idx] = 53
                data['dns_query'][idx] = random.choice(malicious_domains)
            else:  # HTTP or HTTPS
                data['dst_port'][idx] = 80 if data['protocol'][idx] == 'HTTP' else 443
                data['http_endpoint'][idx] = '/gate.php' if random.random() < 0.7 else '/config.bin'
                data['bytes_received'][idx] = np.random.randint(500, 10000)  # Command downloads
                data['bytes_sent'][idx] = np.random.randint(100, 1000)  # Status reports
                
                # Unusual timing pattern - regular intervals
                if idx > 0 and data['timestamp'][idx-1]:
                    time_diff = np.random.randint(5, 15) * 60  # 5-15 minutes
                    data['timestamp'][idx] = data['timestamp'][idx-1] + timedelta(seconds=time_diff)
            
            # Mark as threat
            data['is_threat'][idx] = 1
            data['threat_type'][idx] = 'MALWARE_COMMUNICATION'
        
        return data, attack_indices
    
    def generate_dataset(self, n_samples=10000, attack_ratio=0.05, start_date=None):
        """
        Generate a complete dataset with normal traffic and various attacks.
        
        Args:
            n_samples (int): Number of total records to generate
            attack_ratio (float): Ratio of attack traffic to generate (per attack type)
            start_date (datetime): Starting date for the time series
            
        Returns:
            DataFrame: The generated dataset
        """
        if start_date is None:
            start_date = datetime(2024, 1, 1)
        
        # Generate normal traffic base
        print(f"Generating {n_samples} network traffic records...")
        data = self._generate_normal_traffic(n_samples, start_date)
        
        # Convert lists to arrays for easier manipulation
        for key in data:
            if isinstance(data[key], list):
                data[key] = np.array(data[key])
        
        # Inject different attack types
        attack_types = [
            ('SQL injection attacks', self._inject_sql_injection_attacks),
            ('Brute force attacks', self._inject_brute_force_attacks),
            ('DDoS attacks', self._inject_ddos_attacks),
            ('Port scan attacks', self._inject_port_scan_attacks),
            ('Data exfiltration attacks', self._inject_data_exfiltration),
            ('Malware communication', self._inject_malware_communication)
        ]
        
        all_attack_indices = []
        
        # Apply each attack type
        for attack_name, attack_function in attack_types:
            specific_ratio = attack_ratio / len(attack_types)  # Distribute attacks evenly
            data, attack_indices = attack_function(data, attack_ratio=specific_ratio)
            all_attack_indices.extend(attack_indices)
            print(f"Injected {len(attack_indices)} {attack_name} ({len(attack_indices)/n_samples*100:.2f}%)")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add some metadata features
        df['weekday'] = df['timestamp'].dt.day_name()
        df['hour'] = df['timestamp'].dt.hour
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                                  (df['weekday'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']))).astype(int)
        
        total_attacks = df['is_threat'].sum()
        print(f"Generated dataset with {len(df)} records, including {total_attacks} threats ({total_attacks/len(df)*100:.2f}%)")
        
        return df
    
    def generate_traffic_stream(self, n_samples=100, attack_ratio=0.05):
        """
        Generate a simulated traffic stream (for real-time monitoring).
        
        Args:
            n_samples (int): Number of records to generate
            attack_ratio (float): Ratio of attack traffic to generate
            
        Returns:
            DataFrame: DataFrame with traffic records
        """
        df = self.generate_dataset(n_samples=n_samples, attack_ratio=attack_ratio)
        return df
    
    def save_dataset(self, df, filename="cyber_threat_dataset.csv"):
        """
        Save the generated dataset to a CSV file.
        
        Args:
            df (DataFrame): The dataset to save
            filename (str): Output filename
        """
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")


class CyberThreatDetector:
    """
    Machine learning model for cyber threat detection.
    """
    
    def __init__(self):
        """Initialize the detector."""
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        
    def train(self, df, features=None, n_features=5):
        """
        Train the threat detection model.
        
        Args:
            df (DataFrame): Training data
            features (list): List of feature names to use (if None, select best features)
            n_features (int): Number of best features to select if features is None
        """
        # Prepare data
        if features is None:
            # Automatically select the best features
            print("Training base model for feature selection...")
            
            # Define features to consider for selection
            numeric_features = [
                'src_port', 'dst_port', 'packet_length', 'packet_count', 
                'duration_ms', 'bytes_sent', 'bytes_received', 'login_attempts', 
                'error_rate', 'is_business_hours', 'hour'
            ]
            
            # Convert categorical variables to numeric
            df_prep = df.copy()
            
            # Protocol encoding
            df_prep['protocol'] = df_prep['protocol'].astype('category').cat.codes
            
            # TCP flags encoding
            df_prep['tcp_flags'] = df_prep['tcp_flags'].astype('category').cat.codes
            
            # Combine features
            all_features = numeric_features + ['protocol', 'tcp_flags']
            
            # Get features that actually exist in the dataframe
            valid_features = [f for f in all_features if f in df_prep.columns]
            
            X = df_prep[valid_features]
            y = df_prep['is_threat']
            
            # Select best features
            self.feature_selector = SelectKBest(f_classif, k=n_features)
            self.feature_selector.fit(X, y)
            
            # Get selected feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_features = [valid_features[i] for i in selected_indices]
            
            print(f"Selected {n_features} features: {self.selected_features}")
        else:
            self.selected_features = features
            
        print("Training final model...")
        X = df[self.selected_features]
        y = df['is_threat']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print("Model Training Results:")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print("Classification Report:")
        print(f"  Accuracy: {report['accuracy']:.4f}")
        print(f"  Precision (threat class): {report['1']['precision']:.4f}")
        print(f"  Recall (threat class): {report['1']['recall']:.4f}")
        print(f"  F1 Score (threat class): {report['1']['f1-score']:.4f}")
        
        return report, conf_matrix
    
    def save_model(self, directory="threat_models"):
        """Save the trained model and related components."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(directory, f"threat_model_{timestamp}.joblib")
        
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features
        }
        
        joblib.dump(model_package, model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """Load a saved model."""
        model_package = joblib.load(model_path)
        
        self.model = model_package['model']
        self.scaler = model_package['scaler']
        self.feature_selector = model_package['feature_selector']
        self.selected_features = model_package['selected_features']
        
        print(f"Model loaded from {model_path}")
        print(f"Selected features: {self.selected_features}")
    
    def predict(self, df):
        """
        Predict threats from network traffic data.
        
        Args:
            df (DataFrame): Network traffic data
            
        Returns:
            tuple: (predictions, probabilities)
        """
        # Make sure we're only using the features the model was trained on
        X = df[self.selected_features].copy()
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]  # probability of positive class
        
        return predictions, probabilities


def visualize_results(df, predictions, probabilities, threat_threshold=0.5):
    """
    Visualize threat detection results.
    
    Args:
        df (DataFrame): Original traffic data
        predictions (array): Model predictions (0/1)
        probabilities (array): Prediction probabilities
        threat_threshold (float): Threshold for considering a threat
    """
    # Add results to dataframe
    results = df.copy()
    results['predicted_threat'] = predictions
    results['threat_probability'] = probabilities
    
    # Calculate metrics
    true_positives = ((results['is_threat'] == 1) & (results['predicted_threat'] == 1)).sum()
    false_positives = ((results['is_threat'] == 0) & (results['predicted_threat'] == 1)).sum()
    true_negatives = ((results['is_threat'] == 0) & (results['predicted_threat'] == 0)).sum()
    false_negatives = ((results['is_threat'] == 1) & (results['predicted_threat'] == 0)).sum()
    
    # Create figure with multiple subplots
    plt.figure(figsize=(15, 10))
    
    # Confusion matrix heatmap
    plt.subplot(2, 2, 1)
    conf_matrix = confusion_matrix(results['is_threat'], results['predicted_threat'])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Threat'],
               yticklabels=['Normal', 'Threat'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Threat probability distribution
    plt.subplot(2, 2, 2)
    sns.histplot(data=results, x='threat_probability', hue='is_threat', bins=30, element='step')
    plt.axvline(x=threat_threshold, color='red', linestyle='--')
    plt.title('Threat Probability Distribution')
    plt.xlabel('Threat Probability')
    plt.legend(['Threshold', 'Normal', 'Threat'])
    
    # Threat types detected
    plt.subplot(2, 2, 3)
    threat_counts = results[results['predicted_threat'] == 1]['threat_type'].value_counts()
    colors = ['#ff9999' if threat_type == 'NONE' else '#66b3ff' for threat_type in threat_counts.index]
    threat_counts.plot.bar(color=colors)
    plt.title('Detected Threat Types')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Feature importance
    plt.subplot(2, 2, 4)
    feature_importance = pd.Series(data=0, index=results.columns)
    if hasattr(detector.model, 'feature_importances_'):
        feature_importance[detector.selected_features] = detector.model.feature_importances_
    feature_importance = feature_importance.sort_values(ascending=False).head(10)
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Save plot
    plot_dir = "threat_visualizations"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(plot_dir, f"threat_detection_{timestamp}.png"))
    plt.show()


def simulate(detector, data_generator, n_samples=1000, attack_ratio=0.1):
    """
    Simulate real-time threat detection on a stream of network traffic.
    
    Args:
        detector (CyberThreatDetector): Trained threat detection model
        data_generator (CyberThreatDataGenerator): Data generator for network traffic
        n_samples (int): Number of records to generate for the simulation
        attack_ratio (float): Ratio of attack traffic to generate
        
    Returns:
        DataFrame: Results of the simulation
    """
    # Generate a stream of network traffic
    print(f"Generating {n_samples} records for simulation...")
    df = data_generator.generate_traffic_stream(n_samples=n_samples, attack_ratio=attack_ratio)
    
    # Make predictions
    print("Running threat detection...")
    predictions, probabilities = detector.predict(df)
    
    # Add results to dataframe
    results = df.copy()
    results['predicted_threat'] = predictions
    results['threat_probability'] = probabilities
    
    # Calculate performance metrics
    true_positives = ((results['is_threat'] == 1) & (results['predicted_threat'] == 1)
    false_positives = ((results['is_threat'] == 0) & (results['predicted_threat'] == 1)
    true_negatives = ((results['is_threat'] == 0) & (results['predicted_threat'] == 0))
    false_negatives = ((results['is_threat'] == 1) & (results['predicted_threat'] == 0))
    
    # Print simulation results
    print("\nSimulation Results:")
    print(f"  True Positives: {true_positives.sum()}")
    print(f"  False Positives: {false_positives.sum()}")
    print(f"  True Negatives: {true_negatives.sum()}")
    print(f"  False Negatives: {false_negatives.sum()}")
    print(f"  Accuracy: {(true_positives.sum() + true_negatives.sum()) / len(results):.4f}")
    print(f"  Precision: {true_positives.sum() / (true_positives.sum() + false_positives.sum()):.4f}")
    print(f"  Recall: {true_positives.sum() / (true_positives.sum() + false_negatives.sum()):.4f}")
    
    return results


# Example usage
if __name__ == "__main__":
    # Initialize data generator and detector
    data_generator = CyberThreatDataGenerator(seed=42)
    detector = CyberThreatDetector()
    
    # Generate and save a dataset
    df = data_generator.generate_dataset(n_samples=10000, attack_ratio=0.1)
    data_generator.save_dataset(df, "cyber_threat_dataset.csv")
    
    # Train the detector
    report, conf_matrix = detector.train(df)
    
    # Save the trained model
    model_path = detector.save_model()
    
    # Load the model (optional)
    detector.load_model(model_path)
    
    # Simulate real-time threat detection
    simulation_results = simulate(detector, data_generator, n_samples=1000, attack_ratio=0.1)
    
    # Visualize results
    visualize_results(simulation_results, simulation_results['predicted_threat'], simulation_results['threat_probability'])
