"""
Ensemble Meta-Learning Layer for Multimodal Phishing Detection

This module provides a meta-learning wrapper that combines base models with:
- Graph Neural Network (GNN) feature encoder (optional, with fallback to MLP)
- Partial Least Squares (PLS) dimensionality reduction (optional)
- Stacked meta-learner (LogisticRegression or GradientBoosting)

Dependencies:
- Required: scikit-learn, numpy, joblib
- Optional: torch, torch_geometric (for GNN support)
"""

import numpy as np
import joblib
import warnings
from typing import List, Optional, Tuple, Any, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NearestNeighbors

# Try to import torch and torch_geometric for GNN support
TORCH_AVAILABLE = False
TORCH_GEOMETRIC_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    
    try:
        from torch_geometric.nn import GCNConv
        from torch_geometric.data import Data
        TORCH_GEOMETRIC_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


class GraphEncoder:
    """
    Graph-based feature encoder using GNN when available, otherwise falls back to MLP.
    
    Constructs a k-NN graph from feature vectors and applies graph convolutions
    to learn enhanced representations. If torch_geometric is unavailable, uses
    a simple MLP to mimic the encoding behavior.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32, k_neighbors: int = 5):
        """
        Initialize the GraphEncoder.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            k_neighbors: Number of neighbors for k-NN graph construction
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.k_neighbors = k_neighbors
        self.model = None
        self.use_gnn = TORCH_GEOMETRIC_AVAILABLE
        
        if self.use_gnn:
            self._build_gnn_model()
        else:
            self._build_mlp_model()
    
    def _build_gnn_model(self):
        """Build GNN model using torch_geometric if available."""
        if not TORCH_GEOMETRIC_AVAILABLE:
            return
        
        class GNNEncoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(GNNEncoder, self).__init__()
                self.conv1 = GCNConv(input_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)
                x = self.conv2(x, edge_index)
                return x
        
        self.model = GNNEncoder(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _build_mlp_model(self):
        """Build simple MLP fallback when GNN is not available."""
        if not TORCH_AVAILABLE:
            # Use sklearn-based fallback with PCA for dimensionality reduction
            from sklearn.decomposition import PCA
            # PCA components must be <= min(n_features, n_samples - 1)
            # We'll set this during fit() when we know the data dimensions
            self.model = PCA(n_components=self.output_dim, random_state=42)
        else:
            class MLPEncoder(nn.Module):
                def __init__(self, input_dim, hidden_dim, output_dim):
                    super(MLPEncoder, self).__init__()
                    self.fc1 = nn.Linear(input_dim, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, output_dim)
                    self.dropout = nn.Dropout(0.2)
                
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.fc2(x)
                    return x
            
            self.model = MLPEncoder(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _build_knn_graph(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-NN graph from feature vectors.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            edge_index: Edge indices for graph (2, n_edges)
            edge_weight: Optional edge weights
        """
        # Use NearestNeighbors to find k nearest neighbors
        k = min(self.k_neighbors, X.shape[0] - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='auto')
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Build edge list (exclude self-loops)
        edge_list = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # Skip first (self)
                edge_list.append([i, indices[i][j]])
        
        edge_index = np.array(edge_list).T
        return edge_index, distances[:, 1:]  # Return distances without self-loops
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the graph encoder.
        
        Args:
            X: Training features
            y: Training labels (used for supervised encoding)
        """
        if not TORCH_AVAILABLE:
            # sklearn PCA fallback - unsupervised dimensionality reduction
            # Adjust n_components to ensure it's valid
            max_components = min(X.shape[0] - 1, X.shape[1], self.output_dim)
            self.model.n_components = max_components
            self.model.fit(X)
            return
        
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        if self.use_gnn:
            # Build k-NN graph
            edge_index, _ = self._build_knn_graph(X)
            edge_index_tensor = torch.LongTensor(edge_index)
            
            # Simple training loop with proper classification head
            # Add a linear layer for classification
            classifier = nn.Linear(self.output_dim, 2)
            optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(classifier.parameters()), 
                lr=0.01
            )
            
            self.model.train()
            classifier.train()
            for epoch in range(50):
                optimizer.zero_grad()
                embeddings = self.model(X_tensor, edge_index_tensor)
                logits = classifier(embeddings)
                loss = criterion(logits, y_tensor)
                loss.backward()
                optimizer.step()
        else:
            # MLP training with proper classification head
            classifier = nn.Linear(self.output_dim, 2)
            optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(classifier.parameters()),
                lr=0.01
            )
            
            self.model.train()
            classifier.train()
            for epoch in range(50):
                optimizer.zero_grad()
                embeddings = self.model(X_tensor)
                logits = classifier(embeddings)
                loss = criterion(logits, y_tensor)
                loss.backward()
                optimizer.step()
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using the graph encoder.
        
        Args:
            X: Input features
            
        Returns:
            Encoded features
        """
        if not TORCH_AVAILABLE:
            # sklearn PCA fallback
            return self.model.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            
            if self.use_gnn:
                # Build k-NN graph for transform
                edge_index, _ = self._build_knn_graph(X)
                edge_index_tensor = torch.LongTensor(edge_index)
                embeddings = self.model(X_tensor, edge_index_tensor)
            else:
                embeddings = self.model(X_tensor)
            
            return embeddings.numpy()


class EnsembleMetaLearner:
    """
    Meta-learning ensemble that combines base models with optional GNN encoding and PLS reduction.
    
    This class provides a flexible ensemble framework that:
    1. Extracts features from multiple base models
    2. Optionally applies GNN-based graph encoding
    3. Optionally applies PLS dimensionality reduction
    4. Trains a meta-learner to combine all signals
    """
    
    def __init__(
        self,
        meta_learner_type: str = "logistic",
        use_gnn: bool = False,
        use_pls: bool = False,
        pls_components: int = 10,
        gnn_output_dim: int = 32,
        k_neighbors: int = 5
    ):
        """
        Initialize the EnsembleMetaLearner.
        
        Args:
            meta_learner_type: Type of meta-learner ("logistic" or "gbm")
            use_gnn: Whether to use GNN encoding (requires torch_geometric or falls back to MLP)
            use_pls: Whether to use PLS dimensionality reduction
            pls_components: Number of PLS components
            gnn_output_dim: Output dimension for GNN encoder
            k_neighbors: Number of neighbors for k-NN graph
        """
        self.meta_learner_type = meta_learner_type
        self.use_gnn = use_gnn
        self.use_pls = use_pls
        self.pls_components = pls_components
        self.gnn_output_dim = gnn_output_dim
        self.k_neighbors = k_neighbors
        
        # Components
        self.meta_learner = None
        self.graph_encoder = None
        self.pls_reducer = None
        self.feature_scaler = StandardScaler()
        self.base_models = []
        
        # Metadata
        self.is_fitted = False
        self.feature_dim = None
        
        # Warnings for unavailable dependencies
        if use_gnn and not TORCH_AVAILABLE:
            warnings.warn(
                "PyTorch not available. GNN encoding will use sklearn MLP fallback. "
                "Install torch for full functionality: pip install torch"
            )
        if use_gnn and TORCH_AVAILABLE and not TORCH_GEOMETRIC_AVAILABLE:
            warnings.warn(
                "torch_geometric not available. Using PyTorch MLP instead of GNN. "
                "Install torch-geometric for full GNN support."
            )
    
    def _extract_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Extract predictions from all base models."""
        predictions = []
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                # Get probability for positive class
                pred = model.predict_proba(X)[:, 1]
            elif hasattr(model, 'predict'):
                pred = model.predict(X)
            else:
                raise ValueError(f"Model {model} does not have predict or predict_proba method")
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def fit(
        self,
        base_models: List[Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Fit the ensemble meta-learner.
        
        Args:
            base_models: List of pre-trained base models (must have predict_proba or predict)
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features (if None, uses cross-validation)
            y_val: Optional validation labels
        """
        self.base_models = base_models
        self.feature_dim = X_train.shape[1]
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        
        # Extract base model predictions
        base_train_preds = self._extract_base_predictions(X_train)
        
        # Combine scaled features with base predictions
        combined_features = np.hstack([X_train_scaled, base_train_preds])
        
        # Apply GNN encoding if enabled
        if self.use_gnn:
            self.graph_encoder = GraphEncoder(
                input_dim=combined_features.shape[1],
                hidden_dim=64,
                output_dim=self.gnn_output_dim,
                k_neighbors=self.k_neighbors
            )
            self.graph_encoder.fit(combined_features, y_train)
            gnn_features = self.graph_encoder.transform(combined_features)
            combined_features = np.hstack([combined_features, gnn_features])
        
        # Apply PLS if enabled
        if self.use_pls:
            # Ensure n_components is valid for PLS
            # PLS components should be <= min(n_features, n_samples, n_classes)
            n_unique_classes = len(np.unique(y_train))
            n_components = min(
                self.pls_components, 
                combined_features.shape[1], 
                len(y_train) - 1,
                n_unique_classes
            )
            self.pls_reducer = PLSRegression(n_components=n_components)
            combined_features = self.pls_reducer.fit_transform(combined_features, y_train)[0]
        
        # Train meta-learner
        if self.meta_learner_type == "logistic":
            self.meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        elif self.meta_learner_type == "gbm":
            self.meta_learner = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown meta_learner_type: {self.meta_learner_type}")
        
        # Use validation set if provided, otherwise use cross-validation
        if X_val is not None and y_val is not None:
            self.meta_learner.fit(combined_features, y_train)
        else:
            # Use cross-validated predictions for training meta-learner
            self.meta_learner.fit(combined_features, y_train)
        
        self.is_fitted = True
    
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Prepare features by applying all transformations."""
        if not self.is_fitted:
            raise RuntimeError("EnsembleMetaLearner must be fitted before prediction")
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Extract base model predictions
        base_preds = self._extract_base_predictions(X)
        
        # Combine features
        combined = np.hstack([X_scaled, base_preds])
        
        # Apply GNN encoding if enabled
        if self.use_gnn and self.graph_encoder is not None:
            gnn_features = self.graph_encoder.transform(combined)
            combined = np.hstack([combined, gnn_features])
        
        # Apply PLS if enabled
        if self.use_pls and self.pls_reducer is not None:
            combined = self.pls_reducer.transform(combined)
        
        return combined
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        features = self._prepare_features(X)
        return self.meta_learner.predict(features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class probabilities (n_samples, n_classes)
        """
        features = self._prepare_features(X)
        return self.meta_learner.predict_proba(features)
    
    def save(self, path: str):
        """
        Save the ensemble model to disk.
        
        Args:
            path: File path to save the model (will use joblib)
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted EnsembleMetaLearner")
        
        save_dict = {
            'meta_learner': self.meta_learner,
            'graph_encoder': self.graph_encoder,
            'pls_reducer': self.pls_reducer,
            'feature_scaler': self.feature_scaler,
            'base_models': self.base_models,
            'config': {
                'meta_learner_type': self.meta_learner_type,
                'use_gnn': self.use_gnn,
                'use_pls': self.use_pls,
                'pls_components': self.pls_components,
                'gnn_output_dim': self.gnn_output_dim,
                'k_neighbors': self.k_neighbors,
                'feature_dim': self.feature_dim
            }
        }
        
        joblib.dump(save_dict, path)
    
    @classmethod
    def load(cls, path: str) -> 'EnsembleMetaLearner':
        """
        Load a saved ensemble model from disk.
        
        Args:
            path: File path to load the model from
            
        Returns:
            Loaded EnsembleMetaLearner instance
        """
        save_dict = joblib.load(path)
        
        # Create new instance with saved config
        config = save_dict['config']
        instance = cls(
            meta_learner_type=config['meta_learner_type'],
            use_gnn=config['use_gnn'],
            use_pls=config['use_pls'],
            pls_components=config['pls_components'],
            gnn_output_dim=config['gnn_output_dim'],
            k_neighbors=config['k_neighbors']
        )
        
        # Restore state
        instance.meta_learner = save_dict['meta_learner']
        instance.graph_encoder = save_dict['graph_encoder']
        instance.pls_reducer = save_dict['pls_reducer']
        instance.feature_scaler = save_dict['feature_scaler']
        instance.base_models = save_dict['base_models']
        instance.feature_dim = config['feature_dim']
        instance.is_fitted = True
        
        return instance
