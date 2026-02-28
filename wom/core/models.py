"""Embedding models for metabolite preference prediction.

All models learn embeddings for microbes and metabolites such that
the dot product approximates preference strength.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Base Class
# =============================================================================

class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding-based preference models.
    
    All models learn embeddings for microbes and metabolites such that
    the dot product approximates preference strength.
    """
    
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        self.n_microbes = None
        self.n_metabolites = None
        self._microbe_embeddings = None
        self._metabolite_embeddings = None
        self._fitted = False
    
    @abstractmethod
    def fit(
        self,
        positives: dict[int, set[int]],
        n_microbes: int,
        n_metabolites: int,
        **kwargs
    ) -> "BaseEmbeddingModel":
        """Fit the model to training data.
        
        Args:
            positives: Dict mapping microbe_idx -> set of positive metabolite_idx
            n_microbes: Total number of microbes
            n_metabolites: Total number of metabolites
            **kwargs: Model-specific parameters
        
        Returns:
            self
        """
        pass
    
    @property
    def microbe_embeddings(self) -> np.ndarray:
        """Microbe embedding matrix (n_microbes, embedding_dim)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        return self._microbe_embeddings
    
    @property
    def metabolite_embeddings(self) -> np.ndarray:
        """Metabolite embedding matrix (n_metabolites, embedding_dim)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        return self._metabolite_embeddings
    
    def predict_scores(self, microbe_idx: int) -> np.ndarray:
        """Predict preference scores for all metabolites given a microbe.
        
        Args:
            microbe_idx: Index of the microbe
        
        Returns:
            Array of scores (n_metabolites,)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        return self._microbe_embeddings[microbe_idx] @ self._metabolite_embeddings.T
    
    def predict_scores_from_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Predict scores using a custom embedding (e.g., aggregated taxon embedding).
        
        Args:
            embedding: Embedding vector (embedding_dim,)
        
        Returns:
            Array of scores (n_metabolites,)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        return embedding @ self._metabolite_embeddings.T


# =============================================================================
# BPR Model (Bayesian Personalized Ranking)
# =============================================================================

class BPRModel(BaseEmbeddingModel):
    """Bayesian Personalized Ranking using PyTorch.
    
    Optimizes pairwise ranking: positive items should be ranked higher than
    negative items for each user (microbe).
    
    Reference: Rendle et al. (2009) "BPR: Bayesian Personalized Ranking from
    Implicit Feedback"
    """
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__(embedding_dim)
        self._torch_model = None
    
    def fit(
        self,
        positives: dict[int, set[int]],
        n_microbes: int,
        n_metabolites: int,
        n_epochs: int = 1000,
        batch_size: int = 256,
        learning_rate: float = 0.01,
        verbose: bool = False,
    ) -> "BPRModel":
        """Fit BPR model using stochastic gradient descent.
        
        Args:
            positives: Dict mapping microbe_idx -> set of positive metabolite_idx
            n_microbes: Total number of microbes
            n_metabolites: Total number of metabolites
            n_epochs: Number of training epochs
            batch_size: Batch size for SGD
            learning_rate: Learning rate
            verbose: Whether to show progress
        
        Returns:
            self
        """
        self.n_microbes = n_microbes
        self.n_metabolites = n_metabolites
        
        # Create PyTorch model
        model = _BPRTorchModel(n_microbes, n_metabolites, self.embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        microbe_indices = [m for m in positives.keys() if len(positives[m]) > 0]
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Sample triplets
            for _ in range(len(microbe_indices) // batch_size + 1):
                m_batch = []
                pos_batch = []
                neg_batch = []
                
                for _ in range(batch_size):
                    if not microbe_indices:
                        break
                    m_idx = np.random.choice(microbe_indices)
                    pos_idx = np.random.choice(list(positives[m_idx]))
                    
                    neg_idx = np.random.randint(n_metabolites)
                    while neg_idx in positives[m_idx]:
                        neg_idx = np.random.randint(n_metabolites)
                    
                    m_batch.append(m_idx)
                    pos_batch.append(pos_idx)
                    neg_batch.append(neg_idx)
                
                if not m_batch:
                    continue
                
                # Forward pass
                m_tensor = torch.tensor(m_batch, dtype=torch.long)
                pos_tensor = torch.tensor(pos_batch, dtype=torch.long)
                neg_tensor = torch.tensor(neg_batch, dtype=torch.long)
                
                pos_scores, neg_scores = model(m_tensor, pos_tensor, neg_tensor)
                
                # BPR loss: -log(sigmoid(pos_score - neg_score))
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if verbose and (epoch + 1) % 100 == 0:
                avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
                logger.info(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        # Extract embeddings
        self._torch_model = model
        self._microbe_embeddings = model.microbe_emb.weight.detach().numpy()
        self._metabolite_embeddings = model.metabolite_emb.weight.detach().numpy()
        self._fitted = True
        
        return self
    
    def get_torch_model(self) -> nn.Module:
        """Return underlying PyTorch model (for saving/loading)."""
        return self._torch_model
    
    def load_from_torch(self, torch_model: nn.Module) -> "BPRModel":
        """Load embeddings from a PyTorch model."""
        self._torch_model = torch_model
        self._microbe_embeddings = torch_model.microbe_emb.weight.detach().numpy()
        self._metabolite_embeddings = torch_model.metabolite_emb.weight.detach().numpy()
        self.n_microbes = self._microbe_embeddings.shape[0]
        self.n_metabolites = self._metabolite_embeddings.shape[0]
        self._fitted = True
        return self


class _BPRTorchModel(nn.Module):
    """Internal PyTorch module for BPR."""
    
    def __init__(self, n_microbes: int, n_metabolites: int, embedding_dim: int):
        super().__init__()
        self.microbe_emb = nn.Embedding(n_microbes, embedding_dim)
        self.metabolite_emb = nn.Embedding(n_metabolites, embedding_dim)
        nn.init.normal_(self.microbe_emb.weight, std=0.01)
        nn.init.normal_(self.metabolite_emb.weight, std=0.01)
    
    def forward(self, m_idx, pos_idx, neg_idx):
        m = self.microbe_emb(m_idx)
        pos = self.metabolite_emb(pos_idx)
        neg = self.metabolite_emb(neg_idx)
        return (m * pos).sum(1), (m * neg).sum(1)


# =============================================================================
# WR-MF Model (Weighted Regularized Matrix Factorization)
# =============================================================================

class WRMFModel(BaseEmbeddingModel):
    """Weighted Regularized Matrix Factorization using ALS.
    
    Uses the implicit library for efficient ALS optimization.
    
    Reference: Hu et al. (2008) "Collaborative Filtering for Implicit Feedback
    Datasets"
    """
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__(embedding_dim)
    
    def fit(
        self,
        positives: dict[int, set[int]],
        n_microbes: int,
        n_metabolites: int,
        iterations: int = 50,
        regularization: float = 0.1,
        verbose: bool = False,
    ) -> "WRMFModel":
        """Fit WR-MF model using ALS.
        
        Args:
            positives: Dict mapping microbe_idx -> set of positive metabolite_idx
            n_microbes: Total number of microbes
            n_metabolites: Total number of metabolites
            iterations: Number of ALS iterations
            regularization: L2 regularization strength
            verbose: Whether to show progress
        
        Returns:
            self
        """
        try:
            from implicit.als import AlternatingLeastSquares
            from scipy.sparse import csr_matrix
        except ImportError:
            raise ImportError("WR-MF requires 'implicit' library: pip install implicit")
        
        self.n_microbes = n_microbes
        self.n_metabolites = n_metabolites
        
        # Build sparse matrix
        rows, cols, data = [], [], []
        for m_idx, pos_set in positives.items():
            for c_idx in pos_set:
                rows.append(m_idx)
                cols.append(c_idx)
                data.append(1.0)
        
        sparse_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_microbes, n_metabolites),
        )
        
        model = AlternatingLeastSquares(
            factors=self.embedding_dim,
            regularization=regularization,
            iterations=iterations,
            random_state=42,
        )
        
        model.fit(sparse_matrix, show_progress=verbose)
        
        self._microbe_embeddings = model.user_factors
        self._metabolite_embeddings = model.item_factors
        self._fitted = True
        
        return self


# =============================================================================
# Implicit BPR Model (from implicit library)
# =============================================================================

class ImplicitBPRModel(BaseEmbeddingModel):
    """BPR using the implicit library's optimized C++ implementation.
    
    Bayesian Personalized Ranking optimized with efficient sampling.
    
    Reference: Rendle et al. (2009) "BPR: Bayesian Personalized Ranking from
    Implicit Feedback"
    """
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__(embedding_dim)
    
    def fit(
        self,
        positives: dict[int, set[int]],
        n_microbes: int,
        n_metabolites: int,
        iterations: int = 100,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        verbose: bool = False,
    ) -> "ImplicitBPRModel":
        """Fit BPR model using implicit library.
        
        Args:
            positives: Dict mapping microbe_idx -> set of positive metabolite_idx
            n_microbes: Total number of microbes
            n_metabolites: Total number of metabolites
            iterations: Number of training iterations
            learning_rate: Learning rate
            regularization: L2 regularization strength
            verbose: Whether to show progress
        
        Returns:
            self
        """
        try:
            from implicit.bpr import BayesianPersonalizedRanking
            from scipy.sparse import csr_matrix
        except ImportError:
            raise ImportError("ImplicitBPR requires 'implicit' library: pip install implicit")
        
        self.n_microbes = n_microbes
        self.n_metabolites = n_metabolites
        
        # Build sparse matrix (microbes × metabolites)
        rows, cols, data = [], [], []
        for m_idx, pos_set in positives.items():
            for met_idx in pos_set:
                rows.append(m_idx)
                cols.append(met_idx)
                data.append(1.0)
        
        sparse_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_microbes, n_metabolites),
        )
        
        model = BayesianPersonalizedRanking(
            factors=self.embedding_dim,
            learning_rate=learning_rate,
            regularization=regularization,
            iterations=iterations,
            random_state=42,
        )
        
        model.fit(sparse_matrix, show_progress=verbose)
        
        self._microbe_embeddings = model.user_factors
        self._metabolite_embeddings = model.item_factors
        self._fitted = True
        
        return self


# =============================================================================
# LMF Model (Logistic Matrix Factorization)
# =============================================================================

class LMFModel(BaseEmbeddingModel):
    """Logistic Matrix Factorization using implicit library.
    
    Probabilistic approach to matrix factorization with maximum-likelihood
    estimation for implicit feedback data.
    """
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__(embedding_dim)
    
    def fit(
        self,
        positives: dict[int, set[int]],
        n_microbes: int,
        n_metabolites: int,
        iterations: int = 30,
        learning_rate: float = 1.0,
        regularization: float = 0.1,
        verbose: bool = False,
    ) -> "LMFModel":
        """Fit LMF model using implicit library.
        
        Args:
            positives: Dict mapping microbe_idx -> set of positive metabolite_idx
            n_microbes: Total number of microbes
            n_metabolites: Total number of metabolites
            iterations: Number of training iterations
            learning_rate: Learning rate
            regularization: L2 regularization strength
            verbose: Whether to show progress
        
        Returns:
            self
        """
        try:
            from implicit.lmf import LogisticMatrixFactorization
            from scipy.sparse import csr_matrix
        except ImportError:
            raise ImportError("LMF requires 'implicit' library: pip install implicit")
        
        self.n_microbes = n_microbes
        self.n_metabolites = n_metabolites
        
        # Build sparse matrix (microbes × metabolites)
        rows, cols, data = [], [], []
        for m_idx, pos_set in positives.items():
            for met_idx in pos_set:
                rows.append(m_idx)
                cols.append(met_idx)
                data.append(1.0)
        
        sparse_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_microbes, n_metabolites),
        )
        
        model = LogisticMatrixFactorization(
            factors=self.embedding_dim,
            learning_rate=learning_rate,
            regularization=regularization,
            iterations=iterations,
            random_state=42,
        )
        
        model.fit(sparse_matrix, show_progress=verbose)
        
        self._microbe_embeddings = model.user_factors
        self._metabolite_embeddings = model.item_factors
        self._fitted = True
        
        return self


# =============================================================================
# SVD Model
# =============================================================================

class SVDModel(BaseEmbeddingModel):
    """Truncated SVD for matrix factorization baseline.
    
    Simple baseline that factorizes the binary interaction matrix directly.
    """
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__(embedding_dim)
    
    def fit(
        self,
        positives: dict[int, set[int]],
        n_microbes: int,
        n_metabolites: int,
        **kwargs
    ) -> "SVDModel":
        """Fit truncated SVD.
        
        Args:
            positives: Dict mapping microbe_idx -> set of positive metabolite_idx
            n_microbes: Total number of microbes
            n_metabolites: Total number of metabolites
        
        Returns:
            self
        """
        from sklearn.decomposition import TruncatedSVD
        
        self.n_microbes = n_microbes
        self.n_metabolites = n_metabolites
        
        # Build dense matrix
        M = np.zeros((n_microbes, n_metabolites))
        for m_idx, pos_set in positives.items():
            for c_idx in pos_set:
                M[m_idx, c_idx] = 1.0
        
        # Fit SVD
        n_components = min(self.embedding_dim, min(M.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Compute embeddings
        U = svd.fit_transform(M)  # (n_microbes, n_components)
        V = svd.components_  # (n_components, n_metabolites)
        
        self._microbe_embeddings = U
        self._metabolite_embeddings = V.T  # (n_metabolites, n_components)
        
        self._fitted = True
        
        return self
