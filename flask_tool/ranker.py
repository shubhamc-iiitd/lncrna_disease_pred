"""Low-rank factorization (truncated SVD) for ranking lncRNA–disease candidates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse as sp
from sklearn.decomposition import TruncatedSVD


@dataclass
class LatentRanker:
    U: np.ndarray  # (n_lnc, k)
    Vt: np.ndarray  # (k, n_disease)

    def scores_for_disease(self, disease_index: int) -> np.ndarray:
        return self.U @ self.Vt[:, disease_index]

    @classmethod
    def fit(cls, Y: sp.csr_matrix, n_components: int = 48, random_state: int = 0) -> "LatentRanker":
        n_l, n_d = Y.shape
        max_k = min(n_components, max(1, n_l - 1), max(1, n_d - 1))
        k = max(1, max_k)
        svd = TruncatedSVD(n_components=k, random_state=random_state)
        U = svd.fit_transform(Y.astype(np.float64))
        Vt = svd.components_
        return cls(U=U.astype(np.float32), Vt=Vt.astype(np.float32))
