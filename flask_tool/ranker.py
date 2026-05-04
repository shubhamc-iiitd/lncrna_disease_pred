"""Link prediction scores for bipartite lncRNA–disease graphs.

- **LatentRanker**: truncated SVD (good global low-rank signal, weak alone on tiny/sparse graphs).
- **HybridLinkScorer**: blends SVD with a **co-occurrence / 3-path** score ``(Y @ Y.T @ Y)``,
  which counts paths lnc–disease–lnc–disease and usually separates positives much better
  on sparse association data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse as sp
from sklearn.decomposition import TruncatedSVD


def _zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    s = v.std()
    if s < 1e-12:
        return np.zeros_like(v, dtype=np.float64)
    return (v - v.mean()) / s


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


@dataclass
class HybridLinkScorer:
    """Per-disease column: w_svd * zscore(SVD) + w_cn * zscore(log1p(Y @ Y.T @ Y))."""

    svd: LatentRanker
    cn_log: np.ndarray  # (n_lnc, n_disease), log1p(Y @ Y.T @ Y)
    w_svd: float = 0.3
    w_cn: float = 0.7

    def scores_for_disease(self, disease_index: int) -> np.ndarray:
        svd_col = self.svd.scores_for_disease(disease_index).astype(np.float64)
        cn_col = self.cn_log[:, disease_index].astype(np.float64)
        return (self.w_svd * _zscore(svd_col) + self.w_cn * _zscore(cn_col)).astype(np.float32)

    @classmethod
    def fit(
        cls,
        Y: sp.csr_matrix,
        n_components: int = 48,
        random_state: int = 0,
        w_svd: float = 0.3,
        w_cn: float = 0.7,
    ) -> "HybridLinkScorer":
        Y = Y.tocsr()
        svd = LatentRanker.fit(Y, n_components=n_components, random_state=random_state)
        # Disease–disease co-occurrence through shared lncRNAs, then map back to lnc × dis scores
        m = Y.T @ Y
        cn = Y @ m
        if sp.issparse(cn):
            cn = cn.toarray()
        cn = np.maximum(np.asarray(cn, dtype=np.float64), 0.0)
        cn_log = np.log1p(cn).astype(np.float32)
        wsum = w_svd + w_cn
        if wsum > 0:
            w_svd, w_cn = w_svd / wsum, w_cn / wsum
        return cls(svd=svd, cn_log=cn_log, w_svd=w_svd, w_cn=w_cn)
