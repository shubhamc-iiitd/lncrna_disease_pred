from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class BipartiteData:
    """Incidence matrix Y of shape (n_lnc, n_disease) with values in {0,1}."""

    Y: sp.csr_matrix
    lnc_ids: list[str]
    disease_ids: list[str]
    disease_names: dict[str, str]
    disease_category: dict[str, str]
    lnc_names: dict[str, str]


def load_bipartite(data_dir: Path | str) -> BipartiteData:
    data_dir = Path(data_dir)
    assoc = pd.read_csv(data_dir / "associations.csv")
    diseases = pd.read_csv(data_dir / "diseases.csv")
    lncs = pd.read_csv(data_dir / "lncrnas.csv")
    if "category" not in diseases.columns:
        diseases = diseases.copy()
        diseases["category"] = "Unknown"

    lnc_ids = sorted(lncs["lncrna_id"].astype(str).unique().tolist())
    disease_ids = sorted(diseases["disease_id"].astype(str).unique().tolist())
    li = {v: i for i, v in enumerate(lnc_ids)}
    dj = {v: j for j, v in enumerate(disease_ids)}

    rows = assoc["lncrna_id"].astype(str).map(li)
    cols = assoc["disease_id"].astype(str).map(dj)
    n_l, n_d = len(lnc_ids), len(disease_ids)
    data = np.ones(len(rows), dtype=np.float32)
    Y = sp.csr_matrix((data, (rows.to_numpy(), cols.to_numpy())), shape=(n_l, n_d))
    Y.data[:] = 1.0
    Y.eliminate_zeros()

    disease_names = dict(zip(diseases["disease_id"].astype(str), diseases["disease_name"].astype(str)))
    disease_category = dict(zip(diseases["disease_id"].astype(str), diseases["category"].astype(str)))
    lnc_names = dict(zip(lncs["lncrna_id"].astype(str), lncs["lncrna_name"].astype(str)))

    return BipartiteData(
        Y=Y,
        lnc_ids=lnc_ids,
        disease_ids=disease_ids,
        disease_names=disease_names,
        disease_category=disease_category,
        lnc_names=lnc_names,
    )
