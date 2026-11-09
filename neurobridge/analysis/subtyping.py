"""Transfer of TCGA transcriptional-subtype labels to cohorts without labels (e.g. CGGA).

Design choices that prevent leakage and circularity:
- the classifier is trained on TCGA only; validation cohorts never contribute to training or feature selection;
- genes are z-scored within each cohort separately (removes platform/batch location-scale differences);
- candidate neuronal-activity genes and the neuron-marker control genes are EXCLUDED from the features, so a later
  association between the candidate score and the projected subtype cannot be produced by the classifier itself;
- the Neural subtype is excluded from training by default because it is now regarded as normal-tissue contamination
  (Wang et al. 2017) and would be circular with neuronal signals;
- accuracy is estimated by stratified cross-validation inside TCGA and reported with the projection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neurobridge.analysis.common import finish, new_record
from neurobridge.analysis.preprocessing import filter_expressed_genes
from neurobridge.config import get_settings
from neurobridge.data.cohort import Cohort
from neurobridge.schemas.analysis import AnalysisKind


def _within_z(expr: pd.DataFrame) -> pd.DataFrame:
    sd = expr.std(axis=1, ddof=1).replace(0, np.nan)
    return expr.sub(expr.mean(axis=1), axis=0).div(sd, axis=0)


def train_and_project(train: Cohort, targets: dict[str, Cohort], exclude_genes: list[str], *, n_features: int = 1500, include_neural: bool = False, seed: int | None = None) -> dict:
    seed = get_settings().seed if seed is None else seed
    tr = train.primary_tumours()
    y = tr.clinical["subtype"].dropna()
    if not include_neural:
        y = y[y != "Neural"]
    shared = set(filter_expressed_genes(tr.expression).index)
    for t in targets.values():
        shared &= set(filter_expressed_genes(t.expression).index)
    shared -= set(exclude_genes)
    Ztr = _within_z(tr.expression.loc[sorted(shared)])
    # feature selection uses training cohort only
    feats = tr.expression.loc[sorted(shared), y.index].var(axis=1).sort_values(ascending=False).index[:n_features]
    X = Ztr.loc[feats, y.index].T.fillna(0).values
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred_cv = cross_val_predict(clf, X, y.values, cv=cv)
    bacc = balanced_accuracy_score(y.values, pred_cv)
    labels = sorted(y.unique())
    cm = confusion_matrix(y.values, pred_cv, labels=labels)
    clf.fit(X, y.values)
    rec = new_record("subtype_projection", AnalysisKind.EXPLORATORY, [train.dataset.dataset_id, *targets.keys()],
                     "L2-regularised multinomial logistic regression on within-cohort z-scored expression; trained on TCGA primary tumours only; 5-fold stratified CV",
                     {"n_features": n_features, "include_neural": include_neural, "C": 0.05, "excluded_genes": sorted(exclude_genes), "n_shared_genes": len(shared)}, seed=seed)
    projections = {}
    for tid, t in targets.items():
        Zt = _within_z(t.expression.loc[feats])
        Xt = Zt.T.fillna(0).values
        proba = clf.predict_proba(Xt)
        pred = pd.Series(clf.classes_[proba.argmax(1)], index=t.expression.columns)
        conf = pd.Series(proba.max(1), index=t.expression.columns)
        projections[tid] = {"labels": pred.to_dict(), "confidence": conf.to_dict(), "counts": pred.value_counts().to_dict(), "median_confidence": float(conf.median())}
    rec.n_used = int(len(y))
    rec.limitations += [
        f"Cross-validated balanced accuracy in TCGA = {bacc:.2f}; projected labels in other cohorts carry at least this error, plus platform/population shift.",
        "Verhaak subtypes were defined for IDH-wildtype-dominated TCGA tumours; projection onto IDH-mutant GBM is not meaningful, so analyses using projected labels should be stratified by IDH.",
        "Projected labels are model outputs, not ground truth.",
    ]
    return {"record": finish(rec), "cv_balanced_accuracy": float(bacc), "cv_confusion": {"labels": labels, "matrix": cm.tolist()}, "n_train": int(len(y)), "features": list(feats), "projections": projections}
