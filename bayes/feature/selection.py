from __future__ import annotations

from functools import reduce
from typing import Literal

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import LabelEncoder

from bayes.utils.logging import log


def compute_association(
    data: pd.DataFrame,
    target: str,
    features: str | list[str],
    task: str = "regression",
    threshold: float | None = None,
    pivot: bool = True,
) -> pd.DataFrame:
    """Compute task-appropriate association between features and target.

    For regression: absolute Pearson correlation.
    For classification: ANOVA F-score (proxy for group separation).

    Args:
        data: Input DataFrame containing features and target.
        target: Name of the target column.
        features: Single feature name or list of feature names.
        task: Either "regression" or "classification".
        threshold: Minimum association score to keep. Defaults to None (keep all).
        pivot: If True, pivot result with features as index. Defaults to True.

    Returns:
        DataFrame with columns ["feature", "association"] (long format if pivot=False)
        or pivoted with features as index.

    Raises:
        KeyError: If target or any feature is not in data.
        ValueError: If task is not "regression" or "classification".

    Example:
        >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> compute_association(df, "y", "x", task="regression")
           association
        feature
        x            1.0
    """
    if isinstance(features, str):
        features = [features]
    if not all(f in data.columns for f in features):
        missing = [f for f in features if f not in data.columns]
        raise KeyError(f"Features not found: {missing}")
    if target not in data.columns:
        raise KeyError(f"Target '{target}' not found.")

    if task == "regression":
        corr = data[features].corrwith(data[target]).abs()
        result = corr.reset_index()
        result.columns = ["feature", "association"]
    elif task == "classification":
        f_scores, _ = f_classif(data[features], data[target])
        result = pd.DataFrame({"feature": features, "association": f_scores})
    else:
        raise ValueError("task must be 'regression' or 'classification'")

    if threshold is not None:
        result = result[result["association"] >= threshold]

    result = result.sort_values(by="association", ascending=False)
    if pivot:
        result = result.set_index("feature")[["association"]]
    return result


def compute_anova_f(
    data: pd.DataFrame,
    target: str,
    features: str | list[str],
    task: str = "regression",
    normalize: bool = False,
    pivot: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute ANOVA F-scores and p-values for features.

    Args:
        data: Input DataFrame.
        target: Target column name.
        features: Feature column(s).
        task: "regression" (f_regression) or "classification" (f_classif).
        normalize: If True, scale F-scores to [0, 1]. Defaults to False.
        pivot: If True, pivot results. Defaults to True.

    Returns:
        Tuple of (f_scores_df, p_values_df), each with:
        - Long format: ["target", "feature", "f_score"] or ["target", "feature", "p_value"]
        - Pivoted: features as index, target as column.

    Raises:
        ValueError: If task is invalid.

    Example:
        >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        >>> f_df, p_df = compute_anova_f(df, "y", "x", task="regression")
    """
    if isinstance(features, str):
        features = [features]
    inputs = data[features].values
    y = data[target].values

    if task == "regression":
        f_scores, p_values = f_regression(inputs, y)
    elif task == "classification":
        f_scores, p_values = f_classif(inputs, y)
    else:
        raise ValueError("task must be 'regression' or 'classification'")

    if normalize and len(f_scores) > 0:
        max_score = f_scores.max()
        if max_score > 0:
            f_scores = f_scores / max_score

    df = pd.DataFrame({"feature": features, "f_score": f_scores, "p_value": p_values})
    df["target"] = target

    f_scores_df = df[["target", "feature", "f_score"]].sort_values("f_score", ascending=False)
    p_values_df = df[["target", "feature", "p_value"]].sort_values("p_value")

    if pivot:
        f_scores_df = f_scores_df.pivot(index="feature", columns="target", values="f_score")
        p_values_df = p_values_df.pivot(index="feature", columns="target", values="p_value")

    return f_scores_df, p_values_df


def compute_mutual_information(
    data: pd.DataFrame,
    target: str,
    features: str | list[str],
    task: Literal["regression", "classification"] = "regression",
    threshold: float | None = None,
    random_state: int | None = 200,
    pivot: bool = True,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Compute mutual information in parallel per feature.

    Args:
        data: Input DataFrame.
        target: Target column name.
        features: Feature column(s).
        task: "regression" or "classification".
        threshold: Minimum MI score to keep.
        random_state: Random seed.
        pivot: If True, pivot result.
        n_jobs: Parallel jobs (-1 = all cores).

    Returns:
        DataFrame with ["target", "feature", "score"] or pivoted.

    Example:
        >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        >>> compute_mutual_information(df, "y", "x", task="regression")
    """
    if isinstance(features, str):
        features = [features]

    inputs = data[features].values.astype(np.float64)
    target_vals = data[target].values
    n_features = inputs.shape[1]

    mi_func = mutual_info_regression if task == "regression" else mutual_info_classif

    # ---- CORRECT PARALLEL MI ----
    mi_scores = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(lambda col: mi_func(col, target_vals, random_state=random_state)[0])(inputs[:, [i]])
            for i in range(n_features)
        )
    )

    result = pd.DataFrame({"feature": features, "score": mi_scores, "target": target})
    result = result[["target", "feature", "score"]].sort_values("score", ascending=False)

    if threshold is not None:
        result = result[result["score"] >= threshold]

    if pivot:
        result = result.pivot(index="feature", columns="target", values="score")

    return result


def compute_chi2(
    data: pd.DataFrame,
    target: str,
    features: str | list[str],
    pivot: bool = True,
) -> pd.DataFrame:
    """Compute Chi-square statistic for categorical features (classification only).

    Requires non-negative feature values (e.g., counts, binary flags).

    Args:
        data: Input DataFrame.
        target: Target column (must be categorical/integer-encoded).
        features: Feature column(s).
        pivot: If True, return with features as index.

    Returns:
        DataFrame with ["feature", "chi2_score"] or pivoted.

    Raises:
        ValueError: If any feature has negative values.

    Example:
        >>> df = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 0], "y": [0, 1, 0]})
        >>> compute_chi2(df, "y", ["a", "b"])
             chi2_score
        feature
        a            1.0
        b            1.0
    """
    if isinstance(features, str):
        features = [features]
    inputs = data[features].values
    y = data[target].values

    if inputs.min() < 0:
        raise ValueError("Chi-square requires non-negative feature values.")

    scores, _ = chi2(inputs, y)
    result = pd.DataFrame({"feature": features, "chi2_score": scores})
    result = result.sort_values("chi2_score", ascending=False)
    if pivot:
        result = result.set_index("feature")
    return result


def compute_rf_importance(
    data: pd.DataFrame,
    target: str,
    features: str | list[str],
    task: str = "regression",
    n_estimators: int = 50,
    random_state: int | None = 42,
    n_jobs: int = -1,
    pivot: bool = True,
) -> pd.DataFrame:
    """Compute Random Forest feature importance using SelectFromModel.

    Uses mean importance threshold internally.

    Args:
        data: Input DataFrame.
        target: Target column.
        features: Feature column(s).
        task: "regression" or "classification".
        n_estimators: Number of trees in the forest.
        random_state: Random seed.
        n_jobs: number of jobs for parralisation.
        pivot: If True, pivot result.

    Returns:
        DataFrame with ["feature", "rf_importance"] or pivoted.

    Example:
        >>> df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [0, 1, 0]})
        >>> compute_rf_importance(df, "y", ["x1", "x2"], task="classification")
    """
    if isinstance(features, str):
        features = [features]
    inputs = data[features].values
    y = data[target].values

    estimator = (
        RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
        if task == "regression"
        else RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
    )

    selector = SelectFromModel(estimator, threshold="mean")
    selector.fit(inputs, y)
    importances = selector.estimator_.feature_importances_

    result = pd.DataFrame({"feature": features, "rf_importance": importances})
    result = result.sort_values("rf_importance", ascending=False)
    if pivot:
        result = result.set_index("feature")
    return result


def select_top_features(
    data: pd.DataFrame,
    features: str | list[str],
    target: str,
    task: Literal["regression", "classification"] = "regression",
    top_k: int = 2,
    alpha: float = 0.05,
    use_percentile_ranking: bool = False,
    rank_aggregation: Literal["borda_count", "geom_rank"] = "borda_count",
    normalize_scores: bool = True,
    return_scores: bool = False,
    random_state: int | None = 200,
    include_chi2: bool = True,
    include_rf: bool = True,
    fast_mode: bool = False,
    n_jobs: int = -1,
    var_threshold:float = 1e-8,
) -> list[str] | tuple[list[str], pd.DataFrame]:
    """Selects top-k features using multiple metrics and rank aggregation.

    Automatically encodes string categorical targets. Supports Chi-square and RF.

    Args:
        data: Input DataFrame.
        features: Feature column(s).
        target: Target column.
        task: "regression" or "classification".
        top_k: Number of top features to return.
        alpha: Significance threshold for p-values.
        use_percentile_ranking: Rank using percentiles.
        rank_aggregation: One of: "borda_count", "geom_rank", "arith_rank", "med_rank", "sum_rank".
        normalize_scores: Normalize final rank scores.
        return_scores: If True, return (features, scores_df).
        random_state: For MI and RF.
        include_chi2: Compute Chi-square if applicable.
        include_rf: Compute Random Forest importance.
        fast_mode: Skip MI and RF for fast screening.
        n_jobs: Parallel jobs.
        var_threshold: Variance threshold to identify constant features.

    Returns:
        List of top feature names, or tuple with full scores if return_scores=True.

    Raises:
        ValueError: If top_k <= 0 or invalid task/aggregation.
    """
    if isinstance(features, str):
        features = [features]
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'")

    data = data.copy()
    label_encoder: LabelEncoder | None = None
    if task == "classification" and data[target].dtype == "object":
        label_encoder = LabelEncoder()
        data[target] = label_encoder.fit_transform(data[target])
        log.info("Auto-encoded target '%s': %s", target, dict(enumerate(label_encoder.classes_)))


    variances = data[features].var(numeric_only=True)
    constant_features = variances[variances <= var_threshold].index.tolist()
    if constant_features:
        log.info(f"Removing {len(constant_features)} constant/near-constant features")
        features = [f for f in features if f not in constant_features]

    inputs = data[features].values

    scores_dfs: list[pd.DataFrame] = []

    # 1. Association
    assoc_df = compute_association(data, target, features, task=task, pivot=False)
    scores_dfs.append(assoc_df[["feature", "association"]])

    # 2. ANOVA F
    f_scores_df, p_values_df = compute_anova_f(data, target, features, task=task, pivot=False)
    scores_dfs.append(f_scores_df[["feature", "f_score"]])
    scores_dfs.append(p_values_df[["feature", "p_value"]])

    # 3. Mutual Information
    if not fast_mode:
        mi_df = compute_mutual_information(
            data, target, features, task=task, pivot=False, random_state=random_state, n_jobs=n_jobs
        )
        scores_dfs.append(mi_df[["feature", "score"]].rename(columns={"score": "mutual_info"}))
    else:
        scores_dfs.append(pd.DataFrame({"feature": features, "mutual_info": 0.0}))

    # 4. Chi-square
    if task == "classification" and include_chi2:
        try:
            if inputs.min() >= 0:
                chi2_df = compute_chi2(data, target, features, pivot=False)
                scores_dfs.append(chi2_df[["feature", "chi2_score"]])
            else:
                log.info("Chi-square skipped: negative values.")
        except Exception as e:
            log.exception("Chi-square failed: %s", e)

    # 5. Random Forest
    if include_rf and not fast_mode:
        try:
            rf_df = compute_rf_importance(
                data, target, features, task=task, random_state=random_state, n_jobs=n_jobs, pivot=False
            )
            scores_dfs.append(rf_df[["feature", "rf_importance"]])
        except Exception as e:
            log.exception("RF importance failed: %s", e)

    #  Merge all using reduce
    scores = reduce(
        lambda left, right: pd.merge(left, right, on="feature", how="left"),
        scores_dfs,
        pd.DataFrame({"feature": features}),
    )

    # Filter by p-value
    if "p_value" in scores.columns:
        scores = scores[scores["p_value"] <= alpha].copy()
    if scores.empty:
        return ([], scores) if return_scores else []

    # Rank aggregation
    base_cols = ["mutual_info", "association", "f_score"]
    extra_cols = [c for c in ["chi2_score", "rf_importance"] if c in scores.columns]
    rank_cols = base_cols + extra_cols

    ranks = scores[rank_cols].rank(ascending=True, pct=use_percentile_ranking)
    ranks["arith_rank"] = ranks.mean(axis=1)
    ranks["med_rank"] = ranks.median(axis=1)
    ranks["sum_rank"] = ranks.sum(axis=1)
    log_ranks = np.log(ranks.replace(0, np.nan))
    ranks["geom_rank"] = np.exp(log_ranks.mean(axis=1))

    borda = ranks[["arith_rank", "med_rank", "sum_rank", "geom_rank"]].rank(ascending=True, pct=use_percentile_ranking)
    ranks["borda_count"] = borda.sum(axis=1)

    scores = pd.concat([scores, ranks], axis=1)

    if rank_aggregation not in ranks.columns:
        raise ValueError(f"Invalid rank_aggregation: {rank_aggregation}")

    if normalize_scores and scores[rank_aggregation].max() > 0:
        scores[rank_aggregation] = scores[rank_aggregation] / scores[rank_aggregation].max()

    scores = scores.sort_values(by=rank_aggregation, ascending=False)
    top_features = scores.head(top_k)["feature"].tolist()

    return (top_features, scores) if return_scores else top_features
