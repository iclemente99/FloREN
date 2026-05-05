import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
from collections import Counter
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from scipy.stats import spearmanr
import pandas as pd  # Assuming metadata is a pandas DataFrame
from scipy.spatial.distance import pdist, squareform
from typing import List, Optional, Union

def compute_knn_predictions(embeddings, labels, k=3, is_categorical=True):
    """
    Compute KNN-based predictions for labels.
    :param embeddings: np.array of shape (n_samples, n_features)
    :param labels: np.array of shape (n_samples,) - categorical or numerical labels
    :param k: number of nearest neighbors
    :param is_categorical: bool, True for categorical, False for numerical/ordinal
    :return: np.array of predicted labels
    """
    n_samples = embeddings.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)
    #
    pred_labels = []
    for i in range(n_samples):
        nn_indices = indices[i, 1:k + 1]  # exclude self
        nn_labels = labels[nn_indices]
        #
        if is_categorical:
            # Majority vote for categorical
            pred = Counter(nn_labels).most_common(1)[0][0]
        else:
            # Average for numerical
            pred = np.mean(nn_labels)
        #
        pred_labels.append(pred)
    #
    return np.array(pred_labels)


def compute_information_retention(embeddings, metadata, covariate, is_categorical=True):
    """
    Compute information retention score I for a given covariate.
    :param embeddings: np.array of shape (n_samples, n_features)
    :param metadata: pd.DataFrame with samples as rows
    :param covariate: str, column name in metadata
    :param is_categorical: bool, True for categorical, False for numerical/ordinal
    :return: float, information retention score
    """
    labels = metadata[covariate].values
    pred_labels = compute_knn_predictions(embeddings, labels, is_categorical=is_categorical)
    #
    if is_categorical:
        unique_labels = np.unique(labels)
        L = len(unique_labels)
        if L <= 1:
            return 1.0  # Degenerate case
        f_macro = f1_score(labels, pred_labels, average='macro')
        I = (L / (L - 1)) * (f_macro - (1 / L))
        I = max(I, 0.0)  # Clip negative values to 0 as per Appendix A.5
    else:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(labels, pred_labels)
        I = abs(rho)
    #
    return I


def compute_diffusion_pseudotime(embeddings, root_index, n_components=10, t=1):
    """
    Compute diffusion pseudotime (DPT) from a root sample in the embedding space.
    :param embeddings: np.array of shape (n_samples, n_features)
    :param root_index: int, index of the root sample (earliest point)
    :param n_components: int, number of diffusion components to use
    :param t: int or float, diffusion time parameter
    :return: np.array of shape (n_samples,), pseudotime values
    """
    n = embeddings.shape[0]
    dist = cdist(embeddings, embeddings, 'euclidean')
    #
    # Choose sigma as median of positive distances
    positive_dist = dist[dist > 0]
    sigma = np.median(positive_dist) if len(positive_dist) > 0 else 1.0
    #
    # Affinity matrix
    K = np.exp(-dist ** 2 / (2 * sigma ** 2))
    #
    # Degree
    row_sums = np.sum(K, axis=1)
    #
    # Handle potential zero sums (though unlikely)
    row_sums[row_sums == 0] = 1.0
    #
    D_inv_sqrt = np.diag(1.0 / np.sqrt(row_sums))
    #
    # Symmetric normalized kernel
    K_sym = D_inv_sqrt @ K @ D_inv_sqrt
    #
    # Eigendecomposition
    evals, evecs = eigh(K_sym)
    #
    # Sort by descending eigenvalues
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    #
    # Right eigenvectors
    phi = D_inv_sqrt @ evecs
    #
    # Diffusion coordinates (skip the first trivial component)
    lambdas_pow_t = evals[1:n_components + 1] ** t
    coords = phi[:, 1:n_components + 1] * lambdas_pow_t
    #
    # Pseudotime as Euclidean distance in diffusion space from root
    T = np.linalg.norm(coords - coords[root_index, :], axis=1)
    #
    return T


def compute_trajectory_preservation(embeddings, metadata, covariate='Age', n_components=10, t=1):
    """
    Compute trajectory preservation score TS.
    :param embeddings: np.array of shape (n_samples, n_features)
    :param metadata: pd.DataFrame with samples as rows
    :param covariate: str, column name for the trajectory covariate (e.g., 'Age')
    :param n_components: int, number of diffusion components
    :param t: int or float, diffusion time
    :return: float, trajectory preservation score
    """
    R = metadata[covariate].values.astype(float)
    root_index = np.argmin(R)  # Assume minimum value is the earliest point
    #
    T = compute_diffusion_pseudotime(embeddings, root_index, n_components=n_components, t=t)
    #
    rho, _ = spearmanr(T, R)
    TS = abs(rho)
    #
    return TS

def compute_replicate_robustness(
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        patient_col: str = "patient_id",  # column that identifies the individual/patient
        replicate_group_col: str = None,  # optional: if replicates are grouped differently (e.g. timepoint)
        distance_metric: str = "euclidean",
        verbose: bool = True
) -> float:
    """
    Compute the SPARE replicate robustness score (RS).
    This metric evaluates how well replicates from the same individual are placed
    closer to each other in the embedding space than to other (non-replicate) samples.
    The score is averaged over all identified replicate pairs.
    RS(i,j) = fraction of other samples k that are LESS similar (farther) to Xi than Xj is.
            → high value = replicate Xj is among the closest samples to Xi
    Final RS = average RS(i,j) over all replicate pairs (considering both directions or unique pairs)
    Parameters
    ----------
    embeddings : np.ndarray (n_samples, n_features)
        Sample representation (e.g. MrVI latent space)
    metadata : pd.DataFrame
        Must contain a column identifying replicates (usually same patient/individual)
    patient_col : str
        Column name in metadata that groups replicates (same value = same patient)
    replicate_group_col : str, optional
        If replicates are only within certain conditions (e.g. same timepoint),
        provide this column name. Pairs are only considered within the same group.
    distance_metric : str
        Metric passed to scipy.spatial.distance.pdist ('euclidean', 'cosine', ...)
    Returns
    -------
    float
        Average replicate robustness score ∈ [0, 1]
        1 = perfect (replicates are always the closest)
        0 = terrible (replicates are never close)
    """
    n_samples = embeddings.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 samples to compute replicate robustness")
    #
    # Compute full pairwise distance matrix
    dist_condensed = pdist(embeddings, metric=distance_metric)
    dist_matrix = squareform(dist_condensed)  # (n × n) symmetric distance matrix
    #
    # Identify replicate pairs
    group_col = replicate_group_col if replicate_group_col is not None else patient_col
    groups = metadata[group_col].values
    #
    # Find all unique pairs of replicates (i < j)
    replicate_pairs = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) >= 2:
            # All pairs within this group are considered replicates
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    replicate_pairs.append((idx[a], idx[b]))
    #
    if len(replicate_pairs) == 0:
        raise ValueError(f"No replicate pairs found using column '{group_col}'")
    #
    if verbose:
        print(f"Found {len(replicate_pairs)} replicate pairs")
    #
    robustness_scores = []
    #
    for i, j in replicate_pairs:
        # For sample i: how many other samples k are farther than dist(i,j)?
        dij = dist_matrix[i, j]
        distances_from_i = dist_matrix[i, :]  # includes self and j
        # Exclude self (i) and the replicate (j)
        mask_other = (np.arange(n_samples) != i) & (np.arange(n_samples) != j)
        n_other = np.sum(mask_other)
        #
        if n_other == 0:
            continue  # degenerate case
        #
        n_less_similar = np.sum(distances_from_i[mask_other] > dij)
        RS_i = n_less_similar / n_other
        #
        # We can do the symmetric direction (from j), or just one way.
        # Paper implies computing per pair direction → we average both
        distances_from_j = dist_matrix[j, :]
        n_less_similar_j = np.sum(distances_from_j[mask_other] > dij)  # dij is symmetric
        RS_j = n_less_similar_j / n_other
        #
        # Average the two directions for this pair
        robustness_scores.append((RS_i + RS_j) / 2)
    #
    if len(robustness_scores) == 0:
        return np.nan
    #
    final_RS = np.mean(robustness_scores)
    #
    if verbose:
        print(f"Replicate robustness score (averaged over {len(robustness_scores)} directed pairs): "
              f"{final_RS:.4f}")
    #
    return final_RS


def safe_replicate_robustness(colname: str) -> float:
    if colname not in metadata.columns:
        print(f"Column '{colname}' not found → RS_{colname} = 0")
        return 0.0
    try:
        rs = compute_replicate_robustness(
            embeddings,
            metadata,
            patient_col=colname,
            verbose=False   # reduce spam
        )
        print(f"Replicate robustness ({colname}): {rs:.4f}")
        return rs
    except Exception as e:
        print(f"Error computing replicate robustness for '{colname}': {e}")
        return 0.0

def spare_aggregate_score(
    information_retention_scores: Union[float, List[float], np.ndarray],
    batch_removal_scores: Union[float, List[float], np.ndarray],
    trajectory_preservation: Optional[Union[float, List[float]]] = None,
    replicate_robustness: Optional[float] = None,
    batch_weight: float = 0.5,
    normalize_denominator: bool = True
) -> float:
    """
    Aggregate SPARE metrics into a single final score following the paper's logic.
    Parameters
    ----------
    information_retention_scores : float | list | np.ndarray
        Score(s) for relevant metadata (I). Usually a list if multiple covariates.
    batch_removal_scores : float | list | np.ndarray
        Score(s) for technical metadata / batch effect removal (B = 1 - I_technical).
    trajectory_preservation : float | list | None, optional
        Trajectory preservation score(s) TS. If None, treated as 0.
    replicate_robustness : float | None, optional
        Replicate robustness score RS. If None, treated as 0.
    batch_weight : float, default 0.5
        Weight for the batch removal part (paper uses 1/2)
    normalize_denominator : bool, default True
        If True, divide by sum of weights (3.5 when all four parts are present).
        If False, just sum the weighted terms (can exceed 1 if all perfect).
    Returns
    -------
    float
        Final aggregated SPARE score, typically ∈ [0, 1]
    """
    # Convert to np.array and compute averages where needed
    I = np.mean(np.atleast_1d(information_retention_scores))
    B = np.mean(np.atleast_1d(batch_removal_scores))
    #
    TS = 0.0
    if trajectory_preservation is not None:
        TS = np.mean(np.atleast_1d(trajectory_preservation))
    #
    RS = 0.0
    if replicate_robustness is not None:
        RS = float(replicate_robustness)
    #
    # Weighted sum
    weighted_sum = (
        1.0 * I +
        batch_weight * B +
        1.0 * TS +
        1.0 * RS
    )
    #
    if normalize_denominator:
        # Paper implies scaling so max = 1 when all components are perfect
        total_weight = 1.0 + batch_weight + 1.0 + 1.0  # 3.5
        final_score = weighted_sum / total_weight
    else:
        final_score = weighted_sum
    #
    return float(np.clip(final_score, 0.0, 1.0))  # safeguard








# ────────────────────────────────────────────────────────────────────────────────
# Example usage - relevant and technical metadata
# ────────────────────────────────────────────────────────────────────────────────
# Assume:
#    embeddings is your MrVI results matrix: np.array (n_samples, 30)
#    metadata is pd.DataFrame with columns ['Disease', 'Age', 'Batch', 'Ethnicity', 'Tissue']

# For relevant metadata (Disease, assuming categorical)
# relevant_score = compute_information_retention(embeddings, metadata, 'Disease', is_categorical=True)
# For technical metadata (Batch, assuming categorical)
# technical_score = 1 - compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)

# ────────────────────────────────────────────────────────────────────────────────
# Example usage - trajectory score
# ────────────────────────────────────────────────────────────────────────────────
# Assume:
# Assume 'embeddings' is your MrVI results matrix: np.array (n_samples, 30)
# Assume 'metadata' is pd.DataFrame with columns ['Disease', 'Age', 'Batch', 'Ethnicity', 'Tissue']

# trajectory_score = compute_trajectory_preservation(embeddings, metadata, covariate='Age')

# ────────────────────────────────────────────────────────────────────────────────
# Example usage - replicate robustness
# ────────────────────────────────────────────────────────────────────────────────
# Assume:
#   embeddings = your MrVI output (n_samples × 30)
#   metadata   = pd.DataFrame with columns including e.g. 'patient_id', 'Disease', 'Batch', ...

# If your dataset has a column that directly identifies the patient/individual:
#rs = compute_replicate_robustness(embeddings, metadata, patient_col="patient_id")
# If replicates are only valid within the same condition (e.g. same time point or visit):
#rs = compute_replicate_robustness(embeddings, metadata,
#                                   patient_col="patient_id",
#                                   replicate_group_col="timepoint")

# ────────────────────────────────────────────────────────────────────────────────
# Example usage — integrating previous functions
# ────────────────────────────────────────────────────────────────────────────────

# Suppose you have already computed:
# 1. Relevant metadata (e.g. Disease, maybe Ethnicity if relevant)
#I_disease   = compute_information_retention(embeddings, metadata, 'Disease',   is_categorical=True)
#I_ethnicity = compute_information_retention(embeddings, metadata, 'Ethnicity', is_categorical=True)
#relevant_scores = [I_disease, I_ethnicity]          # or just [I_disease] if only one
# 2. Technical metadata (e.g. Batch)
#B_batch = 1.0 - compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
#technical_scores = [B_batch]                        # can be more
# 3. Trajectory (e.g. Age)
#TS_age = compute_trajectory_preservation(embeddings, metadata, covariate='Age')
# 4. Replicate robustness (COPD-style, needs patient/replicate grouping)
#RS = compute_replicate_robustness(embeddings, metadata, patient_col="patient_id")
# Now aggregate
#final_spare_score = spare_aggregate_score(
#    information_retention_scores = relevant_scores,
#    batch_removal_scores         = technical_scores,
#    trajectory_preservation      = TS_age,
#    replicate_robustness         = RS
#)

#print(f"Final SPARE score: {final_spare_score:.4f}")






# ────────────────────────────────────────────────────────────────────────────────
# BINVIGNAT — PSEUDOBULK
# ────────────────────────────────────────────────────────────────────────────────

embeddings_df = pd.read_csv("C:/Users/Inigo/Desktop/Samp_Emb_benchmark/pseudobulk/binvignat_pseudobulk_embs.csv")
embeddings = embeddings_df.iloc[:,1:].values
embeddings_patient_id = (
    embeddings_df["Unnamed: 0"]
    .str.split("__")
    .str[0]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Binvignat/samples_metadata.csv")
metadata = metadata.iloc[:,1:]
metadata.columns = ["patient_id", "Disease", "Sex", "Age", "Ethnicity", "Batch"]
metadata = (
    metadata
    .drop_duplicates(subset="patient_id", keep="first")
    .assign(Batch=lambda df: df["Batch"].where(
        ~df["patient_id"].eq("Control07_1_C_F"), 0
    ))
)
cat_cols = ["Disease", "Sex", "Ethnicity", "Batch"]
for col in cat_cols:
    metadata[col] = metadata[col].astype("category")

metadata = (
    metadata
    .set_index("patient_id")
    .loc[embeddings_patient_id]
    .reset_index()
)
# Suppose you have already computed:
# 1. Relevant metadata (e.g. Disease, maybe Ethnicity if relevant)
#I_disease   = compute_information_retention(embeddings, metadata, 'Disease',   is_categorical=True)
#I_ethnicity = compute_information_retention(embeddings, metadata, 'Ethnicity', is_categorical=True)
#relevant_scores = [I_disease]          # or just [I_disease] if only one
# 2. Technical metadata (e.g. Batch)
#B_batch = 1.0 - compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
#technical_scores = [B_batch]                        # can be more
# 3. Trajectory (e.g. Age)
#TS_age = compute_trajectory_preservation(embeddings, metadata, covariate='Age')
# 4. Replicate robustness (COPD-style, needs patient/replicate grouping)
#RS_ethnicity = compute_replicate_robustness(embeddings, metadata, patient_col="Ethnicity")
#RS_sex = compute_replicate_robustness(embeddings, metadata, patient_col="Sex")
#RS_tissue = compute_replicate_robustness(embeddings, metadata, patient_col="Tissue")
# Now aggregate
#final_spare_score = spare_aggregate_score(
#    information_retention_scores = relevant_scores,
#    batch_removal_scores         = technical_scores,
#    trajectory_preservation      = TS_age,
#    replicate_robustness         = RS
#)

#print(f"Final SPARE score: {final_spare_score:.4f}")


# 1. Relevant metadata (Disease is expected — but we check anyway)
relevant_scores = []
if "Disease" in metadata.columns:
    try:
        I_disease = compute_information_retention(embeddings, metadata, 'Disease', is_categorical=True)
        relevant_scores.append(I_disease)
    except Exception as e:
        print(f"Error computing I_Disease: {e}")
        relevant_scores.append(0.0)
else:
    print("Column 'Disease' not found → setting I_Disease = 0")
    relevant_scores.append(0.0)

# 2. Technical / batch removal
technical_scores = []
if "Batch" in metadata.columns:
    try:
        I_batch = compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
        B_batch = 1.0 - I_batch
        technical_scores.append(B_batch)
    except Exception as e:
        print(f"Error computing B_batch: {e}")
        technical_scores.append(0.0)
else:
    print("Column 'Batch' not found → setting B_batch = 0")
    technical_scores.append(0.0)

# 3. Trajectory preservation — only if Age exists **and** is numeric
TS_age = 0.0
if "Age" in metadata.columns:
    age_series = metadata["Age"]
    # Check if we can reasonably treat it as numeric
    if pd.api.types.is_numeric_dtype(age_series) or age_series.dtype.name == 'category':
        # Try to convert — will raise if truly non-numeric
        try:
            age_numeric = pd.to_numeric(age_series, errors='coerce')
            if age_numeric.notna().sum() >= 5:  # arbitrary small threshold
                metadata_for_ts = metadata.copy()
                metadata_for_ts["Age"] = age_numeric
                TS_age = compute_trajectory_preservation(
                    embeddings,
                    metadata_for_ts,
                    covariate='Age'
                )
                print(f"Trajectory preservation (Age): {TS_age:.4f}")
            else:
                print("Too few valid numeric Age values → skipping (TS_age = 0)")
        except Exception as e:
            print(f"Could not convert Age to numeric: {e} → TS_age = 0")
    else:
        print("Age is not numeric → skipping trajectory preservation (TS_age = 0)")
else:
    print("Column 'Age' not found → TS_age = 0")

# 4. Replicate robustness — only compute if column exists and has replicates
RS_ethnicity = safe_replicate_robustness("Ethnicity")
RS_sex       = safe_replicate_robustness("Sex")
RS_tissue    = safe_replicate_robustness("Tissue")   # ← Tissue not in your metadata
# You probably want to average the replicate scores you actually computed
replicate_scores = [rs for rs in [RS_ethnicity, RS_sex] if rs > 0]  # or include 0s
RS_final = np.mean(replicate_scores) if replicate_scores else 0.0
print(f"Average replicate robustness: {RS_final:.4f}")
# ────────────────────────────────────────────────
# Final aggregation
# ────────────────────────────────────────────────
final_spare_score = spare_aggregate_score(
    information_retention_scores = relevant_scores,
    batch_removal_scores         = technical_scores,
    trajectory_preservation      = TS_age,
    replicate_robustness         = RS_final,          # ← use averaged or chosen one
)
print("\n" + "="*60)
print(f"Final SPARE score: {final_spare_score:.4f}")
print(f"  • Information retention: {np.mean(relevant_scores):.4f}")
print(f"  • Batch removal:         {np.mean(technical_scores):.4f}")
print(f"  • Trajectory (Age):      {TS_age:.4f}")
print(f"  • Replicate robustness:  {RS_final:.4f}")
print("="*60)










# ────────────────────────────────────────────────────────────────────────────────
# BINVIGNAT — GLOSCOPE
# ────────────────────────────────────────────────────────────────────────────────

embeddings_df = pd.read_csv("C:/Users/Inigo/Desktop/Samp_Emb_benchmark/gloscope/binvignat_gloscope_embs.csv")
embeddings = embeddings_df.iloc[:,3:].values
embeddings_patient_id = (
    embeddings_df["Unnamed: 0"]
    #.str.split("__")
    #.str[0]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Binvignat/samples_metadata.csv")
metadata = metadata.iloc[:,1:]
metadata.columns = ["patient_id", "Disease", "Sex", "Age", "Ethnicity", "Batch"]
metadata = (
    metadata
    .drop_duplicates(subset="patient_id", keep="first")
    .assign(Batch=lambda df: df["Batch"].where(
        ~df["patient_id"].eq("Control07_1_C_F"), 0
    ))
)
cat_cols = ["Disease", "Sex", "Ethnicity", "Batch"]
for col in cat_cols:
    metadata[col] = metadata[col].astype("category")

metadata = (
    metadata
    .set_index("patient_id")
    .loc[embeddings_patient_id]
    .reset_index()
)

# 1. Relevant metadata (Disease is expected — but we check anyway)
relevant_scores = []
if "Disease" in metadata.columns:
    try:
        I_disease = compute_information_retention(embeddings, metadata, 'Disease', is_categorical=True)
        relevant_scores.append(I_disease)
    except Exception as e:
        print(f"Error computing I_Disease: {e}")
        relevant_scores.append(0.0)
else:
    print("Column 'Disease' not found → setting I_Disease = 0")
    relevant_scores.append(0.0)

# 2. Technical / batch removal
technical_scores = []
if "Batch" in metadata.columns:
    try:
        I_batch = compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
        B_batch = 1.0 - I_batch
        technical_scores.append(B_batch)
    except Exception as e:
        print(f"Error computing B_batch: {e}")
        technical_scores.append(0.0)
else:
    print("Column 'Batch' not found → setting B_batch = 0")
    technical_scores.append(0.0)

# 3. Trajectory preservation — only if Age exists **and** is numeric
TS_age = 0.0
if "Age" in metadata.columns:
    age_series = metadata["Age"]
    # Check if we can reasonably treat it as numeric
    if pd.api.types.is_numeric_dtype(age_series) or age_series.dtype.name == 'category':
        # Try to convert — will raise if truly non-numeric
        try:
            age_numeric = pd.to_numeric(age_series, errors='coerce')
            if age_numeric.notna().sum() >= 5:  # arbitrary small threshold
                metadata_for_ts = metadata.copy()
                metadata_for_ts["Age"] = age_numeric
                TS_age = compute_trajectory_preservation(
                    embeddings,
                    metadata_for_ts,
                    covariate='Age'
                )
                print(f"Trajectory preservation (Age): {TS_age:.4f}")
            else:
                print("Too few valid numeric Age values → skipping (TS_age = 0)")
        except Exception as e:
            print(f"Could not convert Age to numeric: {e} → TS_age = 0")
    else:
        print("Age is not numeric → skipping trajectory preservation (TS_age = 0)")
else:
    print("Column 'Age' not found → TS_age = 0")

# 4. Replicate robustness — only compute if column exists and has replicates
RS_ethnicity = safe_replicate_robustness("Ethnicity")
RS_sex       = safe_replicate_robustness("Sex")
RS_tissue    = safe_replicate_robustness("Tissue")   # ← Tissue not in your metadata
# You probably want to average the replicate scores you actually computed
replicate_scores = [rs for rs in [RS_ethnicity, RS_sex] if rs > 0]  # or include 0s
RS_final = np.mean(replicate_scores) if replicate_scores else 0.0
print(f"Average replicate robustness: {RS_final:.4f}")
# Final aggregation
final_spare_score = spare_aggregate_score(
    information_retention_scores = relevant_scores,
    batch_removal_scores         = technical_scores,
    trajectory_preservation      = TS_age,
    replicate_robustness         = RS_final,          # ← use averaged or chosen one
)
print("\n" + "="*60)
print(f"Final SPARE score: {final_spare_score:.4f}")
print(f"  • Information retention: {np.mean(relevant_scores):.4f}")
print(f"  • Batch removal:         {np.mean(technical_scores):.4f}")
print(f"  • Trajectory (Age):      {TS_age:.4f}")
print(f"  • Replicate robustness:  {RS_final:.4f}")
print("="*60)










# ────────────────────────────────────────────────────────────────────────────────
# BINVIGNAT — MRVI
# ────────────────────────────────────────────────────────────────────────────────

embeddings_df = pd.read_csv("C:/Users/Inigo/Desktop/Samp_Emb_benchmark/mrvi/binvignat_mrvi_zmean_representations.csv")
embeddings_df = embeddings_df.groupby("sample_id").mean()
embeddings = np.array(embeddings_df)
embeddings_patient_id = (
    embeddings_df.index
    .str.split("__")
    .str[0]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Binvignat/samples_metadata.csv")
metadata = metadata.iloc[:,1:]
metadata.columns = ["patient_id", "Disease", "Sex", "Age", "Ethnicity", "Batch"]
metadata = (
    metadata
    .drop_duplicates(subset="patient_id", keep="first")
    .assign(Batch=lambda df: df["Batch"].where(
        ~df["patient_id"].eq("Control07_1_C_F"), 0
    ))
)
cat_cols = ["Disease", "Sex", "Ethnicity", "Batch"]
for col in cat_cols:
    metadata[col] = metadata[col].astype("category")

metadata = (
    metadata
    .set_index("patient_id")
    .loc[embeddings_patient_id]
    .reset_index()
)

# 1. Relevant metadata (Disease is expected — but we check anyway)
relevant_scores = []
if "Disease" in metadata.columns:
    try:
        I_disease = compute_information_retention(embeddings, metadata, 'Disease', is_categorical=True)
        relevant_scores.append(I_disease)
    except Exception as e:
        print(f"Error computing I_Disease: {e}")
        relevant_scores.append(0.0)
else:
    print("Column 'Disease' not found → setting I_Disease = 0")
    relevant_scores.append(0.0)

# 2. Technical / batch removal
technical_scores = []
if "Batch" in metadata.columns:
    try:
        I_batch = compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
        B_batch = 1.0 - I_batch
        technical_scores.append(B_batch)
    except Exception as e:
        print(f"Error computing B_batch: {e}")
        technical_scores.append(0.0)
else:
    print("Column 'Batch' not found → setting B_batch = 0")
    technical_scores.append(0.0)

# 3. Trajectory preservation — only if Age exists **and** is numeric
TS_age = 0.0
if "Age" in metadata.columns:
    age_series = metadata["Age"]
    # Check if we can reasonably treat it as numeric
    if pd.api.types.is_numeric_dtype(age_series) or age_series.dtype.name == 'category':
        # Try to convert — will raise if truly non-numeric
        try:
            age_numeric = pd.to_numeric(age_series, errors='coerce')
            if age_numeric.notna().sum() >= 5:  # arbitrary small threshold
                metadata_for_ts = metadata.copy()
                metadata_for_ts["Age"] = age_numeric
                TS_age = compute_trajectory_preservation(
                    embeddings,
                    metadata_for_ts,
                    covariate='Age'
                )
                print(f"Trajectory preservation (Age): {TS_age:.4f}")
            else:
                print("Too few valid numeric Age values → skipping (TS_age = 0)")
        except Exception as e:
            print(f"Could not convert Age to numeric: {e} → TS_age = 0")
    else:
        print("Age is not numeric → skipping trajectory preservation (TS_age = 0)")
else:
    print("Column 'Age' not found → TS_age = 0")

# 4. Replicate robustness — only compute if column exists and has replicates
RS_ethnicity = safe_replicate_robustness("Ethnicity")
RS_sex       = safe_replicate_robustness("Sex")
RS_tissue    = safe_replicate_robustness("Tissue")   # ← Tissue not in your metadata
# You probably want to average the replicate scores you actually computed
replicate_scores = [rs for rs in [RS_ethnicity, RS_sex] if rs > 0]  # or include 0s
RS_final = np.mean(replicate_scores) if replicate_scores else 0.0
print(f"Average replicate robustness: {RS_final:.4f}")
# Final aggregation
final_spare_score = spare_aggregate_score(
    information_retention_scores = relevant_scores,
    batch_removal_scores         = technical_scores,
    trajectory_preservation      = TS_age,
    replicate_robustness         = RS_final,          # ← use averaged or chosen one
)
print("\n" + "="*60)
print(f"Final SPARE score: {final_spare_score:.4f}")
print(f"  • Information retention: {np.mean(relevant_scores):.4f}")
print(f"  • Batch removal:         {np.mean(technical_scores):.4f}")
print(f"  • Trajectory (Age):      {TS_age:.4f}")
print(f"  • Replicate robustness:  {RS_final:.4f}")
print("="*60)










# ────────────────────────────────────────────────────────────────────────────────
# BINVIGNAT — SAMPLECLR
# ────────────────────────────────────────────────────────────────────────────────

embeddings_df = pd.read_csv("C:/Users/Inigo/Desktop/Samp_Emb_benchmark/SampleCLR/binvignat_sampleclr_tiny.csv")
embeddings = embeddings_df.iloc[:,1:].values
embeddings_patient_id = (
    embeddings_df["Unnamed: 0"]
    .str.split("__")
    .str[0]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Binvignat/samples_metadata.csv")
metadata = metadata.iloc[:,1:]
metadata.columns = ["patient_id", "Disease", "Sex", "Age", "Ethnicity", "Batch"]
metadata = (
    metadata
    .drop_duplicates(subset="patient_id", keep="first")
    .assign(Batch=lambda df: df["Batch"].where(
        ~df["patient_id"].eq("Control07_1_C_F"), 0
    ))
)
cat_cols = ["Disease", "Sex", "Ethnicity", "Batch"]
for col in cat_cols:
    metadata[col] = metadata[col].astype("category")

metadata = (
    metadata
    .set_index("patient_id")
    .loc[embeddings_patient_id]
    .reset_index()
)

# 1. Relevant metadata (Disease is expected — but we check anyway)
relevant_scores = []
if "Disease" in metadata.columns:
    try:
        I_disease = compute_information_retention(embeddings, metadata, 'Disease', is_categorical=True)
        relevant_scores.append(I_disease)
    except Exception as e:
        print(f"Error computing I_Disease: {e}")
        relevant_scores.append(0.0)
else:
    print("Column 'Disease' not found → setting I_Disease = 0")
    relevant_scores.append(0.0)

# 2. Technical / batch removal
technical_scores = []
if "Batch" in metadata.columns:
    try:
        I_batch = compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
        B_batch = 1.0 - I_batch
        technical_scores.append(B_batch)
    except Exception as e:
        print(f"Error computing B_batch: {e}")
        technical_scores.append(0.0)
else:
    print("Column 'Batch' not found → setting B_batch = 0")
    technical_scores.append(0.0)

# 3. Trajectory preservation — only if Age exists **and** is numeric
TS_age = 0.0
if "Age" in metadata.columns:
    age_series = metadata["Age"]
    # Check if we can reasonably treat it as numeric
    if pd.api.types.is_numeric_dtype(age_series) or age_series.dtype.name == 'category':
        # Try to convert — will raise if truly non-numeric
        try:
            age_numeric = pd.to_numeric(age_series, errors='coerce')
            if age_numeric.notna().sum() >= 5:  # arbitrary small threshold
                metadata_for_ts = metadata.copy()
                metadata_for_ts["Age"] = age_numeric
                TS_age = compute_trajectory_preservation(
                    embeddings,
                    metadata_for_ts,
                    covariate='Age'
                )
                print(f"Trajectory preservation (Age): {TS_age:.4f}")
            else:
                print("Too few valid numeric Age values → skipping (TS_age = 0)")
        except Exception as e:
            print(f"Could not convert Age to numeric: {e} → TS_age = 0")
    else:
        print("Age is not numeric → skipping trajectory preservation (TS_age = 0)")
else:
    print("Column 'Age' not found → TS_age = 0")

# 4. Replicate robustness — only compute if column exists and has replicates
RS_ethnicity = safe_replicate_robustness("Ethnicity")
RS_sex       = safe_replicate_robustness("Sex")
RS_tissue    = safe_replicate_robustness("Tissue")   # ← Tissue not in your metadata
# You probably want to average the replicate scores you actually computed
replicate_scores = [rs for rs in [RS_ethnicity, RS_sex] if rs > 0]  # or include 0s
RS_final = np.mean(replicate_scores) if replicate_scores else 0.0
print(f"Average replicate robustness: {RS_final:.4f}")
# Final aggregation
final_spare_score = spare_aggregate_score(
    information_retention_scores = relevant_scores,
    batch_removal_scores         = technical_scores,
    trajectory_preservation      = TS_age,
    replicate_robustness         = RS_final,          # ← use averaged or chosen one
)
print("\n" + "="*60)
print(f"Final SPARE score: {final_spare_score:.4f}")
print(f"  • Information retention: {np.mean(relevant_scores):.4f}")
print(f"  • Batch removal:         {np.mean(technical_scores):.4f}")
print(f"  • Trajectory (Age):      {TS_age:.4f}")
print(f"  • Replicate robustness:  {RS_final:.4f}")
print("="*60)











# ────────────────────────────────────────────────────────────────────────────────
# BINVIGNAT — FLOREN SSL
# ────────────────────────────────────────────────────────────────────────────────

base_dir = "C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_patient_embeddings/split/"
dfs = []
for fname in os.listdir(base_dir):
    if fname.startswith("sample_"):
        continue
    #
    if fname.endswith(".csv") and "_emb_ssl" in fname:
        patient_id = fname.replace("_emb_ssl.csv", "")
        df = pd.read_csv(os.path.join(base_dir, fname), header=None).T
        if df.shape[0] != 1:
            raise ValueError(f"{fname} has {df.shape[0]} rows (expected 1)")
        #
        df.index = [patient_id]
        dfs.append(df)

embeddings_df = pd.concat(dfs, axis=0)
embeddings = np.array(embeddings_df)
embeddings_patient_id = (
    embeddings_df.index
    #.str.split("_epoch_")
    #.str[0]
    #.str.split("sample_")
    #.str[0]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Binvignat/samples_metadata.csv")
metadata = metadata.iloc[:,1:]
metadata.columns = ["patient_id", "Disease", "Sex", "Age", "Ethnicity", "Batch"]
metadata = (
    metadata
    .drop_duplicates(subset="patient_id", keep="first")
    .assign(Batch=lambda df: df["Batch"].where(
        ~df["patient_id"].eq("Control07_1_C_F"), 0
    ))
)
cat_cols = ["Disease", "Sex", "Ethnicity", "Batch"]
for col in cat_cols:
    metadata[col] = metadata[col].astype("category")

metadata = (
    metadata
    .set_index("patient_id")
    .loc[embeddings_patient_id]
    .reset_index()
)

# 1. Relevant metadata (Disease is expected — but we check anyway)
relevant_scores = []
if "Disease" in metadata.columns:
    try:
        I_disease = compute_information_retention(embeddings, metadata, 'Disease', is_categorical=True)
        relevant_scores.append(I_disease)
    except Exception as e:
        print(f"Error computing I_Disease: {e}")
        relevant_scores.append(0.0)
else:
    print("Column 'Disease' not found → setting I_Disease = 0")
    relevant_scores.append(0.0)

# 2. Technical / batch removal
technical_scores = []
if "Batch" in metadata.columns:
    try:
        I_batch = compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
        B_batch = 1.0 - I_batch
        technical_scores.append(B_batch)
    except Exception as e:
        print(f"Error computing B_batch: {e}")
        technical_scores.append(0.0)
else:
    print("Column 'Batch' not found → setting B_batch = 0")
    technical_scores.append(0.0)

# 3. Trajectory preservation — only if Age exists **and** is numeric
TS_age = 0.0
if "Age" in metadata.columns:
    age_series = metadata["Age"]
    # Check if we can reasonably treat it as numeric
    if pd.api.types.is_numeric_dtype(age_series) or age_series.dtype.name == 'category':
        # Try to convert — will raise if truly non-numeric
        try:
            age_numeric = pd.to_numeric(age_series, errors='coerce')
            if age_numeric.notna().sum() >= 5:  # arbitrary small threshold
                metadata_for_ts = metadata.copy()
                metadata_for_ts["Age"] = age_numeric
                TS_age = compute_trajectory_preservation(
                    embeddings,
                    metadata_for_ts,
                    covariate='Age'
                )
                print(f"Trajectory preservation (Age): {TS_age:.4f}")
            else:
                print("Too few valid numeric Age values → skipping (TS_age = 0)")
        except Exception as e:
            print(f"Could not convert Age to numeric: {e} → TS_age = 0")
    else:
        print("Age is not numeric → skipping trajectory preservation (TS_age = 0)")
else:
    print("Column 'Age' not found → TS_age = 0")

# 4. Replicate robustness — only compute if column exists and has replicates
RS_ethnicity = safe_replicate_robustness("Ethnicity")
RS_sex       = safe_replicate_robustness("Sex")
RS_tissue    = safe_replicate_robustness("Tissue")   # ← Tissue not in your metadata
# You probably want to average the replicate scores you actually computed
replicate_scores = [rs for rs in [RS_ethnicity, RS_sex] if rs > 0]  # or include 0s
RS_final = np.mean(replicate_scores) if replicate_scores else 0.0
print(f"Average replicate robustness: {RS_final:.4f}")
# Final aggregation
final_spare_score = spare_aggregate_score(
    information_retention_scores = relevant_scores,
    batch_removal_scores         = technical_scores,
    trajectory_preservation      = TS_age,
    replicate_robustness         = RS_final,          # ← use averaged or chosen one
)
print("\n" + "="*60)
print(f"Final SPARE score: {final_spare_score:.4f}")
print(f"  • Information retention: {np.mean(relevant_scores):.4f}")
print(f"  • Batch removal:         {np.mean(technical_scores):.4f}")
print(f"  • Trajectory (Age):      {TS_age:.4f}")
print(f"  • Replicate robustness:  {RS_final:.4f}")
print("="*60)











# ────────────────────────────────────────────────────────────────────────────────
# BINVIGNAT — FLOREN L128
# ────────────────────────────────────────────────────────────────────────────────

base_dir = "C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_patient_embeddings/split/"
dfs = []
for fname in os.listdir(base_dir):
    if fname.startswith("sample_"):
        continue
    #
    if fname.endswith(".csv") and "_emb_l128" in fname:
        patient_id = fname.replace("_emb_l128.csv", "")
        df = pd.read_csv(os.path.join(base_dir, fname), header=None).T
        if df.shape[0] != 1:
            raise ValueError(f"{fname} has {df.shape[0]} rows (expected 1)")
        #
        df.index = [patient_id]
        dfs.append(df)

embeddings_df = pd.concat(dfs, axis=0)
embeddings = np.array(embeddings_df)
embeddings_patient_id = (
    embeddings_df.index
    #.str.split("__")
    #.str[0]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Binvignat/samples_metadata.csv")
metadata = metadata.iloc[:,1:]
metadata.columns = ["patient_id", "Disease", "Sex", "Age", "Ethnicity", "Batch"]
metadata = (
    metadata
    .drop_duplicates(subset="patient_id", keep="first")
    .assign(Batch=lambda df: df["Batch"].where(
        ~df["patient_id"].eq("Control07_1_C_F"), 0
    ))
)
cat_cols = ["Disease", "Sex", "Ethnicity", "Batch"]
for col in cat_cols:
    metadata[col] = metadata[col].astype("category")

metadata = (
    metadata
    .set_index("patient_id")
    .loc[embeddings_patient_id]
    .reset_index()
)

# 1. Relevant metadata (Disease is expected — but we check anyway)
relevant_scores = []
if "Disease" in metadata.columns:
    try:
        I_disease = compute_information_retention(embeddings, metadata, 'Disease', is_categorical=True)
        relevant_scores.append(I_disease)
    except Exception as e:
        print(f"Error computing I_Disease: {e}")
        relevant_scores.append(0.0)
else:
    print("Column 'Disease' not found → setting I_Disease = 0")
    relevant_scores.append(0.0)

# 2. Technical / batch removal
technical_scores = []
if "Batch" in metadata.columns:
    try:
        I_batch = compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
        B_batch = 1.0 - I_batch
        technical_scores.append(B_batch)
    except Exception as e:
        print(f"Error computing B_batch: {e}")
        technical_scores.append(0.0)
else:
    print("Column 'Batch' not found → setting B_batch = 0")
    technical_scores.append(0.0)

# 3. Trajectory preservation — only if Age exists **and** is numeric
TS_age = 0.0
if "Age" in metadata.columns:
    age_series = metadata["Age"]
    # Check if we can reasonably treat it as numeric
    if pd.api.types.is_numeric_dtype(age_series) or age_series.dtype.name == 'category':
        # Try to convert — will raise if truly non-numeric
        try:
            age_numeric = pd.to_numeric(age_series, errors='coerce')
            if age_numeric.notna().sum() >= 5:  # arbitrary small threshold
                metadata_for_ts = metadata.copy()
                metadata_for_ts["Age"] = age_numeric
                TS_age = compute_trajectory_preservation(
                    embeddings,
                    metadata_for_ts,
                    covariate='Age'
                )
                print(f"Trajectory preservation (Age): {TS_age:.4f}")
            else:
                print("Too few valid numeric Age values → skipping (TS_age = 0)")
        except Exception as e:
            print(f"Could not convert Age to numeric: {e} → TS_age = 0")
    else:
        print("Age is not numeric → skipping trajectory preservation (TS_age = 0)")
else:
    print("Column 'Age' not found → TS_age = 0")

# 4. Replicate robustness — only compute if column exists and has replicates
RS_ethnicity = safe_replicate_robustness("Ethnicity")
RS_sex       = safe_replicate_robustness("Sex")
RS_tissue    = safe_replicate_robustness("Tissue")   # ← Tissue not in your metadata
# You probably want to average the replicate scores you actually computed
replicate_scores = [rs for rs in [RS_ethnicity, RS_sex] if rs > 0]  # or include 0s
RS_final = np.mean(replicate_scores) if replicate_scores else 0.0
print(f"Average replicate robustness: {RS_final:.4f}")
# Final aggregation
final_spare_score = spare_aggregate_score(
    information_retention_scores = relevant_scores,
    batch_removal_scores         = technical_scores,
    trajectory_preservation      = TS_age,
    replicate_robustness         = RS_final,          # ← use averaged or chosen one
)
print("\n" + "="*60)
print(f"Final SPARE score: {final_spare_score:.4f}")
print(f"  • Information retention: {np.mean(relevant_scores):.4f}")
print(f"  • Batch removal:         {np.mean(technical_scores):.4f}")
print(f"  • Trajectory (Age):      {TS_age:.4f}")
print(f"  • Replicate robustness:  {RS_final:.4f}")
print("="*60)










# ────────────────────────────────────────────────────────────────────────────────
# BINVIGNAT — FLOREN G128
# ────────────────────────────────────────────────────────────────────────────────

base_dir = "C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_patient_embeddings/split/"
emb_key = "g128"
dfs = []
for fname in os.listdir(base_dir):
    if fname.startswith("sample_"):
        continue
    #
    if fname.endswith(".csv") and f"_emb_{emb_key}" in fname:
        patient_id = fname.replace(f"_emb_{emb_key}.csv", "")
        df = pd.read_csv(os.path.join(base_dir, fname), header=None).T
        if df.shape[0] != 1:
            raise ValueError(f"{fname} has {df.shape[0]} rows (expected 1)")
        #
        df.index = [patient_id]
        dfs.append(df)

embeddings_df = pd.concat(dfs, axis=0)
embeddings = np.array(embeddings_df)
embeddings_patient_id = (
    embeddings_df.index
    #.str.split("__")
    #.str[0]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Binvignat/samples_metadata.csv")
metadata = metadata.iloc[:,1:]
metadata.columns = ["patient_id", "Disease", "Sex", "Age", "Ethnicity", "Batch"]
metadata = (
    metadata
    .drop_duplicates(subset="patient_id", keep="first")
    .assign(Batch=lambda df: df["Batch"].where(
        ~df["patient_id"].eq("Control07_1_C_F"), 0
    ))
)
cat_cols = ["Disease", "Sex", "Ethnicity", "Batch"]
for col in cat_cols:
    metadata[col] = metadata[col].astype("category")

metadata = (
    metadata
    .set_index("patient_id")
    .loc[embeddings_patient_id]
    .reset_index()
)

# 1. Relevant metadata (Disease is expected — but we check anyway)
relevant_scores = []
if "Disease" in metadata.columns:
    try:
        I_disease = compute_information_retention(embeddings, metadata, 'Disease', is_categorical=True)
        relevant_scores.append(I_disease)
    except Exception as e:
        print(f"Error computing I_Disease: {e}")
        relevant_scores.append(0.0)
else:
    print("Column 'Disease' not found → setting I_Disease = 0")
    relevant_scores.append(0.0)

# 2. Technical / batch removal
technical_scores = []
if "Batch" in metadata.columns:
    try:
        I_batch = compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
        B_batch = 1.0 - I_batch
        technical_scores.append(B_batch)
    except Exception as e:
        print(f"Error computing B_batch: {e}")
        technical_scores.append(0.0)
else:
    print("Column 'Batch' not found → setting B_batch = 0")
    technical_scores.append(0.0)

# 3. Trajectory preservation — only if Age exists **and** is numeric
TS_age = 0.0
if "Age" in metadata.columns:
    age_series = metadata["Age"]
    # Check if we can reasonably treat it as numeric
    if pd.api.types.is_numeric_dtype(age_series) or age_series.dtype.name == 'category':
        # Try to convert — will raise if truly non-numeric
        try:
            age_numeric = pd.to_numeric(age_series, errors='coerce')
            if age_numeric.notna().sum() >= 5:  # arbitrary small threshold
                metadata_for_ts = metadata.copy()
                metadata_for_ts["Age"] = age_numeric
                TS_age = compute_trajectory_preservation(
                    embeddings,
                    metadata_for_ts,
                    covariate='Age'
                )
                print(f"Trajectory preservation (Age): {TS_age:.4f}")
            else:
                print("Too few valid numeric Age values → skipping (TS_age = 0)")
        except Exception as e:
            print(f"Could not convert Age to numeric: {e} → TS_age = 0")
    else:
        print("Age is not numeric → skipping trajectory preservation (TS_age = 0)")
else:
    print("Column 'Age' not found → TS_age = 0")

# 4. Replicate robustness — only compute if column exists and has replicates
RS_ethnicity = safe_replicate_robustness("Ethnicity")
RS_sex       = safe_replicate_robustness("Sex")
RS_tissue    = safe_replicate_robustness("Tissue")   # ← Tissue not in your metadata
# You probably want to average the replicate scores you actually computed
replicate_scores = [rs for rs in [RS_ethnicity, RS_sex] if rs > 0]  # or include 0s
RS_final = np.mean(replicate_scores) if replicate_scores else 0.0
print(f"Average replicate robustness: {RS_final:.4f}")
# Final aggregation
final_spare_score = spare_aggregate_score(
    information_retention_scores = relevant_scores,
    batch_removal_scores         = technical_scores,
    trajectory_preservation      = TS_age,
    replicate_robustness         = RS_final,          # ← use averaged or chosen one
)
print("\n" + "="*60)
print(f"Final SPARE score: {final_spare_score:.4f}")
print(f"  • Information retention: {np.mean(relevant_scores):.4f}")
print(f"  • Batch removal:         {np.mean(technical_scores):.4f}")
print(f"  • Trajectory (Age):      {TS_age:.4f}")
print(f"  • Replicate robustness:  {RS_final:.4f}")
print("="*60)











# ────────────────────────────────────────────────────────────────────────────────
# BINVIGNAT — FLOREN L64
# ────────────────────────────────────────────────────────────────────────────────

base_dir = "C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_patient_embeddings/split/"
emb_key = "l64"
dfs = []
for fname in os.listdir(base_dir):
    if fname.startswith("sample_"):
        continue
    #
    if fname.endswith(".csv") and f"_emb_{emb_key}" in fname:
        patient_id = fname.replace(f"_emb_{emb_key}.csv", "")
        df = pd.read_csv(os.path.join(base_dir, fname), header=None).T
        if df.shape[0] != 1:
            raise ValueError(f"{fname} has {df.shape[0]} rows (expected 1)")
        #
        df.index = [patient_id]
        dfs.append(df)

embeddings_df = pd.concat(dfs, axis=0)
embeddings = np.array(embeddings_df)
embeddings_patient_id = (
    embeddings_df.index
    #.str.split("__")
    #.str[0]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Binvignat/samples_metadata.csv")
metadata = metadata.iloc[:,1:]
metadata.columns = ["patient_id", "Disease", "Sex", "Age", "Ethnicity", "Batch"]
metadata = (
    metadata
    .drop_duplicates(subset="patient_id", keep="first")
    .assign(Batch=lambda df: df["Batch"].where(
        ~df["patient_id"].eq("Control07_1_C_F"), 0
    ))
)
cat_cols = ["Disease", "Sex", "Ethnicity", "Batch"]
for col in cat_cols:
    metadata[col] = metadata[col].astype("category")

metadata = (
    metadata
    .set_index("patient_id")
    .loc[embeddings_patient_id]
    .reset_index()
)

# 1. Relevant metadata (Disease is expected — but we check anyway)
relevant_scores = []
if "Disease" in metadata.columns:
    try:
        I_disease = compute_information_retention(embeddings, metadata, 'Disease', is_categorical=True)
        relevant_scores.append(I_disease)
    except Exception as e:
        print(f"Error computing I_Disease: {e}")
        relevant_scores.append(0.0)
else:
    print("Column 'Disease' not found → setting I_Disease = 0")
    relevant_scores.append(0.0)

# 2. Technical / batch removal
technical_scores = []
if "Batch" in metadata.columns:
    try:
        I_batch = compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
        B_batch = 1.0 - I_batch
        technical_scores.append(B_batch)
    except Exception as e:
        print(f"Error computing B_batch: {e}")
        technical_scores.append(0.0)
else:
    print("Column 'Batch' not found → setting B_batch = 0")
    technical_scores.append(0.0)

# 3. Trajectory preservation — only if Age exists **and** is numeric
TS_age = 0.0
if "Age" in metadata.columns:
    age_series = metadata["Age"]
    # Check if we can reasonably treat it as numeric
    if pd.api.types.is_numeric_dtype(age_series) or age_series.dtype.name == 'category':
        # Try to convert — will raise if truly non-numeric
        try:
            age_numeric = pd.to_numeric(age_series, errors='coerce')
            if age_numeric.notna().sum() >= 5:  # arbitrary small threshold
                metadata_for_ts = metadata.copy()
                metadata_for_ts["Age"] = age_numeric
                TS_age = compute_trajectory_preservation(
                    embeddings,
                    metadata_for_ts,
                    covariate='Age'
                )
                print(f"Trajectory preservation (Age): {TS_age:.4f}")
            else:
                print("Too few valid numeric Age values → skipping (TS_age = 0)")
        except Exception as e:
            print(f"Could not convert Age to numeric: {e} → TS_age = 0")
    else:
        print("Age is not numeric → skipping trajectory preservation (TS_age = 0)")
else:
    print("Column 'Age' not found → TS_age = 0")

# 4. Replicate robustness — only compute if column exists and has replicates
RS_ethnicity = safe_replicate_robustness("Ethnicity")
RS_sex       = safe_replicate_robustness("Sex")
RS_tissue    = safe_replicate_robustness("Tissue")   # ← Tissue not in your metadata
# You probably want to average the replicate scores you actually computed
replicate_scores = [rs for rs in [RS_ethnicity, RS_sex] if rs > 0]  # or include 0s
RS_final = np.mean(replicate_scores) if replicate_scores else 0.0
print(f"Average replicate robustness: {RS_final:.4f}")
# Final aggregation
final_spare_score = spare_aggregate_score(
    information_retention_scores = relevant_scores,
    batch_removal_scores         = technical_scores,
    trajectory_preservation      = TS_age,
    replicate_robustness         = RS_final,          # ← use averaged or chosen one
)
print("\n" + "="*60)
print(f"Final SPARE score: {final_spare_score:.4f}")
print(f"  • Information retention: {np.mean(relevant_scores):.4f}")
print(f"  • Batch removal:         {np.mean(technical_scores):.4f}")
print(f"  • Trajectory (Age):      {TS_age:.4f}")
print(f"  • Replicate robustness:  {RS_final:.4f}")
print("="*60)










# ────────────────────────────────────────────────────────────────────────────────
# BINVIGNAT — FLOREN G64
# ────────────────────────────────────────────────────────────────────────────────

base_dir = "C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_patient_embeddings/split/"
emb_key = "g64"
dfs = []
for fname in os.listdir(base_dir):
    if fname.startswith("sample_"):
        continue
    #
    if fname.endswith(".csv") and f"_emb_{emb_key}" in fname:
        patient_id = fname.replace(f"_emb_{emb_key}.csv", "")
        df = pd.read_csv(os.path.join(base_dir, fname), header=None).T
        if df.shape[0] != 1:
            raise ValueError(f"{fname} has {df.shape[0]} rows (expected 1)")
        #
        df.index = [patient_id]
        dfs.append(df)

embeddings_df = pd.concat(dfs, axis=0)
embeddings = np.array(embeddings_df)
embeddings_patient_id = (
    embeddings_df.index
    #.str.split("__")
    #.str[0]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Binvignat/samples_metadata.csv")
metadata = metadata.iloc[:,1:]
metadata.columns = ["patient_id", "Disease", "Sex", "Age", "Ethnicity", "Batch"]
metadata = (
    metadata
    .drop_duplicates(subset="patient_id", keep="first")
    .assign(Batch=lambda df: df["Batch"].where(
        ~df["patient_id"].eq("Control07_1_C_F"), 0
    ))
)
cat_cols = ["Disease", "Sex", "Ethnicity", "Batch"]
for col in cat_cols:
    metadata[col] = metadata[col].astype("category")

metadata = (
    metadata
    .set_index("patient_id")
    .loc[embeddings_patient_id]
    .reset_index()
)

# 1. Relevant metadata (Disease is expected — but we check anyway)
relevant_scores = []
if "Disease" in metadata.columns:
    try:
        I_disease = compute_information_retention(embeddings, metadata, 'Disease', is_categorical=True)
        relevant_scores.append(I_disease)
    except Exception as e:
        print(f"Error computing I_Disease: {e}")
        relevant_scores.append(0.0)
else:
    print("Column 'Disease' not found → setting I_Disease = 0")
    relevant_scores.append(0.0)

# 2. Technical / batch removal
technical_scores = []
if "Batch" in metadata.columns:
    try:
        I_batch = compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
        B_batch = 1.0 - I_batch
        technical_scores.append(B_batch)
    except Exception as e:
        print(f"Error computing B_batch: {e}")
        technical_scores.append(0.0)
else:
    print("Column 'Batch' not found → setting B_batch = 0")
    technical_scores.append(0.0)

# 3. Trajectory preservation — only if Age exists **and** is numeric
TS_age = 0.0
if "Age" in metadata.columns:
    age_series = metadata["Age"]
    # Check if we can reasonably treat it as numeric
    if pd.api.types.is_numeric_dtype(age_series) or age_series.dtype.name == 'category':
        # Try to convert — will raise if truly non-numeric
        try:
            age_numeric = pd.to_numeric(age_series, errors='coerce')
            if age_numeric.notna().sum() >= 5:  # arbitrary small threshold
                metadata_for_ts = metadata.copy()
                metadata_for_ts["Age"] = age_numeric
                TS_age = compute_trajectory_preservation(
                    embeddings,
                    metadata_for_ts,
                    covariate='Age'
                )
                print(f"Trajectory preservation (Age): {TS_age:.4f}")
            else:
                print("Too few valid numeric Age values → skipping (TS_age = 0)")
        except Exception as e:
            print(f"Could not convert Age to numeric: {e} → TS_age = 0")
    else:
        print("Age is not numeric → skipping trajectory preservation (TS_age = 0)")
else:
    print("Column 'Age' not found → TS_age = 0")

# 4. Replicate robustness — only compute if column exists and has replicates
RS_ethnicity = safe_replicate_robustness("Ethnicity")
RS_sex       = safe_replicate_robustness("Sex")
RS_tissue    = safe_replicate_robustness("Tissue")   # ← Tissue not in your metadata
# You probably want to average the replicate scores you actually computed
replicate_scores = [rs for rs in [RS_ethnicity, RS_sex] if rs > 0]  # or include 0s
RS_final = np.mean(replicate_scores) if replicate_scores else 0.0
print(f"Average replicate robustness: {RS_final:.4f}")
# Final aggregation
final_spare_score = spare_aggregate_score(
    information_retention_scores = relevant_scores,
    batch_removal_scores         = technical_scores,
    trajectory_preservation      = TS_age,
    replicate_robustness         = RS_final,          # ← use averaged or chosen one
)
print("\n" + "="*60)
print(f"Final SPARE score: {final_spare_score:.4f}")
print(f"  • Information retention: {np.mean(relevant_scores):.4f}")
print(f"  • Batch removal:         {np.mean(technical_scores):.4f}")
print(f"  • Trajectory (Age):      {TS_age:.4f}")
print(f"  • Replicate robustness:  {RS_final:.4f}")
print("="*60)










# ────────────────────────────────────────────────────────────────────────────────
# BINVIGNAT — FLOREN L32
# ────────────────────────────────────────────────────────────────────────────────

base_dir = "C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_patient_embeddings/split/"
emb_key = "l32"
dfs = []
for fname in os.listdir(base_dir):
    if fname.startswith("sample_"):
        continue
    #
    if fname.endswith(".csv") and f"_emb_{emb_key}" in fname:
        patient_id = fname.replace(f"_emb_{emb_key}.csv", "")
        df = pd.read_csv(os.path.join(base_dir, fname), header=None).T
        if df.shape[0] != 1:
            raise ValueError(f"{fname} has {df.shape[0]} rows (expected 1)")
        #
        df.index = [patient_id]
        dfs.append(df)

embeddings_df = pd.concat(dfs, axis=0)
embeddings = np.array(embeddings_df)
embeddings_patient_id = (
    embeddings_df.index
    #.str.split("__")
    #.str[0]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Binvignat/samples_metadata.csv")
metadata = metadata.iloc[:,1:]
metadata.columns = ["patient_id", "Disease", "Sex", "Age", "Ethnicity", "Batch"]
metadata = (
    metadata
    .drop_duplicates(subset="patient_id", keep="first")
    .assign(Batch=lambda df: df["Batch"].where(
        ~df["patient_id"].eq("Control07_1_C_F"), 0
    ))
)
cat_cols = ["Disease", "Sex", "Ethnicity", "Batch"]
for col in cat_cols:
    metadata[col] = metadata[col].astype("category")

metadata = (
    metadata
    .set_index("patient_id")
    .loc[embeddings_patient_id]
    .reset_index()
)

# 1. Relevant metadata (Disease is expected — but we check anyway)
relevant_scores = []
if "Disease" in metadata.columns:
    try:
        I_disease = compute_information_retention(embeddings, metadata, 'Disease', is_categorical=True)
        relevant_scores.append(I_disease)
    except Exception as e:
        print(f"Error computing I_Disease: {e}")
        relevant_scores.append(0.0)
else:
    print("Column 'Disease' not found → setting I_Disease = 0")
    relevant_scores.append(0.0)

# 2. Technical / batch removal
technical_scores = []
if "Batch" in metadata.columns:
    try:
        I_batch = compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
        B_batch = 1.0 - I_batch
        technical_scores.append(B_batch)
    except Exception as e:
        print(f"Error computing B_batch: {e}")
        technical_scores.append(0.0)
else:
    print("Column 'Batch' not found → setting B_batch = 0")
    technical_scores.append(0.0)

# 3. Trajectory preservation — only if Age exists **and** is numeric
TS_age = 0.0
if "Age" in metadata.columns:
    age_series = metadata["Age"]
    # Check if we can reasonably treat it as numeric
    if pd.api.types.is_numeric_dtype(age_series) or age_series.dtype.name == 'category':
        # Try to convert — will raise if truly non-numeric
        try:
            age_numeric = pd.to_numeric(age_series, errors='coerce')
            if age_numeric.notna().sum() >= 5:  # arbitrary small threshold
                metadata_for_ts = metadata.copy()
                metadata_for_ts["Age"] = age_numeric
                TS_age = compute_trajectory_preservation(
                    embeddings,
                    metadata_for_ts,
                    covariate='Age'
                )
                print(f"Trajectory preservation (Age): {TS_age:.4f}")
            else:
                print("Too few valid numeric Age values → skipping (TS_age = 0)")
        except Exception as e:
            print(f"Could not convert Age to numeric: {e} → TS_age = 0")
    else:
        print("Age is not numeric → skipping trajectory preservation (TS_age = 0)")
else:
    print("Column 'Age' not found → TS_age = 0")

# 4. Replicate robustness — only compute if column exists and has replicates
RS_ethnicity = safe_replicate_robustness("Ethnicity")
RS_sex       = safe_replicate_robustness("Sex")
RS_tissue    = safe_replicate_robustness("Tissue")   # ← Tissue not in your metadata
# You probably want to average the replicate scores you actually computed
replicate_scores = [rs for rs in [RS_ethnicity, RS_sex] if rs > 0]  # or include 0s
RS_final = np.mean(replicate_scores) if replicate_scores else 0.0
print(f"Average replicate robustness: {RS_final:.4f}")
# Final aggregation
final_spare_score = spare_aggregate_score(
    information_retention_scores = relevant_scores,
    batch_removal_scores         = technical_scores,
    trajectory_preservation      = TS_age,
    replicate_robustness         = RS_final,          # ← use averaged or chosen one
)
print("\n" + "="*60)
print(f"Final SPARE score: {final_spare_score:.4f}")
print(f"  • Information retention: {np.mean(relevant_scores):.4f}")
print(f"  • Batch removal:         {np.mean(technical_scores):.4f}")
print(f"  • Trajectory (Age):      {TS_age:.4f}")
print(f"  • Replicate robustness:  {RS_final:.4f}")
print("="*60)










# ────────────────────────────────────────────────────────────────────────────────
# BINVIGNAT — FLOREN G32
# ────────────────────────────────────────────────────────────────────────────────

base_dir = "C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_patient_embeddings/split/"
emb_key = "g32"
dfs = []
for fname in os.listdir(base_dir):
    if fname.startswith("sample_"):
        continue
    #
    if fname.endswith(".csv") and f"_emb_{emb_key}" in fname:
        patient_id = fname.replace(f"_emb_{emb_key}.csv", "")
        df = pd.read_csv(os.path.join(base_dir, fname), header=None).T
        if df.shape[0] != 1:
            raise ValueError(f"{fname} has {df.shape[0]} rows (expected 1)")
        #
        df.index = [patient_id]
        dfs.append(df)

embeddings_df = pd.concat(dfs, axis=0)
embeddings = np.array(embeddings_df)
embeddings_patient_id = (
    embeddings_df.index
    #.str.split("__")
    #.str[0]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Binvignat/samples_metadata.csv")
metadata = metadata.iloc[:,1:]
metadata.columns = ["patient_id", "Disease", "Sex", "Age", "Ethnicity", "Batch"]
metadata = (
    metadata
    .drop_duplicates(subset="patient_id", keep="first")
    .assign(Batch=lambda df: df["Batch"].where(
        ~df["patient_id"].eq("Control07_1_C_F"), 0
    ))
)
cat_cols = ["Disease", "Sex", "Ethnicity", "Batch"]
for col in cat_cols:
    metadata[col] = metadata[col].astype("category")

metadata = (
    metadata
    .set_index("patient_id")
    .loc[embeddings_patient_id]
    .reset_index()
)

# 1. Relevant metadata (Disease is expected — but we check anyway)
relevant_scores = []
if "Disease" in metadata.columns:
    try:
        I_disease = compute_information_retention(embeddings, metadata, 'Disease', is_categorical=True)
        relevant_scores.append(I_disease)
    except Exception as e:
        print(f"Error computing I_Disease: {e}")
        relevant_scores.append(0.0)
else:
    print("Column 'Disease' not found → setting I_Disease = 0")
    relevant_scores.append(0.0)

# 2. Technical / batch removal
technical_scores = []
if "Batch" in metadata.columns:
    try:
        I_batch = compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
        B_batch = 1.0 - I_batch
        technical_scores.append(B_batch)
    except Exception as e:
        print(f"Error computing B_batch: {e}")
        technical_scores.append(0.0)
else:
    print("Column 'Batch' not found → setting B_batch = 0")
    technical_scores.append(0.0)

# 3. Trajectory preservation — only if Age exists **and** is numeric
TS_age = 0.0
if "Age" in metadata.columns:
    age_series = metadata["Age"]
    # Check if we can reasonably treat it as numeric
    if pd.api.types.is_numeric_dtype(age_series) or age_series.dtype.name == 'category':
        # Try to convert — will raise if truly non-numeric
        try:
            age_numeric = pd.to_numeric(age_series, errors='coerce')
            if age_numeric.notna().sum() >= 5:  # arbitrary small threshold
                metadata_for_ts = metadata.copy()
                metadata_for_ts["Age"] = age_numeric
                TS_age = compute_trajectory_preservation(
                    embeddings,
                    metadata_for_ts,
                    covariate='Age'
                )
                print(f"Trajectory preservation (Age): {TS_age:.4f}")
            else:
                print("Too few valid numeric Age values → skipping (TS_age = 0)")
        except Exception as e:
            print(f"Could not convert Age to numeric: {e} → TS_age = 0")
    else:
        print("Age is not numeric → skipping trajectory preservation (TS_age = 0)")
else:
    print("Column 'Age' not found → TS_age = 0")

# 4. Replicate robustness — only compute if column exists and has replicates
RS_ethnicity = safe_replicate_robustness("Ethnicity")
RS_sex       = safe_replicate_robustness("Sex")
RS_tissue    = safe_replicate_robustness("Tissue")   # ← Tissue not in your metadata
# You probably want to average the replicate scores you actually computed
replicate_scores = [rs for rs in [RS_ethnicity, RS_sex] if rs > 0]  # or include 0s
RS_final = np.mean(replicate_scores) if replicate_scores else 0.0
print(f"Average replicate robustness: {RS_final:.4f}")
# Final aggregation
final_spare_score = spare_aggregate_score(
    information_retention_scores = relevant_scores,
    batch_removal_scores         = technical_scores,
    trajectory_preservation      = TS_age,
    replicate_robustness         = RS_final,          # ← use averaged or chosen one
)
print("\n" + "="*60)
print(f"Final SPARE score: {final_spare_score:.4f}")
print(f"  • Information retention: {np.mean(relevant_scores):.4f}")
print(f"  • Batch removal:         {np.mean(technical_scores):.4f}")
print(f"  • Trajectory (Age):      {TS_age:.4f}")
print(f"  • Replicate robustness:  {RS_final:.4f}")
print("="*60)










# ────────────────────────────────────────────────────────────────────────────────
# BINVIGNAT — FLOREN CLASS
# ────────────────────────────────────────────────────────────────────────────────

base_dir = "C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_patient_embeddings/split/"
emb_key = "class"
dfs = []
for fname in os.listdir(base_dir):
    if fname.startswith("sample_"):
        continue
    #
    if fname.endswith(".csv") and f"_emb_{emb_key}" in fname:
        patient_id = fname.replace(f"_emb_{emb_key}.csv", "")
        df = pd.read_csv(os.path.join(base_dir, fname), header=None).T
        if df.shape[0] != 1:
            raise ValueError(f"{fname} has {df.shape[0]} rows (expected 1)")
        #
        df.index = [patient_id]
        dfs.append(df)

embeddings_df = pd.concat(dfs, axis=0)
embeddings = np.array(embeddings_df)
embeddings_patient_id = (
    embeddings_df.index
    #.str.split("__")
    #.str[0]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Binvignat/samples_metadata.csv")
metadata = metadata.iloc[:,1:]
metadata.columns = ["patient_id", "Disease", "Sex", "Age", "Ethnicity", "Batch"]
metadata = (
    metadata
    .drop_duplicates(subset="patient_id", keep="first")
    .assign(Batch=lambda df: df["Batch"].where(
        ~df["patient_id"].eq("Control07_1_C_F"), 0
    ))
)
cat_cols = ["Disease", "Sex", "Ethnicity", "Batch"]
for col in cat_cols:
    metadata[col] = metadata[col].astype("category")

metadata = (
    metadata
    .set_index("patient_id")
    .loc[embeddings_patient_id]
    .reset_index()
)

# 1. Relevant metadata (Disease is expected — but we check anyway)
relevant_scores = []
if "Disease" in metadata.columns:
    try:
        I_disease = compute_information_retention(embeddings, metadata, 'Disease', is_categorical=True)
        relevant_scores.append(I_disease)
    except Exception as e:
        print(f"Error computing I_Disease: {e}")
        relevant_scores.append(0.0)
else:
    print("Column 'Disease' not found → setting I_Disease = 0")
    relevant_scores.append(0.0)

# 2. Technical / batch removal
technical_scores = []
if "Batch" in metadata.columns:
    try:
        I_batch = compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
        B_batch = 1.0 - I_batch
        technical_scores.append(B_batch)
    except Exception as e:
        print(f"Error computing B_batch: {e}")
        technical_scores.append(0.0)
else:
    print("Column 'Batch' not found → setting B_batch = 0")
    technical_scores.append(0.0)

# 3. Trajectory preservation — only if Age exists **and** is numeric
TS_age = 0.0
if "Age" in metadata.columns:
    age_series = metadata["Age"]
    # Check if we can reasonably treat it as numeric
    if pd.api.types.is_numeric_dtype(age_series) or age_series.dtype.name == 'category':
        # Try to convert — will raise if truly non-numeric
        try:
            age_numeric = pd.to_numeric(age_series, errors='coerce')
            if age_numeric.notna().sum() >= 5:  # arbitrary small threshold
                metadata_for_ts = metadata.copy()
                metadata_for_ts["Age"] = age_numeric
                TS_age = compute_trajectory_preservation(
                    embeddings,
                    metadata_for_ts,
                    covariate='Age'
                )
                print(f"Trajectory preservation (Age): {TS_age:.4f}")
            else:
                print("Too few valid numeric Age values → skipping (TS_age = 0)")
        except Exception as e:
            print(f"Could not convert Age to numeric: {e} → TS_age = 0")
    else:
        print("Age is not numeric → skipping trajectory preservation (TS_age = 0)")
else:
    print("Column 'Age' not found → TS_age = 0")

# 4. Replicate robustness — only compute if column exists and has replicates
RS_ethnicity = safe_replicate_robustness("Ethnicity")
RS_sex       = safe_replicate_robustness("Sex")
RS_tissue    = safe_replicate_robustness("Tissue")   # ← Tissue not in your metadata
# You probably want to average the replicate scores you actually computed
replicate_scores = [rs for rs in [RS_ethnicity, RS_sex] if rs > 0]  # or include 0s
RS_final = np.mean(replicate_scores) if replicate_scores else 0.0
print(f"Average replicate robustness: {RS_final:.4f}")
# Final aggregation
final_spare_score = spare_aggregate_score(
    information_retention_scores = relevant_scores,
    batch_removal_scores         = technical_scores,
    trajectory_preservation      = TS_age,
    replicate_robustness         = RS_final,          # ← use averaged or chosen one
)
print("\n" + "="*60)
print(f"Final SPARE score: {final_spare_score:.4f}")
print(f"  • Information retention: {np.mean(relevant_scores):.4f}")
print(f"  • Batch removal:         {np.mean(technical_scores):.4f}")
print(f"  • Trajectory (Age):      {TS_age:.4f}")
print(f"  • Replicate robustness:  {RS_final:.4f}")
print("="*60)










# ────────────────────────────────────────────────────────────────────────────────
# BINVIGNAT — FLOREN o(CLASS)
# ────────────────────────────────────────────────────────────────────────────────

base_dir = "C:/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_patient_embeddings/split/"
emb_key = "class"
dfs = []
for fname in os.listdir(base_dir):
    if fname.startswith("sample_"):
        continue
    #
    if fname.endswith(".csv") and f"_emb_{emb_key}" in fname:
        patient_id = fname.replace(f"_emb_{emb_key}.csv", "")
        df = pd.read_csv(os.path.join(base_dir, fname), header=None).T
        if df.shape[0] != 1:
            raise ValueError(f"{fname} has {df.shape[0]} rows (expected 1)")
        #
        df.index = [patient_id]
        dfs.append(df)

embeddings_df = pd.concat(dfs, axis=0)
embeddings = np.array(embeddings_df)
embeddings = np.exp(embeddings) / np.sum(np.exp(embeddings), axis=1, keepdims=True)  # [36, 2]
embeddings_patient_id = (
    embeddings_df.index
    #.str.split("__")
    #.str[0]
)
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Binvignat/samples_metadata.csv")
metadata = metadata.iloc[:,1:]
metadata.columns = ["patient_id", "Disease", "Sex", "Age", "Ethnicity", "Batch"]
metadata = (
    metadata
    .drop_duplicates(subset="patient_id", keep="first")
    .assign(Batch=lambda df: df["Batch"].where(
        ~df["patient_id"].eq("Control07_1_C_F"), 0
    ))
)
cat_cols = ["Disease", "Sex", "Ethnicity", "Batch"]
for col in cat_cols:
    metadata[col] = metadata[col].astype("category")

metadata = (
    metadata
    .set_index("patient_id")
    .loc[embeddings_patient_id]
    .reset_index()
)

# 1. Relevant metadata (Disease is expected — but we check anyway)
relevant_scores = []
if "Disease" in metadata.columns:
    try:
        I_disease = compute_information_retention(embeddings, metadata, 'Disease', is_categorical=True)
        relevant_scores.append(I_disease)
    except Exception as e:
        print(f"Error computing I_Disease: {e}")
        relevant_scores.append(0.0)
else:
    print("Column 'Disease' not found → setting I_Disease = 0")
    relevant_scores.append(0.0)

# 2. Technical / batch removal
technical_scores = []
if "Batch" in metadata.columns:
    try:
        I_batch = compute_information_retention(embeddings, metadata, 'Batch', is_categorical=True)
        B_batch = 1.0 - I_batch
        technical_scores.append(B_batch)
    except Exception as e:
        print(f"Error computing B_batch: {e}")
        technical_scores.append(0.0)
else:
    print("Column 'Batch' not found → setting B_batch = 0")
    technical_scores.append(0.0)

# 3. Trajectory preservation — only if Age exists **and** is numeric
TS_age = 0.0
if "Age" in metadata.columns:
    age_series = metadata["Age"]
    # Check if we can reasonably treat it as numeric
    if pd.api.types.is_numeric_dtype(age_series) or age_series.dtype.name == 'category':
        # Try to convert — will raise if truly non-numeric
        try:
            age_numeric = pd.to_numeric(age_series, errors='coerce')
            if age_numeric.notna().sum() >= 5:  # arbitrary small threshold
                metadata_for_ts = metadata.copy()
                metadata_for_ts["Age"] = age_numeric
                TS_age = compute_trajectory_preservation(
                    embeddings,
                    metadata_for_ts,
                    covariate='Age'
                )
                print(f"Trajectory preservation (Age): {TS_age:.4f}")
            else:
                print("Too few valid numeric Age values → skipping (TS_age = 0)")
        except Exception as e:
            print(f"Could not convert Age to numeric: {e} → TS_age = 0")
    else:
        print("Age is not numeric → skipping trajectory preservation (TS_age = 0)")
else:
    print("Column 'Age' not found → TS_age = 0")

# 4. Replicate robustness — only compute if column exists and has replicates
RS_ethnicity = safe_replicate_robustness("Ethnicity")
RS_sex       = safe_replicate_robustness("Sex")
RS_tissue    = safe_replicate_robustness("Tissue")   # ← Tissue not in your metadata
# You probably want to average the replicate scores you actually computed
replicate_scores = [rs for rs in [RS_ethnicity, RS_sex] if rs > 0]  # or include 0s
RS_final = np.mean(replicate_scores) if replicate_scores else 0.0
print(f"Average replicate robustness: {RS_final:.4f}")
# Final aggregation
final_spare_score = spare_aggregate_score(
    information_retention_scores = relevant_scores,
    batch_removal_scores         = technical_scores,
    trajectory_preservation      = TS_age,
    replicate_robustness         = RS_final,          # ← use averaged or chosen one
)
print("\n" + "="*60)
print(f"Final SPARE score: {final_spare_score:.4f}")
print(f"  • Information retention: {np.mean(relevant_scores):.4f}")
print(f"  • Batch removal:         {np.mean(technical_scores):.4f}")
print(f"  • Trajectory (Age):      {TS_age:.4f}")
print(f"  • Replicate robustness:  {RS_final:.4f}")
print("="*60)











# ────────────────────────────────────────────────────────────────────────────────
# BINVIGNAT — ML scores
# ────────────────────────────────────────────────────────────────────────────────
# === UMAP SAMPLES PLOT ===
vector_path = "/Users/Inigo/Desktop/FloREN3.0/binvignat_outputs/floren_patient_embeddings/"  # Path to saved patient embeddings
vector_files = [f for f in os.listdir(vector_path) if f.endswith(".csv")]
vector_files = [f for f in vector_files if "n_hid_128_nheads_8" in f]
all_vectors = []
labels = []
for file in vector_files:
    df = pd.read_csv(os.path.join(vector_path, file), header=None)  # No index column
    all_vectors.append(df.values.squeeze())  # Shape: [552]
    label = file.split("_epoch")[0].split("sample_")[1]
    labels.append(label)

all_vectors = np.array(all_vectors)  # Shape: [num_patients, 552]
num_groups = 2  # Control vs. RA
# Split all_vectors into components
ret_class = all_vectors[:, 576:]  # [36, 2]
softmax_ret_class = np.exp(ret_class) / np.sum(np.exp(ret_class), axis=1, keepdims=True)  # [36, 2]
metadata = pd.read_csv("C:/Users/Inigo/Desktop/FloREN/Binvignat/samples_metadata.csv")
metadata = metadata.iloc[:,1:]
metadata.columns = ["patient_id", "Disease", "Sex", "Age", "Ethnicity", "Batch"]
metadata = (
    metadata
    .drop_duplicates(subset="patient_id", keep="first")
    .assign(Batch=lambda df: df["Batch"].where(
        ~df["patient_id"].eq("Control07_1_C_F"), 0
    ))
)
cat_cols = ["Disease", "Sex", "Ethnicity", "Batch"]
for col in cat_cols:
    metadata[col] = metadata[col].astype("category")

metadata = (
    metadata
    .set_index("patient_id")
    .loc[labels]
    .reset_index()
)

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# === MODEL EVALUATION ===
print("\n=== MODEL EVALUATION METRICS ===")

# Convert labels (RA=1, CNT=0)
y_true = np.array([1 if l == metadata.Disease.unique()[1] else 0 for l in metadata.Disease])
y_pred_probs = softmax_ret_class[:, 1]  # Probability of RA
y_pred = (y_pred_probs > 0.5).astype(int)  # Classify with 0.5 threshold

# Compute metrics
f1_macro = f1_score(y_true, y_pred, average="macro")
f1_micro = f1_score(y_true, y_pred, average="micro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")

try:
    auc_score = roc_auc_score(y_true, y_pred_probs)
except ValueError:
    auc_score = np.nan  # Handles the case if only one class is predicted

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["CNT", "SLE"])

# Print metrics
print(f"F1 (macro):     {f1_macro:.4f}")
print(f"F1 (micro):     {f1_micro:.4f}")
print(f"F1 (weighted):  {f1_weighted:.4f}")
print(f"ROC-AUC:        {auc_score:.4f}")
print(f"Confusion matrix:\n{cm}")