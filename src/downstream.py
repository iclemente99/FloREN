
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#
#                         PLOT CELL TYPE IMPORTANCE
#
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


def cells_attention_ranking(
    floren_results_path: str,
    h5ad_path: str,
    output_pdf: str = None,
    celltype_col: str = "Celltype.Lev1.manuscript",
    sample_col: str = "Sample",
    agg: str = "mean",          # "mean", "median", or a quantile float e.g. 0.75
    top_n: int = None,          # None = show all cell types
    figsize_per_row: float = 0.35,
):
    """
    Rank cell types by mean attention saliency across all patients in a
    FloREN run, and save a lollipop plot.

    Parameters
    ----------
    floren_results_path : str
        Root output folder of a FloREN run. Must contain:
          - "interpretability/<patient>/cells_atts.csv" for each patient
          - "All_AUC_Cell_names.csv" (column "0" = full cell barcodes,
            formatted "<patient>__..." )
    h5ad_path : str
        Path to the AnnData (.h5ad) file with cell-type / sample metadata.
    output_pdf : str, optional
        Where to save the plot. Defaults to
        "<floren_results_path>/plots/cells_attention_ranking.pdf"
    celltype_col, sample_col : str
        Column names in adata.obs.
    agg : str or float
        How to summarize saliency per cell type per patient before
        averaging across patients: "mean", "median", or a quantile in (0,1).
    top_n : int, optional
        Only plot the top N cell types by saliency.

    Returns
    -------
    pd.Series
        Mean saliency per cell type across patients, sorted descending.
    """
    inter_dir = os.path.join(floren_results_path, "interpretability")
    cell_names_path = os.path.join(floren_results_path, "All_AUC_Cell_names.csv")
    if output_pdf is None:
        plots_dir = os.path.join(floren_results_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        output_pdf = os.path.join(plots_dir, "cells_attention_ranking.pdf")

    # ---- load required inputs ----
    patients = sorted(
        d for d in os.listdir(inter_dir) if os.path.isdir(os.path.join(inter_dir, d))
    )
    if not patients:
        raise FileNotFoundError(f"No patient subfolders found in {inter_dir}")

    cell_names = pd.read_csv(cell_names_path)

    adata = sc.read_h5ad(h5ad_path)
    meta = pd.DataFrame({
        "celltype": adata.obs[celltype_col].astype(str).values,
        "patient": adata.obs[sample_col].astype(str).values,
    }, index=adata.obs.index)
    meta["barcode"] = meta["patient"] + "__" + meta.index.astype(str)

    # ---- per-patient cell-type saliency ----
    all_patients_cell_df = []
    for patient in patients:
        att_path = os.path.join(inter_dir, patient, "cells_atts.csv")
        cell_scores = pd.read_csv(att_path, index_col=0).iloc[:, 0].values

        mask = cell_names["0"].str.startswith(patient + "__")
        patient_cell_barcodes = cell_names.loc[mask, "0"].values

        if len(patient_cell_barcodes) != len(cell_scores):
            print(f"WARNING: cell count mismatch for {patient} "
                  f"({len(patient_cell_barcodes)} names vs {len(cell_scores)} scores)")

        cell_df = pd.DataFrame({"barcode": patient_cell_barcodes, "saliency": cell_scores})
        merged = cell_df.merge(meta[["barcode", "celltype"]], on="barcode", how="left")

        if agg == "mean":
            grouped = merged.groupby("celltype")["saliency"].mean()
        elif agg == "median":
            grouped = merged.groupby("celltype")["saliency"].median()
        elif isinstance(agg, float):
            grouped = merged.groupby("celltype")["saliency"].quantile(agg)
        else:
            raise ValueError("agg must be 'mean', 'median', or a float quantile")

        celltype_df = grouped.sort_values(ascending=False).reset_index()
        celltype_df["patient"] = patient
        all_patients_cell_df.append(celltype_df)

    all_cells = pd.concat(all_patients_cell_df, ignore_index=True)
    mean_celltype_saliency = (
        all_cells.groupby("celltype")["saliency"].mean().sort_values(ascending=False)
    )

    df_plot = mean_celltype_saliency
    if top_n is not None:
        df_plot = df_plot.head(top_n)
    df_plot = df_plot.sort_values(ascending=True).reset_index()

    # ---- plot ----
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    plt.rcParams['pdf.fonttype'] = 42

    fig, ax = plt.subplots(figsize=(5, max(4, len(df_plot) * figsize_per_row)))
    ax.hlines(
        y=df_plot['celltype'],
        xmin=df_plot['saliency'].min() - abs(df_plot['saliency'].min()) * 0.1,
        xmax=df_plot['saliency'],
        color='#d1d1d1', linewidth=1, zorder=1,
    )
    ax.scatter(
        df_plot['saliency'], df_plot['celltype'],
        color='#2c3e50', s=60, edgecolors='white', linewidth=0.25, zorder=2,
    )
    ax.set_xlabel(f"{agg if isinstance(agg, str) else f'{agg:.0%} quantile'} saliency score",
                  fontsize=10, fontweight='bold')
    ax.set_title("Ranked Cell Type Saliency", loc='left', fontsize=12, pad=15)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='both', labelsize=9)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)

    plt.tight_layout()
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_pdf}")

    return mean_celltype_saliency



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#
#                         PLOT GENE SIGNATURES
#
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns


def gene_signatures(
    floren_results_path: str,
    h5ad_path: str,
    gene_names = None,                      # list/array of gene names, OR a path to a reference count-matrix CSV
    att_subdir: str = "att",
    cell_types: list = None,         # None = auto-derive from adata
    celltype_col: str = "Celltype.Lev1.manuscript",
    sample_col: str = "Sample",
    top_n_genes: int = 50,
    output_pdf: str = None,
    attention_matrix_path : str = "floren_annotated_signatures.csv",
    return_matrix: bool = False,
):
    """
    Build per-cell-type gene attention signatures from FloREN edge attention
    scores, and plot the top N genes (ranked by mean attention across cell
    types) as one line per cell type.

    Parameters
    ----------
    floren_results_path : str
        Root FloREN output folder. Must contain "<att_subdir>/<patient>_edge_att.csv"
        for each patient, with columns [idx, src, tgt, att] (src/tgt are global
        node indices: genes are [0, n_genes), cells are [n_genes, ...)).
        Also must contain "All_AUC_Cell_names.csv" (column "0" = full barcodes,
        formatted "<patient>__...").
    h5ad_path : str
        AnnData file with cell-type / sample metadata.
    gene_names : list-like or str
        Either the ordered list of gene names (length = n_genes, matching the
        node indexing used when the graph was built), or a path to the
        reference count-matrix CSV used at graph-construction time
        (genes as rows, "Unnamed: 0" as gene-name column).
    cell_types : list, optional
        Cell types to build signatures for. Defaults to all unique values
        in adata.obs[celltype_col].
    top_n_genes : int
        Number of top genes (by mean attention across cell types) to plot.
    output_pdf : str, optional
        Defaults to "<floren_results_path>/plots/gene_signatures_{top_n_genes}.pdf"

    Returns
    -------
    pd.DataFrame
        celltype x gene attention matrix (only returned if return_matrix=True;
        otherwise still available as the first return value).
    """
    att_dir = os.path.join(floren_results_path, att_subdir)
    cell_names_path = os.path.join(floren_results_path, "All_AUC_Cell_names.csv")
    if output_pdf is None:
        plots_dir = os.path.join(floren_results_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        output_pdf = os.path.join(
            plots_dir, f"gene_signatures_{top_n_genes}.pdf"
        )

    # ---- patients (derived from available edge-attention files) ----
    edge_files = sorted(glob.glob(os.path.join(att_dir, "*_edge_att.csv")))
    if not edge_files:
        raise FileNotFoundError(f"No '*_edge_att.csv' files found in {att_dir}")
    patients = [os.path.basename(f).replace("_edge_att.csv", "") for f in edge_files]

    # ---- cell metadata ----
    cell_names = pd.read_csv(cell_names_path)
    adata = sc.read_h5ad(h5ad_path)
    meta = pd.DataFrame({
        "celltype": adata.obs[celltype_col].astype(str).values,
        "patient": adata.obs[sample_col].astype(str).values,
    }, index=adata.obs.index)
    meta["barcode"] = meta["patient"] + "__" + meta.index.astype(str)

    if cell_types is None:
        cell_types = sorted(meta["celltype"].unique().tolist())
    
    # ---- gene names processing (moved below AnnData loading) ----
    if gene_names is None:
        # Fallback to AnnData features if nothing is passed
        gene_names = adata.var_names.values
    elif isinstance(gene_names, str):
        reference = pd.read_csv(gene_names)
        reference = reference.set_index("Unnamed: 0").T.reset_index().rename(columns={"index": "gene"})
        gene_names = reference[reference.columns[0]].values
        
    gene_names = np.asarray(gene_names)
    n_genes = len(gene_names)

    # ---- build per-celltype gene attention vectors ----
    celltype_vectors = {}
    for target_celltype in cell_types:
        all_patients_gene_df = []
        for patient in patients:
            df2 = pd.read_csv(
                os.path.join(att_dir, f"{patient}_edge_att.csv"), delimiter=","
            ).iloc[:, 1:]
            df2.columns = ["src", "tgt", "att"]
            df2[["src", "tgt"]] = df2[["src", "tgt"]].astype(int)

            patient_mask = cell_names["0"].str.startswith(patient + "__")
            patient_cell_names = cell_names.loc[patient_mask, "0"].reset_index(drop=True)
            cell_df = pd.DataFrame({
                "local_cell_index": np.arange(len(patient_cell_names)),
                "barcode": patient_cell_names,
                "patient": patient,
            })
            cell_df["node_index"] = cell_df["local_cell_index"] + n_genes
            cell_df = cell_df.merge(
                meta[["barcode", "celltype", "patient"]], on=["barcode", "patient"], how="left"
            )

            target_cells = cell_df[
                (cell_df["celltype"] == target_celltype) & (cell_df["patient"] == patient)
            ]["node_index"].values

            mask = (
                (df2["src"].isin(target_cells) & (df2["tgt"] < n_genes)) |
                (df2["tgt"].isin(target_cells) & (df2["src"] < n_genes))
            )
            df_ct_edges = df2[mask]

            gene_attention = np.zeros(n_genes, dtype=np.float64)
            src_is_gene = df_ct_edges["src"] < n_genes
            np.add.at(gene_attention, df_ct_edges.loc[src_is_gene, "src"].values,
                      df_ct_edges.loc[src_is_gene, "att"].values)
            tgt_is_gene = df_ct_edges["tgt"] < n_genes
            np.add.at(gene_attention, df_ct_edges.loc[tgt_is_gene, "tgt"].values,
                      df_ct_edges.loc[tgt_is_gene, "att"].values)

            if len(target_cells) > 0:
                gene_attention /= len(target_cells)

            all_patients_gene_df.append(pd.DataFrame({
                "gene": gene_names, "saliency": gene_attention,
                "patient": patient, "celltype": target_celltype,
            }))

        if not all_patients_gene_df:
            print(f"  -> No data for {target_celltype}")
            continue

        agg = (
            pd.concat(all_patients_gene_df, ignore_index=True)
            .groupby("gene")["saliency"].mean().reset_index()
        )
        vector_wide = agg.set_index("gene")["saliency"].to_frame().T
        vector_wide.index = [target_celltype]
        celltype_vectors[target_celltype] = vector_wide

    if not celltype_vectors:
        raise RuntimeError("No cell types had valid data.")

    celltype_attention_matrix = pd.concat(celltype_vectors.values(), axis=0)
    celltype_attention_matrix = celltype_attention_matrix.reindex(columns=gene_names, fill_value=0.0)
    celltype_attention_matrix = celltype_attention_matrix.reindex(cell_types)
    if return_matrix:
        celltype_attention_matrix.to_csv(attention_matrix_path)

    # ---- select & rank top genes, plot ----
    gene_means = celltype_attention_matrix.mean(axis=0)
    top_genes = gene_means.sort_values(ascending=False).head(top_n_genes).index
    top_matrix = celltype_attention_matrix[top_genes]
    top_matrix = top_matrix.loc[:, gene_means[top_genes].sort_values(ascending=False).index]

    plot_df = (
        top_matrix.reset_index().melt(id_vars="index", var_name="gene", value_name="value")
        .rename(columns={"index": "celltype"})
    )
    plot_df["gene"] = pd.Categorical(plot_df["gene"], categories=top_matrix.columns, ordered=True)

    sns.set_style("white")
    palette = sns.color_palette("tab10", len(top_matrix.index))
    fig, ax = plt.subplots(figsize=(16, 6))
    for i, ct in enumerate(top_matrix.index):
        ct_df = plot_df[plot_df["celltype"] == ct]
        ax.plot(ct_df["gene"], ct_df["value"], marker="o", markersize=4,
                linewidth=1.5, color=palette[i], label=ct)

    ax.set_xlabel(f"Top {top_n_genes} Genes (Ranked by Mean Attention)")
    ax.set_ylabel("Attention Score")
    ax.set_title("Celltype Attention Across Top Genes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.legend(title="Celltype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_pdf}")

    return celltype_attention_matrix



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#
#                         PLOT DIFFERENTIAL ABUNDANCE
#
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def differential_abundance_analysis(
    floren_results_path: str,
    h5ad_path: str,
    group_assignment,
    celltype_col: str = "Celltype.Lev1.manuscript",
    sample_col: str = "Sample",
    agg: str = "sum",              # "mean" or "sum" — how to summarize saliency per celltype per patient
    scale: str = "both",            # "linear", "log", or "both"
    output_pdf: str = None,
    da_matrix_path: str = "floren_DA_matrix.csv",
    return_matrix: bool = False,
):
    """
    Compare per-cell-type attention saliency between two patient groups
    (e.g. disease vs control) using Mann-Whitney U tests with FDR correction,
    and plot boxplot+stripplot per cell type.

    Parameters
    ----------
    floren_results_path : str
        Root FloREN output folder. Must contain "interpretability/<patient>/cells_atts.csv"
        for each patient, and "All_AUC_Cell_names.csv" (column "0" = barcodes,
        formatted "<patient>__...").
    h5ad_path : str
        AnnData file with cell-type / sample metadata.
    group_assignment : dict or tuple
        How to assign each patient to a group. Either:
          - dict {patient_id: group_label}, or
          - tuple (substring, [label_if_match, label_if_not]),
            e.g. ("MS", ["MS", "HC"]) reproduces the original substring logic.
    agg : str
        "mean" or "sum" — how to aggregate saliency per cell type per patient
        before comparing groups. NOTE: "sum" is sensitive to cell counts per
        patient; "mean" is the safer default for comparing across patients.
    scale : str
        "linear", "log", or "both" (both saved as separate pages in one PDF).
    output_pdf : str, optional
        Defaults to "<floren_results_path>/plots/differential_abundance_analysis.pdf"

    Returns
    -------
    results_df : pd.DataFrame
        Mann-Whitney U results per cell type, with FDR-adjusted p-values
        (columns: celltype, n_group0, n_group1, U_statistic, p_value,
        <group0>_median, <group1>_median, p_adj).
    all_cells : pd.DataFrame
        Long-format per-patient-per-celltype saliency table used for the test/plot.
    """
    inter_dir = os.path.join(floren_results_path, "interpretability")
    cell_names_path = os.path.join(floren_results_path, "All_AUC_Cell_names.csv")
    if output_pdf is None:
        plots_dir = os.path.join(floren_results_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        output_pdf = os.path.join(plots_dir, "differential_abundance_analysis.pdf")

    if scale not in ("linear", "log", "both"):
        raise ValueError("scale must be 'linear', 'log', or 'both'")

    # ---- resolve group assignment ----
    if isinstance(group_assignment, tuple):
        substring, (label_match, label_nomatch) = group_assignment
        def assign_group(patient_id):
            return label_match if substring in patient_id else label_nomatch
    elif isinstance(group_assignment, dict):
        def assign_group(patient_id):
            return group_assignment.get(patient_id, "unknown")
    else:
        raise ValueError("group_assignment must be a dict or (substring, [label, label]) tuple")

    # ---- load inputs ----
    patients = sorted(
        d for d in os.listdir(inter_dir) if os.path.isdir(os.path.join(inter_dir, d))
    )
    if not patients:
        raise FileNotFoundError(f"No patient subfolders found in {inter_dir}")

    cell_names = pd.read_csv(cell_names_path)

    adata = sc.read_h5ad(h5ad_path)
    meta = pd.DataFrame({
        "celltype": adata.obs[celltype_col].astype(str).values,
        "patient": adata.obs[sample_col].astype(str).values,
    }, index=adata.obs.index)
    meta["barcode"] = meta["patient"] + "__" + meta.index.astype(str)

    # ---- per-patient cell-type saliency ----
    all_patients_cell_df = []
    for patient in patients:
        att_path = os.path.join(inter_dir, patient, "cells_atts.csv")
        cell_scores = pd.read_csv(att_path, index_col=0).iloc[:, 0].values

        mask = cell_names["0"].str.startswith(patient + "__")
        patient_cell_barcodes = cell_names.loc[mask, "0"].values

        if len(patient_cell_barcodes) != len(cell_scores):
            print(f"WARNING: cell count mismatch for {patient} "
                  f"({len(patient_cell_barcodes)} names vs {len(cell_scores)} scores)")

        cell_df = pd.DataFrame({"barcode": patient_cell_barcodes, "saliency": cell_scores})
        merged = cell_df.merge(meta[["barcode", "celltype"]], on="barcode", how="left")

        grouped = (
            merged.groupby("celltype")["saliency"].sum()
            if agg == "sum" else
            merged.groupby("celltype")["saliency"].mean()
        )
        celltype_df = grouped.sort_values(ascending=False).reset_index()
        celltype_df["patient"] = patient
        all_patients_cell_df.append(celltype_df)

    all_cells = pd.concat(all_patients_cell_df, ignore_index=True)
    all_cells["group"] = all_cells["patient"].map(assign_group)

    mean_celltype_saliency = (
        all_cells.groupby("celltype")["saliency"].mean().sort_values(ascending=False)
    )
    ordered_cells = mean_celltype_saliency.index.tolist()

    if return_matrix:
        mean_celltype_saliency.to_csv(da_matrix_path)

    group_labels = sorted(all_cells["group"].unique())
    if len(group_labels) != 2:
        raise ValueError(f"Expected exactly 2 groups, got {group_labels}")
    g0, g1 = group_labels

    # ---- Mann-Whitney U test per cell type ----
    results = []
    for celltype, df_ct in all_cells.groupby("celltype"):
        v0 = df_ct.loc[df_ct["group"] == g0, "saliency"]
        v1 = df_ct.loc[df_ct["group"] == g1, "saliency"]
        if len(v0) > 0 and len(v1) > 0:
            stat, pval = mannwhitneyu(v0, v1, alternative="two-sided")
            results.append({
                "celltype": celltype,
                f"n_{g0}": len(v0), f"n_{g1}": len(v1),
                "U_statistic": stat, "p_value": pval,
                f"{g0}_median": v0.median(), f"{g1}_median": v1.median(),
            })
    results_df = pd.DataFrame(results)
    results_df["p_adj"] = multipletests(results_df["p_value"], method="fdr_bh")[1]

    # ---- plot ----
    plot_df = all_cells.copy()
    plot_df["celltype"] = pd.Categorical(plot_df["celltype"], categories=ordered_cells, ordered=True)

    palette = {g0: "#3A7FB0", g1: "#C23B3B"}
    sns.set_style("white")

    scales_to_plot = ["linear", "log"] if scale == "both" else [scale]

    with PdfPages(output_pdf) as pdf:
        for sc_type in scales_to_plot:
            fig, ax = plt.subplots(figsize=(22, 8))
            sns.boxplot(
                data=plot_df, x="celltype", y="saliency", hue="group",
                palette=palette, dodge=True, width=0.6, showcaps=True, showfliers=False,
                boxprops=dict(facecolor='none', linewidth=1.5),
                whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                medianprops=dict(linewidth=2), ax=ax,
            )
            sns.stripplot(
                data=plot_df, x="celltype", y="saliency", hue="group",
                palette=palette, dodge=True, size=4, alpha=0.7, linewidth=0, ax=ax,
            )
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:2], labels[:2], frameon=False)

            if sc_type == "log":
                ax.set_yscale("log")
                ax.set_title(f"Cell Type Saliency Distribution ({g0} vs {g1}, Log Scale)", fontsize=16)
                ax.set_ylabel("Log Saliency Score")
            else:
                ax.set_title(f"Cell Type Saliency Distribution ({g0} vs {g1})", fontsize=16)
                ax.set_ylabel("Saliency Score")

            ax.set_xlabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved: {output_pdf}")
    return results_df, all_cells



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#
#                         PLOT DIFFERENTIAL GENE EXPRESSION
#
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests


def run_differential_gene_expression(
    gene_embeddings_dir: str,
    gene_names,                      # list-like or path to reference count-matrix CSV
    group_assignment,                # dict {patient_id: group} or (substring, [label_if_match, label_if_not])
    map_to_symbols: bool = True,
    p_threshold: float = 0.05,
    top_n_labels: int = 10,
    output_dir: str = None,
    make_plots: bool = True,
):
    """
    Compare per-gene embedding shift between two patient groups, using the
    Euclidean distance between group-mean gene embeddings as an effect size,
    and an empirical (rank-based) p-value for significance.

    NOTE ON STATISTICS: the p-value here is NOT computed against an
    independent null distribution. It is an empirical/rank-based p-value:
    p(gene) = (# genes with distance >= this gene's distance + 1) / (N + 1).
    This assumes most genes are not differentially represented and uses the
    bulk of the distance distribution as an implicit null. It is a useful
    ranking/triage tool, not a substitute for a permutation test against
    shuffled group labels. Treat results as exploratory.

    Parameters
    ----------
    gene_embeddings_dir : str
        Folder containing "<patient_id>_gene_embs.csv" per patient (headerless,
        rows = genes in the same order as `gene_names`, columns = embedding dims).
    gene_names : list-like or str
        Ordered gene identifiers matching the row order of the embedding CSVs,
        OR a path to the reference count-matrix CSV used at graph-construction
        time (genes as rows, "Unnamed: 0" as gene-name column).
    group_assignment : dict or tuple
        Either dict {patient_id: group_label}, or
        (substring, [label_if_match, label_if_not]) e.g. ("RA", ["RA", "HD"]).
    map_to_symbols : bool
        If True, attempts to map gene IDs to symbols via mygene for readable
        labels. Falls back to raw IDs if mygene is unavailable or a gene
        isn't found. Purely cosmetic — does not affect distances/p-values.
    p_threshold : float
        Significance threshold used for plot coloring/labeling.
    top_n_labels : int
        Number of top-distance genes to annotate on each plot.
    output_dir : str, optional
        Where to save plots. Defaults to "<gene_embeddings_dir>/../plots".
    make_plots : bool
        If False, skips plot generation and only returns the results table.

    Returns
    -------
    gene_results : pd.DataFrame
        Index = gene (symbol or ID), columns: distance, pval, padj.
        Sorted by distance, descending.
    group_means : dict[str, pd.DataFrame]
        {group_label: mean_embedding_df} for the two groups, aligned to the
        same gene index used in gene_results.
    """
    # ---- gene names ----
    if isinstance(gene_names, str):
        reference = pd.read_csv(gene_names)
        reference = reference.set_index("Unnamed: 0").T.reset_index().rename(columns={"index": "gene"})
        gene_names = reference[reference.columns[0]].values
    gene_names = np.asarray(gene_names)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(gene_embeddings_dir.rstrip("/\\")), "plots")
    os.makedirs(output_dir, exist_ok=True)

    # ---- optional symbol mapping (cosmetic only) ----
    gene_labels = pd.Series(gene_names, index=gene_names)  # identity fallback
    if map_to_symbols:
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            gene_map = mg.querymany(
                gene_names.tolist(), scopes="ensembl.gene", fields="symbol",
                species="human", as_dataframe=True,
            )
            gene_labels = gene_map["symbol"].fillna(gene_map.index.to_series())
            gene_labels = gene_labels.reindex(gene_names).fillna(pd.Series(gene_names, index=gene_names))
        except Exception as e:
            print(f"Warning: gene symbol mapping failed ({e}); using raw gene IDs.")

    # ---- resolve group assignment ----
    if isinstance(group_assignment, tuple):
        substring, (label_match, label_nomatch) = group_assignment
        def assign_group(patient_id):
            return label_match if substring in patient_id else label_nomatch
    elif isinstance(group_assignment, dict):
        def assign_group(patient_id):
            return group_assignment.get(patient_id, "unknown")
    else:
        raise ValueError("group_assignment must be a dict or (substring, [label, label]) tuple")

    # ---- load per-patient gene embeddings ----
    emb_files = sorted(glob.glob(os.path.join(gene_embeddings_dir, "*_gene_embs.csv")))
    if not emb_files:
        raise FileNotFoundError(f"No '*_gene_embs.csv' files found in {gene_embeddings_dir}")

    group_dfs = {}
    for f in emb_files:
        patient_id = os.path.basename(f).replace("_gene_embs.csv", "")
        group = assign_group(patient_id)
        df = pd.read_csv(f, index_col=0)
        if len(df) != len(gene_names):
            raise ValueError(
                f"{f}: {len(df)} rows but gene_names has {len(gene_names)} entries"
            )
        df.index = gene_labels.values
        group_dfs.setdefault(group, []).append(df)

    group_labels = [g for g in group_dfs if g != "unknown"]
    if len(group_labels) != 2:
        raise ValueError(f"Expected exactly 2 groups, got {group_labels}")
    g0, g1 = group_labels
    print(f"{g0} patients: {len(group_dfs[g0])} | {g1} patients: {len(group_dfs[g1])}")

    mean_df_0 = pd.concat(group_dfs[g0]).groupby(level=0).mean()
    mean_df_1 = pd.concat(group_dfs[g1]).groupby(level=0).mean()

    # ---- align gene index between groups (fixes silent misalignment risk) ----
    common_genes = mean_df_0.index.intersection(mean_df_1.index)
    dropped = set(mean_df_0.index).symmetric_difference(set(mean_df_1.index))
    if dropped:
        print(f"Warning: {len(dropped)} genes present in only one group's embeddings; dropping from comparison.")
    mean_df_0 = mean_df_0.loc[common_genes]
    mean_df_1 = mean_df_1.loc[common_genes]

    # ---- effect size: Euclidean distance between group-mean embeddings ----
    distances = np.linalg.norm(mean_df_0.values - mean_df_1.values, axis=1)
    gene_distances = pd.Series(distances, index=common_genes, name="euclidean_distance")

    # ---- empirical (rank-based) p-values + FDR ----
    dist_vals = gene_distances.values
    N = len(dist_vals)
    pvals = np.array([(np.sum(dist_vals >= d) + 1) / (N + 1) for d in dist_vals])
    padj = multipletests(pvals, method="fdr_bh")[1]

    gene_results = pd.DataFrame({
        "distance": gene_distances,
        "pval": pvals,
        "padj": padj,
    }, index=gene_distances.index).sort_values("distance", ascending=False)

    if make_plots:
        _plot_distance_distribution(gene_results, g0, g1, top_n_labels, output_dir)
        _plot_distance_volcano(gene_results, g0, g1, p_threshold, top_n_labels, output_dir)

    return gene_results, {g0: mean_df_0, g1: mean_df_1}


def _plot_distance_distribution(gene_results, g0, g1, top_n_labels, output_dir):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.sans-serif'] = 'Arial'
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(
        gene_results["distance"], bins=60, kde=True,
        color="#2c3e50", edgecolor='white', line_kws={'linewidth': 2}, ax=ax,
    )
    outliers = gene_results["distance"].sort_values(ascending=False).head(top_n_labels)
    for i, (gene, dist) in enumerate(outliers.items()):
        ax.axvline(dist, color='red', alpha=0.2, linestyle='--', linewidth=0.8)
        ax.text(
            dist, ax.get_ylim()[1] * (0.9 - i * 0.07), f" {gene}",
            color='red', fontsize=8, fontweight='bold', va='center',
        )

    ax.set_xlabel(f"Euclidean Distance ({g0} vs {g1} Mean Embeddings)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Number of Genes", fontsize=10, fontweight='bold')
    ax.set_title("Shift in Gene Representation Space", loc='left', fontsize=12, pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()

    out = os.path.join(output_dir, "gene_embedding_distance_annotated.pdf")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def _plot_distance_volcano(gene_results, g0, g1, p_threshold, top_n_labels, output_dir):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.sans-serif'] = 'Arial'
    sns.set_style("white")

    gr = gene_results.copy()
    gr["neg_log10_pval"] = -np.log10(gr["pval"])
    sig_mask = gr["pval"] < p_threshold

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        gr.loc[~sig_mask, "distance"], gr.loc[~sig_mask, "neg_log10_pval"],
        color="#e0e0e0", alpha=0.5, s=15, zorder=2, edgecolors='none',
    )
    ax.scatter(
        gr.loc[sig_mask, "distance"], gr.loc[sig_mask, "neg_log10_pval"],
        color="#C23B3B", alpha=0.9, s=40, zorder=3, edgecolors='white', linewidth=0.3,
    )

    top_hits = gr[sig_mask].sort_values("distance", ascending=False).head(top_n_labels)
    y_range = gr["neg_log10_pval"].max() - gr["neg_log10_pval"].min()
    offset_step = y_range * 0.03 if y_range > 0 else 0.1
    for i, (gene, row) in enumerate(top_hits.iterrows()):
        stagger = offset_step if i % 2 == 0 else -offset_step
        ax.annotate(
            gene, xy=(row["distance"], row["neg_log10_pval"]),
            xytext=(5, stagger), textcoords="offset points",
            fontsize=6, fontweight='bold', color='black', va='center',
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7),
        )

    ax.axhline(-np.log10(p_threshold), color='black', linestyle='--', alpha=0.3, lw=1)
    ax.text(
        ax.get_xlim()[0], -np.log10(p_threshold), f' P={p_threshold}',
        va='bottom', ha='left', fontsize=8, color='gray',
    )

    ax.set_xlabel(f"Euclidean Distance ({g0} vs {g1})", fontsize=11, fontweight='bold')
    ax.set_ylabel(r"$-log_{10}(p\text{-value})$", fontsize=11, fontweight='bold')
    ax.set_title("Gene Embedding Shift Significance", loc='left', fontsize=14, pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out = os.path.join(output_dir, "embedding_volcano_manual_clean.pdf")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#
#                             PLOT IMMUNE NETWORK
#
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colorbar import ColorbarBase


def _normalize_pos(pos_dict, xmin=0.05, xmax=0.95, ymin=0.05, ymax=0.95):
    """Rescale a layout dict to a fixed [0,1]-ish box so nothing clips."""
    xs = np.array([v[0] for v in pos_dict.values()])
    ys = np.array([v[1] for v in pos_dict.values()])
    xr = xs.max() - xs.min() or 1.0
    yr = ys.max() - ys.min() or 1.0
    nodes = list(pos_dict.keys())
    xs_n = xmin + (xs - xs.min()) / xr * (xmax - xmin)
    ys_n = ymin + (ys - ys.min()) / yr * (ymax - ymin)
    return {n: (xs_n[i], ys_n[i]) for i, n in enumerate(nodes)}


def _build_two_ring_layout(G, cell_nodes, gene_nodes):
    """Deterministic two-ring layout: cells outer ring, genes inner ring,
    sorted by degree so high-degree nodes face each other."""
    cell_sorted = sorted(cell_nodes, key=lambda n: -G.degree(n))
    gene_sorted = sorted(gene_nodes, key=lambda n: -G.degree(n))
    pos = {}
    nc = len(cell_sorted)
    for i, n in enumerate(cell_sorted):
        a = 2 * np.pi * i / nc - np.pi / 2
        pos[n] = (np.cos(a) * 2.8, np.sin(a) * 2.8)
    ng = len(gene_sorted)
    for i, n in enumerate(gene_sorted):
        a = 2 * np.pi * i / ng - np.pi / 2 + np.pi / ng
        pos[n] = (np.cos(a) * 1.3, np.sin(a) * 1.3)
    return pos


def immune_network_plot(
    G,
    cell_nodes,
    gene_nodes=None,
    layout: str = "kamada",       # "kamada", "two_ring", "spring", or "spectral"
    figsize=(16, 16),
    label_fontsize_cell: float = 7.5,
    label_fontsize_gene: float = 6.0,
    save_path: str = "immune_network_plot.pdf",
):
    """
    Plot a cell-type/gene attention network with a shared blue->red colormap
    for node/edge scores, plus node labels (black text, white background).

    Node color and size both encode the node's saliency/attention score
    (G.nodes[n]['size'], expected pre-normalized to [0, 1]).
    Edge color encodes the mean score of its two endpoints; edge width
    encodes G.edges[u,v]['weight']. Edges with type == "gene-gene" are drawn
    on top of cell-cell edges, which are drawn on top of cell-gene edges.

    Parameters
    ----------
    G : networkx.Graph
        Must have node attribute 'size' (float, ideally in [0,1]) and edge
        attributes 'weight' (float) and optionally 'type' ("gene-gene" or
        unset/other for cell-cell and cell-gene edges).
    cell_nodes : iterable
        Node keys that represent cell types.
    gene_nodes : iterable, optional
        Node keys that represent genes. Defaults to all nodes in G not in
        cell_nodes.
    layout : str
        "kamada" (default, usually most readable), "two_ring", "spring",
        or "spectral".
    save_path : str
        Output PDF path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cell_nodes = set(cell_nodes)
    if gene_nodes is None:
        gene_nodes = set(G.nodes()) - cell_nodes
    else:
        gene_nodes = set(gene_nodes)

    CMAP = cm.get_cmap("RdBu_r")  # blue = low, red = high

    # ---- layout ----
    if layout == "two_ring":
        raw_pos = _build_two_ring_layout(G, cell_nodes, gene_nodes)
    elif layout == "spring":
        seed = _build_two_ring_layout(G, cell_nodes, gene_nodes)
        raw_pos = nx.spring_layout(G, pos=seed, fixed=None, k=2.0, iterations=150,
                                    seed=42, weight=None)
    elif layout == "kamada":
        raw_pos = nx.kamada_kawai_layout(G, weight=None)
    else:
        raw_pos = nx.spectral_layout(G)
    pos = _normalize_pos(raw_pos)

    def node_score(n):
        return float(G.nodes[n].get("size", 0.0))

    # ---- node visual attributes ----
    node_list = list(G.nodes())
    sizes, colors, edgecs = [], [], []
    for n in node_list:
        s = node_score(n)
        rgba = CMAP(s)
        dark = tuple(max(0, c * 0.65) for c in rgba[:3]) + (1.0,)
        sizes.append(800 + 1400 * s if n in cell_nodes else 200 + 500 * s)
        colors.append(rgba)
        edgecs.append(dark)

    # ---- edge visual attributes ----
    e_col, e_wid, e_alp = [], [], []
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.3))
        t = d.get("type", "")
        rgba = CMAP((node_score(u) + node_score(v)) / 2.0)
        if t == "gene-gene":
            e_col.append(rgba); e_wid.append(0.8 + w * 2); e_alp.append(0.70)
        elif u in cell_nodes and v in cell_nodes:
            e_col.append(rgba); e_wid.append(1.0 + w * 2.5); e_alp.append(0.60)
        else:
            e_col.append(rgba); e_wid.append(0.4 + w * 1.2); e_alp.append(0.40)

    # ---- figure layout with room for colorbars ----
    fig = plt.figure(figsize=figsize, facecolor="#FFFFFF")
    ax = fig.add_axes([0.02, 0.05, 0.78, 0.88])
    ax_cb1 = fig.add_axes([0.83, 0.55, 0.025, 0.35])
    ax_cb2 = fig.add_axes([0.83, 0.12, 0.025, 0.35])
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    # ---- edges, drawn in layers so gene-gene sits on top ----
    for etype, z in [("ct_gene", 1), ("cell_cell", 2), ("gene_gene", 3)]:
        el = []
        for (u, v, d), c, w, a in zip(G.edges(data=True), e_col, e_wid, e_alp):
            t = d.get("type", "")
            is_cc = u in cell_nodes and v in cell_nodes
            if etype == "gene_gene" and t == "gene-gene": el.append((u, v, c, w, a))
            elif etype == "cell_cell" and is_cc and t != "gene-gene": el.append((u, v, c, w, a))
            elif etype == "ct_gene" and not is_cc and t != "gene-gene": el.append((u, v, c, w, a))
        for u, v, c, w, a in el:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color=c, lw=w, alpha=a, zorder=z)

    # ---- nodes ----
    ax.scatter(
        [pos[n][0] for n in node_list], [pos[n][1] for n in node_list],
        s=sizes, c=colors, edgecolors=edgecs, linewidths=1.1, zorder=5,
    )

    # ---- node labels: black text, white background ----
    cx = np.mean([pos[n][0] for n in node_list])
    cy = np.mean([pos[n][1] for n in node_list])
    for n in node_list:
        x, y = pos[n]
        is_cell = n in cell_nodes
        dx, dy = x - cx, y - cy
        dist = max(np.hypot(dx, dy), 1e-3)
        off = 0.06 if is_cell else 0.045
        lx = np.clip(x + dx / dist * off, 0.02, 0.98)
        ly = np.clip(y + dy / dist * off, 0.02, 0.98)
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        ax.annotate(
            str(n), xy=(x, y), xytext=(lx, ly),
            fontsize=label_fontsize_cell if is_cell else label_fontsize_gene,
            fontweight="semibold" if is_cell else "normal",
            color="black",
            ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
            zorder=6,
        )

    # ---- colorbars ----
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cb1 = ColorbarBase(ax_cb1, cmap=CMAP, norm=norm, orientation="vertical")
    cb1.set_label("Cell type saliency score", fontsize=9, color="#111111")
    cb1.ax.yaxis.set_tick_params(color="#111111", labelcolor="#111111", labelsize=8)
    ax_cb1.set_title("Cell type", fontsize=8, color="#111111", pad=6)

    cb2 = ColorbarBase(ax_cb2, cmap=CMAP, norm=norm, orientation="vertical")
    cb2.set_label("Gene attention score", fontsize=9, color="#111111")
    cb2.ax.yaxis.set_tick_params(color="#111111", labelcolor="#111111", labelsize=8)
    ax_cb2.set_title("Gene", fontsize=8, color="#111111", pad=6)

    # ---- legend ----
    legend_elements = [
        Line2D([0], [0], color="#888888", lw=0.6, alpha=0.6, label="Cell-gene attention (thin=low w)"),
        Line2D([0], [0], color="#888888", lw=2.5, alpha=0.6, label="Cell-gene attention (thick=high w)"),
        Line2D([0], [0], color="#444444", lw=1.5, alpha=0.7, label="Cell-cell co-attention"),
        Line2D([0], [0], color="#444444", lw=1.5, alpha=0.7, label="Gene-gene interaction"),
        mpatches.Patch(fc="none", ec="none", label=""),
        mpatches.Patch(fc="none", ec="none", label="Color: blue=low - red=high score"),
        mpatches.Patch(fc="none", ec="none", label="Size: small=low - large=high score"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower left", framealpha=0.9,
        facecolor="#F8F8F8", edgecolor="#CCCCCC", fontsize=8,
        labelcolor="#111111", title="Edge width = attention weight", title_fontsize=8,
    )

    ax.set_title(
        f"GNN Attention Network - Single Cell Transcriptomics\n"
        f"({G.number_of_nodes()} nodes - {G.number_of_edges()} edges - "
        f"{len(cell_nodes)} cell types - {len(gene_nodes)} genes)",
        color="#111111", fontsize=13, pad=14,
    )

    fig.patch.set_facecolor("#FFFFFF")
    try:
        os.remove(save_path)
    except OSError:
        pass
    fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight",
                facecolor="#FFFFFF", edgecolor="none")
    print(f"Saved: {save_path}")
    return fig



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#
#                            CELL NICHES ANALYSIS
#
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def _row_zscore(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1
    return (X - mean) / std


def _hierarchical_order(similarity_matrix, method="average"):
    """Leaf order from a [-1,1]-bounded similarity matrix, via 1-sim distance."""
    dist = 1 - similarity_matrix
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    link = linkage(squareform(dist, checks=False), method=method)
    return leaves_list(link)


def cell_niches_analysis(
    cell_embeddings_dir: str,
    h5ad_path: str,
    group_assignment,
    patient_col: str = "ind_cov",
    celltype_col: str = "cg_cov",
    metrics=("cosine", "euclidean", "pearson"),
    cluster: bool = True,
    output_pdf: str = None,
    figsize=(15, 5),
    correlation_cmap: str = "RdBu_r",
    distance_cmap: str = "rocket_r",
):
    """
    Compare cell-type "niches" (mean embedding profile per cell type) between
    two patient groups: within-group A, within-group B, and cross-group A-vs-B
    similarity/distance heatmaps, on a shared cell-type ordering and color scale.

    Parameters
    ----------
    cell_embeddings_dir : str
        Folder with "<patient_id>_cell_embs.csv" per patient (headerless,
        rows = cells in the same order as that patient's cells in adata).
    h5ad_path : str
        AnnData with a patient-ID column and a cell-type column in .obs.
    group_assignment : dict or tuple
        Either dict {patient_id: group_label}, or
        (match_condition, [label_if_match, label_if_not]) where
        match_condition is a substring (checked via `in`) or a list/tuple
        of prefixes (checked via `str.startswith`).
        e.g. (["HC-", "IGTB"], ["HC", "SLE"])  or  ("RA", ["RA", "HD"])
    patient_col, celltype_col : str
        Column names in adata.obs.
    metrics : tuple of str
        Any of "cosine", "pearson", "euclidean". One PDF page per metric.
    cluster : bool
        If True, order cell types by hierarchical clustering of their
        averaged (group A + group B) profile, applied consistently to all
        three panels so rows/columns are comparable across panels.
    output_pdf : str, optional
        Defaults to "<cell_embeddings_dir>/../plots/cell_niches_analysis.pdf"

    Returns
    -------
    dict with keys:
        "group_means": {group_label: mean_embedding_df} (aligned, ordered)
        "celltypes_ordered": list of cell types in plotted order
    """
    if output_pdf is None:
        output_pdf = os.path.join(
            os.path.dirname(cell_embeddings_dir.rstrip("/\\")), "plots", "cell_niches_analysis.pdf"
        )
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    # ---- resolve group assignment ----
    if isinstance(group_assignment, tuple):
        match_condition, (label_match, label_nomatch) = group_assignment
        if isinstance(match_condition, (list, tuple)):
            prefixes = tuple(match_condition)
            def assign_group(pid): return label_match if str(pid).startswith(prefixes) else label_nomatch
        else:
            def assign_group(pid): return label_match if match_condition in str(pid) else label_nomatch
    elif isinstance(group_assignment, dict):
        def assign_group(pid): return group_assignment.get(pid, "unknown")
    else:
        raise ValueError("group_assignment must be a dict or (match_condition, [label, label]) tuple")

    # ---- metadata ----
    adata = sc.read_h5ad(h5ad_path)
    meta_full = adata.obs[[patient_col, celltype_col]].copy()
    meta_full.columns = ["patient", "celltype"]

    # ---- load & aggregate per-patient cell embeddings by cell type ----
    emb_files = sorted(glob.glob(os.path.join(cell_embeddings_dir, "*_cell_embs.csv")))
    if not emb_files:
        raise FileNotFoundError(f"No '*_cell_embs.csv' files found in {cell_embeddings_dir}")

    patient_agg = {}
    for fpath in emb_files:
        patient_id = os.path.basename(fpath).replace("_cell_embs.csv", "")
        cells = pd.read_csv(fpath, index_col=0)
        meta_p = meta_full[meta_full["patient"].astype(str).str.contains(patient_id)].reset_index(drop=True)
        if len(cells) != len(meta_p):
            print(f"Warning: row mismatch for {patient_id} ({len(cells)} vs {len(meta_p)}); skipping.")
            continue
        patient_agg[patient_id] = cells.groupby(meta_p["celltype"].values).mean().fillna(0)

    print(f"Loaded {len(patient_agg)} patients.")

    groups = {}
    for pid, df in patient_agg.items():
        groups.setdefault(assign_group(pid), []).append(df)

    group_labels = [g for g in groups if g != "unknown"]
    if len(group_labels) != 2:
        raise ValueError(f"Expected exactly 2 groups, got {group_labels}")
    g0, g1 = group_labels
    print(f"{g0}: {len(groups[g0])} patients | {g1}: {len(groups[g1])} patients")

    def group_mean(dfs):
        combined = pd.concat(dfs)
        return combined.groupby(combined.index).mean()

    mean0, mean1 = group_mean(groups[g0]), group_mean(groups[g1])

    celltypes = sorted(set(mean0.index) & set(mean1.index))
    dropped = set(mean0.index).symmetric_difference(set(mean1.index))
    if dropped:
        print(f"Warning: {len(dropped)} cell types not shared between groups, excluded: {sorted(dropped)}")
    mean0, mean1 = mean0.loc[celltypes], mean1.loc[celltypes]

    # ---- one shared cell-type ordering for all three panels ----
    if cluster:
        avg_profile = (mean0.values + mean1.values) / 2
        order = _hierarchical_order(cosine_similarity(avg_profile))
        celltypes_ordered = [celltypes[i] for i in order]
    else:
        celltypes_ordered = celltypes
    mean0, mean1 = mean0.loc[celltypes_ordered], mean1.loc[celltypes_ordered]

    # ---- journal-style aesthetics ----
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "font.size": 9,
    })

    metric_cfg = {
        "cosine":    dict(cmap=correlation_cmap, vmin=-1, vmax=1, label="Cosine similarity"),
        "pearson":   dict(cmap=correlation_cmap, vmin=-1, vmax=1, label="Pearson correlation"),
        "euclidean": dict(cmap=distance_cmap,    vmin=None, vmax=None, label="Euclidean distance"),
    }

    with PdfPages(output_pdf) as pdf:
        for metric in metrics:
            cfg = metric_cfg[metric]
            h0, h1 = mean0.values, mean1.values

            if metric == "cosine":
                mat0, mat1 = cosine_similarity(h0), cosine_similarity(h1)
                cross = cosine_similarity(h0, h1)
            elif metric == "pearson":
                mat0, mat1 = np.corrcoef(h0), np.corrcoef(h1)
                cross = np.corrcoef(np.vstack([h0, h1]))[:len(celltypes_ordered), len(celltypes_ordered):]
            elif metric == "euclidean":
                z0, z1 = _row_zscore(h0), _row_zscore(h1)
                mat0, mat1 = euclidean_distances(z0), euclidean_distances(z1)
                cross = euclidean_distances(z0, z1)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            vmin, vmax = cfg["vmin"], cfg["vmax"]
            if vmin is None:
                vmin, vmax = 0, max(mat0.max(), mat1.max(), cross.max())

            fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
            panels = [(cross, f"{g0} vs {g1}"), (mat0, g0), (mat1, g1)]
            for ax, (mat, title) in zip(axes, panels):
                sns.heatmap(
                    mat, ax=ax, cmap=cfg["cmap"], vmin=vmin, vmax=vmax,
                    square=True, cbar=False,
                    xticklabels=celltypes_ordered, yticklabels=celltypes_ordered,
                    linewidths=0.0,
                )
                ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
                ax.tick_params(axis="x", labelsize=6, rotation=90)
                ax.tick_params(axis="y", labelsize=6, rotation=0)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.8)
                    spine.set_color("#333333")

            sm = plt.cm.ScalarMappable(cmap=cfg["cmap"], norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=25)
            cbar.set_label(cfg["label"], fontsize=9, fontweight="bold")
            cbar.ax.tick_params(labelsize=8)

            fig.suptitle(f"Cell-type niche similarity — {cfg['label']}", fontsize=13, fontweight="bold", y=1.04)
            fig.patch.set_facecolor("white")
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"Added panel: {metric}")

    print(f"Saved: {output_pdf}")
    return {"group_means": {g0: mean0, g1: mean1}, "celltypes_ordered": celltypes_ordered}



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#
#                            GENE PROGRAMS ANALYSIS
#
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_grn_leiden_network(
    att_dir: str,
    gene_names,                 # list-like or path to reference count-matrix CSV
    target_genes: list,         # genes to include in the network (subset of gene_names)
    gene_scores: dict,          # {group_label: pd.Series(index=gene, value=score)}
    group_assignment,           # dict {patient_id: group} or (match_condition, [label, label])
    leiden_resolution: float = 1.0,
    min_module_size: int = 4,
    layout: str = "kamada_kawai",   # "kamada_kawai", "spring", or "fruchterman_reingold"
    figsize=(26, 13),
    output_pdf: str = "GRN_leiden_comparison.pdf",
):
    """
    Build gene-gene co-attention networks (restricted to `target_genes`) for
    two patient groups, detect Leiden communities within each, and plot them
    side by side with nodes colored by module and sized by `gene_scores`.

    Parameters
    ----------
    att_dir : str
        Folder with "<patient_id>_edge_att.csv" per patient (columns after
        the first: src, tgt, att — global node indices, genes < n_genes).
    gene_names : list-like or str
        Ordered gene identifiers matching edge-file node indexing, or a path
        to the reference count-matrix CSV used at graph-construction time.
    target_genes : list
        Genes (matching `gene_names`' identifier system) to restrict the
        network to. Typically a union of top-attention genes and
        differentially-embedded genes from earlier analyses.
    gene_scores : dict
        {group_label: pd.Series(index=gene, value=score)}. Used for node
        size/color within each group's panel. Must use the same two group
        labels produced by `group_assignment`.
    group_assignment : dict or tuple
        dict {patient_id: group_label}, or
        (match_condition, [label_if_match, label_if_not]) where
        match_condition is a substring or list of prefixes.
    leiden_resolution : float
        Resolution parameter for Leiden community detection.
    min_module_size : int
        Modules smaller than this are dropped from the plot.
    layout : str
        Graph layout algorithm.
    output_pdf : str
        Output PDF path.

    Returns
    -------
    dict with keys:
        "graphs": {group_label: networkx.DiGraph} (post module-size filtering)
        "modules": {group_label: {module_id: [genes]}}
    """
    # ---- gene names / index map ----
    if isinstance(gene_names, str):
        reference = pd.read_csv(gene_names)
        reference = reference.set_index("Unnamed: 0").T.reset_index().rename(columns={"index": "gene"})
        gene_names = reference[reference.columns[0]].values
    gene_names = np.asarray(gene_names)
    n_genes = len(gene_names)
    gene_to_index = {g: i for i, g in enumerate(gene_names)}

    target_genes = [g for g in target_genes if g in gene_to_index]
    target_indices = set(gene_to_index[g] for g in target_genes)
    if not target_genes:
        raise ValueError("None of target_genes were found in gene_names.")

    # ---- resolve group assignment ----
    if isinstance(group_assignment, tuple):
        match_condition, (label_match, label_nomatch) = group_assignment
        if isinstance(match_condition, (list, tuple)):
            prefixes = tuple(match_condition)
            def assign_group(pid): return label_match if str(pid).startswith(prefixes) else label_nomatch
        else:
            def assign_group(pid): return label_match if match_condition in str(pid) else label_nomatch
    elif isinstance(group_assignment, dict):
        def assign_group(pid): return group_assignment.get(pid, "unknown")
    else:
        raise ValueError("group_assignment must be a dict or (match_condition, [label, label]) tuple")

    edge_files = sorted(glob.glob(os.path.join(att_dir, "*_edge_att.csv")))
    if not edge_files:
        raise FileNotFoundError(f"No '*_edge_att.csv' files found in {att_dir}")

    # ---- gather gene-gene edges restricted to target genes, per condition ----
    all_edges = []
    for f in edge_files:
        patient = os.path.basename(f).replace("_edge_att.csv", "")
        condition = assign_group(patient)
        if condition == "unknown":
            continue
        df = pd.read_csv(f, delimiter=",").iloc[:, 1:]
        df.columns = ["src", "tgt", "att"]
        df[["src", "tgt"]] = df[["src", "tgt"]].astype(int)
        df = df[(df["src"] < n_genes) & (df["tgt"] < n_genes)]
        df = df[df["src"].isin(target_indices) & df["tgt"].isin(target_indices)]
        df["condition"] = condition
        all_edges.append(df)

    all_edges = pd.concat(all_edges, ignore_index=True)
    group_labels = [g for g in all_edges["condition"].unique()]
    if len(group_labels) != 2:
        raise ValueError(f"Expected exactly 2 groups in edge data, got {group_labels}")
    g0, g1 = group_labels

    edge_mean = all_edges.groupby(["condition", "src", "tgt"], as_index=False)["att"].mean()

    graphs, modules_out = {}, {}
    for group in (g0, g1):
        edges_g = edge_mean.query("condition == @group")
        edges_g = edges_g[edges_g["src"] != edges_g["tgt"]]  # drop self-loops

        if group not in gene_scores:
            raise ValueError(f"gene_scores missing entry for group '{group}'")
        scores = gene_scores[group].reindex(target_genes).fillna(0.0)

        # local index map (0..len(target_genes)-1), in target_genes order
        local_idx = {gene_to_index[g]: i for i, g in enumerate(target_genes)}
        local_edges = edges_g[
            edges_g["src"].isin(local_idx) & edges_g["tgt"].isin(local_idx)
        ].copy()
        local_edges["src_local"] = local_edges["src"].map(local_idx)
        local_edges["tgt_local"] = local_edges["tgt"].map(local_idx)

        # ---- Leiden community detection ----
        G_ig = ig.Graph(directed=False)
        G_ig.add_vertices(len(target_genes))
        G_ig.vs["gene"] = target_genes
        G_ig.vs["score"] = scores.values.tolist()
        G_ig.add_edges(list(zip(local_edges["src_local"], local_edges["tgt_local"])))
        G_ig.es["weight"] = local_edges["att"].tolist()

        partition = leidenalg.find_partition(
            G_ig, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=leiden_resolution,
        )

        # ---- build networkx graph with module + score attrs ----
        G_nx = nx.DiGraph()
        for i, gene in enumerate(target_genes):
            G_nx.add_node(gene, score=float(scores.iloc[i]), module=partition.membership[i])
        for _, row in local_edges.iterrows():
            src_gene = target_genes[int(row["src_local"])]
            tgt_gene = target_genes[int(row["tgt_local"])]
            G_nx.add_edge(src_gene, tgt_gene, weight=row["att"])

        # ---- filter small modules ----
        module_sizes = pd.Series([G_nx.nodes[n]["module"] for n in G_nx.nodes()]).value_counts()
        keep_nodes = [n for n in G_nx.nodes() if module_sizes[G_nx.nodes[n]["module"]] >= min_module_size]
        G_filtered = G_nx.subgraph(keep_nodes).copy()

        graphs[group] = G_filtered
        modules_by_id = {}
        for n in G_filtered.nodes():
            modules_by_id.setdefault(G_filtered.nodes[n]["module"], []).append(n)
        modules_out[group] = modules_by_id

        print(f"{group}: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges, "
              f"{len(modules_by_id)} modules (>= {min_module_size} genes)")

    _plot_grn_pair(graphs[g0], graphs[g1], g0, g1, layout, figsize, output_pdf)

    return {"graphs": graphs, "modules": modules_out}


def _plot_grn_pair(G0, G1, label0, label1, layout, figsize, output_pdf):
    all_modules = sorted({G0.nodes[n]["module"] for n in G0.nodes()} |
                          {G1.nodes[n]["module"] for n in G1.nodes()}) or [-1]
    norm = mcolors.Normalize(vmin=min(all_modules), vmax=max(all_modules))
    n_mod = len(all_modules)
    cmap = cm.get_cmap("tab10" if n_mod <= 10 else "tab20" if n_mod <= 20 else "viridis", n_mod)

    def get_layout(G):
        if G.number_of_nodes() == 0:
            return {}
        if layout == "kamada_kawai":
            return nx.kamada_kawai_layout(G)
        elif layout == "spring":
            return nx.spring_layout(G, seed=42, k=0.2, iterations=150)
        elif layout == "fruchterman_reingold":
            return nx.fruchterman_reingold_layout(G, seed=42, k=0.25)
        return nx.spring_layout(G, seed=42)

    def draw(G, ax, title):
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "No nodes after filtering", ha="center", va="center")
            ax.set_title(title); ax.axis("off"); return
        pos = get_layout(G)
        node_sizes = [max(40, min(900, G.nodes[n]["score"] * 140)) for n in G.nodes()]
        node_colors = [cmap(norm(G.nodes[n]["module"])) for n in G.nodes()]
        edge_widths = [max(0.4, G[u][v]["weight"] * 4.5) for u, v in G.edges()]

        nx.draw_networkx_edges(
            G, pos, width=edge_widths, arrows=True, ax=ax, alpha=0.55,
            arrowsize=8, connectionstyle="arc3,rad=0.08",
        )
        nx.draw_networkx_nodes(
            G, pos, node_size=node_sizes, node_color=node_colors, ax=ax,
            edgecolors="gray", linewidths=0.4, alpha=0.95,
        )
        nx.draw_networkx_labels(
            G, pos, labels={n: n for n in G.nodes()}, font_size=5.8,
            font_family="Arial", ax=ax,
        )
        ax.set_title(
            f"{title} — {G.number_of_nodes()} genes, modules >= min size — {layout}",
            fontsize=13, pad=12,
        )
        ax.axis("off")

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    draw(G0, axs[0], label0)
    draw(G1, axs[1], label1)
    plt.tight_layout()
    fig.savefig(output_pdf, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_pdf}")



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#
#                    PLOT CELL-TYPE x CELL-TYPE ATTENTION HEATMAPS
#
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def cell_communication_profiling(
    att_dir: str,
    cell_names_path: str,
    meta: pd.DataFrame,          # columns: barcode, celltype, patient
    gene_names,                  # list-like or path to reference count-matrix CSV
    group_assignment,            # dict {patient_id: group} or (match_condition, [label, label])
    output_pdf: str = "cell_communication_profiling.pdf",
    cmap: str = "viridis",
    figsize=(12, 6),
):
    """
    Build cell-type x cell-type attention matrices for two patient groups
    from FloREN edge-attention scores, and plot them side by side on a
    shared color scale.

    For each patient: sum attention across all cell-cell edges within each
    (celltype_src, celltype_tgt) pair. Average that per-patient sum across
    patients within each group. Self-loops (same celltype) are dropped.

    Parameters
    ----------
    att_dir : str
        Folder with "<patient_id>_edge_att.csv" per patient (columns after
        the first: src, tgt, att — global node indices; cells are indices
        >= n_genes).
    cell_names_path : str
        Path to "All_AUC_Cell_names.csv" (column "0" = full barcodes,
        formatted "<patient>__...").
    meta : pd.DataFrame
        Must have columns "barcode", "celltype", "patient" — however you
        construct barcodes for your dataset (this varies across pipelines,
        so it's left to the caller rather than assumed).
    gene_names : list-like or str
        Ordered gene identifiers matching edge-file node indexing, or a
        path to the reference count-matrix CSV used at graph-construction
        time. Only its length (n_genes) is used here.
    group_assignment : dict or tuple
        dict {patient_id: group_label}, or
        (match_condition, [label_if_match, label_if_not]) where
        match_condition is a substring or list of prefixes.
    output_pdf : str
        Output PDF path.

    Returns
    -------
    dict {group_label: pd.DataFrame} — the two celltype x celltype matrices,
    in the same cell-type order used for plotting.
    """
    # ---- n_genes ----
    if isinstance(gene_names, str):
        reference = pd.read_csv(gene_names)
        reference = reference.set_index("Unnamed: 0").T.reset_index().rename(columns={"index": "gene"})
        gene_names = reference[reference.columns[0]].values
    n_genes = len(gene_names)

    # ---- resolve group assignment ----
    if isinstance(group_assignment, tuple):
        match_condition, (label_match, label_nomatch) = group_assignment
        if isinstance(match_condition, (list, tuple)):
            prefixes = tuple(match_condition)
            def assign_group(pid): return label_match if str(pid).startswith(prefixes) else label_nomatch
        else:
            def assign_group(pid): return label_match if match_condition in str(pid) else label_nomatch
    elif isinstance(group_assignment, dict):
        def assign_group(pid): return group_assignment.get(pid, "unknown")
    else:
        raise ValueError("group_assignment must be a dict or (match_condition, [label, label]) tuple")

    cell_names = pd.read_csv(cell_names_path)
    edge_files = sorted(glob.glob(os.path.join(att_dir, "*_edge_att.csv")))
    if not edge_files:
        raise FileNotFoundError(f"No '*_edge_att.csv' files found in {att_dir}")

    # ---- per-patient celltype-celltype attention aggregation ----
    all_celltype_edges = []
    for f in edge_files:
        patient = os.path.basename(f).replace("_edge_att.csv", "")
        condition = assign_group(patient)
        if condition == "unknown":
            continue

        df2 = pd.read_csv(f, delimiter=",").iloc[:, 1:]
        df2.columns = ["src", "tgt", "att"]
        df2[["src", "tgt"]] = df2[["src", "tgt"]].astype(int)

        patient_mask = cell_names["0"].str.startswith(patient + "__")
        patient_cell_names = cell_names.loc[patient_mask, "0"].reset_index(drop=True)
        cell_df = pd.DataFrame({
            "local_cell_index": np.arange(len(patient_cell_names)),
            "barcode": patient_cell_names,
            "patient": patient,
        })
        cell_df["node_index"] = cell_df["local_cell_index"] + n_genes
        cell_df = cell_df.merge(
            meta[["barcode", "celltype", "patient"]], on=["barcode", "patient"], how="left"
        )
        node_to_celltype = dict(zip(cell_df["node_index"], cell_df["celltype"]))

        df_cells = df2[(df2["src"] >= n_genes) & (df2["tgt"] >= n_genes)].copy()
        df_cells["celltype_src"] = df_cells["src"].map(node_to_celltype)
        df_cells["celltype_tgt"] = df_cells["tgt"].map(node_to_celltype)
        df_cells = df_cells.dropna(subset=["celltype_src", "celltype_tgt"])

        agg = df_cells.groupby(["celltype_src", "celltype_tgt"], as_index=False)["att"].sum()
        agg["patient"] = patient
        agg["condition"] = condition
        all_celltype_edges.append(agg)

    all_celltype_edges = pd.concat(all_celltype_edges, ignore_index=True)
    group_labels = [g for g in all_celltype_edges["condition"].unique()]
    if len(group_labels) != 2:
        raise ValueError(f"Expected exactly 2 groups, got {group_labels}")
    g0, g1 = group_labels

    celltype_edge_mean = all_celltype_edges.groupby(
        ["condition", "celltype_src", "celltype_tgt"], as_index=False
    )["att"].mean()

    edges0 = celltype_edge_mean.query("condition == @g0").copy()
    edges1 = celltype_edge_mean.query("condition == @g1").copy()
    edges0 = edges0[edges0["celltype_src"] != edges0["celltype_tgt"]]  # drop self-loops
    edges1 = edges1[edges1["celltype_src"] != edges1["celltype_tgt"]]

    celltypes = sorted(set(all_celltype_edges["celltype_src"]) | set(all_celltype_edges["celltype_tgt"]))

    def build_matrix(edges):
        mat = pd.DataFrame(0.0, index=celltypes, columns=celltypes)
        for _, row in edges.iterrows():
            mat.loc[row["celltype_src"], row["celltype_tgt"]] = row["att"]
        return mat

    matrix0 = build_matrix(edges0)
    matrix1 = build_matrix(edges1)

    # ── hierarchical ordering on the pooled mean ──────────────────────────────
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    combined_vals = (matrix0.values + matrix1.values) / 2.0
    if combined_vals.max() > 0 and len(celltypes) > 1:
        try:
            link  = linkage(pdist(combined_vals, metric="euclidean"), method="average")
            order = leaves_list(link)
        except Exception:
            order = list(range(len(celltypes)))
    else:
        order = list(range(len(celltypes)))

    ct_ord  = [celltypes[i] for i in order]
    matrix0 = matrix0.loc[ct_ord, ct_ord]
    matrix1 = matrix1.loc[ct_ord, ct_ord]

    # patient counts per group
    counts = all_celltype_edges.groupby("condition")["patient"].nunique()
    n0 = int(counts.get(g0, 0))
    n1 = int(counts.get(g1, 0))

    # ── journal-quality plot ──────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.8, "font.size": 9,
    })

    vmin = 0.0
    vmax = float(max(matrix0.values.max(), matrix1.values.max()))
    if vmax == 0:
        vmax = 1.0

    n_ct = len(ct_ord)
    cell_px    = max(0.3, min(0.65, 7.0 / max(n_ct, 1)))
    panel_side = max(3.5, cell_px * n_ct + 1.8)
    fig_w      = panel_side * 2 + 1.5

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, panel_side))

    _hm_kw = dict(
        cmap=cmap, vmin=vmin, vmax=vmax,
        square=True, cbar=False,
        linewidths=0.25, linecolor="white",
    )

    # Left panel (g0) — show both x and y labels
    sns.heatmap(matrix0, ax=axes[0],
                xticklabels=ct_ord, yticklabels=ct_ord, **_hm_kw)
    axes[0].set_title(f"{g0}  (n = {n0} patients)",
                      fontsize=11, fontweight="bold", pad=10)
    axes[0].set_xlabel("Target cell type", fontsize=9, fontweight="bold", labelpad=6)
    axes[0].set_ylabel("Source cell type", fontsize=9, fontweight="bold", labelpad=6)
    axes[0].tick_params(axis="x", labelsize=6, rotation=90)
    axes[0].tick_params(axis="y", labelsize=6, rotation=0)
    for spine in axes[0].spines.values():
        spine.set_visible(True); spine.set_linewidth(0.6); spine.set_color("#444444")

    # Right panel (g1) — show x labels; y labels omitted (shared axis)
    sns.heatmap(matrix1, ax=axes[1],
                xticklabels=ct_ord, yticklabels=False, **_hm_kw)
    axes[1].set_title(f"{g1}  (n = {n1} patients)",
                      fontsize=11, fontweight="bold", pad=10)
    axes[1].set_xlabel("Target cell type", fontsize=9, fontweight="bold", labelpad=6)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", labelsize=6, rotation=90)
    for spine in axes[1].spines.values():
        spine.set_visible(True); spine.set_linewidth(0.6); spine.set_color("#444444")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.55, pad=0.03, aspect=28, location="right")
    cbar.set_label("Mean attention score", fontsize=9, fontweight="bold", labelpad=8)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.6)

    fig.suptitle("Cell–Cell Communication Profiling",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    try:
        os.remove(output_pdf)
    except OSError:
        pass
    fig.savefig(output_pdf, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_pdf}")

    return {g0: matrix0, g1: matrix1}