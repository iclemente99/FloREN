#!/usr/bin/env python3
"""
FloREN end-to-end test — measures wall time and peak RAM for every pipeline step.

Runs from the repository root directory.

Usage:
    python test_pipeline.py                        # quick test (50 AE epochs, 10 GNN epochs)
    python test_pipeline.py --full                 # full parameters (1000 / 100 epochs)
    python test_pipeline.py --skip-step1           # skip graph construction (reuse existing output)
    python test_pipeline.py --skip-step2           # skip GNN training (reuse existing output)
    python test_pipeline.py --cuda 0               # use GPU 0
    python test_pipeline.py --output-dir ./my_out  # custom output directory
"""

import os, sys, time, subprocess, threading, argparse, shutil, zipfile, traceback, importlib
import numpy as np
import pandas as pd
from pathlib import Path

# ── psutil (RAM monitoring) ────────────────────────────────────────────────────
try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False
    print("WARNING: psutil not installed — RAM tracking disabled. "
          "Install with: pip install psutil")

# ── matplotlib non-interactive backend (needed for headless envs) ──────────────
import matplotlib
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR   = REPO_ROOT / "src"
DATA_DIR  = REPO_ROOT / "data"

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="FloREN end-to-end test")
ap.add_argument("--full",       action="store_true",
                help="Use full epoch counts (AE=1000, GNN=100). Slow but realistic.")
ap.add_argument("--output-dir", default=str(REPO_ROOT / "test_output"),
                help="Directory for pipeline outputs and test report")
ap.add_argument("--cuda",       type=int, default=1,
                help="0 = GPU 0, 1 = CPU (default)")
ap.add_argument("--skip-step1", action="store_true",
                help="Skip floren_input.py (graph construction). Reuse existing output.")
ap.add_argument("--skip-step2", action="store_true",
                help="Skip floren_training.py. Reuse existing output.")
ap.add_argument("--skip-step3", action="store_true",
                help="Skip floren_visualization.py.")
cli = ap.parse_args()

OUTPUT_DIR = Path(cli.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)

AE_EPOCHS  = 1000 if cli.full else 50
GNN_EPOCHS = 100  if cli.full else 10
PYTHON     = sys.executable


# ──────────────────────────────────────────────────────────────────────────────
# Results table
# ──────────────────────────────────────────────────────────────────────────────
_report: list[dict] = []

def _record(step: str, status: str, elapsed_s: float, peak_ram_mb: float, note: str = ""):
    _report.append(dict(step=step, status=status,
                        elapsed_s=round(elapsed_s, 1),
                        peak_ram_mb=round(peak_ram_mb),
                        note=note))
    icon = "✓" if status == "OK" else "✗"
    ram_str = f"{peak_ram_mb:.0f} MB" if peak_ram_mb > 0 else "n/a"
    print(f"\n{icon}  {step}  |  {status}  |  {elapsed_s:.1f}s  |  peak RAM {ram_str}"
          + (f"\n    ↳ {note}" if note else ""), flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Peak-RAM context manager (background thread, 300 ms poll)
# ──────────────────────────────────────────────────────────────────────────────
class PeakRAM:
    """
    Polls the RSS of a process (and its children) every 300 ms.
    Use as a context manager.  target_pid=None → current process.
    """
    def __init__(self, target_pid: int = None):
        self._pid  = target_pid
        self.peak_mb     = 0.0
        self.baseline_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        if _HAVE_PSUTIL:
            pid = self._pid or os.getpid()
            try:
                self.baseline_mb = psutil.Process(pid).memory_info().rss / 1024 ** 2
            except Exception:
                pass
            self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if _HAVE_PSUTIL:
            self._thread.join(timeout=2)

    def _run(self):
        pid = self._pid or os.getpid()
        while not self._stop.wait(0.3):
            try:
                p = psutil.Process(pid)
                rss = p.memory_info().rss
                for child in p.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.peak_mb = max(self.peak_mb, rss / 1024 ** 2)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    @property
    def delta_mb(self) -> float:
        """Net RAM increase during the monitored block."""
        return max(0.0, self.peak_mb - self.baseline_mb)


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess runner (live stdout + RAM monitor of child process)
# ──────────────────────────────────────────────────────────────────────────────
def run_subprocess(label: str, cmd: list, cwd=None) -> tuple[bool, str]:
    print(f"\n{'━'*66}")
    print(f"  STEP : {label}")
    print(f"  CMD  : {' '.join(str(c) for c in cmd)}")
    print(f"  CWD  : {cwd or REPO_ROOT}")
    print('━'*66, flush=True)

    t0   = time.perf_counter()
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(cwd or REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    monitor = PeakRAM(proc.pid)
    monitor.__enter__()
    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    monitor.__exit__()

    elapsed = time.perf_counter() - t0
    ok = (proc.returncode == 0)
    _record(label,
            "OK" if ok else f"FAIL  rc={proc.returncode}",
            elapsed,
            monitor.peak_mb)
    return ok, "".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# In-process runner (downstream functions, same Python process)
# ──────────────────────────────────────────────────────────────────────────────
def run_inprocess(label: str, fn):
    print(f"\n  ▸ {label} ...", flush=True)
    result = None
    note   = ""
    with PeakRAM() as ram:
        t0 = time.perf_counter()
        try:
            result = fn()
            status = "OK"
        except Exception as exc:
            status = "FAIL"
            note   = str(exc)[:150]
            traceback.print_exc()
        elapsed = time.perf_counter() - t0
    _record(label, status, elapsed, ram.delta_mb, note=note)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# PREFLIGHT — unzip Prior Knowledge reference if not already present
# ──────────────────────────────────────────────────────────────────────────────
prior_csv = DATA_DIR / "Prior_Knowledge_PRECISEADS.csv"
if not prior_csv.exists():
    print(f"\nPreflight: extracting Prior Knowledge reference CSV …")
    zip1 = DATA_DIR / "Prior_Knowledge_PRECISEADS_compI_compII.zip"
    if zip1.exists():
        tmp = DATA_DIR / "_pk_tmp"
        tmp.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip1) as z:
            z.extractall(tmp)
        inner_zips = list(tmp.rglob("*.zip"))
        if inner_zips:
            inner = tmp / "inner"
            inner.mkdir(exist_ok=True)
            with zipfile.ZipFile(inner_zips[0]) as z:
                z.extractall(inner)
            csvs = list(inner.rglob("*.csv"))
        else:
            csvs = list(tmp.rglob("*.csv"))
        if csvs:
            shutil.copy(csvs[0], prior_csv)
            print(f"  ✓ Extracted → {prior_csv}")
        shutil.rmtree(tmp, ignore_errors=True)
    if not prior_csv.exists():
        print("  WARNING: Prior_Knowledge_PRECISEADS.csv could not be extracted. "
              "GRN prior knowledge step in floren_input.py may fail.")
else:
    print(f"Preflight: Prior Knowledge CSV found  ({prior_csv})")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Graph construction  (floren_input.py)
# ──────────────────────────────────────────────────────────────────────────────
if cli.skip_step1:
    print("\nStep 1 skipped (--skip-step1).")
else:
    run_subprocess(
        "Step 1 — floren_input.py  (AE graph construction)",
        [PYTHON, "src/floren_input.py",
         "--adata_path",     str(DATA_DIR / "binvignat_example.h5ad"),
         "--cell_comm_path", str(DATA_DIR / "cell_connections"),
         "--output_path",    str(OUTPUT_DIR),
         "--epochs",         str(AE_EPOCHS),
         "--grn_cutoff",     "0.9",
         "--cuda",           str(cli.cuda),
         ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — GNN training  (floren_training.py)
# ──────────────────────────────────────────────────────────────────────────────
if cli.skip_step2:
    print("\nStep 2 skipped (--skip-step2).")
else:
    run_subprocess(
        "Step 2 — floren_training.py  (HGT self-supervised + classifier)",
        [PYTHON, "src/floren_training.py",
         "--adata_path",     str(DATA_DIR / "binvignat_example.h5ad"),
         "--cell_comm_path", str(DATA_DIR / "cell_connections"),
         "--output_path",    str(OUTPUT_DIR),
         "--result_dir",     str(OUTPUT_DIR),
         "--epochs",         str(GNN_EPOCHS),
         "--patient_id",     "patient_id",
         "--metadata_group", "disease",
         "--cuda",           str(cli.cuda),
         "--min_count",      "0",
         ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Visualization  (floren_visualization.py)
# ──────────────────────────────────────────────────────────────────────────────
if cli.skip_step3:
    print("\nStep 3 skipped (--skip-step3).")
else:
    # --epochs / --n_hid / --n_heads / --n_batch must match training (used to locate the saved model)
    run_subprocess(
        "Step 3 — floren_visualization.py  (UMAP + attention plots)",
        [PYTHON, "src/floren_visualization.py",
         "--adata_path",     str(DATA_DIR / "binvignat_example.h5ad"),
         "--output_path",    str(OUTPUT_DIR),
         "--result_dir",     str(OUTPUT_DIR),
         "--patient_id",     "patient_id",
         "--metadata_group", "disease",
         "--epochs",         str(GNN_EPOCHS),
         ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Downstream analysis  (downstream.py functions imported in-process)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'━'*66}")
print("  STEP 4 — downstream.py analysis functions")
print('━'*66, flush=True)

sys.path.insert(0, str(SRC_DIR))

# ── load adata metadata (backed mode = no full matrix in RAM) ──────────────────
import scanpy as sc
adata_meta = sc.read_h5ad(str(DATA_DIR / "binvignat_example.h5ad"))
gene_names  = list(adata_meta.var_names)
patients    = list(adata_meta.obs["patient_id"].unique())

# Build group_assignment: try adata.obs["disease"], fall back to name heuristic
if "disease" in adata_meta.obs.columns:
    _tmp = (adata_meta.obs[["patient_id", "disease"]]
            .drop_duplicates("patient_id")
            .set_index("patient_id")["disease"])
    group_assignment = {p: str(_tmp.get(p, "Unknown")) for p in patients}
else:
    group_assignment = {p: ("RA" if "RA" in str(p) else "Control") for p in patients}

del adata_meta

groups        = sorted(set(group_assignment.values()))
results_path  = str(OUTPUT_DIR)
h5ad_path     = str(DATA_DIR / "binvignat_example.h5ad")
gene_emb_dir  = str(OUTPUT_DIR / "floren_gene_embeddings")
cell_emb_dir  = str(OUTPUT_DIR / "floren_cell_embeddings")
att_dir_str   = str(OUTPUT_DIR / "floren_attention_embeddings")

print(f"\n  Patients  : {patients}")
print(f"  Groups    : {group_assignment}")
print(f"  Gene count: {len(gene_names)}", flush=True)

try:
    ds = importlib.import_module("downstream")
except Exception as exc:
    print(f"\n  FAIL — could not import downstream.py: {exc}")
    ds = None


# 4-A ── Cell-type saliency ranking ────────────────────────────────────────────
saliency_series = None
if ds:
    saliency_series = run_inprocess(
        "Step 4a — plot_celltype_saliency_ranking",
        lambda: ds.plot_celltype_saliency_ranking(
            floren_results_path=results_path,
            h5ad_path=h5ad_path,
            celltype_col="cell_type",
            sample_col="patient_id",
            agg="mean",
            top_n=10,
        ),
    )

# 4-B ── Per-cell-type gene attention signatures ────────────────────────────────
if ds:
    run_inprocess(
        "Step 4b — plot_celltype_gene_signatures",
        lambda: ds.plot_celltype_gene_signatures(
            floren_results_path=results_path,
            h5ad_path=h5ad_path,
            gene_names=gene_names,
            att_subdir="floren_attention_embeddings",
            celltype_col="cell_type",
            sample_col="patient_id",
            top_n_genes=20,
        ),
    )

# 4-C ── Differential cell-type abundance (Mann-Whitney) ───────────────────────
if ds:
    run_inprocess(
        "Step 4c — plot_celltype_differential_abundance",
        lambda: ds.plot_celltype_differential_abundance(
            floren_results_path=results_path,
            h5ad_path=h5ad_path,
            group_assignment=group_assignment,
            celltype_col="cell_type",
            sample_col="patient_id",
            agg="mean",
            scale="both",
        ),
    )

# 4-D ── Differential gene expression (embedding-space) ────────────────────────
dge_result = None
if ds:
    dge_result = run_inprocess(
        "Step 4d — run_differential_gene_expression",
        lambda: ds.run_differential_gene_expression(
            gene_embeddings_dir=gene_emb_dir,
            gene_names=gene_names,
            group_assignment=group_assignment,
            map_to_symbols=False,
            make_plots=True,
            output_dir=str(OUTPUT_DIR / "plots"),
        ),
    )

# 4-E ── Cell-type niche similarity heatmaps ───────────────────────────────────
if ds:
    run_inprocess(
        "Step 4e — plot_celltype_niche_heatmaps",
        lambda: ds.plot_celltype_niche_heatmaps(
            cell_embeddings_dir=cell_emb_dir,
            h5ad_path=h5ad_path,
            group_assignment=group_assignment,
            patient_col="patient_id",
            celltype_col="cell_type",
            metrics=("cosine", "euclidean"),
        ),
    )

# 4-F ── GRN Leiden community network ──────────────────────────────────────────
# target_genes: top 40 genes by mean attention from DGE result, or first 40 genes
if ds:
    try:
        # run_differential_gene_expression returns (gene_results_df, group_means_dict)
        gene_results_df = None
        if dge_result is not None:
            gene_results_df = dge_result[0] if isinstance(dge_result, tuple) else dge_result
        if gene_results_df is not None and "distance" in gene_results_df.columns:
            gene_col = "gene" if "gene" in gene_results_df.columns else gene_results_df.index.name or "gene"
            target_genes = list(
                gene_results_df.sort_values("distance", ascending=False)
                .head(40)
                .index if gene_col not in gene_results_df.columns
                else gene_results_df.sort_values("distance", ascending=False).head(40)[gene_col]
            )
        else:
            target_genes = gene_names[:40]

        # gene_scores: uniform score of 1.0 for every target gene per group
        gene_scores = {g: pd.Series(np.ones(len(target_genes)), index=target_genes)
                       for g in groups}

        run_inprocess(
            "Step 4f — plot_grn_leiden_network",
            lambda: ds.plot_grn_leiden_network(
                att_dir=att_dir_str,
                gene_names=gene_names,
                target_genes=target_genes,
                gene_scores=gene_scores,
                group_assignment=group_assignment,
                leiden_resolution=0.5,
                output_pdf=str(OUTPUT_DIR / "plots" / "GRN_leiden_test.pdf"),
            ),
        )
    except Exception as exc:
        _record("Step 4f — plot_grn_leiden_network", "SKIP", 0, 0,
                note=f"Could not prepare inputs: {exc}")


# 4-G ── Cell-type x cell-type attention heatmaps ─────────────────────────────
if ds:
    try:
        import scanpy as _sc_g
        _adata_g = _sc_g.read_h5ad(h5ad_path)
        _meta_g = pd.DataFrame({
            "celltype": _adata_g.obs["cell_type"].astype(str).values,
            "patient":  _adata_g.obs["patient_id"].astype(str).values,
        }, index=_adata_g.obs.index)
        _meta_g["barcode"] = _meta_g["patient"] + "__" + _meta_g.index.astype(str)
        del _adata_g

        run_inprocess(
            "Step 4g — plot_celltype_attention_heatmaps",
            lambda: ds.plot_celltype_attention_heatmaps(
                att_dir=att_dir_str,
                cell_names_path=str(OUTPUT_DIR / "All_AUC_Cell_names.csv"),
                meta=_meta_g,
                gene_names=gene_names,
                group_assignment=group_assignment,
                output_pdf=str(OUTPUT_DIR / "plots" / "celltype_attention_heatmaps_test.pdf"),
            ),
        )
    except Exception as exc:
        _record("Step 4g — plot_celltype_attention_heatmaps", "SKIP", 0, 0,
                note=f"Could not prepare inputs: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n\n{'═'*72}")
print("  FloREN Pipeline Test Report")
print('═'*72)

df = pd.DataFrame(_report)
if not df.empty:
    df["time"] = df["elapsed_s"].apply(lambda x: f"{x:>7.1f} s")
    df["RAM"]  = df["peak_ram_mb"].apply(lambda x: f"{x:>6.0f} MB" if x > 0 else "     n/a")
    fmt = df[["step", "status", "time", "RAM", "note"]]
    print(fmt.to_string(index=False, max_colwidth=52))
    print('═'*72)

    total_s    = df["elapsed_s"].sum()
    ok_count   = (df["status"] == "OK").sum()
    fail_count = (df["status"] != "OK").sum()

    print(f"\n  Total wall time : {total_s/60:.1f} min ({total_s:.0f} s)")
    print(f"  Passed          : {ok_count} / {ok_count + fail_count}")
    if fail_count:
        print(f"  Failed          : {fail_count}")
        for _, row in df[df["status"] != "OK"].iterrows():
            print(f"    ✗  {row['step']}: {row['status']}  — {row['note']}")

report_csv = OUTPUT_DIR / "test_report.csv"
df.to_csv(report_csv, index=False)
print(f"\n  Report saved → {report_csv}")
print('═'*72 + "\n")

if fail_count:
    sys.exit(1)
