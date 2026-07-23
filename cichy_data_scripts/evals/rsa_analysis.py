"""Compare saved model RDMs with Cichy IT fMRI RDMs.

Run after extract_activations.py:

    python -m cichy_data_scripts.rsa_analysis
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import (
    spearmanr,
    ttest_1samp,
    ttest_rel,
    wilcoxon,
)

from .config import CHECKPOINT_DIR, CICHY_DATA, SEED


# ---------------------------------------------------------------------
# Paths and analysis selection
# ---------------------------------------------------------------------

FMRI_FILE = CICHY_DATA / "target_fmri.mat"

RDM_ROOT = (
    CHECKPOINT_DIR.parent
    / "rdms"
    / "cichy_data_models"
)

OUTPUT_DIR = RDM_ROOT / "rsa_analysis"


# Set to None to automatically use every model directory found.
SELECTED_MODELS: list[str] | None = [
    "standard",
    "expert",
]

# Set to None to use every matching layer found.
SELECTED_LAYERS: list[str] | None = [
    "fc7_post_relu",
    "fc8_logits",
]


# ---------------------------------------------------------------------
# Statistical settings
# ---------------------------------------------------------------------

EXPECTED_STIMULI = 92

N_PERMUTATIONS = 100_000
ALPHA = 0.05
RANDOM_SEED = SEED


# ---------------------------------------------------------------------
# MATLAB and brain-RDM loading
# ---------------------------------------------------------------------

def load_mat_file(
    mat_path: Path,
) -> dict[str, np.ndarray]:
    """Load MATLAB v7 files or HDF5-based MATLAB v7.3 files."""

    if not mat_path.exists():
        raise FileNotFoundError(
            f"MATLAB file not found:\n{mat_path}"
        )

    try:
        with h5py.File(mat_path, "r") as file:
            return {
                name: np.transpose(
                    np.asarray(file[name])
                )
                for name in file.keys()
            }

    except OSError:
        return sio.loadmat(mat_path)


def standardize_subject_rdm_axes(
    subject_rdms: np.ndarray,
) -> np.ndarray:
    """Convert IT RDMs to [subjects, stimuli, stimuli]."""

    subject_rdms = np.asarray(
        subject_rdms,
        dtype=np.float64,
    )

    if subject_rdms.ndim != 3:
        raise ValueError(
            "IT_RDMs must have three dimensions. "
            f"Received {subject_rdms.shape}."
        )

    expected_pair = (
        EXPECTED_STIMULI,
        EXPECTED_STIMULI,
    )

    # Already [subjects, stimuli, stimuli].
    if subject_rdms.shape[1:] == expected_pair:
        return subject_rdms

    # MATLAB may load them as [stimuli, stimuli, subjects].
    if subject_rdms.shape[:2] == expected_pair:
        return np.moveaxis(
            subject_rdms,
            source=2,
            destination=0,
        )

    raise ValueError(
        "Could not identify the IT_RDMs axes. "
        f"Received shape {subject_rdms.shape}; expected either "
        f"[subjects, {EXPECTED_STIMULI}, {EXPECTED_STIMULI}] "
        f"or [{EXPECTED_STIMULI}, {EXPECTED_STIMULI}, subjects]."
    )


def load_it_rdms(
    fmri_path: Path,
) -> np.ndarray:
    """Load subject-level IT RDMs."""

    data = load_mat_file(
        fmri_path
    )

    if "IT_RDMs" not in data:
        raise KeyError(
            "Could not find IT_RDMs in target_fmri.mat. "
            f"Available keys: {list(data.keys())}"
        )

    subject_rdms = standardize_subject_rdm_axes(
        data["IT_RDMs"]
    )

    if not np.isfinite(subject_rdms).all():
        raise ValueError(
            "The IT RDMs contain NaN or infinite values."
        )

    print(f"IT RDM shape: {subject_rdms.shape}")
    print(f"Subjects: {subject_rdms.shape[0]}")

    return subject_rdms


# ---------------------------------------------------------------------
# Model-RDM discovery
# ---------------------------------------------------------------------

def discover_model_rdms(
    root: Path,
) -> dict[str, dict[str, Path]]:
    """Find saved model RDMs organized by model and layer."""

    if not root.exists():
        raise FileNotFoundError(
            f"Model RDM directory not found:\n{root}"
        )

    discovered: dict[str, dict[str, Path]] = {}

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue

        if model_dir.name == "rsa_analysis":
            continue

        model_name = model_dir.name

        if (
            SELECTED_MODELS is not None
            and model_name not in SELECTED_MODELS
        ):
            continue

        layer_paths = {}

        for rdm_path in sorted(
            model_dir.glob(
                "rdm_*_correlation.npy"
            )
        ):
            prefix = "rdm_"
            suffix = "_correlation.npy"

            layer_name = rdm_path.name[
                len(prefix):-len(suffix)
            ]

            if (
                SELECTED_LAYERS is not None
                and layer_name not in SELECTED_LAYERS
            ):
                continue

            layer_paths[layer_name] = rdm_path

        if layer_paths:
            discovered[model_name] = layer_paths

    if not discovered:
        raise RuntimeError(
            f"No matching model RDMs were found under:\n{root}"
        )

    return discovered


def load_model_rdms(
    discovered_paths: dict[str, dict[str, Path]],
) -> dict[str, dict[str, np.ndarray]]:
    """Load and validate all discovered model RDMs."""

    loaded = {}

    for model_name, layer_paths in discovered_paths.items():
        loaded[model_name] = {}

        for layer_name, rdm_path in layer_paths.items():
            rdm = np.load(
                rdm_path
            ).astype(
                np.float64,
                copy=False,
            )

            expected_shape = (
                EXPECTED_STIMULI,
                EXPECTED_STIMULI,
            )

            if rdm.shape != expected_shape:
                raise ValueError(
                    f"{model_name}/{layer_name}: expected "
                    f"{expected_shape}, received {rdm.shape}."
                )

            if not np.isfinite(rdm).all():
                raise ValueError(
                    f"{model_name}/{layer_name} contains "
                    "NaN or infinite values."
                )

            if not np.allclose(
                rdm,
                rdm.T,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{model_name}/{layer_name} is not symmetric."
                )

            loaded[model_name][layer_name] = rdm

    return loaded


# ---------------------------------------------------------------------
# RSA
# ---------------------------------------------------------------------

def vectorize_rdm(
    rdm: np.ndarray,
) -> np.ndarray:
    """Return the lower triangle without the diagonal."""

    if (
        rdm.ndim != 2
        or rdm.shape[0] != rdm.shape[1]
    ):
        raise ValueError(
            f"RDM must be square. Received {rdm.shape}."
        )

    indices = np.tril_indices(
        rdm.shape[0],
        k=-1,
    )

    return rdm[indices]


def spearman_rsa(
    model_rdm: np.ndarray,
    brain_rdm: np.ndarray,
) -> float:
    """Compute Spearman correlation between vectorized RDMs."""

    result = spearmanr(
        vectorize_rdm(model_rdm),
        vectorize_rdm(brain_rdm),
    )

    return float(result.statistic)


def calculate_subject_rsa(
    model_rdms: dict[str, dict[str, np.ndarray]],
    subject_it_rdms: np.ndarray,
) -> pd.DataFrame:
    """Calculate one RSA correlation per subject/model/layer."""

    rows = []

    for model_name, layer_rdms in model_rdms.items():
        for layer_name, model_rdm in layer_rdms.items():
            for subject_index, subject_rdm in enumerate(
                subject_it_rdms
            ):
                rho = spearman_rsa(
                    model_rdm=model_rdm,
                    brain_rdm=subject_rdm,
                )

                rows.append(
                    {
                        "subject": subject_index + 1,
                        "model": model_name,
                        "layer": layer_name,
                        "rho": rho,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------

def fisher_z(
    correlations: np.ndarray,
) -> np.ndarray:
    """Apply the Fisher transformation safely."""

    correlations = np.asarray(
        correlations,
        dtype=np.float64,
    )

    correlations = np.clip(
        correlations,
        -0.999999,
        0.999999,
    )

    return np.arctanh(correlations)


def sign_flip_permutation_test(
    differences: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Two-sided paired sign-flip permutation test.

    The statistic is the mean paired difference.
    """

    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    observed_difference = float(
        differences.mean()
    )

    if np.allclose(differences, 0.0):
        return observed_difference, 1.0

    number_of_subjects = len(differences)

    permutation_statistics = np.empty(
        n_permutations,
        dtype=np.float64,
    )

    for permutation_index in range(n_permutations):
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=number_of_subjects,
            replace=True,
        )

        permutation_statistics[permutation_index] = (
            differences * signs
        ).mean()

    extreme_count = np.sum(
        np.abs(permutation_statistics)
        >= abs(observed_difference)
    )

    p_value = (
        extreme_count + 1
    ) / (
        n_permutations + 1
    )

    return observed_difference, float(p_value)


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = 20_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a subject-level mean."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    sample_indices = rng.integers(
        low=0,
        high=len(values),
        size=(n_bootstrap, len(values)),
    )

    bootstrap_means = values[
        sample_indices
    ].mean(axis=1)

    tail = (1.0 - confidence) / 2.0

    lower = np.quantile(
        bootstrap_means,
        tail,
    )

    upper = np.quantile(
        bootstrap_means,
        1.0 - tail,
    )

    return float(lower), float(upper)


def fdr_bh(
    p_values: np.ndarray,
) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""

    p_values = np.asarray(
        p_values,
        dtype=np.float64,
    )

    number_of_tests = len(p_values)

    if number_of_tests == 0:
        return p_values.copy()

    order = np.argsort(p_values)
    ranked_p = p_values[order]

    adjusted_ranked = (
        ranked_p
        * number_of_tests
        / np.arange(
            1,
            number_of_tests + 1,
        )
    )

    adjusted_ranked = np.minimum.accumulate(
        adjusted_ranked[::-1]
    )[::-1]

    adjusted_ranked = np.clip(
        adjusted_ranked,
        0.0,
        1.0,
    )

    adjusted = np.empty_like(
        adjusted_ranked
    )

    adjusted[order] = adjusted_ranked

    return adjusted


def one_model_statistics(
    rsa_table: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Test whether each model/layer has RSA above zero."""

    rows = []

    grouped = rsa_table.groupby(
        ["model", "layer"],
        sort=True,
    )

    for (model_name, layer_name), group in grouped:
        correlations = group["rho"].to_numpy()

        transformed = fisher_z(
            correlations
        )

        t_result = ttest_1samp(
            transformed,
            popmean=0.0,
        )

        try:
            wilcoxon_result = wilcoxon(
                correlations,
                alternative="greater",
                zero_method="wilcox",
            )

            wilcoxon_statistic = float(
                wilcoxon_result.statistic
            )
            wilcoxon_p = float(
                wilcoxon_result.pvalue
            )

        except ValueError:
            wilcoxon_statistic = np.nan
            wilcoxon_p = np.nan

        observed_mean, permutation_p = (
            sign_flip_permutation_test(
                differences=correlations,
                n_permutations=N_PERMUTATIONS,
                rng=rng,
            )
        )

        ci_low, ci_high = bootstrap_mean_ci(
            values=correlations,
            rng=rng,
        )

        rows.append(
            {
                "model": model_name,
                "layer": layer_name,
                "n_subjects": len(correlations),
                "mean_rho": correlations.mean(),
                "std_rho": correlations.std(ddof=1),
                "median_rho": np.median(correlations),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "fisher_t": float(t_result.statistic),
                "fisher_t_p_two_sided": float(
                    t_result.pvalue
                ),
                "wilcoxon_greater_statistic": (
                    wilcoxon_statistic
                ),
                "wilcoxon_greater_p": wilcoxon_p,
                "permutation_mean": observed_mean,
                "permutation_p_two_sided": permutation_p,
            }
        )

    result = pd.DataFrame(rows)

    result["permutation_p_fdr"] = fdr_bh(
        result["permutation_p_two_sided"].to_numpy()
    )

    return result


def paired_model_statistics(
    rsa_table: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compare models within each matching layer.

    Correlations are paired because every model is compared with the
    same fMRI participant.
    """

    rows = []

    available_layers = sorted(
        rsa_table["layer"].unique()
    )

    for layer_name in available_layers:
        layer_table = rsa_table[
            rsa_table["layer"] == layer_name
        ]

        available_models = sorted(
            layer_table["model"].unique()
        )

        for model_a, model_b in combinations(
            available_models,
            2,
        ):
            model_a_values = (
                layer_table[
                    layer_table["model"] == model_a
                ][["subject", "rho"]]
                .rename(
                    columns={
                        "rho": "rho_a",
                    }
                )
            )

            model_b_values = (
                layer_table[
                    layer_table["model"] == model_b
                ][["subject", "rho"]]
                .rename(
                    columns={
                        "rho": "rho_b",
                    }
                )
            )

            paired = model_a_values.merge(
                model_b_values,
                on="subject",
                how="inner",
                validate="one_to_one",
            )

            if paired.empty:
                raise RuntimeError(
                    f"No paired subjects for {model_a} and "
                    f"{model_b}, layer {layer_name}."
                )

            correlations_a = paired[
                "rho_a"
            ].to_numpy()

            correlations_b = paired[
                "rho_b"
            ].to_numpy()

            raw_differences = (
                correlations_b - correlations_a
            )

            fisher_differences = (
                fisher_z(correlations_b)
                - fisher_z(correlations_a)
            )

            t_result = ttest_rel(
                fisher_z(correlations_b),
                fisher_z(correlations_a),
            )

            try:
                wilcoxon_result = wilcoxon(
                    raw_differences,
                    alternative="two-sided",
                    zero_method="wilcox",
                )

                wilcoxon_statistic = float(
                    wilcoxon_result.statistic
                )
                wilcoxon_p = float(
                    wilcoxon_result.pvalue
                )

            except ValueError:
                wilcoxon_statistic = np.nan
                wilcoxon_p = np.nan

            (
                observed_difference,
                permutation_p,
            ) = sign_flip_permutation_test(
                differences=raw_differences,
                n_permutations=N_PERMUTATIONS,
                rng=rng,
            )

            ci_low, ci_high = bootstrap_mean_ci(
                values=raw_differences,
                rng=rng,
            )

            rows.append(
                {
                    "layer": layer_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "difference_definition": (
                        "model_b minus model_a"
                    ),
                    "n_subjects": len(paired),
                    "model_a_mean_rho": correlations_a.mean(),
                    "model_b_mean_rho": correlations_b.mean(),
                    "mean_rho_difference": (
                        raw_differences.mean()
                    ),
                    "median_rho_difference": (
                        np.median(raw_differences)
                    ),
                    "difference_ci95_low": ci_low,
                    "difference_ci95_high": ci_high,
                    "paired_fisher_t": float(
                        t_result.statistic
                    ),
                    "paired_fisher_t_p": float(
                        t_result.pvalue
                    ),
                    "wilcoxon_statistic": (
                        wilcoxon_statistic
                    ),
                    "wilcoxon_p": wilcoxon_p,
                    "permutation_mean_difference": (
                        observed_difference
                    ),
                    "permutation_p_two_sided": (
                        permutation_p
                    ),
                    "mean_fisher_z_difference": (
                        fisher_differences.mean()
                    ),
                }
            )

    result = pd.DataFrame(rows)

    if not result.empty:
        result["permutation_p_fdr"] = fdr_bh(
            result[
                "permutation_p_two_sided"
            ].to_numpy()
        )

        result["wilcoxon_p_fdr"] = fdr_bh(
            result[
                "wilcoxon_p"
            ].fillna(1.0).to_numpy()
        )

    return result


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def save_results(
    rsa_table: pd.DataFrame,
    one_model_results: pd.DataFrame,
    paired_results: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rsa_table.to_csv(
        OUTPUT_DIR / "subject_rsa_values.csv",
        index=False,
    )

    one_model_results.to_csv(
        OUTPUT_DIR / "model_significance_tests.csv",
        index=False,
    )

    paired_results.to_csv(
        OUTPUT_DIR / "paired_model_comparisons.csv",
        index=False,
    )

    summary = {
        "fmri_file": str(FMRI_FILE),
        "rdm_root": str(RDM_ROOT),
        "selected_models": SELECTED_MODELS,
        "selected_layers": SELECTED_LAYERS,
        "number_of_permutations": N_PERMUTATIONS,
        "alpha": ALPHA,
        "primary_comparison_test": (
            "paired subject-level sign-flip permutation test"
        ),
        "multiple_comparison_correction": (
            "Benjamini-Hochberg FDR"
        ),
        "difference_direction": (
            "positive means model_b has higher IT RSA than model_a"
        ),
        "files": {
            "subject_rsa_values": (
                "subject_rsa_values.csv"
            ),
            "individual_model_tests": (
                "model_significance_tests.csv"
            ),
            "paired_model_tests": (
                "paired_model_comparisons.csv"
            ),
        },
    }

    with (
        OUTPUT_DIR / "analysis_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("IT RSA analysis")
    print("=" * 72)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    subject_it_rdms = load_it_rdms(
        FMRI_FILE
    )

    discovered_paths = discover_model_rdms(
        RDM_ROOT
    )

    print("\nDiscovered RDMs:")

    for model_name, layers in discovered_paths.items():
        for layer_name, rdm_path in layers.items():
            print(
                f"  {model_name:12s} | "
                f"{layer_name:20s} | "
                f"{rdm_path.name}"
            )

    model_rdms = load_model_rdms(
        discovered_paths
    )

    rsa_table = calculate_subject_rsa(
        model_rdms=model_rdms,
        subject_it_rdms=subject_it_rdms,
    )

    individual_results = one_model_statistics(
        rsa_table=rsa_table,
        rng=rng,
    )

    paired_results = paired_model_statistics(
        rsa_table=rsa_table,
        rng=rng,
    )

    save_results(
        rsa_table=rsa_table,
        one_model_results=individual_results,
        paired_results=paired_results,
    )

    print("\n" + "=" * 72)
    print("Mean subject-level RSA")
    print("=" * 72)

    mean_table = (
        rsa_table
        .groupby(
            ["model", "layer"]
        )["rho"]
        .agg(
            ["mean", "std", "median"]
        )
        .reset_index()
    )

    print(
        mean_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 72)
    print("Paired model comparisons")
    print("=" * 72)

    if paired_results.empty:
        print(
            "No matching model/layer comparisons were available."
        )
    else:
        display_columns = [
            "layer",
            "model_a",
            "model_b",
            "model_a_mean_rho",
            "model_b_mean_rho",
            "mean_rho_difference",
            "difference_ci95_low",
            "difference_ci95_high",
            "permutation_p_two_sided",
            "permutation_p_fdr",
        ]

        print(
            paired_results[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.5f}",
            )
        )

    print("\n" + "=" * 72)
    print("RSA analysis complete")
    print(f"Results saved under:\n{OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()