"""Create publication-style visualizations for the Cichy IT RSA analysis.

Figures created:

1. Standard, expert, and mean IT RDMs for each selected layer.
2. Paired subject-level RSA correlations for both models.
3. Paired expert-minus-standard RSA differences.
4. Summary figure showing mean RSA with 95% confidence intervals.

Run from the project root:

    python -m cichy_data_scripts.evals.visualize_rsa
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

from ..config import CHECKPOINT_DIR, CICHY_DATA


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RDM_ROOT = (
    CHECKPOINT_DIR.parent
    / "rdms"
    / "cichy_data_models"
)

RSA_DIR = RDM_ROOT / "rsa_analysis"

SUBJECT_RSA_FILE = (
    RSA_DIR
    / "subject_rsa_values.csv"
)

PAIRED_COMPARISONS_FILE = (
    RSA_DIR
    / "paired_model_comparisons.csv"
)

FMRI_FILE = (
    CICHY_DATA
    / "target_fmri.mat"
)

FIGURE_DIR = (
    RSA_DIR
    / "figures"
)


# ---------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------

MODELS = [
    "standard",
    "expert",
]

LAYERS = [
    "fc7_post_relu"
]

MODEL_DISPLAY_NAMES = {
    "standard": "Standard AlexNet",
    "expert": "Mixed expert",
}

LAYER_DISPLAY_NAMES = {
    "fc7_post_relu": "FC7 post-ReLU"
}


# ---------------------------------------------------------------------
# Figure settings
# ---------------------------------------------------------------------

FIGURE_DPI = 300
EXPECTED_STIMULI = 92

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    }
)


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_mat_file(
    mat_path: Path,
) -> dict[str, np.ndarray]:
    """Load MATLAB v7 or HDF5-based MATLAB v7.3 files."""

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
    """Return IT RDMs as [subjects, stimuli, stimuli]."""

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

    if subject_rdms.shape[1:] == expected_pair:
        return subject_rdms

    if subject_rdms.shape[:2] == expected_pair:
        return np.moveaxis(
            subject_rdms,
            source=2,
            destination=0,
        )

    raise ValueError(
        "Could not determine the IT_RDMs axis order. "
        f"Received {subject_rdms.shape}."
    )


def load_mean_it_rdm() -> np.ndarray:
    """Load and average the 15 subject-level IT RDMs."""

    data = load_mat_file(
        FMRI_FILE
    )

    if "IT_RDMs" not in data:
        raise KeyError(
            "Could not find IT_RDMs in target_fmri.mat. "
            f"Available keys: {list(data.keys())}"
        )

    subject_rdms = standardize_subject_rdm_axes(
        data["IT_RDMs"]
    )

    return subject_rdms.mean(
        axis=0
    )


def load_model_rdm(
    model_name: str,
    layer_name: str,
) -> np.ndarray:
    """Load one saved model RDM."""

    rdm_path = (
        RDM_ROOT
        / model_name
        / f"rdm_{layer_name}_correlation.npy"
    )

    if not rdm_path.exists():
        raise FileNotFoundError(
            f"Model RDM not found:\n{rdm_path}"
        )

    rdm = np.load(
        rdm_path
    )

    expected_shape = (
        EXPECTED_STIMULI,
        EXPECTED_STIMULI,
    )

    if rdm.shape != expected_shape:
        raise ValueError(
            f"Expected RDM shape {expected_shape}, "
            f"but {rdm_path} has shape {rdm.shape}."
        )

    return rdm


def load_statistics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load subject RSA values and paired model comparisons."""

    if not SUBJECT_RSA_FILE.exists():
        raise FileNotFoundError(
            f"Subject RSA CSV not found:\n{SUBJECT_RSA_FILE}"
        )

    if not PAIRED_COMPARISONS_FILE.exists():
        raise FileNotFoundError(
            "Paired-comparison CSV not found:\n"
            f"{PAIRED_COMPARISONS_FILE}"
        )

    subject_rsa = pd.read_csv(
        SUBJECT_RSA_FILE
    )

    paired_comparisons = pd.read_csv(
        PAIRED_COMPARISONS_FILE
    )

    required_subject_columns = {
        "subject",
        "model",
        "layer",
        "rho",
    }

    missing_columns = (
        required_subject_columns
        - set(subject_rsa.columns)
    )

    if missing_columns:
        raise ValueError(
            "subject_rsa_values.csv is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return subject_rsa, paired_comparisons


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 20_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate a percentile bootstrap CI for the mean."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.integers(
        low=0,
        high=len(values),
        size=(
            n_bootstrap,
            len(values),
        ),
    )

    bootstrap_means = values[
        indices
    ].mean(axis=1)

    tail_probability = (
        1.0 - confidence
    ) / 2.0

    lower = np.quantile(
        bootstrap_means,
        tail_probability,
    )

    upper = np.quantile(
        bootstrap_means,
        1.0 - tail_probability,
    )

    return float(lower), float(upper)


def significance_label(
    p_value: float,
) -> str:
    """Convert a p-value into a standard significance label."""

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return "n.s."


def get_pairwise_p_value(
    paired_results: pd.DataFrame,
    layer_name: str,
) -> float:
    """Read the FDR-corrected model-comparison p-value."""

    row = paired_results[
        paired_results["layer"] == layer_name
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected exactly one paired comparison for "
            f"{layer_name}, but found {len(row)}."
        )

    return float(
        row.iloc[0]["permutation_p_fdr"]
    )


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    """Save PNG and PDF versions of a figure."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        FIGURE_DIR
        / f"{filename}.png"
    )

    pdf_path = (
        FIGURE_DIR
        / f"{filename}.pdf"
    )

    figure.savefig(
        png_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ---------------------------------------------------------------------
# RDM figures
# ---------------------------------------------------------------------

def plot_rdm_comparison(
    layer_name: str,
    mean_it_rdm: np.ndarray,
) -> None:
    """Plot standard, expert, and mean IT RDMs side by side."""

    standard_rdm = load_model_rdm(
        model_name="standard",
        layer_name=layer_name,
    )

    expert_rdm = load_model_rdm(
        model_name="expert",
        layer_name=layer_name,
    )

    model_min = min(
        standard_rdm.min(),
        expert_rdm.min(),
    )

    model_max = max(
        standard_rdm.max(),
        expert_rdm.max(),
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(14, 4.5),
    )

    standard_image = axes[0].imshow(
        standard_rdm,
        aspect="equal",
        interpolation="nearest",
        vmin=model_min,
        vmax=model_max,
        cmap="viridis",
    )

    axes[0].set_title(
        "Standard AlexNet"
    )

    expert_image = axes[1].imshow(
        expert_rdm,
        aspect="equal",
        interpolation="nearest",
        vmin=model_min,
        vmax=model_max,
        cmap="viridis",
    )

    axes[1].set_title(
        "Mixed expert"
    )

    brain_image = axes[2].imshow(
        mean_it_rdm,
        aspect="equal",
        interpolation="nearest",
        cmap="viridis",
    )

    axes[2].set_title(
        "Mean IT fMRI RDM"
    )

    for axis in axes:
        axis.set_xlabel(
            "Stimulus"
        )
        axis.set_ylabel(
            "Stimulus"
        )

    model_colorbar = figure.colorbar(
        expert_image,
        ax=axes[:2],
        fraction=0.025,
        pad=0.03,
    )

    model_colorbar.set_label(
        "Correlation distance"
    )

    brain_colorbar = figure.colorbar(
        brain_image,
        ax=axes[2],
        fraction=0.045,
        pad=0.04,
    )

    brain_colorbar.set_label(
        "fMRI dissimilarity"
    )

    figure.suptitle(
        f"Representational dissimilarity matrices: "
        f"{LAYER_DISPLAY_NAMES[layer_name]}"
    )

    figure.subplots_adjust(
        top=0.83,
        wspace=0.32,
    )

    save_figure(
        figure,
        f"rdm_comparison_{layer_name}",
    )

    plt.close(figure)


# ---------------------------------------------------------------------
# Paired RSA plot
# ---------------------------------------------------------------------

def plot_paired_rsa(
    subject_rsa: pd.DataFrame,
    paired_results: pd.DataFrame,
) -> None:
    """Plot paired subject correlations for each model and layer."""

    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(LAYERS),
        figsize=(11, 5),
        sharey=True,
    )

    if len(LAYERS) == 1:
        axes = [axes]

    model_positions = {
        "standard": 0,
        "expert": 1,
    }

    for axis, layer_name in zip(
        axes,
        LAYERS,
    ):
        layer_data = subject_rsa[
            subject_rsa["layer"] == layer_name
        ]

        pivot = layer_data.pivot(
            index="subject",
            columns="model",
            values="rho",
        )

        pivot = pivot[
            MODELS
        ].dropna()

        for _, subject_values in pivot.iterrows():
            axis.plot(
                [
                    model_positions["standard"],
                    model_positions["expert"],
                ],
                [
                    subject_values["standard"],
                    subject_values["expert"],
                ],
                marker="o",
                linewidth=0.8,
                alpha=0.45,
            )

        for model_name in MODELS:
            values = pivot[
                model_name
            ].to_numpy()

            mean_value = values.mean()

            ci_low, ci_high = bootstrap_mean_ci(
                values
            )

            axis.errorbar(
                model_positions[model_name],
                mean_value,
                yerr=[
                    [mean_value - ci_low],
                    [ci_high - mean_value],
                ],
                fmt="s",
                markersize=8,
                capsize=5,
                linewidth=2.0,
                label=MODEL_DISPLAY_NAMES[model_name],
            )

        p_value = get_pairwise_p_value(
            paired_results=paired_results,
            layer_name=layer_name,
        )

        all_values = pivot.to_numpy()

        y_max = np.nanmax(
            all_values
        )

        y_min = np.nanmin(
            all_values
        )

        y_range = max(
            y_max - y_min,
            0.05,
        )

        bracket_y = (
            y_max
            + 0.12 * y_range
        )

        bracket_height = (
            0.04 * y_range
        )

        axis.plot(
            [
                0,
                0,
                1,
                1,
            ],
            [
                bracket_y,
                bracket_y + bracket_height,
                bracket_y + bracket_height,
                bracket_y,
            ],
            linewidth=1.2,
        )

        axis.text(
            0.5,
            bracket_y + bracket_height,
            (
                f"{significance_label(p_value)}\n"
                f"$p_{{FDR}}$ = {p_value:.4g}"
            ),
            ha="center",
            va="bottom",
        )

        axis.set_xticks(
            [0, 1],
            [
                "Standard",
                "Mixed expert",
            ],
        )

        axis.set_title(
            LAYER_DISPLAY_NAMES[layer_name]
        )

        axis.set_xlim(
            -0.35,
            1.35,
        )

        axis.axhline(
            0.0,
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
        )

    axes[0].set_ylabel(
        "Spearman RSA correlation with IT"
    )

    figure.suptitle(
        "Subject-level correspondence between model and IT RDMs"
    )

    figure.tight_layout()

    save_figure(
        figure,
        "paired_subject_rsa",
    )

    plt.close(figure)


# ---------------------------------------------------------------------
# Paired-difference plot
# ---------------------------------------------------------------------

def plot_rsa_differences(
    subject_rsa: pd.DataFrame,
    paired_results: pd.DataFrame,
) -> None:
    """Plot expert-minus-standard differences for each participant."""

    differences_by_layer = {}

    for layer_name in LAYERS:
        layer_data = subject_rsa[
            subject_rsa["layer"] == layer_name
        ]

        pivot = layer_data.pivot(
            index="subject",
            columns="model",
            values="rho",
        )

        pivot = pivot[
            MODELS
        ].dropna()

        differences_by_layer[layer_name] = (
            pivot["expert"]
            - pivot["standard"]
        ).to_numpy()

    figure, axis = plt.subplots(
        figsize=(8, 5.5)
    )

    positions = np.arange(
        len(LAYERS)
    )

    boxplot_values = [
        differences_by_layer[layer_name]
        for layer_name in LAYERS
    ]

    axis.boxplot(
        boxplot_values,
        positions=positions,
        widths=0.45,
        showfliers=False,
        medianprops={
            "linewidth": 2,
        },
    )

    rng = np.random.default_rng(
        42
    )

    for position, layer_name in zip(
        positions,
        LAYERS,
    ):
        differences = differences_by_layer[
            layer_name
        ]

        jitter = rng.normal(
            loc=0.0,
            scale=0.035,
            size=len(differences),
        )

        axis.scatter(
            position + jitter,
            differences,
            alpha=0.75,
            zorder=3,
        )

        mean_difference = differences.mean()

        ci_low, ci_high = bootstrap_mean_ci(
            differences
        )

        axis.errorbar(
            position,
            mean_difference,
            yerr=[
                [mean_difference - ci_low],
                [ci_high - mean_difference],
            ],
            fmt="D",
            markersize=7,
            capsize=5,
            linewidth=2,
            zorder=4,
        )

        p_value = get_pairwise_p_value(
            paired_results=paired_results,
            layer_name=layer_name,
        )

        annotation_y = max(
            differences.max(),
            ci_high,
        )

        axis.text(
            position,
            annotation_y + 0.001,
            (
                f"{significance_label(p_value)}\n"
                f"$p_{{FDR}}$={p_value:.4g}"
            ),
            ha="center",
            va="bottom",
        )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_xticks(
        positions,
        [
            LAYER_DISPLAY_NAMES[layer]
            for layer in LAYERS
        ],
    )

    axis.set_ylabel(
        "RSA difference: mixed expert − standard"
    )

    axis.set_title(
        "Paired change in model–IT correspondence"
    )

    figure.tight_layout()

    save_figure(
        figure,
        "rsa_expert_minus_standard_differences",
    )

    plt.close(figure)


# ---------------------------------------------------------------------
# Mean summary plot
# ---------------------------------------------------------------------

def plot_mean_rsa_summary(
    subject_rsa: pd.DataFrame,
) -> None:
    """Plot model means and bootstrap confidence intervals."""

    figure, axis = plt.subplots(
        figsize=(8.5, 5.5)
    )

    layer_positions = np.arange(
        len(LAYERS)
    )

    model_offsets = {
        "standard": -0.13,
        "expert": 0.13,
    }

    for model_name in MODELS:
        means = []
        lower_errors = []
        upper_errors = []

        for layer_name in LAYERS:
            values = subject_rsa[
                (
                    subject_rsa["model"]
                    == model_name
                )
                & (
                    subject_rsa["layer"]
                    == layer_name
                )
            ]["rho"].to_numpy()

            mean_value = values.mean()

            ci_low, ci_high = bootstrap_mean_ci(
                values
            )

            means.append(
                mean_value
            )

            lower_errors.append(
                mean_value - ci_low
            )

            upper_errors.append(
                ci_high - mean_value
            )

        x_positions = (
            layer_positions
            + model_offsets[model_name]
        )

        axis.errorbar(
            x_positions,
            means,
            yerr=[
                lower_errors,
                upper_errors,
            ],
            fmt="o-",
            markersize=8,
            capsize=5,
            linewidth=2,
            label=MODEL_DISPLAY_NAMES[model_name],
        )

    axis.set_xticks(
        layer_positions,
        [
            LAYER_DISPLAY_NAMES[layer]
            for layer in LAYERS
        ],
    )

    axis.set_ylabel(
        "Mean Spearman RSA correlation with IT"
    )

    axis.set_title(
        "Model correspondence with IT across classifier layers"
    )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )

    axis.legend(
        frameon=False
    )

    figure.tight_layout()

    save_figure(
        figure,
        "mean_rsa_summary",
    )

    plt.close(figure)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("RSA visualization")
    print("=" * 72)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    subject_rsa, paired_results = (
        load_statistics()
    )

    mean_it_rdm = load_mean_it_rdm()

    print(
        f"Loaded subject RSA rows: "
        f"{len(subject_rsa)}"
    )

    print(
        f"Mean IT RDM shape: "
        f"{mean_it_rdm.shape}"
    )

    for layer_name in LAYERS:
        print(
            f"Creating RDM comparison for "
            f"{layer_name}..."
        )

        plot_rdm_comparison(
            layer_name=layer_name,
            mean_it_rdm=mean_it_rdm,
        )

    print(
        "Creating paired subject RSA plot..."
    )

    plot_paired_rsa(
        subject_rsa=subject_rsa,
        paired_results=paired_results,
    )

    print(
        "Creating paired-difference plot..."
    )

    plot_rsa_differences(
        subject_rsa=subject_rsa,
        paired_results=paired_results,
    )

    print(
        "Creating mean RSA summary..."
    )

    plot_mean_rsa_summary(
        subject_rsa=subject_rsa,
    )

    print("\n" + "=" * 72)
    print("Visualization complete")
    print(f"Figures saved under:\n{FIGURE_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()