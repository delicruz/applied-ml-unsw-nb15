from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORT_DIR = ROOT / "deliverables" / "final_report"
FIGURE_DIR = REPORT_DIR / "figures"


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class ProgressionPoint:
    label: str
    average_precision: float
    mcc: float
    mcc_interval: Interval | None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required experiment artifact is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _one(rows: Iterable[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key, "")) == str(value) for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def _metric_long(path: Path, *, model: str, dataset: str | None = None) -> dict[str, float]:
    rows = _read_csv(path)
    selected = [
        row
        for row in rows
        if row.get("model") == model and (dataset is None or row.get("dataset") == dataset)
    ]
    if not selected:
        raise ValueError(f"No metrics found for model={model!r}, dataset={dataset!r} in {path}")
    return {row["metric"]: float(row["value"]) for row in selected}


def _intervals(path: Path, *, model: str, dataset: str) -> dict[str, Interval]:
    rows = _read_csv(path)
    selected = [
        row for row in rows if row.get("model") == model and row.get("dataset") == dataset
    ]
    if not selected:
        raise ValueError(f"No intervals found for model={model!r}, dataset={dataset!r} in {path}")
    return {
        row["metric"]: Interval(
            estimate=float(row["estimate"]),
            low=float(row["ci_lower"]),
            high=float(row["ci_upper"]),
        )
        for row in selected
    }


def load_progression_points() -> list[ProgressionPoint]:
    robustness = _read_csv(ARTIFACTS / "robustness_audit" / "robustness_metrics.csv")
    validation = _one(
        robustness,
        evaluation_scope="current_validation",
        variant="without_all_ttl",
    )

    holdout_rows = _read_csv(
        ARTIFACTS / "final_holdout_evaluation" / "holdout_metrics.csv"
    )
    holdout = _one(holdout_rows, scope="conservative_primary")
    holdout_intervals = _read_csv(
        ARTIFACTS / "final_holdout_evaluation" / "holdout_bootstrap_intervals.csv"
    )
    holdout_mcc = _one(holdout_intervals, scope="conservative_primary", metric="mcc")

    ton_metrics = _metric_long(
        ARTIFACTS / "external_netflow_validation" / "metrics_long.csv",
        model="random_forest",
        dataset="external",
    )
    ton_intervals = _intervals(
        ARTIFACTS / "external_netflow_validation" / "group_bootstrap_intervals.csv",
        model="random_forest",
        dataset="external",
    )

    bot_metrics = _metric_long(
        ARTIFACTS / "third_dataset_validation" / "metrics_long.csv",
        model="random_forest",
    )
    bot_intervals = _intervals(
        ARTIFACTS / "third_dataset_validation" / "group_bootstrap_intervals.csv",
        model="random_forest",
        dataset="NF-BoT-IoT",
    )

    cse_metrics = _metric_long(
        ARTIFACTS / "fresh_confirmation_evaluation" / "metrics_long.csv",
        model="target_rf_all_features",
    )
    cse_intervals = _intervals(
        ARTIFACTS / "fresh_confirmation_evaluation" / "group_bootstrap_intervals.csv",
        model="target_rf_all_features",
        dataset="sealed_confirmation",
    )

    return [
        ProgressionPoint(
            "UNSW\nvalidation",
            float(validation["average_precision"]),
            float(validation["mcc"]),
            None,
        ),
        ProgressionPoint(
            "UNSW\nholdout",
            float(holdout["average_precision"]),
            float(holdout["mcc"]),
            Interval(
                estimate=float(holdout_mcc["bootstrap_mean"]),
                low=float(holdout_mcc["ci_95_low"]),
                high=float(holdout_mcc["ci_95_high"]),
            ),
        ),
        ProgressionPoint(
            "NF-ToN-IoT",
            ton_metrics["average_precision"],
            ton_metrics["mcc"],
            ton_intervals["mcc"],
        ),
        ProgressionPoint(
            "NF-BoT-IoT",
            bot_metrics["average_precision"],
            bot_metrics["mcc"],
            bot_intervals["mcc"],
        ),
        ProgressionPoint(
            "NF-CSE\nconfirmation",
            cse_metrics["average_precision"],
            cse_metrics["mcc"],
            cse_intervals["mcc"],
        ),
    ]


def load_v2_v3_ranges() -> dict[str, dict[str, tuple[float, float, float]]]:
    specifications = {
        "V2 weighted LR": (
            ARTIFACTS / "v2_generalisation_development" / "fold_metrics.csv",
            "pooled_lr_domain_family_weighted_all_features",
        ),
        "V3 derived LR": (
            ARTIFACTS / "v3_group_balanced_development" / "fold_metrics.csv",
            "group_capped_lr_derived_features",
        ),
    }
    metric_names = ("mcc", "balanced_accuracy", "recall", "false_positive_rate")
    output: dict[str, dict[str, tuple[float, float, float]]] = {}
    for label, (path, candidate) in specifications.items():
        rows = [row for row in _read_csv(path) if row.get("candidate") == candidate]
        if len(rows) != 4:
            raise ValueError(f"Expected four held-out domains for {candidate}, found {len(rows)}")
        output[label] = {}
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            output[label][metric] = (fmean(values), min(values), max(values))
    return output


def _set_cell_fill(cell, fill: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def _set_cell_border(cell, color: str = "6B7280", size: str = "8") -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right"):
        tag = f"w:{edge}"
        node = borders.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), size)
        node.set(qn("w:color"), color)


def add_editable_model_development_flow(doc) -> None:
    """Add Figure 1 as editable Word table cells rather than an embedded image."""
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    stages = [
        ("VERSION 1", "Source models\nand frozen transfer", "EAF2F8"),
        ("AUDIT", "Shortcut, duplicate\nand shift evidence", "F2F2F2"),
        ("VERSION 2", "Weighted\nmultidomain models", "FFF2E6"),
        ("VERSION 3", "Group caps and\nderived features", "E8F4F3"),
        ("DECISION", "No candidate\npromoted", "F9EAEA"),
    ]
    table = doc.add_table(rows=1, cols=9)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = [1.13, 0.20, 1.13, 0.20, 1.13, 0.20, 1.13, 0.20, 1.13]
    row = table.rows[0]
    cant_split = OxmlElement("w:cantSplit")
    row._tr.get_or_add_trPr().append(cant_split)

    stage_index = 0
    for index, cell in enumerate(row.cells):
        cell.width = Inches(widths[index])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        tc_pr = cell._tc.get_or_add_tcPr()
        tc_w = tc_pr.find(qn("w:tcW"))
        if tc_w is not None:
            tc_w.set(qn("w:w"), str(int(widths[index] * 1440)))
            tc_w.set(qn("w:type"), "dxa")
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(3)
        paragraph.paragraph_format.space_after = Pt(3)
        if index % 2:
            arrow = paragraph.add_run("→")
            arrow.font.name = "Times New Roman"
            arrow.font.size = Pt(13)
            continue

        heading, body, fill = stages[stage_index]
        stage_index += 1
        _set_cell_fill(cell, fill)
        _set_cell_border(cell)
        heading_run = paragraph.add_run(heading)
        heading_run.font.name = "Calibri"
        heading_run.font.size = Pt(8.2)
        heading_run.font.bold = True
        heading_run.font.color.rgb = RGBColor(0, 0, 0)
        paragraph.add_run("\n")
        body_run = paragraph.add_run(body)
        body_run.font.name = "Calibri"
        body_run.font.size = Pt(7.2)
        body_run.font.color.rgb = RGBColor(0, 0, 0)


def generate_report_figures() -> dict[str, str]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#777777",
            "axes.linewidth": 0.7,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )

    # Figure 1 remains a rendered visual for PDF output. The DOCX builder inserts
    # the same stages as native editable Word table cells.
    fig, ax = plt.subplots(figsize=(8.4, 1.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("VERSION 1", "Source models\nand frozen transfer"),
        ("AUDIT", "Shortcut, duplicate\nand shift evidence"),
        ("VERSION 2", "Weighted\nmultidomain models"),
        ("VERSION 3", "Group caps and\nderived features"),
        ("DECISION", "No candidate\npromoted"),
    ]
    fills = ["#EAF2F8", "#F2F2F2", "#FFF2E6", "#E8F4F3", "#F9EAEA"]
    xs = np.linspace(0.02, 0.82, len(stages))
    width = 0.16
    for index, ((heading, body), x0, fill) in enumerate(zip(stages, xs, fills)):
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.24),
                width,
                0.54,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.0,
                edgecolor="#6B7280",
                facecolor=fill,
            )
        )
        ax.text(x0 + width / 2, 0.64, heading, ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x0 + width / 2, 0.43, body, ha="center", va="center", fontsize=8.2, linespacing=1.15)
        if index < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x0 + width + 0.006, 0.51),
                    (xs[index + 1] - 0.006, 0.51),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.1,
                    color="#374151",
                )
            )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.05)
    figure1 = FIGURE_DIR / "figure_1_model_development_flow.png"
    fig.savefig(figure1, bbox_inches="tight")
    plt.close(fig)

    points = load_progression_points()
    x = np.arange(len(points))
    ap = np.array([point.average_precision for point in points])
    mcc = np.array([point.mcc for point in points])
    fig, ax = plt.subplots(figsize=(8.4, 4.25))
    ax.scatter(x, ap, marker="o", s=58, color="#2F75B5", label="Average Precision (ranking)", zorder=3)
    ax.scatter(x, mcc, marker="s", s=52, color="#C00000", label="Matthews Correlation Coefficient", zorder=3)
    for index, point in enumerate(points):
        if point.mcc_interval is not None:
            low = point.mcc_interval.low
            high = point.mcc_interval.high
            ax.errorbar(
                index,
                point.mcc,
                yerr=[[point.mcc - low], [high - point.mcc]],
                fmt="none",
                ecolor="#C00000",
                capsize=4,
                linewidth=1.3,
            )
    ax.axhline(0.0, color="#555555", linewidth=0.9)
    ax.set_xticks(x, [point.label for point in points])
    ax.set_ylim(-0.3, 1.05)
    ax.set_ylabel("Metric value")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=2, frameon=False)
    ax.grid(axis="y", color="#D9E2F3", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.30)
    figure2 = FIGURE_DIR / "figure_2_generalisation_progression.png"
    fig.savefig(figure2, bbox_inches="tight")
    plt.close(fig)

    ranges = load_v2_v3_ranges()
    metrics = [
        ("mcc", "MCC"),
        ("balanced_accuracy", "Balanced\naccuracy"),
        ("recall", "Recall"),
        ("false_positive_rate", "False-positive\nrate"),
    ]
    labels = ["V2 weighted LR", "V3 derived LR"]
    colors = ["#ED7D31", "#17807E"]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.4))
    for metric_index, (metric_key, title) in enumerate(metrics):
        ax = axes.flat[metric_index]
        values = [ranges[label][metric_key][0] for label in labels]
        lows = [ranges[label][metric_key][1] for label in labels]
        highs = [ranges[label][metric_key][2] for label in labels]
        for row_index, (value, low, high, color) in enumerate(
            zip(values, lows, highs, colors)
        ):
            ax.errorbar(
                value,
                row_index,
                xerr=[[value - low], [high - value]],
                fmt="o",
                markersize=7,
                color=color,
                ecolor=color,
                capsize=4,
                linewidth=2,
            )
        ax.set_yticks([0, 1], labels)
        ax.set_title(title, loc="left", color="#000000", weight="bold")
        left = min(lows) - 0.07
        right = max(highs) + 0.07
        if metric_index > 0:
            left, right = max(0, left), min(1.05, right)
        ax.set_xlim(left, right)
        if metric_index == 0:
            ax.axvline(0, color="#777777", linewidth=0.8)
        ax.grid(axis="x", color="#D9E2F3", linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.text(
        0.03,
        0.01,
        "Dots are domain means; horizontal lines show minimum-to-maximum domain ranges (not confidence intervals).",
        fontsize=9,
        color="#000000",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    figure4 = FIGURE_DIR / "figure_4_v2_v3_ranges.png"
    fig.savefig(figure4, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "figure_1_model_development_flow.png": {
            "type": "conceptual project-flow diagram",
            "docx_representation": "native editable Word table cells",
        },
        "figure_2_generalisation_progression.png": {
            "type": "generated from experiment artifacts",
            "sources": [
                "artifacts/robustness_audit/robustness_metrics.csv",
                "artifacts/final_holdout_evaluation/holdout_metrics.csv",
                "artifacts/final_holdout_evaluation/holdout_bootstrap_intervals.csv",
                "artifacts/external_netflow_validation/metrics_long.csv",
                "artifacts/external_netflow_validation/group_bootstrap_intervals.csv",
                "artifacts/third_dataset_validation/metrics_long.csv",
                "artifacts/third_dataset_validation/group_bootstrap_intervals.csv",
                "artifacts/fresh_confirmation_evaluation/metrics_long.csv",
                "artifacts/fresh_confirmation_evaluation/group_bootstrap_intervals.csv",
            ],
        },
        "figure_4_v2_v3_ranges.png": {
            "type": "computed directly from held-out-domain fold metrics",
            "sources": [
                "artifacts/v2_generalisation_development/fold_metrics.csv",
                "artifacts/v3_group_balanced_development/fold_metrics.csv",
            ],
        },
    }
    provenance_path = REPORT_DIR / "figure_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {
        "figure_1": str(figure1),
        "figure_2": str(figure2),
        "figure_4": str(figure4),
        "provenance": str(provenance_path),
    }
