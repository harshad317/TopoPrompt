from __future__ import annotations

import math
import random
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Any

import orjson


def build_significance_summary(
    *,
    label_a: str,
    label_b: str,
    program_a_id: str,
    program_b_id: str,
    sample_count: int,
    repeat_results: list[dict[str, Any]],
    confidence_level: float = 0.95,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    repeat_summaries = [
        build_repeat_significance(
            repeat_metrics=row,
            sample_count=sample_count,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + int(row["repeat_index"]),
        )
        for row in repeat_results
    ]
    delta_values = [float(row["score_delta_a_minus_b"]) for row in repeat_summaries]
    # McNemar p-values are None for continuous metrics; filter before aggregating.
    p_values = [float(row["mcnemar_exact_p_value"]) for row in repeat_summaries if row["mcnemar_exact_p_value"] is not None]

    return {
        "label_a": label_a,
        "label_b": label_b,
        "program_a_id": program_a_id,
        "program_b_id": program_b_id,
        "sample_count": sample_count,
        "repeats": len(repeat_summaries),
        "confidence_level": confidence_level,
        "bootstrap_samples": bootstrap_samples,
        "score_delta_a_minus_b_mean": mean(delta_values),
        "score_delta_a_minus_b_std": _std(delta_values),
        "mcnemar_exact_p_value_mean": mean(p_values) if p_values else None,
        "mcnemar_exact_p_value_min": min(p_values) if p_values else None,
        "compiled_better_repeats": sum(1 for row in repeat_summaries if row["score_delta_a_minus_b"] > 0.0),
        "significant_repeats_p_lt_0_05": sum(
            1 for row in repeat_summaries
            if row["mcnemar_exact_p_value"] is not None and row["mcnemar_exact_p_value"] < 0.05
        ),
        "repeat_results": repeat_summaries,
    }


def build_repeat_significance(
    *,
    repeat_metrics: dict[str, Any],
    sample_count: int,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    a_only = int(repeat_metrics["a_only_positive_count"])
    b_only = int(repeat_metrics["b_only_positive_count"])
    both_positive = int(repeat_metrics["both_positive_count"])
    both_zero = int(repeat_metrics["both_zero_count"])
    discordant = a_only + b_only
    binary_supported = _is_binary_accuracy_row(
        repeat_metrics=repeat_metrics,
        sample_count=sample_count,
        a_only=a_only,
        b_only=b_only,
        both_positive=both_positive,
    )

    ci_low = None
    ci_high = None
    probability_a_beats_b = None
    if binary_supported and bootstrap_samples > 0:
        ci_low, ci_high, probability_a_beats_b = _bootstrap_accuracy_delta_ci(
            a_only=a_only,
            b_only=b_only,
            ties=both_positive + both_zero,
            sample_count=sample_count,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )

    score_a = float(repeat_metrics["score_a"])
    score_b = float(repeat_metrics["score_b"])
    delta_a_minus_b = score_a - score_b

    # McNemar's test is only valid for binary (0/1) outcomes. For continuous
    # metrics the test is meaningless, so we emit None rather than a misleading
    # p-value.
    if binary_supported:
        mcnemar_exact_p = _mcnemar_exact_p_value(a_only=a_only, b_only=b_only)
        mcnemar_chi_square_p = _mcnemar_chi_square_p_value(a_only=a_only, b_only=b_only)
    else:
        mcnemar_exact_p = None
        mcnemar_chi_square_p = None

    return {
        "repeat_index": int(repeat_metrics["repeat_index"]),
        "score_a": score_a,
        "score_b": score_b,
        "score_delta_a_minus_b": delta_a_minus_b,
        "relative_lift_over_b": (delta_a_minus_b / score_b) if score_b else None,
        "a_only_positive_count": a_only,
        "b_only_positive_count": b_only,
        "both_positive_count": both_positive,
        "both_zero_count": both_zero,
        "discordant_pair_count": discordant,
        "binary_accuracy_supported": binary_supported,
        "mcnemar_exact_p_value": mcnemar_exact_p,
        "mcnemar_chi_square_p_value": mcnemar_chi_square_p,
        "bootstrap_delta_a_minus_b_ci_low": ci_low,
        "bootstrap_delta_a_minus_b_ci_high": ci_high,
        "bootstrap_probability_a_beats_b": probability_a_beats_b,
    }


def summarize_significance_from_compare_dir(
    compare_dir: str | Path,
    *,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 0,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    compare_path = Path(compare_dir)
    summary = orjson.loads((compare_path / "compare_summary.json").read_bytes())
    repeat_results = [
        orjson.loads(line)
        for line in (compare_path / "repeat_metrics.jsonl").read_bytes().splitlines()
        if line.strip()
    ]
    significance = build_significance_summary(
        label_a=summary["label_a"],
        label_b=summary["label_b"],
        program_a_id=summary["program_a_id"],
        program_b_id=summary["program_b_id"],
        sample_count=int(summary["sample_count"]),
        repeat_results=repeat_results,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )

    target_dir = Path(output_dir) if output_dir is not None else compare_path
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_json(target_dir / "significance_summary.json", significance)
    (target_dir / "significance_summary.md").write_text(render_significance_summary(significance))
    return significance


def render_significance_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# TopoPrompt Significance Summary",
        "",
        f"- Program A: `{summary['label_a']}` / `{summary['program_a_id']}`",
        f"- Program B: `{summary['label_b']}` / `{summary['program_b_id']}`",
        f"- Sample count: `{summary['sample_count']}`",
        f"- Repeats: `{summary['repeats']}`",
        f"- Delta (A - B) mean/std: `{summary['score_delta_a_minus_b_mean']:.4f}` / `{summary['score_delta_a_minus_b_std']:.4f}`",
        f"- McNemar exact p-value mean: `{summary['mcnemar_exact_p_value_mean']:.6f}`" if summary["mcnemar_exact_p_value_mean"] is not None else "- McNemar exact p-value mean: `n/a (continuous metric)`",
        f"- McNemar exact p-value min: `{summary['mcnemar_exact_p_value_min']:.6f}`" if summary["mcnemar_exact_p_value_min"] is not None else "- McNemar exact p-value min: `n/a (continuous metric)`",
        f"- Significant repeats (p < 0.05): `{summary['significant_repeats_p_lt_0_05']}`",
        "",
        "## Per Repeat",
        "",
    ]
    for row in summary["repeat_results"]:
        ci_fragment = "n/a"
        if row["bootstrap_delta_a_minus_b_ci_low"] is not None and row["bootstrap_delta_a_minus_b_ci_high"] is not None:
            ci_fragment = (
                f"[{row['bootstrap_delta_a_minus_b_ci_low']:.4f}, "
                f"{row['bootstrap_delta_a_minus_b_ci_high']:.4f}]"
            )
        lines.append(
            (
                f"- Repeat {row['repeat_index']}: "
                f"delta={row['score_delta_a_minus_b']:.4f}, "
                f"discordant={row['discordant_pair_count']}, "
                f"p_exact={row['mcnemar_exact_p_value']:.6f}, " if row["mcnemar_exact_p_value"] is not None else "p_exact=n/a, "
                f"bootstrap_ci={ci_fragment}"
            )
        )
    return "\n".join(lines) + "\n"


def _is_binary_accuracy_row(
    *,
    repeat_metrics: dict[str, Any],
    sample_count: int,
    a_only: int,
    b_only: int,
    both_positive: int,
) -> bool:
    score_a = float(repeat_metrics["score_a"])
    score_b = float(repeat_metrics["score_b"])
    expected_a = both_positive + a_only
    expected_b = both_positive + b_only
    return (
        abs(score_a * sample_count - expected_a) < 1e-9
        and abs(score_b * sample_count - expected_b) < 1e-9
    )


def _bootstrap_accuracy_delta_ci(
    *,
    a_only: int,
    b_only: int,
    ties: int,
    sample_count: int,
    confidence_level: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[float, float, float]:
    # Reconstruct per-example (score_a, score_b) pairs from concordance counts.
    # a_only  → A correct, B wrong  → (1, 0)
    # b_only  → B correct, A wrong  → (0, 1)
    # ties (both_positive + both_zero) split into equal halves:
    #   both_positive → (1, 1),  both_zero → (0, 0)
    # We don't know the exact split of `ties` here so we treat all as (0, 0)
    # for a conservative estimate (delta = 0 for ties either way).
    pairs: list[tuple[int, int]] = (
        [(1, 0)] * a_only
        + [(0, 1)] * b_only
        + [(0, 0)] * ties
    )
    rng = random.Random(bootstrap_seed)
    n = len(pairs)
    estimates: list[float] = []
    positive = 0
    for _ in range(bootstrap_samples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        delta = sum(pa - pb for pa, pb in sample) / n
        estimates.append(delta)
        if delta > 0:
            positive += 1
    estimates.sort()
    lower_q = (1.0 - confidence_level) / 2.0
    upper_q = 1.0 - lower_q
    lower_index = max(0, min(bootstrap_samples - 1, int(math.floor(lower_q * (bootstrap_samples - 1)))))
    upper_index = max(0, min(bootstrap_samples - 1, int(math.ceil(upper_q * (bootstrap_samples - 1)))))
    return estimates[lower_index], estimates[upper_index], positive / bootstrap_samples


def _mcnemar_exact_p_value(*, a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = min(a_only, b_only)
    log_terms = [_log_binomial_half_pmf(discordant, value) for value in range(tail + 1)]
    log_tail_probability = _logsumexp(log_terms)
    return min(1.0, 2.0 * math.exp(log_tail_probability))


def _mcnemar_chi_square_p_value(*, a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    statistic = ((abs(a_only - b_only) - 1.0) ** 2) / discordant
    return math.erfc(math.sqrt(statistic / 2.0))


def _log_binomial_half_pmf(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - (n * math.log(2.0))


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return stdev(values)


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
