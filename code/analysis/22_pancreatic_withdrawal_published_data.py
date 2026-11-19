"""Descriptive analysis of published pancreatic cancer drug-withdrawal data (Paulsohn et al., 2026, Figure 6F-J).
Manuscript: Figure 6D-F, Figure S15 and Table S9.

Inputs (data/published_experiments/Paulsohn2026_withdrawal/): group-mean trajectories digitised from Figure 6F-H
(FGH.csv; MIA PaCa-2 cells with scrambled siRNA or siCDKN1B), 6I (I.csv; murine wild-type or CDKN1B knockout cells),
6J (J.csv; viability at days 3 and 7) and the axis calibration of the digitisation (calibration.json).
Treatment is withdrawn at day 3. For every genotype and treatment arm the recovery fold change (last reading / day-3
reading) is compared between p27-depleted and p27-intact cells. The values are approximate published group means:
no replicate identities are available, so no statistical test is performed. The digitisation-stress bounds show how
far each ratio can move if every reading is off by 0.1, 0.25 or 0.5 PDF points; they are not confidence intervals.

Outputs (results/tables/Paulsohn2026_withdrawal/):
  all_trajectories_and_normalizations.csv  every trajectory, normalised to day 3 and to the matched vehicle
  all_group_metrics.csv                    recovery metrics of each group
  all_genotype_contrasts.csv               depleted versus intact ratios for every treatment arm
  combination_key_results.csv              the combination arms (palbociclib + KRAS inhibitor)
  digitization_stress_sensitivity.csv      digitisation-stress bounds of the ratios
  excluded_or_missing_positions.csv        digitised positions outside the analysed window"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import csv
import json
import math

from paths import PANC_DATA, PANC_TAB


def read_csv(name):
    with open(os.path.join(PANC_DATA, name), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    assert rows, name
    with open(os.path.join(PANC_TAB, name), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def area_mean(t, y):
    """Time-averaged value of a piecewise-linear curve."""
    assert len(t) == len(y) and all(b > a for a, b in zip(t, t[1:]))
    return sum((b - a) * (u + v) / 2 for a, b, u, v in zip(t, t[1:], y, y[1:])) / (t[-1] - t[0])


def ratio_bounds(numerators, denominators, epsilon):
    """Bounds of a product ratio when every reading may be off by +/- epsilon (not a confidence interval)."""
    if any(v <= epsilon for v in numerators + denominators):
        return None, None
    lo = math.prod(v - epsilon for v in numerators) / math.prod(v + epsilon for v in denominators)
    hi = math.prod(v + epsilon for v in numerators) / math.prod(v - epsilon for v in denominators)
    return lo, hi


def direction(lo, hi):
    if lo is None or lo <= 1 <= hi:
        return "undetermined"
    return "consistent" if lo > 1 else "inconsistent"


def get_trajectories():
    trajectories = {}
    raw_fgh = read_csv("FGH.csv")
    missing = [r for r in raw_fgh if not r["relative_confluence_mean_approx"]]
    assert len(raw_fgh) == 240 and len(missing) == 2
    for r in raw_fgh:
        key = (r["panel"], r["genetic_condition"], r["drug"])
        value = r["relative_confluence_mean_approx"]
        trajectories.setdefault(key, {})[int(r["day_from_treatment_start_plot"])] = (
            float(value) if value else None
        )
    for r in read_csv("I.csv"):
        key = ("I", r["genetic_condition"], r["drug"])
        trajectories.setdefault(key, {})[int(r["day"])] = float(
            r["absolute_confluence_mean_plot_units_approx"]
        )
    for r in read_csv("J.csv"):
        key = ("J", r["genetic_condition"], r["drug"])
        trajectories.setdefault(key, {})[int(r["panel"].split("_D")[1])] = float(
            r["cell_viability_mean_plot_units_approx"]
        )
    assert len(trajectories) == 40
    write_csv(
        "excluded_or_missing_positions.csv",
        [
            dict(
                panel=r["panel"],
                genetic_condition=r["genetic_condition"],
                drug=r["drug"],
                day=r["day_from_treatment_start_plot"],
                reason="Unresolved day-2 position; outside the day-3-to-end window",
            )
            for r in missing
        ],
    )

    metrics, long = [], []
    for (p, g, d), curve in sorted(trajectories.items()):
        end = 10 if p in "FGH" else 7
        days = sorted(t for t in curve if t >= 3)
        assert days[0] == 3 and days[-1] == end and all(curve[t] is not None for t in days)
        y = [curve[t] for t in days]
        control = trajectories[p, g, "DMSO"]
        fold = y[-1] / y[0]
        metrics.append(
            dict(
                panel=p,
                genetic_condition=g,
                drug=d,
                withdrawal_plot_day=3,
                end_plot_day=end,
                value_D3=y[0],
                value_end=y[-1],
                absolute_increment=y[-1] - y[0],
                recovery_fold=fold,
                vehicle_D3=control[3],
                vehicle_end=control[end],
                D3_fraction_of_matched_vehicle=y[0] / control[3],
                endpoint_fraction_of_matched_vehicle=y[-1] / control[end],
                vehicle_adjusted_recovery_fold=fold / (control[end] / control[3]),
                D3_normalized_area_mean=area_mean(days, [v / y[0] for v in y]) if p != "J" else "",
                postwithdrawal_mean_raw=area_mean(days, y) if p != "J" else "",
                units=(
                    "Relative confluence"
                    if p in "FGH"
                    else (
                        "Absolute confluence (0-100 plot scale)"
                        if p == "I"
                        else "Cell viability (plot units)"
                    )
                ),
            )
        )
        for t, v in sorted(curve.items()):
            long.append(
                dict(
                    panel=p,
                    genetic_condition=g,
                    drug=d,
                    day=t,
                    phase="postwithdrawal" if t > 3 else "D3_boundary" if t == 3 else "during_treatment",
                    published_mean_approx=v if v is not None else "",
                    normalized_to_D3=v / y[0] if v is not None and t >= 3 else "",
                    normalized_to_matched_vehicle=v / control[t] if v is not None else "",
                    source_DOI="10.1038/s41419-025-08362-w",
                )
            )
    write_csv("all_group_metrics.csv", metrics)
    write_csv("all_trajectories_and_normalizations.csv", long)
    return metrics


def compare(metrics):
    with open(os.path.join(PANC_DATA, "calibration.json"), encoding="utf-8") as f:
        calibration = json.load(f)
    index = {(r["panel"], r["genetic_condition"], r["drug"]): r for r in metrics}
    contrasts, sensitivity = [], []
    for p in ["F", "G", "H", "I", "J"]:
        intact, depleted = ("scrb", "siCDKN1B") if p in "FGH" else ("WT", "CDKN1B_KO")
        drugs = (
            ["DMSO", "Palbociclib", "Sotorasib", "Palbo_Soto"]
            if p in "FGH"
            else ["DMSO", "Palbociclib", "MRTX1133", "Palbo_MRTX"]
        )
        for d in drugs:
            a, b = index[p, intact, d], index[p, depleted, d]
            ratio = b["recovery_fold"] / a["recovery_fold"]
            contrasts.append(
                dict(
                    panel=p,
                    drug=d,
                    intact_genotype=intact,
                    depleted_genotype=depleted,
                    intact_recovery_fold=a["recovery_fold"],
                    depleted_recovery_fold=b["recovery_fold"],
                    depleted_vs_intact_recovery_fold_ratio=ratio,
                    recovery_direction_center=direction(ratio, ratio),
                    intact_endpoint_vehicle_fraction=a["endpoint_fraction_of_matched_vehicle"],
                    depleted_endpoint_vehicle_fraction=b["endpoint_fraction_of_matched_vehicle"],
                    depleted_vs_intact_endpoint_vehicle_ratio=(
                        b["endpoint_fraction_of_matched_vehicle"] / a["endpoint_fraction_of_matched_vehicle"]
                    ),
                    depleted_vs_intact_vehicle_adjusted_recovery_ratio=(
                        b["vehicle_adjusted_recovery_fold"] / a["vehicle_adjusted_recovery_fold"]
                    ),
                    depleted_vs_intact_D3_vehicle_fraction_ratio=(
                        b["D3_fraction_of_matched_vehicle"] / a["D3_fraction_of_matched_vehicle"]
                    ),
                )
            )
            for pt in [0.1, 0.25, 0.5]:
                keys = [p] if p != "J" else ["J_D3", "J_D7"]  # largest conversion slope of the two J axes
                eps = max(abs(calibration[k]["data_y_per_pdf_point"][0]) for k in keys) * pt
                ratios = {
                    "recovery_fold_ratio": ([b["value_end"], a["value_D3"]], [b["value_D3"], a["value_end"]]),
                    "endpoint_vehicle_ratio": (
                        [b["value_end"], a["vehicle_end"]],
                        [b["vehicle_end"], a["value_end"]],
                    ),
                    "vehicle_adjusted_recovery_ratio": (
                        [b["value_end"], a["value_D3"], b["vehicle_D3"], a["vehicle_end"]],
                        [b["value_D3"], a["value_end"], b["vehicle_end"], a["vehicle_D3"]],
                    ),
                }
                for name, (num, den) in ratios.items():
                    lo, hi = ratio_bounds(num, den, eps)
                    if d == "DMSO" and name != "recovery_fold_ratio":
                        lo = hi = 1.0
                    sensitivity.append(
                        dict(
                            panel=p,
                            drug=d,
                            metric=name,
                            stress_pdf_y_pt=pt,
                            stress_in_plot_units=eps,
                            lower_reading_stress_bound=lo,
                            upper_reading_stress_bound=hi,
                            directional_assessment=direction(lo, hi),
                        )
                    )
    write_csv("all_genotype_contrasts.csv", contrasts)
    write_csv("digitization_stress_sensitivity.csv", sensitivity)
    combination = [r for r in contrasts if r["drug"] in ["Palbo_Soto", "Palbo_MRTX"]]
    write_csv("combination_key_results.csv", combination)
    return combination


def main():
    metrics = get_trajectories()
    combination = compare(metrics)
    for r in combination:
        print(
            f"panel {r['panel']} {r['drug']}: recovery fold ratio (depleted / intact) "
            f"{r['depleted_vs_intact_recovery_fold_ratio']:.2f} ({r['recovery_direction_center']})"
        )


if __name__ == "__main__":
    main()
