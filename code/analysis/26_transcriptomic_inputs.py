"""Transcriptomic derivation of the four inputs and their link to the proliferative core.
Manuscript: Results 2.1, Figure 1, Methods 4.1, Supplementary Table S1.

The selection of KLRB1, IRF5, STAT6 and TSC22D3 from the TCGA-UCEC cohort was made with the R scripts in
bioinformatics/ (limma, univariate Cox regression, penalised Cox regression, stepwise Cox regression). This script
starts from the two cohort tables written by those scripts (data/signature/) and
  1. recomputes the risk score with the published coefficients, the training-median cutoff, Kaplan-Meier curves,
     log-rank tests, hazard ratios between risk groups and concordance indices in both cohorts;
  2. compares Hallmark scores (mean z-score of log2 RSEM of the set members; data/TCGA_UCEC_502_hallmark_scores.csv)
     between high-risk and low-risk tumours (Mann-Whitney U, two-sided, Benjamini-Hochberg across the four sets);
  3. tabulates risk groups inside and outside the seven contexts that contain a proliferative attractor.
Outputs: results/tables/L1_selection_funnel.csv, L2_km_curves.csv, L3_risk_group_tests.csv,
         L4_hallmark_by_risk.csv, L5_risk_by_proliferative_context.csv
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy.stats import false_discovery_control, fisher_exact, mannwhitneyu

from paths import DATA, TAB, need

warnings.filterwarnings("ignore")

COEF = {"KLRB1": -0.805056, "IRF5": 0.584826, "STAT6": -0.585980, "TSC22D3": 0.331543}
HALLMARK = ["E2F_TARGETS", "MYC_TARGETS_V1", "IL6_JAK_STAT3_SIGNALING", "EPITHELIAL_MESENCHYMAL_TRANSITION"]

# 1. selection funnel (counts reported by the R scripts in bioinformatics/)
funnel = pd.DataFrame(
    [
        ("differentially expressed genes (tumour vs normal)", 7977, "01_differential_expression.R"),
        ("immune-related (ImmPort)", 759, "01_differential_expression.R"),
        ("associated with overall survival (univariate Cox)", 108, "02_univariate_cox_volcano.R"),
        ("retained by penalised Cox regression", 24, "03_penalised_cox.R"),
        ("retained by stepwise Cox regression", 4, "04_stepwise_cox.R"),
    ],
    columns=["step", "genes", "script"],
)
funnel.to_csv(os.path.join(TAB, "L1_selection_funnel.csv"), index=False)


def cohort(name):
    d = pd.read_csv(need(os.path.join(DATA, "signature", name)))
    d.columns = ["sample", "time", "status"] + list(d.columns[3:])
    d["score"] = sum(c * d[g] for g, c in COEF.items())
    return d


train, test = cohort("Training_set_raw.csv"), cohort("Validation_set_raw.csv")
cut = train.score.median()
curves, tests = [], []
for name, d in (("training", train), ("testing", test)):
    d["high"] = (d.score > cut).astype(int)
    for g, lab in ((1, "high risk"), (0, "low risk")):
        s = d[d.high == g]
        km = KaplanMeierFitter().fit(s.time, s.status)
        sf = km.survival_function_.reset_index()
        sf.columns = ["time", "survival"]
        ci = km.confidence_interval_.reset_index(drop=True)
        sf["lower"], sf["upper"] = ci.iloc[:, 0], ci.iloc[:, 1]
        sf["cohort"], sf["group"], sf["n"] = name, lab, len(s)
        curves.append(sf)
    lr = logrank_test(d.time[d.high == 1], d.time[d.high == 0], d.status[d.high == 1], d.status[d.high == 0])
    cph = CoxPHFitter().fit(d[["time", "status", "high"]], "time", "status")
    s = cph.summary.loc["high"]
    five = {}
    for g in (1, 0):
        km = KaplanMeierFitter().fit(d.time[d.high == g], d.status[d.high == g])
        five[g] = float(km.predict(5.0))
    tests.append(
        {
            "cohort": name,
            "n": len(d),
            "deaths": int(d.status.sum()),
            "n_high": int(d.high.sum()),
            "cutoff_training_median": cut,
            "HR_high_vs_low": s["exp(coef)"],
            "HR_lower_95": s["exp(coef) lower 95%"],
            "HR_upper_95": s["exp(coef) upper 95%"],
            "log_rank_P": lr.p_value,
            "C_index": concordance_index(d.time, -d.score, d.status),
            "OS5_high": five[1],
            "OS5_low": five[0],
        }
    )
pd.concat(curves).to_csv(os.path.join(TAB, "L2_km_curves.csv"), index=False)
T = pd.DataFrame(tests)
T.to_csv(os.path.join(TAB, "L3_risk_group_tests.csv"), index=False)
print(T.round(4).to_string())

# 2. Hallmark scores by risk group
M = pd.read_csv(need(os.path.join(DATA, "TCGA_UCEC_patient_to_Boolean_context_mapping.csv")))
H = pd.read_csv(need(os.path.join(DATA, "TCGA_UCEC_502_hallmark_scores.csv"))).merge(
    M[["sample", "Risk_group", "Context"]], on="sample"
)
rows = []
for h in HALLMARK:
    a, b = H.loc[H.Risk_group == "High", h], H.loc[H.Risk_group == "Low", h]
    u = mannwhitneyu(a, b, alternative="two-sided")
    rows.append(
        {
            "hallmark": h,
            "n_high": len(a),
            "n_low": len(b),
            "median_high": a.median(),
            "median_low": b.median(),
            "rank_biserial": 2 * u.statistic / (len(a) * len(b)) - 1,
            "P": u.pvalue,
        }
    )
L4 = pd.DataFrame(rows)
L4["FDR"] = false_discovery_control(L4.P)
L4.to_csv(os.path.join(TAB, "L4_hallmark_by_risk.csv"), index=False)
H.to_csv(os.path.join(TAB, "L4b_hallmark_scores_with_groups.csv"), index=False)
print(L4.round(4).to_string())

# 3. risk groups and proliferative contexts
K = pd.read_csv(need(os.path.join(TAB, "K1_context_landscape.csv")))
prolif = set(K.loc[K.n_proliferative_attractors > 0, "context"])
M["proliferative_context"] = M.Context.isin(prolif)
tab = pd.crosstab(M.proliferative_context, M.Risk_group)
odds, p = fisher_exact([[tab.loc[True, "High"], tab.loc[True, "Low"]], [tab.loc[False, "High"], tab.loc[False, "Low"]]])
L5 = tab.reset_index()
L5["fisher_odds_ratio_high_in_proliferative"] = odds
L5["fisher_P"] = p
L5.to_csv(os.path.join(TAB, "L5_risk_by_proliferative_context.csv"), index=False)
print(L5.to_string())
