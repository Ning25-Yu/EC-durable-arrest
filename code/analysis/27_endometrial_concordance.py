"""Comparison of the model with drug-treated endometrial cancer cells and with endometrial tumour proteomes.
Manuscript: Results 2.8, Figure 7a-c, Methods 4.7, Supplementary Table S6.

1. Endometrial cancer cell transcriptomes (GEO). GSE236228: Ishikawa cells, control or penfluridol (three samples
   each). GSE61481: AN3-CA cells, aFGF or aFGF plus the FGFR inhibitor AZD4547 (two samples each). The expression
   of the network genes was binarised per data set with the edge-detector method and compared with each of the 32
   fixed points of the network. The agreement of a sample with a fixed point is the number of mapped genes whose
   binary value equals the node value (data/GSE*_EdgeDetector_external_validation.xlsx, sheet All_EBP_scores).
   For a phenotype, the margin of a sample is the best agreement among fixed points with the phenotype ON minus
   the best agreement among fixed points with it OFF. A negative shift of the Proliferation margin under treatment
   means that the treated cells resemble arrested fixed points more closely.
2. CPTAC-UCEC proteomes (95 tumours). Each tumour is assigned an input context by binarising KLRB1, IRF5, STAT6 and
   TSC22D3 at their within-cohort medians (log2 RSEM). Protein and pathway read-outs are compared between tumours in
   the seven contexts that contain a proliferative attractor and tumours in the other nine contexts (Mann-Whitney U,
   two-sided, Benjamini-Hochberg across the external read-outs).
Outputs: results/tables/M1_cell_line_margins.csv, M1b_cell_line_summary.csv, M2_cptac_contexts.csv,
         M3_cptac_readouts.csv
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, fisher_exact, mannwhitneyu

from paths import DATA, TAB, need

INPUTS = ["KLRB1", "IRF5", "STAT6", "TSC22D3"]

# 1. cell lines: recompute margins from the agreement scores
rows = []
sets = [
    ("GSE236228", "Ishikawa", {"Control": "control", "Penfluridol": "penfluridol"}, ["Proliferation", "Apoptosis"]),
    ("GSE61481", "AN3-CA", {"aFGF": "aFGF", "aFGF_AZD4547": "aFGF + AZD4547"}, ["Proliferation"]),
]
for gse, line, groups, phen in sets:
    S = pd.read_excel(need(os.path.join(DATA, f"{gse}_EdgeDetector_external_validation.xlsx")), sheet_name="All_EBP_scores")
    for (sample, group), g in S.groupby(["sample", "group"], sort=False):
        for p in phen:
            on, off = g.loc[g[p] == 1, "EBP_score"].max(), g.loc[g[p] == 0, "EBP_score"].max()
            rows.append({"dataset": gse, "cell_line": line, "sample": sample, "group": groups[group],
                         "treated": group != list(groups)[0], "phenotype": p,
                         "best_ON": on, "best_OFF": off, "margin": on - off,
                         "n_fixed_points": g.stable_state_id.nunique()})
M1 = pd.DataFrame(rows)
M1.to_csv(os.path.join(TAB, "M1_cell_line_margins.csv"), index=False)
summ = (M1.groupby(["dataset", "cell_line", "phenotype", "treated"]).margin.agg(["mean", "count", list]).reset_index())
wide = summ.pivot_table(index=["dataset", "cell_line", "phenotype"], columns="treated", values="mean").reset_index()
wide.columns = ["dataset", "cell_line", "phenotype", "mean_margin_control", "mean_margin_treated"]
wide["shift"] = wide.mean_margin_treated - wide.mean_margin_control
wide.to_csv(os.path.join(TAB, "M1b_cell_line_summary.csv"), index=False)
print(wide.round(3).to_string())

# 2. CPTAC contexts and read-outs
E = pd.read_csv(need(os.path.join(DATA, "CPTAC_UCEC_95_network_gene_log2RSEM.csv")))
A = pd.read_csv(need(os.path.join(DATA, "CPTAC_UCEC_95_annotation.csv")))
b = (E[INPUTS] > E[INPUTS].median()).astype(int)
E["Context"] = "C" + b.astype(str).agg("".join, axis=1)
K = pd.read_csv(need(os.path.join(TAB, "K1_context_landscape.csv")))
prolif = set(K.loc[K.n_proliferative_attractors > 0, "context"])
C = E[["sample", "Context"]].merge(A, on="sample")
C["proliferative_context"] = C.Context.isin(prolif)
pro = ["prot_MKI67", "prot_MCM2", "prot_TOP2A", "prot_PCNA", "prot_CCNB1"]
Z = C[pro].apply(lambda x: (x - x.mean()) / x.std())
C["protein_proliferation_index"] = Z.mean(axis=1, skipna=True)
C.to_csv(os.path.join(TAB, "M2_cptac_contexts.csv"), index=False)

reads = ["prot_CDKN1B", "prot_CDKN1A", "protein_proliferation_index"] + pro + [
    "prot_STAT3", "PATHWAY_ACTIVITY_PI3K", "PATHWAY_ACTIVITY_P53", "PATHWAY_ACTIVITY_JAK.STAT",
    "PATHWAY_ACTIVITY_NFKB", "PATHWAY_ACTIVITY_TNFA", "STEMNESS_SCORE", "ESTIMATE_IMMUNESCORE", "PURITY_CANCER"]
out = []
for r in reads:
    a = C.loc[C.proliferative_context, r].dropna()
    c = C.loc[~C.proliferative_context, r].dropna()
    u = mannwhitneyu(a, c, alternative="two-sided")
    out.append({"readout": r, "n_proliferative": len(a), "n_other": len(c), "median_proliferative": a.median(),
                "median_other": c.median(), "rank_biserial": 2 * u.statistic / (len(a) * len(c)) - 1, "P": u.pvalue})
M3 = pd.DataFrame(out)
M3["FDR"] = false_discovery_control(M3.P)
for lab, col, val in (("CN-high subtype", "genomic_subtype", "CNV_high"), ("TP53 mutant", "TP53_mut", 1)):
    x = C[col] == val
    t = [[(x & C.proliferative_context).sum(), (~x & C.proliferative_context).sum()],
         [(x & ~C.proliferative_context).sum(), (~x & ~C.proliferative_context).sum()]]
    o, p = fisher_exact(t)
    M3 = pd.concat([M3, pd.DataFrame([{"readout": lab, "n_proliferative": int(C.proliferative_context.sum()),
                                        "n_other": int((~C.proliferative_context).sum()),
                                        "median_proliferative": t[0][0] / sum(t[0]), "median_other": t[1][0] / sum(t[1]),
                                        "rank_biserial": np.nan, "P": p, "FDR": np.nan,
                                        "note": "proportions; Fisher exact test, odds ratio %.2f" % o}])])
M3.to_csv(os.path.join(TAB, "M3_cptac_readouts.csv"), index=False)
print(C.proliferative_context.value_counts().to_dict())
print(M3.round(4).to_string())
