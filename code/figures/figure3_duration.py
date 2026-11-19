"""Figure 3. Exposure duration and durable arrest after withdrawal in context C0101.
(a) protocol; (b) durable arrest probability a(L) from the closed forms (lines) and the matrix exponential (circles);
(c) exposure durations L50 and L90; (d) Proliferation OFF at the end of an L = 8 exposure against durable arrest
after withdrawal for the 13 interventions that reduce Proliferation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

import style as S
from paths import FIGURES, TAB

S.apply()
I2 = pd.read_csv(TAB + "/K2_duration_response.csv")
SUM = pd.read_csv(TAB + "/K2b_duration_summary.csv").set_index("pulse")
D3 = pd.read_csv(TAB + "/D3_exact_pulse_duration_response.csv")
CURVES = [
    ("CDKN1B ON", "p27 ON", S.BLUE, "-"),
    ("MYC OFF + CCND2 OFF", "MYC + CCND2 OFF", S.BLUE, (0, (5, 2))),
    ("CDKN2A ON", "p16 ON", S.GREY, "-"),
    ("MYC OFF", "six other interventions", S.RUST, "-"),
]


def panel_protocol(ax):
    ax.set_xlim(-1.2, 10.6)
    ax.set_ylim(-2.3, 2.6)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0.55), 4, 0.9, fc=S.LIGHT, ec=S.INK, lw=0.6))
    ax.text(2, 1.0, "exposure", ha="center", va="center", fontsize=8.5)
    ax.annotate("", xy=(10.3, 0), xytext=(-1.0, 0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=S.INK))
    for x, t in [(0, "0"), (4, "$L$")]:
        ax.plot([x, x], [-0.12, 0.12], color=S.INK, lw=0.7)
        ax.text(x, -0.25, t, ha="center", va="top", fontsize=8.5)
    ax.text(10.3, -0.25, "time", ha="right", va="top", fontsize=8.5)
    ax.text(-0.6, 1.9, "proliferating", ha="center", fontsize=8)
    ax.text(2, 1.9, "target held ON or OFF", ha="center", fontsize=8)
    ax.text(7.2, 1.9, "withdrawal, network unperturbed", ha="center", fontsize=8)
    ax.plot([4, 4], [0.12, 2.3], color=S.GREY, lw=0.5, ls=(0, (3, 2)))
    ax.plot([0, 0], [0.12, 2.3], color=S.GREY, lw=0.5, ls=(0, (3, 2)))
    for y, lab, col in [(1.0, "durable arrest, probability $a(L)$", S.BLUE), (0.35, "regrowth, $1-a(L)$", S.RUST)]:
        ax.add_patch(FancyArrowPatch((4.15, 0.68), (6.3, y), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=col, connectionstyle="arc3,rad=0.0"))
        ax.text(6.45, y, lab, ha="left", va="center", fontsize=8, color=col)
    ax.text(4.6, -1.25, "$a(L)=\\delta_{s_0^{c}}\\,e^{Q_cL}\\,h$,     $Qh=0$", ha="center", va="center", fontsize=9.5)
    ax.text(4.6, -2.0, "$L_{50}$, $L_{90}$: exposure for $a=0.5$ and $a=0.9$", ha="center", va="center", fontsize=8.5)


def panel_curves(ax, column, ylabel):
    for key, lab, col, ls in CURVES:
        d = I2[I2.pulse == key]
        ax.plot(d.L, d[column], color=col, ls=ls, label=lab, lw=1.3)
    if column == "arrest":
        pts = D3[D3.clamp.isin(["CDKN1B ON", "MYC OFF + CCND2 OFF"])]
        ax.plot(pts.duration, pts.eventual_arrest, ls="none", marker="o", ms=3, mfc="white", mec=S.INK, mew=0.6, zorder=5)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_xlabel("Exposure duration $L$")
    ax.set_ylabel(ylabel)


def panel_l50(ax):
    rows = [("CDKN1B ON", "p27 ON"), ("CDKN2A ON", "p16 ON"), ("MYC OFF + CCND2 OFF", "MYC and\nCCND2 OFF")]
    for i, (key, lab) in enumerate(rows):
        r = SUM.loc[key]
        ax.plot([r.L50, r.L90], [i, i], color=S.GREY, lw=0.8, zorder=1)
        ax.plot(r.L50, i, marker="o", ms=5, mfc="white", mec=S.BLUE, mew=1.0, zorder=3)
        ax.plot(r.L90, i, marker="o", ms=5, mfc=S.BLUE, mec=S.BLUE, zorder=3)
        ax.text(r.L50, i - 0.22, f"{r.L50:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(r.L90, i - 0.22, f"{r.L90:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows], fontsize=8)
    ax.set_ylim(len(rows) - 0.4, -0.75)
    ax.set_xlim(0, 6)
    ax.set_xticks([0, 2, 4, 6])
    ax.set_xlabel("Exposure duration")
    ax.plot([], [], ls="none", marker="o", ms=5, mfc="white", mec=S.BLUE, mew=1.0, label="$L_{50}$")
    ax.plot([], [], ls="none", marker="o", ms=5, mfc=S.BLUE, mec=S.BLUE, label="$L_{90}$")
    ax.legend(loc="lower right", fontsize=7.5, handletextpad=0.1, labelspacing=0.25)


def panel_screen(ax):
    K = pd.read_csv(TAB + "/K3_exposure_against_durable_arrest.csv")
    K = K.sort_values(["eventual_arrest", "proliferation_off_at_end_of_pulse"]).reset_index(drop=True)
    y = np.arange(len(K))
    for i, r in K.iterrows():
        durable = r.eventual_arrest
        col = S.BLUE if durable > 0.5 else (S.GREY if durable > 0 else S.RUST)
        ax.plot([durable, r.proliferation_off_at_end_of_pulse], [i, i], color=col, lw=0.8, zorder=1)
        ax.plot(r.proliferation_off_at_end_of_pulse, i, marker="o", ms=4.5, mfc="white", mec=col, mew=0.9, zorder=2)
        ax.plot(durable, i, marker="o", ms=4.5, mfc=col, mec=col, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([S.pulse_label(p) for p in K.pulse], fontsize=7.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim(-0.7, len(K) - 0.3)
    ax.set_xlabel("Probability")
    ax.plot([], [], ls="none", marker="o", ms=4.5, mfc="white", mec=S.INK, mew=0.9, label="Proliferation OFF, end of exposure")
    ax.plot([], [], ls="none", marker="o", ms=4.5, mfc=S.INK, mec=S.INK, label="durable arrest after withdrawal")
    ax.legend(loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=1, handletextpad=0.2, fontsize=7.5, labelspacing=0.2)


if __name__ == "__main__":
    fig = plt.figure(figsize=(7.2, 5.9))
    ax_a = fig.add_axes([0.03, 0.60, 0.60, 0.37])
    panel_protocol(ax_a)
    S.panel(ax_a, "a", dx=0.035, dy=0.93)
    ax_d = fig.add_axes([0.765, 0.08, 0.22, 0.80])
    panel_screen(ax_d)
    S.panel(ax_d, "d", dx=-0.85, dy=1.09)
    ax_b = fig.add_axes([0.075, 0.08, 0.215, 0.40])
    panel_curves(ax_b, "arrest", "Durable arrest probability $a(L)$")
    for yv in (0.5, 0.9):
        ax_b.axhline(yv, color=S.GREY, lw=0.5, ls=(0, (3, 2)), zorder=0)
    ax_b.legend(loc="lower right", bbox_to_anchor=(1.02, 0.03), ncol=1, fontsize=6.3, handlelength=1.5, labelspacing=0.2, borderaxespad=0.2)
    S.panel(ax_b, "b", dx=-0.22, dy=1.03)
    ax_c = fig.add_axes([0.42, 0.08, 0.17, 0.40])
    panel_l50(ax_c)
    S.panel(ax_c, "c", dx=-0.5, dy=1.03)
    S.save(fig, FIGURES, "Figure3")
