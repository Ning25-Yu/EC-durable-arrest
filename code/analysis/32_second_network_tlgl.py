"""A second network: the reduced T-LGL leukaemia survival network (six nodes; Saadatpour et al. 2011,
PLoS Comput Biol 7:e1002267). The same construction as for the endometrial network is applied without change:
committor of the unperturbed chain and the probability a(L) that a finite pulse leaves the network in apoptosis,
in closed form, with the exposure durations L50 and L90.

Outputs: results/tables/J1_tlgl_closed_forms.json, J2_tlgl_duration_response.csv, J3_tlgl_summary.csv
"""

import itertools
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np
import pandas as pd
import sympy as sp
from scipy.linalg import expm
from scipy.optimize import brentq

from paths import DATA, TAB as TABLES

RULES = {}
for line in open(DATA + "/second_network/TLGL_reduced.bnet"):
    n, r = [x.strip() for x in line.split(",", 1)]
    RULES[n] = compile(re.sub(r"!", " not ", r).replace("&", " and ").replace("|", " or ").strip(), n, "eval")
NODES = list(RULES)
N = len(NODES)
STATES = list(itertools.product([0, 1], repeat=N))
IDX = {s: i for i, s in enumerate(STATES)}


def image(s):
    env = dict(zip(NODES, s))
    return tuple(int(bool(eval(RULES[n], {}, env))) for n in NODES)


def generator(clamp=()):
    """Asynchronous generator with unit rates; clamped nodes have no transitions."""
    Q = sp.zeros(len(STATES), len(STATES))
    for s in STATES:
        f = image(s)
        for i, n in enumerate(NODES):
            if n not in clamp and f[i] != s[i]:
                t = list(s)
                t[i] = f[i]
                Q[IDX[s], IDX[tuple(t)]] += 1
                Q[IDX[s], IDX[s]] -= 1
    return Q


Q = generator()
FIXED = [s for s in STATES if image(s) == s]
SURVIVAL = tuple(int(n in ("S1P", "FLIP")) for n in NODES)
APOPTOSIS = tuple(int(n == "Apoptosis") for n in NODES)
assert sorted(FIXED) == sorted([SURVIVAL, APOPTOSIS]), FIXED

# committor: probability of absorption in the apoptosis fixed point
transient = [s for s in STATES if s not in FIXED]
ti = [IDX[s] for s in transient]
sol = Q.extract(ti, ti).LUsolve(-Q.extract(ti, [IDX[APOPTOSIS]]))
h = sp.zeros(len(STATES), 1)
h[IDX[APOPTOSIS]] = 1
for k, i in enumerate(ti):
    h[i] = sp.nsimplify(sol[k], rational=True)
basin_apoptosis = sp.nsimplify(sum(h) / len(STATES), rational=True)

L, z = sp.Symbol("L", positive=True), sp.Symbol("z")


def reachable(Qc, s0):
    seen, stack = {s0}, [s0]
    while stack:
        s = stack.pop()
        for j, t in enumerate(STATES):
            if t not in seen and Qc[IDX[s], j] != 0:
                seen.add(t)
                stack.append(t)
    return sorted(seen)


def pulse(clamp):
    """Closed form of a(L) from the survival fixed point, and the numerical matrix exponential for checking."""
    s0 = tuple(clamp.get(n, v) for n, v in zip(NODES, SURVIVAL))
    Qc = generator(tuple(clamp))
    R = reachable(Qc, s0)
    ii = [IDX[s] for s in R]
    Qr, hr = Qc.extract(ii, ii), sp.Matrix([h[i] for i in ii])
    e = sp.zeros(1, len(R))
    e[R.index(s0)] = 1
    lap = sp.apart(sp.simplify((e * (z * sp.eye(len(R)) - Qr).inv() * hr)[0]), z)
    a = sp.expand(sp.simplify(sp.inverse_laplace_transform(lap, z, L).subs(sp.Heaviside(L), 1)))
    Qn, hn = np.array(Qr.tolist(), float), np.array(hr.tolist(), float).ravel()
    num = lambda x: float(expm(Qn * x)[R.index(s0)] @ hn)
    return a, len(R), num


PULSES = {
    "S1P OFF": {"S1P": 0},
    "Ceramide ON": {"Ceramide": 1},
    "DISC ON": {"DISC": 1},
    "Fas ON": {"Fas": 1},
    "FLIP OFF": {"FLIP": 0},
    "Fas ON + FLIP OFF": {"Fas": 1, "FLIP": 0},
    "S1P OFF + FLIP OFF": {"S1P": 0, "FLIP": 0},
}

if __name__ == "__main__":
    grid = np.round(np.arange(0, 10.01, 0.05), 2)
    closed, rows, summ = {}, [], []
    for name, clamp in PULSES.items():
        a, nreach, num = pulse(clamp)
        f = sp.lambdify(L, a, "numpy")
        err = max(abs(float(a.subs(L, x)) - num(x)) for x in (0.25, 1.0, 2.5, 6.0))
        assert err < 1e-10, (name, err)
        a0 = sp.nsimplify(sp.limit(a, L, 0, "+"), rational=True)
        ainf = sp.limit(a, L, sp.oo)
        val = lambda x: float(a.subs(L, x))
        l50 = brentq(lambda x: val(x) - 0.5, 1e-9, 60) if ainf == 1 and float(a0) < 0.5 else (0.0 if ainf == 1 else np.nan)
        l90 = brentq(lambda x: val(x) - 0.9, 1e-9, 60) if ainf == 1 else np.nan
        closed[name] = dict(a_of_L=str(a), latex=sp.latex(a), a_at_onset=str(a0), a_long_pulse=str(ainf), reachable_states=nreach)
        summ.append(dict(pulse=name, reachable_states=nreach, a_at_onset=float(a0), L50=l50, L90=l90, a_L4=val(4), a_L8=val(8),
                         max_abs_error_vs_expm=err))
        for x in grid:
            y = float(a.subs(L, x)) if x > 0 else float(a0)
            rows.append(dict(pulse=name, L=x, arrest=y, arrest_expm=num(x) if x > 0 else np.nan))
    meta = dict(nodes=NODES, survival_fixed_point=SURVIVAL, apoptosis_fixed_point=APOPTOSIS,
                committor_values=sorted({str(v) for v in h}, key=lambda v: float(sp.Rational(v))),
                basin_apoptosis_uniform=str(basin_apoptosis),
                closed_forms=closed)
    json.dump(meta, open(TABLES + "/J1_tlgl_closed_forms.json", "w"), indent=1)
    pd.DataFrame(rows).to_csv(TABLES + "/J2_tlgl_duration_response.csv", index=False)
    S = pd.DataFrame(summ)
    S.to_csv(TABLES + "/J3_tlgl_summary.csv", index=False)
    print(S.round(4).to_string(index=False))
    print("committor values:", meta["committor_values"])
    print("P(apoptosis | uniform start) =", basin_apoptosis, "=", float(basin_apoptosis), )
    for k, v in closed.items():
        print(k, ":", v["a_of_L"])
