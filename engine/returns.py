"""
engine/returns.py
==================
Composite Forward Upside (mu_i) estimation -- Section 4.2, Etap 2: Alpha Blend.

This is the ONE function that will, much later, be the seam where a
pluggable ML expected-return model replaces/augments this formula-based
approach (see PROJECT_CONTEXT.md roadmap). Kept deliberately isolated in its
own module for exactly that reason: an ML-based mu_i estimator will present
the same "row dict + params in -> mu_i out" interface, so nothing outside
this file has to change to swap it in.

No Dash / UI / disk-I/O code lives here on purpose.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ALPHA = 0.5  # backward-compat default for snapshots saved before Etap 2
                      # (their "parameters" JSON has no "alpha" key at all). Chosen as
                      # a neutral 50/50 blend that doesn't privilege either signal.
                      # NOTE: this does NOT reproduce the pre-Etap-2 formula's exact
                      # numbers even at alpha=0.5 -- that old formula applied M_i
                      # (EPS revision momentum) ONLY to the G-branch and A_i (analyst
                      # coverage confidence) ONLY to the U-branch, each modifier
                      # scoped to "its own" term. The confirmed Alpha Blend formula
                      # applies BOTH M_i and A_i to the WHOLE (1-alpha)*U + alpha*G
                      # bracket together, per the approved design -- a structural
                      # change independent of alpha's value, not something alpha=0.5
                      # can undo. Reloading an old snapshot in the Sandbox will
                      # therefore show different mu_i values than it originally did;
                      # this is expected under the new formula, not a regression.


def compute_composite_upside_row(row, gamma, kappa, n_ref, alpha=DEFAULT_ALPHA):
    """
    Composite Forward Upside (mu_i) -- Section 4.2, Alpha Blend (Etap 2),
    REVISED 2026-09-19 (confirmed reversal of the earlier "gamma drops out
    at alpha=1.0" design, on explicit instruction from the project owner):

        BaseReturn_i = (1 - alpha) * U_raw_i  +  alpha * G_val_i     [pure weighted average, no discount here]
        M_i          = 1 + kappa * Rev_val_i
        A_i          = 1 - exp(-N_i / N_ref)
        D_i          = exp(-gamma * S_i)                              [NEW: universal dispersion discount]
        mu_i         = BaseReturn_i * M_i * A_i * D_i

    where:
        U_raw_i   = (T_i - P0_i) / P0_i          [target-price upside]
        S_i       = (T_high_i - T_low_i) / T_i   [analyst target-price dispersion]
        G_val_i   = G_2Y_i / 100.0               [EPS 2Y CAGR -- entered as a whole
                                                   percent, e.g. 55 means 55%; the /100
                                                   division happens ONLY here, never in
                                                   the Stage 3 table itself, so data entry
                                                   stays as "55" not "0.55"]
        Rev_val_i = Rev_90d_i / 100.0            [90d EPS revision, same whole-percent
                                                   entry convention as G_2Y_i]

    CONFIRMED DESIGN REVERSAL (2026-09-19): the PREVIOUS version coupled the
    gamma dispersion penalty to the (1-alpha) branch specifically, so it
    dropped out ENTIRELY at alpha=1.0 (pure EPS growth) -- the stated reasoning
    at the time was that S_i is a property of analyst PRICE TARGETS
    specifically, so it has "nothing to discount" once that branch has zero
    weight. The project owner identified a genuine, undesired consequence of
    this in practice: sliding alpha toward 0 heavily discounts the upside (so
    much that, at low alpha, as few as ~4 stocks in a real universe cleared
    an 25% Rf hurdle after discounting) while alpha=1.0 applied almost NO
    discount at all, letting through wide-dispersion, low-forecast-quality
    names (e.g. ARGX, NBIX in the project owner's own real universe) purely
    because gamma had nothing left to act on. Confirmed reasoning for the fix:
    there is no OTHER available signal specifically measuring the reliability
    of the EPS-growth estimate itself, so S_i (analyst price-target dispersion)
    is repurposed as a general forecast-uncertainty proxy applied to the WHOLE
    composite estimate, exactly like M_i and A_i already are -- not just to
    the price-target branch. This is a conscious, explicit choice to
    discount EPS growth by a price-target-derived signal for lack of a more
    specific one, not an oversight.

    `alpha` in [0, 1] is the Growth/Upside Blend parameter (confirmed design,
    2026-09-06 conversation):
        alpha = 0.0 -> pure analyst target-price consensus (U_raw)
        alpha = 0.5 -> equal blend of the two branches
        alpha = 1.0 -> pure fundamental EPS growth (G_val)
        In EVERY case, the result is now discounted by D_i = exp(-gamma*S_i),
        M_i, and A_i identically -- gamma no longer varies in effect with alpha.
        Values outside [0, 1] are clipped -- a defensive guard against a
        stray out-of-range value reaching here from a UI slider or an old/
        malformed snapshot JSON, not an expected code path.

    A_i (analyst coverage confidence) discounts the WHOLE bracket regardless
    of alpha, unchanged from before -- coverage depth is treated as a general
    signal of forecast QUALITY, not specifically a property of the
    target-price branch.

    Divide-by-zero guards: P0<=0 -> U_raw=0; T_i==0 -> S_i=0; N_ref<=0 -> A_i=0.

    Returns
    -------
    dict with keys:
        "U_raw", "S_i", "G_val", "Rev_val", "M_i", "A_i", "D_i" : the intermediate terms above
        "U_component" : (1-alpha) * U_raw          -- target-price contribution to BaseReturn_i (pre-discount)
        "G_component" : alpha * G_val               -- EPS-growth contribution to BaseReturn_i (pre-discount)
        "base_return" : U_component + G_component (pre-M_i/A_i/D_i)
        "alpha"       : the (clipped) alpha actually used
        "mu_i"        : the final composite upside
    """
    P0 = row.get("Current Price (P0)") or 0.0
    Ti = row.get("Target Consensus (Ti)") or 0.0
    Thigh = row.get("Target High (T_high)") or 0.0
    Tlow = row.get("Target Low (T_low)") or 0.0
    Ni = row.get("Analyst Coverage (Ni)") or 0.0
    G2Y_pct = row.get("EPS 2Y CAGR (Gi,2Y)") or 0.0
    Rev_pct = row.get("90d EPS Revision (\u0394EPS90d)") or 0.0

    alpha = DEFAULT_ALPHA if alpha is None else float(alpha)
    alpha = min(max(alpha, 0.0), 1.0)

    U_raw = (Ti - P0) / P0 if P0 > 0 else 0.0
    S_i = (Thigh - Tlow) / Ti if Ti != 0 else 0.0
    G_val = G2Y_pct / 100.0
    Rev_val = Rev_pct / 100.0

    U_component = (1.0 - alpha) * U_raw
    G_component = alpha * G_val
    base_return = U_component + G_component

    M_i = 1.0 + kappa * Rev_val
    A_i = (1.0 - np.exp(-Ni / n_ref)) if n_ref > 0 else 0.0
    D_i = np.exp(-gamma * S_i)

    mu_i = base_return * M_i * A_i * D_i

    return {
        "U_raw": float(U_raw), "S_i": float(S_i), "G_val": float(G_val), "Rev_val": float(Rev_val),
        "M_i": float(M_i), "A_i": float(A_i), "D_i": float(D_i),
        "U_component": float(U_component), "G_component": float(G_component),
        "base_return": float(base_return), "alpha": float(alpha), "mu_i": float(mu_i),
    }