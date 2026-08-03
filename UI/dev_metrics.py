"""
Dev-only diagnostics for the preference learner.

Everything metric-related lives in this module so it can be deleted
cleanly: remove this file plus the DEV_METRICS conditional blocks in
app.py.  The module is only imported when the DEV_METRICS environment
variable is truthy, so with the flag off the /dev/metrics routes are
never registered, no nav link renders, and no collection or timing code
runs anywhere in the request path.
"""
import math
from datetime import datetime, timezone

import numpy as np
from flask import Blueprint, jsonify, render_template
from scipy.stats import kendalltau

METRICS_SCHEMA_VERSION = 1
RANK_HISTORY_N = 10         # rankings kept for Kendall-tau comparisons
MC_SAMPLES = 500            # posterior draws for P(leader is truly top)
LOO_MAX_COMPARISONS = 200   # leave-one-out is quadratic; stop beyond this
SIGMA_COND_LIMIT = 1e8      # condition number above this flags the posterior

metrics_bp = Blueprint("dev_metrics", __name__)

_state_getter = None


def register_dev_metrics(app, state_getter):
    """Register the diagnostics blueprint; called only when DEV_METRICS is on."""
    global _state_getter
    _state_getter = state_getter
    app.register_blueprint(metrics_bp)


# =====================================================================
# METRIC COMPUTATION HELPERS
# =====================================================================
def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _is_skip(outcome):
    return str(outcome).strip().upper() == "SKIP"


def _answered(algo):
    return [(a, b, o) for a, b, o in algo.past_comparisons if not _is_skip(o)]


def _phi_of(algo, vect):
    """Feature row for a population vector, using the algo's own feature map.

    Works for either model: identity features for MythosLinearAlgo,
    one-hot for MythosNonLinearAlgo. Never reconstructs the encoding by
    hand, so it stays correct whatever phi(x) the algo uses.
    """
    idx = algo.index_of(vect)
    if idx is None:
        raise KeyError(f"vector {np.asarray(vect).tolist()} not in population")
    return algo._phi[idx]


def _winner_diff(algo, a_vect, b_vect, outcome):
    """phi(winner) - phi(loser) for an answered comparison."""
    flag = str(outcome).strip().upper()
    winner, loser = (a_vect, b_vect) if flag in ("A", "0") else (b_vect, a_vect)
    return _phi_of(algo, winner) - _phi_of(algo, loser)


def _log_likelihood(algo):
    """Log-likelihood of all answered comparisons at the posterior mean."""
    total = 0.0
    for a, b, outcome in _answered(algo):
        margin = float(_winner_diff(algo, a, b, outcome) @ algo._mu)
        total += float(-np.logaddexp(0.0, -margin))   # log sigmoid(margin)
    return total


def _loo_accuracy(algo):
    """
    Leave-one-out predictive accuracy over answered comparisons: refit
    without comparison k (updates are cheap rank-1 steps), then check
    whether the held-out winner is predicted.  Ties count half.

    The refit clones the *live* learner's class via type(algo), so the
    posterior dimensionality matches whichever feature map is in use
    (identity for the linear model, one-hot for the non-linear one).
    """
    answered = _answered(algo)
    n = len(answered)
    if n < 2 or n > LOO_MAX_COMPARISONS:
        return None
    correct = 0.0
    for k in range(n):
        fresh = type(algo)(algo.population, past_scores=False,
                           prior_std=algo._prior_std,
                           pool_size=algo._pool_size)
        for idx, (a, b, outcome) in enumerate(answered):
            if idx != k:
                fresh.update_score(a, b, outcome)
        margin = float(_winner_diff(algo, *answered[k]) @ fresh._mu)
        if margin > 0:
            correct += 1.0
        elif margin == 0:
            correct += 0.5
    return correct / n


def _posterior_entropy(sigma):
    """Differential entropy of the Gaussian weight posterior, in nats."""
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        return None
    dim = sigma.shape[0]
    return _finite_or_none(0.5 * (dim * math.log(2.0 * math.pi * math.e) + logdet))


def _p_top(algo, top1):
    """Monte Carlo posterior probability that vector `top1` is truly best."""
    vals, vecs = np.linalg.eigh(algo._Sigma)
    root = vecs * np.sqrt(np.clip(vals, 0.0, None))
    # Own seeded RNG: must not consume the learner's stream, or a save/load
    # cycle would stop being bit-identical.
    draws = np.random.default_rng(0).standard_normal((algo._Sigma.shape[0], MC_SAMPLES))
    utilities = algo._phi @ (algo._mu[:, None] + root @ draws)
    return float(np.mean(np.argmax(utilities, axis=0) == top1))


def _rank_positions(scores):
    """Rank position of every vector (0 = best), plus the top two IDs."""
    order = np.argsort(-scores, kind="stable")
    pos = np.empty(len(scores), dtype=int)
    pos[order] = np.arange(len(scores))
    return pos, int(order[0]), int(order[1])


def _kendall(pos_a, pos_b):
    tau = kendalltau(np.asarray(pos_a), np.asarray(pos_b))[0]
    return _finite_or_none(tau)


def _appearance_counts(algo):
    counts = np.zeros(len(algo.population), dtype=int)
    for a, b, _outcome in algo.past_comparisons:
        for vect in (a, b):
            idx = algo.index_of(vect)
            if idx is not None:
                counts[idx] += 1
    return counts


# =====================================================================
# COLLECTOR
# =====================================================================
class MetricsCollector:
    """
    One snapshot per recorded outcome (answer or skip).  Lives in the
    server-side session state and is persisted inside the profile JSON
    under the separate, versioned "dev_metrics" key.
    """

    def __init__(self):
        self.snapshots = []
        self.rank_history = []      # last RANK_HISTORY_N of {"iteration", "pos"}
        self.bald_max = None
        self._pending_acq = None    # acquisition details for the pair on screen

    def note_acquisition(self, seconds, algo):
        """Called right after get_comparison(); remembers timing + BALD gain."""
        selected = getattr(algo, "last_selected", None) or {}
        self._pending_acq = {
            "seconds": float(seconds),
            "gain_bits": selected.get("gain_bits"),
            "a_id": selected.get("a_id"),
            "b_id": selected.get("b_id"),
        }

    def record_iteration(self, algo, outcome, fit_seconds):
        """Snapshot the learner right after an outcome has been folded in."""
        scores = np.asarray(algo.scores, dtype=float)
        stds = np.asarray(algo.score_stds, dtype=float)
        pos, top1, runner = _rank_positions(scores)

        acq = self._pending_acq or {}
        self._pending_acq = None
        gain = acq.get("gain_bits")
        if gain is not None:
            self.bald_max = gain if self.bald_max is None else max(self.bald_max, gain)

        tau_prev = None
        if self.rank_history:
            tau_prev = _kendall(pos, self.rank_history[-1]["pos"])

        diff = algo._phi[top1] - algo._phi[runner]
        var_diff = float(diff @ algo._Sigma @ diff)
        margin = float(scores[top1] - scores[runner])

        streak = 1
        for snap in reversed(self.snapshots):
            if snap.get("top1") != top1:
                break
            streak += 1

        eigs = np.linalg.eigvalsh(algo._Sigma)
        min_eig, max_eig = float(eigs[0]), float(eigs[-1])
        cond = (max_eig / min_eig) if min_eig > 0 else None

        answered = len(_answered(algo))
        skipped = len(algo.past_comparisons) - answered

        snapshot = {
            "iteration": len(self.snapshots) + 1,
            "outcome": str(outcome).strip().upper(),
            "pair": [acq.get("a_id"), acq.get("b_id")],
            # convergence
            "mean_std": float(stds.mean()),
            "max_std": float(stds.max()),
            "min_std": float(stds.min()),
            "entropy": _posterior_entropy(algo._Sigma),
            "bald_gain_bits": gain,
            "bald_running_max": self.bald_max,
            # ranking stability
            "tau_prev": tau_prev,
            "top1": top1,
            "top1_streak": streak,
            "margin": margin,
            "margin_sigmas": _finite_or_none(margin / math.sqrt(var_diff)) if var_diff > 0 else None,
            "p_top": _p_top(algo, top1),
            # model fit
            "log_likelihood": _finite_or_none(_log_likelihood(algo)),
            "loo_accuracy": _loo_accuracy(algo),
            "sigma_min_eig": min_eig,
            "sigma_cond": _finite_or_none(cond) if cond is not None else None,
            "sigma_ok": bool(min_eig > 0 and cond is not None and cond < SIGMA_COND_LIMIT),
            # coverage / interaction
            "answered": answered,
            "skipped": skipped,
            "pairs_presented": int(algo.n_pairs_presented),
            "pairs_total": int(algo.n_pairs_total),
            "explored_frac": algo.n_pairs_presented / algo.n_pairs_total,
            "fit_seconds": float(fit_seconds),
            "acq_seconds": acq.get("seconds"),
        }
        self.snapshots.append(snapshot)
        self.rank_history.append({"iteration": snapshot["iteration"], "pos": pos.tolist()})
        del self.rank_history[:-RANK_HISTORY_N]

    def to_dict(self):
        return {
            "schema_version": METRICS_SCHEMA_VERSION,
            "snapshots": self.snapshots,
            "rank_history": self.rank_history,
            "bald_max": self.bald_max,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Tolerant restore: a missing, foreign-version or malformed payload
        yields a fresh collector instead of blocking the profile load.
        """
        collector = cls()
        if not isinstance(data, dict) or data.get("schema_version") != METRICS_SCHEMA_VERSION:
            return collector
        if isinstance(data.get("snapshots"), list):
            collector.snapshots = data["snapshots"]
        if isinstance(data.get("rank_history"), list):
            collector.rank_history = data["rank_history"]
        bald_max = data.get("bald_max")
        collector.bald_max = float(bald_max) if isinstance(bald_max, (int, float)) else None
        return collector


# =====================================================================
# ROUTES
# =====================================================================
def _build_payload():
    state = _state_getter()
    algo = state["algo"]
    space = state["style_space"]
    collector = state.get("metrics")
    snapshots = collector.snapshots if collector else []
    latest = snapshots[-1] if snapshots else None

    counts = _appearance_counts(algo)
    order = np.argsort(-np.asarray(algo.scores), kind="stable")
    top_vectors = []
    for rank, idx in enumerate(order[:15], start=1):
        idx = int(idx)
        top_vectors.append({
            "rank": rank,
            "id": idx,
            "style": " / ".join(space.devectorize_profile(algo.population[idx]).values()),
            "score": float(algo.scores[idx]),
            "std": float(algo.score_stds[idx]),
            "appearances": int(counts[idx]),
        })

    # Current ranking against each retained previous ranking.
    tau_vs_recent = []
    if collector and len(collector.rank_history) > 1:
        current = collector.rank_history[-1]["pos"]
        for entry in collector.rank_history[:-1]:
            tau_vs_recent.append({"iteration": entry["iteration"],
                                  "tau": _kendall(current, entry["pos"])})

    answered = len(_answered(algo))
    skipped = len(algo.past_comparisons) - answered
    return {
        "generated": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "profile": state.get("profile_name"),
        "schema_version": METRICS_SCHEMA_VERSION,
        "iterations": snapshots,
        "tau_vs_recent": tau_vs_recent,
        "current": {
            "latest": latest,
            "top_vectors": top_vectors,
            "coverage": {
                "answered": answered,
                "skipped": skipped,
                "pairs_presented": int(algo.n_pairs_presented),
                "pairs_total": int(algo.n_pairs_total),
                "pairs_remaining": int(algo.n_pairs_total - algo.n_pairs_presented),
                "explored_frac": algo.n_pairs_presented / algo.n_pairs_total,
            },
            "appearance": {
                "min": int(counts.min()),
                "max": int(counts.max()),
                "mean": float(counts.mean()),
                "never_shown": int((counts == 0).sum()),
                "counts": counts.tolist(),
            },
        },
    }


@metrics_bp.route("/dev/metrics")
def metrics_page():
    return render_template("dev_metrics.html", payload=_build_payload())


@metrics_bp.route("/dev/metrics.json")
def metrics_json():
    return jsonify(_build_payload())