import itertools
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit  # numerically stable sigmoid


# Add a control group (no style) to measure the value add.

ALGO_SCHEMA_VERSION = 1


class PairsExhausted(Exception):
    """Raised by get_comparison() when every unordered pair has already been presented."""

class LearningAlgoTemplate:

    def __init__(self, vectors):
        # Complete 2D integer matrix containing all possible combos
        # i.e.
        # vectors = [[0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 1],
        #            [0, 0, 0, 0, 2],
        #                  ...
        #            [2, 2, 2, 2, 2]]
        self.population = vectors

        # Allocate a parallel 1D array of zeros
        self.scores = np.zeros(len(self.population), dtype=float)
        self.past_comparisons = []

    def get_comparison(self, vectors, scores):
        """select comparison to maximise information gain"""
        # return a_vect, b_vect
        pass
    
    def update_score(self, a_vect, b_vect, preference):
        # save past comparison
        self.past_comparisons.append([a_vect, b_vect, preference])
        # update scores...

    def get_scores(self):
        # Create score and vector data structure
        # return scores
        pass 

class MythosNonLinearAlgo:
    """
    Bayesian active preference learner over an enumerated population of
    categorical vectors.

    Model
    -----
    Each vector is one-hot encoded (categories are nominal, so index j at
    dimension d activates its own weight rather than implying "more than"
    index j-1).  Latent utility is linear in that encoding, U(x) = w.phi(x),
    and preferences follow the Bradley-Terry rule
    P(A > B) = sigmoid(U(A) - U(B)).  Beliefs about w form a Gaussian
    posterior N(mu, Sigma).  self.scores = Phi @ mu is the per-vector
    utility table implied by the current posterior mean.

    Note on interpretation: pairwise data only identifies score
    *differences*, so compare scores against each other; absolute values
    are pinned by the zero-mean prior, not by the data.
    """

    def __init__(self, vectors, past_scores, prior_std: float = 1.0, pool_size: int = 4096, seed: Optional[int] = None):
        # Complete 2D integer matrix
        self.population = np.asarray(vectors, dtype=int)
        if self.population.ndim != 2:
            raise ValueError("vectors must be a 2D array of shape (N, L)")

        n_vectors, self._L = self.population.shape

        # Per-dimension category counts (supports unequal counts per
        # dimension) and the offset of each dimension's block inside the
        # flattened one-hot encoding.
        self._n_categories = self.population.max(axis=0) + 1
        self._offsets = np.concatenate(([0], np.cumsum(self._n_categories)[:-1]))
        self._dim = int(self._n_categories.sum())

        # One-hot encode the whole population once: row i of self._phi is
        # phi(population[i]).  Entry (offsets[d] + value) flags "dimension d
        # holds this category".
        self._phi = np.zeros((n_vectors, self._dim))
        self._phi[np.arange(n_vectors)[:, None], self._offsets + self.population] = 1.0

        # Gaussian posterior over the weights, initialised to the prior
        # N(0, prior_std^2 I).  mu is the master belief; Sigma tracks how
        # certain we are about each weight and how weights co-vary.
        self._mu = np.zeros(self._dim)
        self._Sigma = np.eye(self._dim) * float(prior_std) ** 2

        # Allocate a parallel 1D array of zeros
        # (posterior-mean utility of every population vector; kept in sync
        # with mu by update_score).
        if past_scores == False:
            self.scores = self._phi @ self._mu
        else:
            self.scores = past_scores

        # Parallel 1D array of score uncertainties: posterior std of each
        # vector's utility, sqrt(phi . Sigma . phi).  Shrinks as we learn.
        self.score_stds = np.sqrt(np.einsum(
            "nd,de,ne->n", self._phi, self._Sigma, self._phi))

        self.past_comparisons = []

        self._pool_size = int(pool_size)
        self._rng = np.random.default_rng(seed)
        self._prior_std = float(prior_std)

        # Stable vector IDs: row index within the population enumeration.
        self._index_of = {tuple(int(v) for v in row): k
                          for k, row in enumerate(self.population)}

        # Unordered pairs already shown to the user (answered OR skipped),
        # keyed lo * N + hi over population indices. get_comparison excludes
        # them outright, so a skipped duel can never resurface.
        self._presented_keys = set()

        # Details of the most recent get_comparison() selection
        # (candidate indices + BALD gain), for diagnostics.
        self.last_selected = None

    def get_comparison(self):
        """
        Select the comparison (a_vect, b_vect) that maximises information gain.

        Pairs that were already presented (answered or skipped) are excluded
        from the candidate set; raises PairsExhausted when none remain.

        How the information-gain logic works, in plain English
        ------------------------------------------------------
        For a candidate pair, everything depends on the difference feature
        d = phi(A) - phi(B) through two numbers:

            m = d . mu        = scores[A] - scores[B]  (predicted margin)
            v = d . Sigma . d                          (our doubt about it)

        The pair is scored with BALD, the mutual information (in bits)
        between the user's answer and the weights:

            gain = H[ predictive P(A>B) ] - E_w[ H[ sigmoid(w.d) ] ]

        The first term rewards duels we currently forecast near 50/50; the
        second subtracts the coin-flip noise that would remain even if we
        knew the weights exactly.  The difference is uncertainty caused
        purely by our ignorance, so the winner is a pair with m ~ 0
        *because* v is large - a close call we could actually learn from.
        (This also self-avoids repeats: once a pair is answered, its v has
        collapsed, so its gain drops.)  Both terms use the standard probit
        approximation of the logistic sigmoid, giving a closed form.

        Small populations are searched exhaustively over all pairs;
        large ones via a random pool of `pool_size` sampled pairs.
        """
        X = self.population
        n = len(X)
        if n < 2:
            raise ValueError("need at least two vectors to form a comparison")

        # One-hot encode the candidate set (reuse the cached encoding when
        # we are handed the population itself).
        if X.shape == self.population.shape and np.array_equal(X, self.population):
            phi = self._phi
        else:
            phi = np.zeros((n, self._dim))
            phi[np.arange(n)[:, None], self._offsets + X] = 1.0

        if n <= 1024:
            # --- exhaustive: score every unordered pair -------------------
            i_idx, j_idx = np.triu_indices(n, k=1)
            i_idx, j_idx = self._filter_presented(i_idx, j_idx, n)
            gram = (phi @ self._Sigma) @ phi.T             # phi Sigma phi^T
            diag = np.diag(gram)
            # var(U_A - U_B) = var(U_A) + var(U_B) - 2 cov(U_A, U_B)
            v = diag[i_idx] + diag[j_idx] - 2.0 * gram[i_idx, j_idx]
        else:
            # --- sampled: random candidate pool ---------------------------
            k = self._pool_size
            i_idx = self._rng.integers(0, n, size=k)
            j_idx = self._rng.integers(0, n, size=k)
            clash = i_idx == j_idx                         # forbid A == B
            j_idx[clash] = (j_idx[clash] + 1) % n
            i_idx, j_idx = self._filter_presented(i_idx, j_idx, n)
            d_mat = phi[i_idx] - phi[j_idx]
            v = np.einsum("kd,de,ke->k", d_mat, self._Sigma, d_mat)

        m = self.scores[i_idx] - self.scores[j_idx]                  # predicted margins
        v = np.maximum(v, 0.0)

        # BALD expected information gain, in bits (closed form).
        p_hat = expit(m / np.sqrt(1.0 + np.pi * v / 8.0))
        p_hat = np.clip(p_hat, 1e-12, 1.0 - 1e-12)
        marginal_entropy = -(p_hat * np.log2(p_hat) + (1.0 - p_hat) * np.log2(1.0 - p_hat))
        c2 = 4.0 * np.log(2.0)
        expected_conditional_entropy = (np.sqrt(c2 / (c2 + v)) * np.exp(-(m ** 2) / (2.0 * (c2 + v))))
        gains = marginal_entropy - expected_conditional_entropy

        best = int(np.argmax(gains))
        self.last_selected = {
            "a_id": int(i_idx[best]),
            "b_id": int(j_idx[best]),
            "gain_bits": float(gains[best]),
        }
        a_vect = X[i_idx[best]].copy()
        b_vect = X[j_idx[best]].copy()
        return a_vect, b_vect

    def update_score(self, a_vect, b_vect, preference):
        """
        Fold one observed preference into the posterior, then refresh
        self.scores / self.score_stds.

        Parameters
        ----------
        a_vect, b_vect : the two vectors that were shown.
        preference     : which one the user chose - 'A' or 'B'
                         (case-insensitive), or 0 for A / 1 for B.

        How the Bayesian update works, in plain English
        -----------------------------------------------
        The likelihood of the choice touches the weights only through the
        scalar utility gap s = w.d, with d = phi(winner) - phi(loser)
        (dimensions where A and B agree cancel out of d, so the duel
        teaches nothing about categories it did not test).  A one-step
        Laplace approximation then reduces to:

        1. MAP: the new mean must lie on the ray mu + alpha * Sigma d,
           where alpha solves  alpha = 1 - sigmoid(m + alpha*v)
           (m = d.mu, v = d.Sigma.d; solved with Brent's method - it is
           strictly monotone, so the root is unique).  alpha is the
           *surprise* of the answer: ~0 if the winner was already a
           foregone conclusion, larger for upsets.  Sigma*d acts like a
           Kalman gain, steering the correction toward uncertain weights.

        2. Covariance: the likelihood curvature kappa = p(1-p) at the MAP
           adds kappa * d d^T to the precision; via Sherman-Morrison that
           is the rank-1 shrink below.  Certainty grows exactly along the
           trade-off just tested, fastest when the duel was close.
        """
        a_vect = np.asarray(a_vect, dtype=int)
        b_vect = np.asarray(b_vect, dtype=int)

        # save past comparison and retire the pair from acquisition
        self.past_comparisons.append([a_vect.copy(), b_vect.copy(), preference])
        self._mark_presented(a_vect, b_vect)

        # Normalise the preference flag into (winner, loser).
        flag = str(preference).strip().upper()
        if flag in ("A", "0"):
            winner, loser = a_vect, b_vect
        elif flag in ("B", "1"):
            winner, loser = b_vect, a_vect
        else:
            raise ValueError("preference must be 'A'/'B' or 0/1")

        # Difference feature d = phi(winner) - phi(loser).
        phi_w = np.zeros(self._dim)
        phi_w[self._offsets + winner] = 1.0
        phi_l = np.zeros(self._dim)
        phi_l[self._offsets + loser] = 1.0
        d = phi_w - phi_l
        if not np.any(d):
            raise ValueError("the two vectors are identical - the choice carries no information")

        sigma_d = self._Sigma @ d
        m = float(d @ self._mu)      # prior mean of the utility gap
        v = float(d @ sigma_d)       # prior variance of the utility gap

        # 1. MAP step: solve alpha = 1 - sigmoid(m + alpha*v) on [0, 1].
        # g is strictly increasing (g' = 1 + p(1-p) v > 0), so the root is
        # unique and Brent's method nails it; the two guards handle
        # numerically saturated likelihoods at the interval edges.
        def g(alpha: float) -> float:
            return alpha - (1.0 - expit(m + alpha * v))

        if g(0.0) >= 0.0:
            alpha = 0.0          # outcome already predicted with certainty
        elif g(1.0) <= 0.0:
            alpha = 1.0          # numerically saturated maximal surprise
        else:
            alpha = brentq(g, 0.0, 1.0)

        s_map = m + alpha * v                     # utility gap at the MAP
        p_map = float(expit(s_map))
        kappa = p_map * (1.0 - p_map)             # likelihood curvature

        # 2. Posterior mean shift + rank-1 covariance shrink.
        self._mu = self._mu + alpha * sigma_d
        self._Sigma = self._Sigma - np.outer(sigma_d, sigma_d) * (kappa / (1.0 + kappa * v))
        self._Sigma = 0.5 * (self._Sigma + self._Sigma.T)   # fight numeric drift

        # update scores...
        # (posterior-mean utility and remaining uncertainty of every
        # population vector, recomputed from the fresh posterior)
        self.scores = self._phi @ self._mu
        self.score_stds = np.sqrt(np.maximum(np.einsum(
            "nd,de,ne->n", self._phi, self._Sigma, self._phi), 0.0))

    def record_skip(self, a_vect, b_vect):
        """
        Record a no-preference outcome for a presented pair.

        The skip is logged in past_comparisons (outcome "SKIP") and the
        pair joins the presented set so acquisition never re-offers it,
        but the posterior is untouched: a shrug carries no Bradley-Terry
        signal, so it stays out of the likelihood.
        """
        a_vect = np.asarray(a_vect, dtype=int)
        b_vect = np.asarray(b_vect, dtype=int)
        self.past_comparisons.append([a_vect.copy(), b_vect.copy(), "SKIP"])
        self._mark_presented(a_vect, b_vect)

    def index_of(self, vect):
        """Stable ID of a vector: its row index in the population, or None."""
        return self._index_of.get(tuple(int(v) for v in np.asarray(vect).ravel()))

    @property
    def n_pairs_total(self):
        n = len(self.population)
        return n * (n - 1) // 2

    @property
    def n_pairs_presented(self):
        return len(self._presented_keys)

    def _mark_presented(self, a_vect, b_vect):
        i = self.index_of(a_vect)
        j = self.index_of(b_vect)
        if i is None or j is None:
            return
        lo, hi = (i, j) if i < j else (j, i)
        self._presented_keys.add(lo * len(self.population) + hi)

    def _filter_presented(self, i_idx, j_idx, n):
        """Drop pairs already presented (answered or skipped); raise when none are left."""
        if self._presented_keys:
            lo = np.minimum(i_idx, j_idx).astype(np.int64)
            hi = np.maximum(i_idx, j_idx).astype(np.int64)
            seen = np.fromiter(self._presented_keys, dtype=np.int64,
                               count=len(self._presented_keys))
            mask = ~np.isin(lo * n + hi, seen)
            i_idx, j_idx = i_idx[mask], j_idx[mask]
        if i_idx.size == 0:
            raise PairsExhausted("every unordered pair has already been presented")
        return i_idx, j_idx

    def get_scores(self):
        """
        Return a 2D array of the population vectors with their current
        scores, sorted best-first.

        Shape (N, L+1): columns 0..L-1 hold the vector values and the last
        column holds the posterior-mean utility score, so row 0 is the
        current best-guess vector.  Ties keep their original population
        order (stable sort).
        """
        order = np.argsort(-self.scores, kind="stable")
        return np.column_stack((self.population[order].astype(float),
                                self.scores[order]))

    def to_dict(self):
        """
        JSON-serialisable snapshot of the full learner state.

        Comparisons are stored by stable vector ID (population row index);
        the presented-pair exclusion set is rebuilt from them on load, so
        it is not stored separately.
        """
        return {
            "schema_version": ALGO_SCHEMA_VERSION,
            "population": self.population.tolist(),
            "prior_std": self._prior_std,
            "pool_size": self._pool_size,
            "mu": self._mu.tolist(),
            "sigma": self._Sigma.tolist(),
            "comparisons": [
                {
                    "a_id": self.index_of(a),
                    "b_id": self.index_of(b),
                    "outcome": str(outcome),
                }
                for a, b, outcome in self.past_comparisons
            ],
            "rng_state": self._rng.bit_generator.state,
            "scores": self.scores.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        """
        Rebuild a learner from to_dict() output.  The posterior, comparison
        log, presented-pair set and RNG state are all restored, so the next
        get_comparison() is identical to what the saved instance would have
        produced.
        """
        if data.get("schema_version") != ALGO_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported algorithm schema version {data.get('schema_version')!r} "
                f"(expected {ALGO_SCHEMA_VERSION})")
        algo = cls(vectors=np.asarray(data["population"], dtype=int),
                   past_scores=False,
                   prior_std=float(data["prior_std"]),
                   pool_size=int(data["pool_size"]))
        mu = np.asarray(data["mu"], dtype=float)
        sigma = np.asarray(data["sigma"], dtype=float)
        if mu.shape != (algo._dim,) or sigma.shape != (algo._dim, algo._dim):
            raise ValueError("posterior shape does not match the stored population")
        algo._mu = mu
        algo._Sigma = sigma
        algo._rng.bit_generator.state = data["rng_state"]
        for entry in data["comparisons"]:
            a = algo.population[int(entry["a_id"])]
            b = algo.population[int(entry["b_id"])]
            algo.past_comparisons.append([a.copy(), b.copy(), entry["outcome"]])
            algo._mark_presented(a, b)
        algo.scores = algo._phi @ algo._mu
        algo.score_stds = np.sqrt(np.maximum(np.einsum(
            "nd,de,ne->n", algo._phi, algo._Sigma, algo._phi), 0.0))
        return algo

class MythosLinearAlgo:
    """
    Bayesian active preference learner over an enumerated population of
    integer vectors, assuming a LINEAR relationship between vector values
    and utility.

    Model
    -----
    The feature map is the identity, phi(x) = x: the integer at each
    dimension is treated as a quantity, not a nominal label.  Latent
    utility is U(x) = w . x, so w_d is a per-dimension slope - the utility
    gained per one-step increase of dimension d's value.  Preferences
    follow the Bradley-Terry rule P(A > B) = sigmoid(U(A) - U(B)).
    Beliefs about the L slopes form a Gaussian posterior N(mu, Sigma).
    self.scores = population @ mu is the per-vector utility table implied
    by the current posterior mean.

    Consequence of linearity: within a dimension, option utilities are
    forced onto a line (0, w_d, 2*w_d, ...).  Preferences are therefore
    monotone in the option index, and the implied favourite of every
    dimension is an endpoint (index 0 if w_d < 0, the last index if
    w_d > 0) - a middle option can never be strictly best.  In exchange
    the model has only L parameters instead of one per category, so it
    needs far fewer comparisons.  Use it when the value ordering is
    meaningful.
    """

    def __init__(self, vectors, past_scores, prior_std: float = 1.0, pool_size: int = 4096, seed: Optional[int] = None):
        # Complete 2D integer matrix
        self.population = np.asarray(vectors, dtype=int)
        if self.population.ndim != 2:
            raise ValueError("vectors must be a 2D array of shape (N, L)")

        n_vectors, self._L = self.population.shape

        # Linear feature matrix: phi(x) = x, one column per dimension.
        # (Under the one-hot/nominal variant this would be the expanded
        # indicator matrix; here the values themselves are the features.)
        self._dim = self._L
        self._phi = self.population.astype(float)

        # Gaussian posterior over the slope weights, initialised to the
        # prior N(0, prior_std^2 I).  mu is the master belief; Sigma tracks
        # how certain we are about each slope and how slopes co-vary.
        self._mu = np.zeros(self._dim)
        self._Sigma = np.eye(self._dim) * float(prior_std) ** 2

        # Allocate a parallel 1D array of zeros
        # (posterior-mean utility of every population vector; kept in sync
        # with mu by update_score).
        if past_scores == False:
            self.scores = self._phi @ self._mu
        else:
            self.scores = past_scores

        # Parallel 1D array of score uncertainties: posterior std of each
        # vector's utility, sqrt(x . Sigma . x).  Shrinks as we learn.
        self.score_stds = np.sqrt(np.einsum(
            "nd,de,ne->n", self._phi, self._Sigma, self._phi))

        self.past_comparisons = []

        self._pool_size = int(pool_size)
        self._rng = np.random.default_rng(seed)
        self._prior_std = float(prior_std)

        # Stable vector IDs: row index within the population enumeration.
        self._index_of = {tuple(int(v) for v in row): k
                          for k, row in enumerate(self.population)}

        # Unordered pairs already shown to the user (answered OR skipped),
        # keyed lo * N + hi over population indices. get_comparison excludes
        # them outright, so a skipped duel can never resurface.
        self._presented_keys = set()

        # Details of the most recent get_comparison() selection
        # (candidate indices + BALD gain), for diagnostics.
        self.last_selected = None

    def get_comparison(self):
        """
        Select the comparison (a_vect, b_vect) that maximises information gain.

        Pairs that were already presented (answered or skipped) are excluded
        from the candidate set; raises PairsExhausted when none remain.

        How the information-gain logic works, in plain English
        ------------------------------------------------------
        For a candidate pair, everything depends on the value difference
        d = x_A - x_B through two numbers:

            m = d . mu        = scores[A] - scores[B]  (predicted margin)
            v = d . Sigma . d                          (our doubt about it)

        The pair is scored with BALD, the mutual information (in bits)
        between the user's answer and the slope weights:

            gain = H[ predictive P(A>B) ] - E_w[ H[ sigmoid(w.d) ] ]

        The first term rewards duels we currently forecast near 50/50; the
        second subtracts the coin-flip noise that would remain even if we
        knew the slopes exactly.  The difference is uncertainty caused
        purely by our ignorance, so the winner is a pair with m ~ 0
        *because* v is large - a close call we could actually learn from.
        (This also self-avoids repeats: once a pair is answered, its v has
        collapsed, so its gain drops.  Note that under the linear model a
        big value gap on an uncertain dimension is extra informative,
        since v scales with the square of the gap.)  Both terms use the
        standard probit approximation of the logistic sigmoid, giving a
        closed form.

        Small populations are searched exhaustively over all pairs;
        large ones via a random pool of `pool_size` sampled pairs.
        """
        X = self.population
        n = len(X)
        if n < 2:
            raise ValueError("need at least two vectors to form a comparison")

        # Linear features of the candidate set (reuse the cached matrix
        # when we are handed the population itself).
        if X.shape == self.population.shape and np.array_equal(X, self.population):
            phi = self._phi
        else:
            phi = X.astype(float)

        if n <= 1024:
            # --- exhaustive: score every unordered pair -------------------
            i_idx, j_idx = np.triu_indices(n, k=1)
            i_idx, j_idx = self._filter_presented(i_idx, j_idx, n)
            gram = (phi @ self._Sigma) @ phi.T             # x Sigma x^T
            diag = np.diag(gram)
            # var(U_A - U_B) = var(U_A) + var(U_B) - 2 cov(U_A, U_B)
            v = diag[i_idx] + diag[j_idx] - 2.0 * gram[i_idx, j_idx]
        else:
            # --- sampled: random candidate pool ---------------------------
            k = self._pool_size
            i_idx = self._rng.integers(0, n, size=k)
            j_idx = self._rng.integers(0, n, size=k)
            clash = i_idx == j_idx                         # forbid A == B
            j_idx[clash] = (j_idx[clash] + 1) % n
            i_idx, j_idx = self._filter_presented(i_idx, j_idx, n)
            d_mat = phi[i_idx] - phi[j_idx]
            v = np.einsum("kd,de,ke->k", d_mat, self._Sigma, d_mat)

        m = self.scores[i_idx] - self.scores[j_idx]                  # predicted margins
        v = np.maximum(v, 0.0)

        # BALD expected information gain, in bits (closed form).
        p_hat = expit(m / np.sqrt(1.0 + np.pi * v / 8.0))
        p_hat = np.clip(p_hat, 1e-12, 1.0 - 1e-12)
        marginal_entropy = -(p_hat * np.log2(p_hat)
                             + (1.0 - p_hat) * np.log2(1.0 - p_hat))
        c2 = 4.0 * np.log(2.0)
        expected_conditional_entropy = (np.sqrt(c2 / (c2 + v))
                                        * np.exp(-(m ** 2) / (2.0 * (c2 + v))))
        gains = marginal_entropy - expected_conditional_entropy

        best = int(np.argmax(gains))
        self.last_selected = {
            "a_id": int(i_idx[best]),
            "b_id": int(j_idx[best]),
            "gain_bits": float(gains[best]),
        }
        a_vect = X[i_idx[best]].copy()
        b_vect = X[j_idx[best]].copy()
        return a_vect, b_vect

    def update_score(self, a_vect, b_vect, preference):
        """
        Fold one observed preference into the posterior, then refresh
        self.scores / self.score_stds.

        Parameters
        ----------
        a_vect, b_vect : the two vectors that were shown.
        preference     : which one the user chose - 'A' or 'B'
                         (case-insensitive), or 0 for A / 1 for B.

        How the Bayesian update works, in plain English
        -----------------------------------------------
        The likelihood of the choice touches the slopes only through the
        scalar utility gap s = w.d, with d = x_winner - x_loser
        (dimensions where A and B hold the same value cancel out of d, so
        the duel teaches nothing about slopes it did not exercise).  A
        one-step Laplace approximation then reduces to:

        1. MAP: the new mean must lie on the ray mu + alpha * Sigma d,
           where alpha solves  alpha = 1 - sigmoid(m + alpha*v)
           (m = d.mu, v = d.Sigma.d; solved with Brent's method - it is
           strictly monotone, so the root is unique).  alpha is the
           *surprise* of the answer: ~0 if the winner was already a
           foregone conclusion, larger for upsets.  Sigma*d acts like a
           Kalman gain, steering the correction toward uncertain slopes
           and, via d, toward the dimensions with the largest value gaps.

        2. Covariance: the likelihood curvature kappa = p(1-p) at the MAP
           adds kappa * d d^T to the precision; via Sherman-Morrison that
           is the rank-1 shrink below.  Certainty grows exactly along the
           trade-off just tested, fastest when the duel was close.
        """
        a_vect = np.asarray(a_vect, dtype=int)
        b_vect = np.asarray(b_vect, dtype=int)

        # save past comparison and retire the pair from acquisition
        self.past_comparisons.append([a_vect.copy(), b_vect.copy(), preference])
        self._mark_presented(a_vect, b_vect)

        # Normalise the preference flag into (winner, loser).
        flag = str(preference).strip().upper()
        if flag in ("A", "0"):
            winner, loser = a_vect, b_vect
        elif flag in ("B", "1"):
            winner, loser = b_vect, a_vect
        else:
            raise ValueError("preference must be 'A'/'B' or 0/1")

        # Linear difference feature d = x_winner - x_loser.
        d = (winner - loser).astype(float)
        if not np.any(d):
            raise ValueError("the two vectors are identical - the choice carries no information")

        sigma_d = self._Sigma @ d
        m = float(d @ self._mu)      # prior mean of the utility gap
        v = float(d @ sigma_d)       # prior variance of the utility gap

        # 1. MAP step: solve alpha = 1 - sigmoid(m + alpha*v) on [0, 1].
        # g is strictly increasing (g' = 1 + p(1-p) v > 0), so the root is
        # unique and Brent's method nails it; the two guards handle
        # numerically saturated likelihoods at the interval edges.
        def g(alpha: float) -> float:
            return alpha - (1.0 - expit(m + alpha * v))

        if g(0.0) >= 0.0:
            alpha = 0.0          # outcome already predicted with certainty
        elif g(1.0) <= 0.0:
            alpha = 1.0          # numerically saturated maximal surprise
        else:
            alpha = brentq(g, 0.0, 1.0)

        s_map = m + alpha * v                     # utility gap at the MAP
        p_map = float(expit(s_map))
        kappa = p_map * (1.0 - p_map)             # likelihood curvature

        # 2. Posterior mean shift + rank-1 covariance shrink.
        self._mu = self._mu + alpha * sigma_d
        self._Sigma = self._Sigma - np.outer(sigma_d, sigma_d) * (kappa / (1.0 + kappa * v))
        self._Sigma = 0.5 * (self._Sigma + self._Sigma.T)   # fight numeric drift

        # update scores...
        # (posterior-mean utility and remaining uncertainty of every
        # population vector, recomputed from the fresh posterior)
        self.scores = self._phi @ self._mu
        self.score_stds = np.sqrt(np.maximum(np.einsum(
            "nd,de,ne->n", self._phi, self._Sigma, self._phi), 0.0))

    def record_skip(self, a_vect, b_vect):
        """
        Record a no-preference outcome for a presented pair.

        The skip is logged in past_comparisons (outcome "SKIP") and the
        pair joins the presented set so acquisition never re-offers it,
        but the posterior is untouched: a shrug carries no Bradley-Terry
        signal, so it stays out of the likelihood.
        """
        a_vect = np.asarray(a_vect, dtype=int)
        b_vect = np.asarray(b_vect, dtype=int)
        self.past_comparisons.append([a_vect.copy(), b_vect.copy(), "SKIP"])
        self._mark_presented(a_vect, b_vect)

    def index_of(self, vect):
        """Stable ID of a vector: its row index in the population, or None."""
        return self._index_of.get(tuple(int(v) for v in np.asarray(vect).ravel()))

    @property
    def n_pairs_total(self):
        n = len(self.population)
        return n * (n - 1) // 2

    @property
    def n_pairs_presented(self):
        return len(self._presented_keys)

    def _mark_presented(self, a_vect, b_vect):
        i = self.index_of(a_vect)
        j = self.index_of(b_vect)
        if i is None or j is None:
            return
        lo, hi = (i, j) if i < j else (j, i)
        self._presented_keys.add(lo * len(self.population) + hi)

    def _filter_presented(self, i_idx, j_idx, n):
        """Drop pairs already presented (answered or skipped); raise when none are left."""
        if self._presented_keys:
            lo = np.minimum(i_idx, j_idx).astype(np.int64)
            hi = np.maximum(i_idx, j_idx).astype(np.int64)
            seen = np.fromiter(self._presented_keys, dtype=np.int64,
                               count=len(self._presented_keys))
            mask = ~np.isin(lo * n + hi, seen)
            i_idx, j_idx = i_idx[mask], j_idx[mask]
        if i_idx.size == 0:
            raise PairsExhausted("every unordered pair has already been presented")
        return i_idx, j_idx

    def get_scores(self):
        """
        Return a 2D array of the population vectors with their current
        scores, sorted best-first.

        Shape (N, L+1): columns 0..L-1 hold the vector values and the last
        column holds the posterior-mean utility score, so row 0 is the
        current best-guess vector.  Ties keep their original population
        order (stable sort).  The array is float because scores are
        continuous; recover the integer vectors with
        ``result[:, :-1].astype(int)``.
        """
        order = np.argsort(-self.scores, kind="stable")
        return np.column_stack((self.population[order].astype(float),
                                self.scores[order]))

    def to_dict(self):
        """
        JSON-serialisable snapshot of the full learner state.

        Comparisons are stored by stable vector ID (population row index);
        the presented-pair exclusion set is rebuilt from them on load, so
        it is not stored separately.
        """
        return {
            "schema_version": ALGO_SCHEMA_VERSION,
            "population": self.population.tolist(),
            "prior_std": self._prior_std,
            "pool_size": self._pool_size,
            "mu": self._mu.tolist(),
            "sigma": self._Sigma.tolist(),
            "comparisons": [
                {
                    "a_id": self.index_of(a),
                    "b_id": self.index_of(b),
                    "outcome": str(outcome),
                }
                for a, b, outcome in self.past_comparisons
            ],
            "rng_state": self._rng.bit_generator.state,
            "scores": self.scores.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        """
        Rebuild a learner from to_dict() output.  The posterior, comparison
        log, presented-pair set and RNG state are all restored, so the next
        get_comparison() is identical to what the saved instance would have
        produced.
        """
        if data.get("schema_version") != ALGO_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported algorithm schema version {data.get('schema_version')!r} "
                f"(expected {ALGO_SCHEMA_VERSION})")
        algo = cls(vectors=np.asarray(data["population"], dtype=int),
                   past_scores=False,
                   prior_std=float(data["prior_std"]),
                   pool_size=int(data["pool_size"]))
        mu = np.asarray(data["mu"], dtype=float)
        sigma = np.asarray(data["sigma"], dtype=float)
        if mu.shape != (algo._dim,) or sigma.shape != (algo._dim, algo._dim):
            raise ValueError("posterior shape does not match the stored population")
        algo._mu = mu
        algo._Sigma = sigma
        algo._rng.bit_generator.state = data["rng_state"]
        for entry in data["comparisons"]:
            a = algo.population[int(entry["a_id"])]
            b = algo.population[int(entry["b_id"])]
            algo.past_comparisons.append([a.copy(), b.copy(), entry["outcome"]])
            algo._mark_presented(a, b)
        algo.scores = algo._phi @ algo._mu
        algo.score_stds = np.sqrt(np.maximum(np.einsum(
            "nd,de,ne->n", algo._phi, algo._Sigma, algo._phi), 0.0))
        return algo




if __name__ == "__main__":
    pass