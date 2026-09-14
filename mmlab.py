"""mmlab: machine learning meets Markowitz, laboratory code.

Reproducibility needs only this file plus the notebook
``unified_implied_returns_and_risk_premia.ipynb`` — nothing else.
The notebook already embeds a copy of this module section by section
(the percent-style section markers delimit the cells), so running the
notebook alone regenerates every table and figure. This file is provided
alongside it as the importable single source of truth for reuse
(``import mmlab``). Build scripts, test suites, and precomputed outputs
are developer conveniences, not requirements.

Conventions
-----------
* A return panel is an array of shape (num_stocks, num_days) of *simple* daily returns.
* Every data-generating process (DGP) is a callable ``dgp(num_stocks, num_days, seed) -> Panel``.
* Risk aversion is ``eta`` (mean-variance) or ``gamma`` (CRRA). Nothing else is called eta.
* Weights ``w`` satisfy ``sum(w) == 1``, ``w >= -short_limit`` and ``w <= max_weight``.
"""
# %% [section] imports
from __future__ import annotations

import inspect
import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy import stats
from scipy.stats import skewnorm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

TRADING_DAYS = 252
DT = 1.0 / TRADING_DAYS
ANN_PCT = TRADING_DAYS * 100.0          # daily simple return -> annualised percent
BINDING_TOL = 1e-6

# %% [section] panels
@dataclass
class Panel:
    """A simulated return panel plus whatever ground truth the DGP knows."""

    returns: np.ndarray                      # (num_stocks, num_days) simple returns
    name: str = "panel"
    true_mean: Optional[np.ndarray] = None   # (num_stocks,) E[simple daily return]
    prices: Optional[np.ndarray] = None      # (num_stocks, num_days + 1) if the DGP has prices
    params: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def num_stocks(self) -> int:
        return self.returns.shape[0]

    @property
    def num_days(self) -> int:
        return self.returns.shape[1]

    def sample_moments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample mean vector and covariance matrix of the daily returns."""
        return self.returns.mean(axis=1), np.cov(self.returns)

    def window(self, start: int, stop: int) -> "Panel":
        """Sub-panel on days [start, stop)."""
        return Panel(self.returns[:, start:stop], self.name, self.true_mean, None, self.params)

    def log_returns(self) -> "Panel":
        """The same paths expressed as log returns (identical shocks, no re-simulation)."""
        return Panel(np.log1p(self.returns), self.name + " (log)", None, self.prices, self.params)


def simulate_gbm(num_stocks: int = 10, num_days: int = 100, seed: int = 0,
                 s0_range=(10.0, 500.0), mu_range=(0.01, 1.0), sigma_range=(0.10, 1.0)) -> Panel:
    """Independent GBM price paths, exact discretisation on the daily grid.

    ``S_k = S0 * exp(sum_{j<=k} [(mu - sigma^2/2) dt + sigma sqrt(dt) z_j])``, so the price at
    day 0 is exactly ``S0`` and drift and diffusion share one clock (fixes NB1/3/4/5).
    The true expected *simple* daily return is ``exp(mu dt) - 1`` (fixes NB1).
    """
    rng = np.random.default_rng(seed)
    s0 = rng.uniform(*s0_range, num_stocks)
    mu = rng.uniform(*mu_range, num_stocks)
    sigma = rng.uniform(*sigma_range, num_stocks)
    z = rng.standard_normal((num_stocks, num_days))
    log_incr = (mu - 0.5 * sigma**2)[:, None] * DT + (sigma * math.sqrt(DT))[:, None] * z
    log_prices = np.concatenate([np.zeros((num_stocks, 1)), np.cumsum(log_incr, axis=1)], axis=1)
    with np.errstate(over="ignore"):
        prices = s0[:, None] * np.exp(log_prices)
    returns = np.expm1(log_incr)                 # identical to diff(prices)/prices, immune to overflow
    return Panel(returns, "Gaussian (GBM)", np.exp(mu * DT) - 1.0, prices,
                 dict(mu=mu, sigma=sigma, s0=s0))


def simulate_factor(num_stocks: int = 10, num_days: int = 100, seed: int = 0,
                    beta_range=(0.5, 1.5), mu_f: float = 0.06, sigma_f: float = 0.20,
                    idio_mu_range=(0.01, 1.0), idio_sigma_range=(0.10, 0.50)) -> Panel:
    """One-factor arithmetic model ``r_it = beta_i f_t + eps_it`` (cross-sectionally correlated)."""
    rng = np.random.default_rng(seed)
    beta = rng.uniform(*beta_range, num_stocks)
    f_t = mu_f * DT + sigma_f * math.sqrt(DT) * rng.standard_normal(num_days)
    mu_i = rng.uniform(*idio_mu_range, num_stocks)
    sig_i = rng.uniform(*idio_sigma_range, num_stocks)
    eps = (mu_i * DT)[:, None] + (sig_i * math.sqrt(DT))[:, None] * rng.standard_normal((num_stocks, num_days))
    returns = beta[:, None] * f_t[None, :] + eps
    true_mean = beta * mu_f * DT + mu_i * DT
    return Panel(returns, "One-factor", true_mean, None, dict(beta=beta, mu=mu_i, sigma=sig_i))


# ---- standardised innovation families (mean 0, variance 1 in-sample) -------------------------
def standardise(raw: np.ndarray) -> np.ndarray:
    return (raw - raw.mean()) / raw.std()


def innov_gaussian(size: int, rng: np.random.Generator) -> np.ndarray:
    return standardise(rng.standard_normal(size))


def innov_student_t(nu: float) -> Callable[[int, np.random.Generator], np.ndarray]:
    def _inner(size: int, rng: np.random.Generator) -> np.ndarray:
        return standardise(rng.standard_t(nu, size))
    _inner.__name__ = f"innov_student_t_{nu:g}"
    return _inner


def innov_skew_normal(shape: float = -6.0) -> Callable[[int, np.random.Generator], np.ndarray]:
    def _inner(size: int, rng: np.random.Generator) -> np.ndarray:
        return standardise(skewnorm.rvs(shape, size=size, random_state=rng))
    _inner.__name__ = f"innov_skew_normal_{shape:g}"
    return _inner


def innov_jump_mixture(p_jump: float = 0.15, jump_mean: float = -2.0, jump_sd: float = 1.5,
                       body_mean: float = 0.4) -> Callable[[int, np.random.Generator], np.ndarray]:
    """Two-component mixture: a Gaussian body plus rare negative jumps (skew and fat tails)."""
    def _inner(size: int, rng: np.random.Generator) -> np.ndarray:
        jump = rng.random(size) < p_jump
        z = body_mean + rng.standard_normal(size)
        z[jump] = jump_mean + jump_sd * rng.standard_normal(int(jump.sum()))
        return standardise(z)
    _inner.__name__ = "innov_jump_mixture"
    return _inner


def make_innovation_dgp(innov_fn: Callable[[int, np.random.Generator], np.ndarray], name: str,
                        mu_range=(0.01, 1.0), sigma_range=(0.10, 1.0)) -> Callable[..., Panel]:
    """Direct arithmetic DGP ``r_it = mu_i dt + sigma_i sqrt(dt) eps_it`` with a chosen innovation.

    Only the *shape* of the innovation changes between families; mean and variance are fixed,
    which makes NB4/NB5/NB6 comparisons free of drift confounding (fixes NB6).
    """
    def dgp(num_stocks: int = 10, num_days: int = 100, seed: int = 0) -> Panel:
        rng = np.random.default_rng(seed)
        mu_i = rng.uniform(*mu_range, num_stocks)
        sig_i = rng.uniform(*sigma_range, num_stocks)
        eps = innov_fn(num_stocks * num_days, rng).reshape(num_stocks, num_days)
        returns = (mu_i * DT)[:, None] + (sig_i * math.sqrt(DT))[:, None] * eps
        return Panel(returns, name, mu_i * DT, None, dict(mu=mu_i, sigma=sig_i))
    dgp.__name__ = "dgp_" + name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    return dgp


def standard_dgps() -> Dict[str, Callable[..., Panel]]:
    """The DGP menu used throughout Part I."""
    return {
        "Gaussian (GBM)": simulate_gbm,
        "Gaussian (arith.)": make_innovation_dgp(innov_gaussian, "Gaussian (arith.)"),
        "One-factor": simulate_factor,
        "Student-t (nu=5)": make_innovation_dgp(innov_student_t(5), "Student-t (nu=5)"),
        "Student-t (nu=3)": make_innovation_dgp(innov_student_t(3), "Student-t (nu=3)"),
        "Skew-normal": make_innovation_dgp(innov_skew_normal(-6), "Skew-normal"),
        "Jump mixture": make_innovation_dgp(innov_jump_mixture(), "Jump mixture"),
    }


def panel_shape_stats(dgp: Callable[..., Panel], n_universes: int = 50, num_stocks: int = 10,
                      num_days: int = 100) -> Dict[str, float]:
    """Pooled skewness, excess kurtosis and mean pairwise correlation of a DGP."""
    pool, corrs = [], []
    for s in range(n_universes):
        p = dgp(num_stocks, num_days, seed=s)
        pool.append(p.returns.ravel())
        c = np.corrcoef(p.returns)
        corrs.append(c[np.triu_indices(num_stocks, 1)])
    pool = np.concatenate(pool)
    return dict(skew=float(stats.skew(pool)), excess_kurtosis=float(stats.kurtosis(pool)),
                mean_pairwise_corr=float(np.concatenate(corrs).mean()))

# %% [section] utilities
def mean_variance_objective(portfolio_return, eta: float, portfolio_risk):
    """Decision loss to MINIMISE: ``-w'r + eta w'Sigma w`` (works on floats and cvxpy expressions)."""
    return -portfolio_return + eta * portfolio_risk


def mv_utility(w: np.ndarray, r: np.ndarray, Sigma: np.ndarray, eta: float) -> float:
    """Realised mean-variance utility ``w'r - eta w'Sigma w``."""
    return float(w @ r - eta * (w @ Sigma @ w))


def realised_mv_utility(panel_oos: Panel, w: np.ndarray, eta: float) -> float:
    """Mean-variance utility evaluated on the sample moments of an out-of-sample window."""
    r, Sigma = panel_oos.sample_moments()
    return mv_utility(w, r, Sigma, eta)


def crra_utility(w: np.ndarray, returns: np.ndarray, gamma: float) -> float:
    """Sample expected CRRA utility of the daily gross portfolio return ``1 + w'r_t``."""
    gross = 1.0 + w @ returns
    if np.any(gross <= 0):
        return -np.inf
    if gamma == 1:
        return float(np.mean(np.log(gross)))
    return float(np.mean(gross ** (1.0 - gamma)) / (1.0 - gamma))

# %% [section] optimisers
@dataclass
class MVSolution:
    weights: np.ndarray
    budget_dual: float                  # nu in r_implied = 2 eta Sigma w + nu
    lower_dual: Optional[np.ndarray]    # multipliers of w >= -short_limit (None if unbounded)
    upper_dual: Optional[np.ndarray]    # multipliers of w <= max_weight   (None if unbounded)
    status: str
    eta: float
    short_limit: Optional[float]
    max_weight: Optional[float]

    @property
    def ok(self) -> bool:
        return self.status in ("optimal", "optimal_inaccurate") and self.weights is not None


def _bound_constraints(w, short_limit, max_weight):
    cons = [cp.sum(w) == 1]
    if short_limit is not None:
        cons.append(w >= -short_limit)
    if max_weight is not None:
        cons.append(w <= max_weight)
    return cons


OK_STATUS = ("optimal", "optimal_inaccurate")


def _solve(prob: cp.Problem) -> str:
    """CLARABEL first, SCS as fallback (fixes the solver drift between notebooks).

    Returns the final status string; ``"solver_error"`` if both solvers raised.
    """
    status = "solver_error"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            prob.solve(solver=cp.CLARABEL)
            status = prob.status
        except (cp.error.SolverError, ValueError, ArithmeticError):
            status = "solver_error"
        if status not in OK_STATUS:
            try:
                prob.solve(solver=cp.SCS, eps=1e-8, max_iters=50000)
                status = prob.status
            except (cp.error.SolverError, ValueError, ArithmeticError):
                status = "solver_error"
    return status


def _active_set_polish(r: np.ndarray, Sigma: np.ndarray, eta: float, short_limit, max_weight,
                       w0: np.ndarray, nu0: float = 0.0, tol: float = 1e-5, max_iter: int = 60):
    """Exact KKT solve on the active set identified by the interior-point solution.

    Interior-point solutions sit ~1e-8 inside the bounds and carry O(1e-7) stationarity error;
    because daily covariance matrices are tiny, that maps to O(1e-3) weight error and breaks the
    exact-inverse property. Solving the equality-constrained QP on the active set restores
    machine precision. Returns ``(w, nu, mu_lower, lambda_upper)`` or ``None`` if the active
    set does not settle (caller keeps the solver solution).
    """
    n = len(r)
    lo = -short_limit if short_limit is not None else -np.inf
    hi = max_weight if max_weight is not None else np.inf
    L = set(np.where(w0 <= lo + tol)[0]) if short_limit is not None else set()
    U = set(np.where(w0 >= hi - tol)[0]) if max_weight is not None else set()
    for _ in range(max_iter):
        A_idx = sorted(L | U)
        F = [i for i in range(n) if i not in L and i not in U]
        w = np.zeros(n)
        w[sorted(L)] = lo
        w[sorted(U)] = hi
        if not F:
            # Every asset is pinned: w is a vertex and nu is any value in [max_L g, min_U g],
            # g = r - 2 eta Sigma w. Keep the solver's nu if admissible, else the midpoint.
            if abs(w.sum() - 1.0) > 1e-9:
                return None
            g = r - 2.0 * eta * (Sigma @ w)
            lo_nu = max([g[i] for i in L], default=-np.inf)
            hi_nu = min([g[i] for i in U], default=np.inf)
            if lo_nu > hi_nu + 1e-10:
                return None
            nu = float(np.clip(nu0, lo_nu, hi_nu)) if np.isfinite(lo_nu) and np.isfinite(hi_nu) else float(nu0)
            grad = g - nu
            mu = np.where(np.isin(np.arange(n), list(L)), -grad, 0.0) if short_limit is not None else None
            lam = np.where(np.isin(np.arange(n), list(U)), grad, 0.0) if max_weight is not None else None
            return w, nu, mu, lam
        K = np.zeros((len(F) + 1, len(F) + 1))
        K[:-1, :-1] = 2.0 * eta * Sigma[np.ix_(F, F)]
        K[:-1, -1] = 1.0
        K[-1, :-1] = 1.0
        rhs = np.concatenate([r[F] - 2.0 * eta * Sigma[np.ix_(F, A_idx)] @ w[A_idx], [1.0 - w[A_idx].sum()]])
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            return None
        w[F] = sol[:-1]
        nu = float(sol[-1])
        grad = r - 2.0 * eta * (Sigma @ w) - nu          # = lambda_i - mu_i at optimum
        viol_lo = np.array([lo - w[i] for i in F])
        viol_hi = np.array([w[i] - hi for i in F])
        if viol_lo.size and viol_lo.max() > 1e-12:
            L.add(F[int(np.argmax(viol_lo))])
            continue
        if viol_hi.size and viol_hi.max() > 1e-12:
            U.add(F[int(np.argmax(viol_hi))])
            continue
        bad_L = [(grad[i], i) for i in L if grad[i] > 1e-12]      # mu_i = -grad_i must be >= 0
        bad_U = [(-grad[i], i) for i in U if grad[i] < -1e-12]    # lambda_i = grad_i must be >= 0
        if bad_L or bad_U:
            worst = max(bad_L + bad_U)
            (L if worst[1] in L else U).discard(worst[1])
            continue
        mu = np.where(np.isin(np.arange(n), list(L)), -grad, 0.0) if short_limit is not None else None
        lam = np.where(np.isin(np.arange(n), list(U)), grad, 0.0) if max_weight is not None else None
        return w, nu, mu, lam
    return None


def solve_mean_variance(r_hat: np.ndarray, Sigma: np.ndarray, eta: float,
                        short_limit: Optional[float] = 0.5, max_weight: Optional[float] = 1.0,
                        polish: bool = True) -> MVSolution:
    """Forward Markowitz: ``min -w'r + eta w'Sigma w  s.t.  sum w = 1, -short_limit <= w <= max_weight``.

    The problem is rescaled so the risk term is O(1) before it reaches the conic solver, and the
    solution is polished on its active set so that the KKT inverse is exact to machine precision.
    """
    r_hat, Sigma = np.asarray(r_hat, dtype=float), np.asarray(Sigma, dtype=float)
    n = len(r_hat)
    scale = 1.0 / max(float(np.mean(np.diag(Sigma))) * eta, 1e-12)
    w = cp.Variable(n)
    cons = _bound_constraints(w, short_limit, max_weight)
    prob = cp.Problem(cp.Minimize(mean_variance_objective(w @ (scale * r_hat), eta,
                                                          cp.quad_form(w, scale * Sigma, assume_PSD=True))), cons)
    status = _solve(prob)
    if w.value is None:
        return MVSolution(None, float("nan"), None, None, status, eta, short_limit, max_weight)
    w_val = np.asarray(w.value).ravel()
    nu = float(cons[0].dual_value) / scale
    k, lower, upper = 1, None, None
    if short_limit is not None:
        lower = np.asarray(cons[k].dual_value).ravel() / scale
        k += 1
    if max_weight is not None:
        upper = np.asarray(cons[k].dual_value).ravel() / scale
    if polish:
        out = _active_set_polish(r_hat, Sigma, eta, short_limit, max_weight, w_val, nu)
        if out is not None:
            w_pol, nu_pol, mu_pol, lam_pol = out
            obj_pol = mean_variance_objective(w_pol @ r_hat, eta, w_pol @ Sigma @ w_pol)
            obj_ip = mean_variance_objective(w_val @ r_hat, eta, w_val @ Sigma @ w_val)
            if obj_pol <= obj_ip + 1e-12 * max(1.0, abs(obj_ip)):
                w_val, nu, lower, upper = w_pol, nu_pol, mu_pol, lam_pol
    return MVSolution(w_val, nu, lower, upper, status, eta, short_limit, max_weight)


def solve_crra(returns: np.ndarray, gamma: float, short_limit: Optional[float] = 0.5,
               max_weight: Optional[float] = 1.0) -> Tuple[Optional[np.ndarray], str]:
    """Maximise sample expected CRRA utility of ``1 + w'r_t`` (concave programme)."""
    n = returns.shape[0]
    w = cp.Variable(n)
    gross = 1.0 + returns.T @ w
    if gamma == 1:
        util = cp.sum(cp.log(gross))
    else:
        util = cp.sum(cp.power(gross, 1.0 - gamma, approx=False)) / (1.0 - gamma)
    prob = cp.Problem(cp.Maximize(util), _bound_constraints(w, short_limit, max_weight))
    status = _solve(prob)
    return (None if w.value is None else np.asarray(w.value).ravel()), status


def corner_fraction(W: np.ndarray, short_limit: Optional[float], max_weight: Optional[float],
                    tol: float = BINDING_TOL) -> float:
    """Fraction of weights pinned at the floor OR the cap (one definition for NB2 and NB5)."""
    W = np.asarray(W)
    at = np.zeros(W.shape, dtype=bool)
    if short_limit is not None:
        at |= W <= -short_limit + tol
    if max_weight is not None:
        at |= W >= max_weight - tol
    return float(at.mean())

# %% [section] kkt_inverse
STATUS_SLACK, STATUS_LOWER, STATUS_UPPER = "SLACK", "BINDING_LOWER", "BINDING_UPPER"


def kkt_status(w: np.ndarray, short_limit: Optional[float], max_weight: Optional[float],
               tol: float = BINDING_TOL) -> np.ndarray:
    """Three-way complementary-slackness label per asset (fixes the missing upper bound)."""
    status = np.full(len(w), STATUS_SLACK, dtype=object)
    if short_limit is not None:
        status[w <= -short_limit + tol] = STATUS_LOWER
    if max_weight is not None:
        status[w >= max_weight - tol] = STATUS_UPPER
    return status.astype(str)


def kkt_implied_returns(Sigma: np.ndarray, w: np.ndarray, eta: float, budget_dual: float) -> np.ndarray:
    """Vectorised KKT inverse ``r_implied = 2 eta Sigma w + nu``.

    Exact (equals the input expected return) for SLACK assets; an upper bound for assets at the
    floor and a lower bound for assets at the cap. Re-optimising ``r_implied`` reproduces ``w``
    exactly in every case because all bound multipliers can be set to zero.
    """
    return 2.0 * eta * (Sigma @ w) + budget_dual


def invert_solution(Sigma: np.ndarray, sol: MVSolution) -> Tuple[np.ndarray, np.ndarray]:
    """Implied returns and status for a solved forward problem."""
    return (kkt_implied_returns(Sigma, sol.weights, sol.eta, sol.budget_dual),
            kkt_status(sol.weights, sol.short_limit, sol.max_weight))


def level_shift_family(r_implied: np.ndarray, shifts: Sequence[float]) -> np.ndarray:
    """The gauge family ``r_implied + d * 1`` (rows = shifts); all members rationalise the same w."""
    return np.asarray(r_implied)[None, :] + np.asarray(shifts)[:, None]

# %% [section] surrogate
FEATURE_NAMES = ["w_i", "Sigma_ii", "(Sigma w)_i", "r_hat_i", "eta"]


def surrogate_features(w: np.ndarray, Sigma: np.ndarray, r_hat: np.ndarray, eta: float) -> np.ndarray:
    """Per-asset feature vector ``[w_i, Sigma_ii, (Sigma w)_i, r_hat_i, eta]``."""
    return np.column_stack([w, np.diag(Sigma), Sigma @ w, r_hat, np.full(len(w), eta)])


STATUS_CODES = {STATUS_SLACK: 0, STATUS_LOWER: 1, STATUS_UPPER: 2}


def build_kkt_dataset(dgp: Callable[..., Panel], num_universes: int = 120, etas=(1.0, 5.0, 10.0),
                      num_stocks: int = 10, num_days: int = 100, short_limit: Optional[float] = 0.5,
                      max_weight: Optional[float] = 1.0, seed0: int = 0):
    """Supervised data for the inverse map: features, implied return, status code (0 slack, 1 floor, 2 cap)."""
    X, y, b = [], [], []
    for seed in range(seed0, seed0 + num_universes):
        r_hat, Sigma = dgp(num_stocks, num_days, seed=seed).sample_moments()
        for eta in etas:
            sol = solve_mean_variance(r_hat, Sigma, eta, short_limit, max_weight)
            if not sol.ok:
                continue
            r_imp, status = invert_solution(Sigma, sol)
            X.append(surrogate_features(sol.weights, Sigma, r_hat, eta))
            y.append(r_imp)
            b.append(np.array([STATUS_CODES[s] for s in status]))
    return np.vstack(X), np.concatenate(y), np.concatenate(b)


@dataclass
class Surrogate:
    regressor: Pipeline
    classifier: object
    metrics: Dict[str, float]

    def predict_returns(self, X: np.ndarray) -> np.ndarray:
        return self.regressor.predict(X)

    def predict_binding(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)


def train_surrogate(X: np.ndarray, y: np.ndarray, b: np.ndarray, test_size: float = 0.25,
                    random_state: int = 1) -> Surrogate:
    """Two heads: Huber regression for the level, multinomial logistic regression for the status.

    The status (slack / at the floor / at the cap) is a deterministic threshold function of the
    weight, so the classifier measures how well a linear boundary recovers that threshold; it is a
    diagnostic and does not feed decisions. Three classes are needed because a two-class linear
    boundary cannot separate the floor and the cap from the slack middle (both bounds bind).
    Features are standardised (fixes the lbfgs convergence warnings of NB1/3/4).
    """
    Xtr, Xte, ytr, yte, btr, bte = train_test_split(X, y, b, test_size=test_size, random_state=random_state)
    reg = make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=5000)).fit(Xtr, ytr)
    trivial = len(np.unique(btr)) < 2
    if trivial:
        clf = DummyClassifier(strategy="most_frequent").fit(Xtr, btr)
    else:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, btr)
    metrics = dict(clf_acc=float(accuracy_score(bte, clf.predict(Xte))),
                   r2=float(reg.score(Xte, yte)),
                   rmse=float(np.sqrt(mean_squared_error(yte, reg.predict(Xte)))),
                   trivial=bool(trivial), binding_frac=float((b != 0).mean()), n_train=int(len(ytr)))
    return Surrogate(reg, clf, metrics)

# %% [section] evaluation
@dataclass
class PipelineWeights:
    two_stage: np.ndarray
    exact_kkt: np.ndarray
    surrogate: np.ndarray
    Sigma: np.ndarray
    r_hat: np.ndarray
    r_implied: np.ndarray
    r_surrogate: np.ndarray
    status: np.ndarray


def fit_pipelines(panel_fit: Panel, surrogate: Optional[Surrogate], eta: float,
                  short_limit: Optional[float] = 0.5, max_weight: Optional[float] = 1.0) -> PipelineWeights:
    """Two-stage, exact-KKT and surrogate portfolios from one estimation window (one QP each)."""
    r_hat, Sigma = panel_fit.sample_moments()
    sol = solve_mean_variance(r_hat, Sigma, eta, short_limit, max_weight)
    r_imp, status = invert_solution(Sigma, sol)
    w_kk = solve_mean_variance(r_imp, Sigma, eta, short_limit, max_weight).weights
    if surrogate is None:
        r_ml, w_ml = r_hat.copy(), sol.weights.copy()
    else:
        r_ml = surrogate.predict_returns(surrogate_features(sol.weights, Sigma, r_hat, eta))
        w_ml = solve_mean_variance(r_ml, Sigma, eta, short_limit, max_weight).weights
    return PipelineWeights(sol.weights, w_kk, w_ml, Sigma, r_hat, r_imp, r_ml, status)


def paired_test(U_a: Sequence[float], U_b: Sequence[float]) -> Dict[str, float]:
    """One-sided paired t-test of ``H0: mean(U_a - U_b) <= 0`` with a 95 percent CI and win rate."""
    d = np.asarray(U_a, dtype=float) - np.asarray(U_b, dtype=float)
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    if n < 2 or se == 0 or not np.isfinite(se):
        t = 0.0 if (n < 2 or np.allclose(d, 0)) else float("inf") * np.sign(d.mean())
        p_two = 1.0 if t == 0.0 else 0.0
    else:
        t, p_two = stats.ttest_rel(U_a, U_b)
        t, p_two = float(t), float(p_two)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return dict(mean=float(d.mean()), se=float(se), t=t, p_two=p_two, p_one=float(p_one),
                ci_lo=float(d.mean() - 1.96 * se), ci_hi=float(d.mean() + 1.96 * se),
                win=float((d > 0).mean()), n=int(n))


def vol_match(w: np.ndarray, Sigma: np.ndarray, target: float = 0.20) -> np.ndarray:
    """Rescale weights so the portfolio has a fixed annualised volatility."""
    ann_var = TRADING_DAYS * float(w @ Sigma @ w)
    return w * (target / math.sqrt(max(ann_var, 1e-12)))


def newey_west_mean_test(x: np.ndarray, lags: Optional[int] = None) -> Dict[str, float]:
    """HAC t-test that the mean of a daily series is zero (lags default to n^(1/4))."""
    x = np.asarray(x, dtype=float)
    if lags is None:
        lags = int(len(x) ** 0.25)
    res = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return dict(mean=float(x.mean()), t=float(res.tvalues[0]), p_two=float(res.pvalues[0]),
                p_one=float(res.pvalues[0] / 2 if res.tvalues[0] > 0 else 1 - res.pvalues[0] / 2),
                lags=int(lags), n=int(len(x)))


def run_oos_battery(dgp: Callable[..., Panel], surrogate: Surrogate, eta: float = 5.0,
                    n_universes: int = 300, seed0: int = 1000, lookback: int = 130, num_days: int = 260,
                    num_stocks: int = 10, short_limit: Optional[float] = 0.5, max_weight: Optional[float] = 1.0,
                    vol_target: float = 0.20) -> Dict[str, object]:
    """Out-of-sample protocol of NB1: fit on days [0, lookback), realise on [lookback, num_days).

    Returns the per-universe utilities of the three pipelines, the paired tests (main and null),
    and the pooled volatility-matched daily return differences with a Newey-West test.
    """
    U = {"two_stage": [], "exact_kkt": [], "surrogate": []}
    D = []
    for i in range(n_universes):
        panel = dgp(num_stocks, num_days, seed=seed0 + i)
        pw = fit_pipelines(panel.window(0, lookback), surrogate, eta, short_limit, max_weight)
        oos = panel.window(lookback, num_days)
        U["two_stage"].append(realised_mv_utility(oos, pw.two_stage, eta))
        U["exact_kkt"].append(realised_mv_utility(oos, pw.exact_kkt, eta))
        U["surrogate"].append(realised_mv_utility(oos, pw.surrogate, eta))
        D.append(oos.returns.T @ (vol_match(pw.surrogate, pw.Sigma, vol_target) - vol_match(pw.two_stage, pw.Sigma, vol_target)))
    D = np.concatenate(D)
    return dict(utilities={k: np.asarray(v) for k, v in U.items()},
                main=paired_test(U["surrogate"], U["two_stage"]),
                null=paired_test(U["exact_kkt"], U["two_stage"]),
                vol_matched_diff=D, newey_west=newey_west_mean_test(D), eta=eta, n=n_universes)


def eta_sweep(dgp: Callable[..., Panel], surrogate: Surrogate, etas: Sequence[float], n_universes: int = 150,
              seed0: int = 2000, **kw) -> pd.DataFrame:
    """Paired test of surrogate vs two-stage for each eta, with Bonferroni correction."""
    rows = []
    for eta in etas:
        res = run_oos_battery(dgp, surrogate, eta=eta, n_universes=n_universes, seed0=seed0, **kw)
        rows.append(dict(eta=eta, **res["main"]))
    df = pd.DataFrame(rows)
    reject, p_bonf, _, _ = multipletests(df["p_one"].values, method="bonferroni")
    df["p_bonferroni"], df["reject_5pct"] = p_bonf, reject
    return df

# %% [section] crra_sweep
def mv_vs_crra_sweep(dgp: Callable[..., Panel], etas: Sequence[float], gammas: Sequence[float],
                     n_universes: int = 150, num_stocks: int = 10, num_days: int = 252, seed0: int = 0,
                     short_limit: float = 0.5, max_weight: float = 1.0,
                     keep_samples: bool = False) -> Tuple[pd.DataFrame, Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]]:
    """Compare implied-return distributions of MV and CRRA investors on identical markets.

    The MV problem is solved once per (universe, eta) and the CRRA problem once per
    (universe, gamma); results are combined for every pair (fixes the 25x redundancy of NB5).
    Both implied vectors use the MV budget dual, so their difference is gauge-free.
    """
    mv = {eta: [] for eta in etas}      # eta -> list of (Sigma, w_mv, nu, r_imp_mv)
    cr = {g: [] for g in gammas}        # gamma -> list of w_cr (or None)
    for seed in range(seed0, seed0 + n_universes):
        panel = dgp(num_stocks, num_days, seed=seed)
        r_hat, Sigma = panel.sample_moments()
        for eta in etas:
            sol = solve_mean_variance(r_hat, Sigma, eta, short_limit, max_weight)
            mv[eta].append((Sigma, sol.weights, sol.budget_dual) if sol.ok else None)
        for g in gammas:
            w_cr, st = solve_crra(panel.returns, g, short_limit, max_weight)
            cr[g].append(w_cr if st in ("optimal", "optimal_inaccurate") and w_cr is not None else None)
    rows, samples = [], {}
    for eta in etas:
        for g in gammas:
            imp_mv, imp_cr, w_mv_all, w_cr_all = [], [], [], []
            for m, w_cr in zip(mv[eta], cr[g]):
                if m is None or w_cr is None:
                    continue
                Sigma, w_mv, nu = m
                imp_mv.append(kkt_implied_returns(Sigma, w_mv, eta, nu))
                imp_cr.append(kkt_implied_returns(Sigma, w_cr, eta, nu))
                w_mv_all.append(w_mv)
                w_cr_all.append(w_cr)
            pm, pc = np.concatenate(imp_mv), np.concatenate(imp_cr)
            Wm, Wc = np.vstack(w_mv_all), np.vstack(w_cr_all)
            rows.append(dict(eta=eta, gamma=g, n=len(imp_mv), corr=float(np.corrcoef(pm, pc)[0, 1]),
                             std_ratio=float(pc.std() / pm.std()), mad_ann_pct=float(np.abs(pc - pm).mean() * ANN_PCT),
                             skew_mv=float(stats.skew(pm)), skew_cr=float(stats.skew(pc)),
                             corner_mv=corner_fraction(Wm, short_limit, max_weight),
                             corner_cr=corner_fraction(Wc, short_limit, max_weight),
                             weight_corr=float(np.corrcoef(Wm.ravel(), Wc.ravel())[0, 1]),
                             on_diagonal=bool(np.isclose(g, 2 * eta))))
            if keep_samples:
                samples[(eta, g)] = (pm, pc)
    return pd.DataFrame(rows), samples


def corner_bridge(df: pd.DataFrame) -> Dict[str, float]:
    """Divergence vs corner-fraction mismatch (NB2 section 8)."""
    mism = (df["corner_cr"] - df["corner_mv"]).abs().values * 100
    lrat = np.abs(np.log(df["std_ratio"].values))
    r_lin = stats.pearsonr(mism, lrat)
    r_sp = stats.spearmanr(mism, lrat)
    r_mad = stats.pearsonr(mism, df["mad_ann_pct"].values)
    r_raw = stats.pearsonr(df["corner_cr"].values * 100, lrat)
    return dict(pearson_mismatch_logratio=float(r_lin[0]), p_pearson=float(r_lin[1]),
                spearman_mismatch_logratio=float(r_sp[0]), pearson_mismatch_mad=float(r_mad[0]),
                pearson_rawcorner_logratio=float(r_raw[0]), mismatch_pp=mism, abs_log_ratio=lrat)

# %% [section] premium_dgps
GAMMA_MERTON = 2.0
RF_ANNUAL = 0.03
MONTH = 21


@dataclass
class PremiumSeries:
    """A single-asset return series with a known Merton premium ``pi_t = gamma sigma_t^2``."""

    returns: np.ndarray        # per-period simple returns
    premium: np.ndarray        # per-period *annualised* premium gamma sigma_t^2
    sigma: np.ndarray          # per-period annualised volatility (ground truth)
    name: str = "series"
    periods_per_year: int = TRADING_DAYS
    realized_vol: Optional[np.ndarray] = None   # observable annualised realized vol inside each period


def _arith_series(sigma: np.ndarray, eps: np.ndarray, name: str, gamma: float = GAMMA_MERTON) -> PremiumSeries:
    premium = gamma * sigma**2
    returns = (RF_ANNUAL + premium) * DT + sigma * math.sqrt(DT) * eps
    return PremiumSeries(returns, premium, sigma, name)


def premium_iid_sigma(T: int = 5000, seed: int = 42, innov: str = "gaussian", nu: float = 5.0,
                      skew_shape: float = -6.0, sigma_range=(0.15, 0.25)) -> PremiumSeries:
    """Arithmetic diffusion with i.i.d. daily volatility: the premium is NOT predictable from the past."""
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(*sigma_range, T)
    if innov == "gaussian":
        eps = rng.standard_normal(T)
    elif innov == "student_t":
        eps = standardise(rng.standard_t(nu, T))
    elif innov == "skew_normal":
        eps = standardise(skewnorm.rvs(skew_shape, size=T, random_state=rng))
    else:
        raise ValueError(innov)
    return _arith_series(sigma, eps, f"iid-sigma {innov}")


def premium_garch(T: int = 5000, seed: int = 42, omega: float = 1e-6, alpha: float = 0.09, beta: float = 0.90) -> PremiumSeries:
    """GARCH(1,1) daily variance; the premium is predictable through volatility clustering."""
    if alpha + beta >= 1:
        raise ValueError("GARCH must be stationary: alpha + beta < 1")
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T)
    h = np.empty(T)
    h[0] = omega / (1 - alpha - beta)
    for t in range(1, T):
        h[t] = omega + alpha * h[t - 1] * eps[t - 1] ** 2 + beta * h[t - 1]
    sigma = np.sqrt(h * TRADING_DAYS)
    return _arith_series(sigma, eps, "GARCH(1,1)")


def premium_ou_logvol(T: int = 6000, seed: int = 42, kappa: float = 2.0, sigma_base: float = 0.20,
                      stationary_std: float = 0.35) -> PremiumSeries:
    """Log-volatility OU: ``d x = -kappa x dt + xi dW``, ``sigma_t = sigma_base exp(x_t)``.

    ``xi = stationary_std * sqrt(2 kappa)`` keeps the stationary spread of log-vol fixed while
    ``kappa`` (per year) changes persistence only. Volatility is positive by construction
    (fixes the clipping degeneracy of NB8). Half-life in years is ``ln 2 / kappa``.
    """
    rng = np.random.default_rng(seed)
    xi = stationary_std * math.sqrt(2 * kappa)
    x = np.empty(T)
    x[0] = rng.standard_normal() * stationary_std
    z = rng.standard_normal(T)
    a = math.exp(-kappa * DT)
    cond_sd = stationary_std * math.sqrt(1 - a * a)          # exact OU transition
    for t in range(1, T):
        x[t] = a * x[t - 1] + cond_sd * z[t]
    sigma = sigma_base * np.exp(x)
    eps = rng.standard_normal(T)
    return _arith_series(sigma, eps, f"OU log-vol (kappa={kappa:g})")


def aggregate_monthly(series: PremiumSeries, month: int = MONTH) -> PremiumSeries:
    """Non-overlapping ``month``-day blocks: compounded return, mean annualised premium and sigma."""
    n = len(series.returns) // month
    daily = series.returns[:n * month].reshape(n, month)
    ret = np.prod(1 + daily, axis=1) - 1
    prem = series.premium[:n * month].reshape(n, month).mean(axis=1)
    sig = series.sigma[:n * month].reshape(n, month).mean(axis=1)
    ppy = TRADING_DAYS // month
    rv = np.sqrt((daily**2).sum(axis=1) * ppy)      # realized variance of the month, annualised
    return PremiumSeries(ret, prem, sig, series.name + " monthly", ppy, rv)

# %% [section] premium_models
MERTON_VOL_COL = "vol_merton_ann"


def make_features(returns: np.ndarray, short: int, long: int, periods_per_year: int,
                  realized_vol: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Nowcast features; row t uses returns at indices <= t only (same convention as NB7/NB8).

    ``vol_merton_ann`` is the observable annualised volatility that feeds the feasible Merton
    model: the annualised rolling ``short``-window std, or, when intra-period returns exist
    (monthly data built from daily returns), the realized volatility of period t.
    """
    r = pd.Series(np.asarray(returns, dtype=float))
    f = pd.DataFrame(index=r.index)
    f["ret_lag1"] = r.shift(1)
    f[f"ret_lag{short}"] = r.shift(short)
    f[f"vol_{short}"] = r.rolling(short).std()
    f[f"vol_{long}"] = r.rolling(long).std()
    f[f"mom_{short}"] = r.rolling(short).mean()
    f[f"mom_{long}"] = r.rolling(long).mean()
    f[f"max_{short}"] = r.rolling(short).max()
    f[f"min_{short}"] = r.rolling(short).min()
    f["ret_sq"] = (r**2).rolling(short).mean()
    if realized_vol is not None:
        rv = pd.Series(np.asarray(realized_vol, dtype=float))
        f["rv_1"] = rv
        f[f"rv_{short}"] = np.sqrt((rv**2).rolling(short).mean())
        f[MERTON_VOL_COL] = rv
    else:
        f[MERTON_VOL_COL] = f[f"vol_{short}"] * math.sqrt(periods_per_year)
    return f


def prepare_premium_data(series: PremiumSeries, short: int, long: int):
    """Aligned (features, target premium, true sigma, excess returns) after dropping warm-up rows."""
    feats = make_features(series.returns, short, long, series.periods_per_year, series.realized_vol)
    valid = feats.dropna().index
    prem = pd.Series(series.premium)[valid]
    sig = pd.Series(series.sigma)[valid]
    excess = pd.Series(series.returns)[valid] - RF_ANNUAL / series.periods_per_year
    return feats.loc[valid], prem, sig, excess


RF_PARAMS = dict(n_estimators=300, max_depth=6, min_samples_leaf=20, max_features="sqrt", random_state=42, n_jobs=-1)


def fit_rf(X, y, criterion: str = "squared_error", sample_weight=None, params: Optional[dict] = None):
    rf = RandomForestRegressor(**(params or RF_PARAMS), criterion=criterion)
    rf.fit(X, y, sample_weight=sample_weight)
    return rf


class HuberReweightedRF:
    """Two-pass Huber: fit, then down-weight observations with large first-pass residuals.

    ``delta`` defaults to the median absolute residual (scale-free; fixes the ad hoc 0.01/0.001).
    """

    def __init__(self, params: Optional[dict] = None, delta: Optional[float] = None):
        self.params, self.delta, self.rf = params or RF_PARAMS, delta, None

    def fit(self, X, y):
        resid = np.abs(y - fit_rf(X, y, params=self.params).predict(X))
        delta = self.delta if self.delta is not None else max(float(np.median(resid)), 1e-12)
        self.delta_used_ = delta
        w = np.where(resid <= delta, 1.0, delta / np.maximum(resid, 1e-300))
        self.rf = fit_rf(X, y, sample_weight=w, params=self.params)
        return self

    def predict(self, X):
        return self.rf.predict(X)


class QuantileGB:
    """Quantile gradient boosting (sklearn random forests have no pinball criterion)."""

    def __init__(self, tau: float, min_samples_leaf: int = 20):
        self.tau = tau
        self.gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, min_samples_leaf=min_samples_leaf,
                                            learning_rate=0.1, loss="quantile", alpha=tau, random_state=42)

    def fit(self, X, y):
        self.gb.fit(X, y)
        return self

    def predict(self, X):
        return self.gb.predict(X)


class MertonStructuralRF:
    """Two-stage model that injects the Merton form ``gamma * vol^2`` using OBSERVABLE volatility.

    Stage 1 learns ``gamma * vol_long_ann^2`` (a feature-derived quantity, never the truth) from
    the features; stage 2 learns the premium from features plus the stage-1 prediction.
    Fixes the ground-truth leak of NB7/NB8.
    """

    def __init__(self, vol_col: int, params: Optional[dict] = None, gamma: float = GAMMA_MERTON):
        self.vol_col, self.params, self.gamma = vol_col, params or RF_PARAMS, gamma
        self.rf1 = self.rf2 = None

    def fit(self, X, y):
        X = np.asarray(X)
        merton_target = self.gamma * X[:, self.vol_col] ** 2
        self.rf1 = fit_rf(X, merton_target, params=self.params)
        self.rf2 = fit_rf(np.hstack([X, self.rf1.predict(X)[:, None]]), y, params=self.params)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self.rf2.predict(np.hstack([X, self.rf1.predict(X)[:, None]]))


class SDFWeightedRF:
    """Heuristic Euler-residual reweighting: ``m = 1 - pi_hat/(1 + R^e)``, weights ``|m R^e|`` (NB8)."""

    def __init__(self, params: Optional[dict] = None):
        self.params, self.rf1, self.rf2 = params or RF_PARAMS, None, None

    def fit(self, X, y, excess_returns):
        self.rf1 = fit_rf(X, y, params=self.params)
        sdf = 1.0 - self.rf1.predict(X) / (1.0 + np.asarray(excess_returns))
        w = np.abs(sdf * np.asarray(excess_returns))
        w = w / (w.mean() + 1e-12)
        self.weights_ = w
        self.rf2 = fit_rf(X, y, sample_weight=w, params=self.params)
        return self

    def predict(self, X):
        return self.rf2.predict(X)


def merton_from_vol(vol_ann: np.ndarray, gamma: float = GAMMA_MERTON) -> np.ndarray:
    """Closed-form Merton premium ``gamma * vol^2`` for any volatility input (feasible or oracle)."""
    return gamma * np.asarray(vol_ann) ** 2


def pinball_loss(y_true, y_pred, tau: float) -> float:
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


def empirical_coverage(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) <= np.asarray(y_pred)))


MEAN_MODELS = ["MSE", "MAE", "Huber", "Merton-structural", "SDF-weighted", "Merton (rolling vol)", "Merton (oracle sigma)"]
QUANT_MODELS = {"Q10": 0.1, "Q50": 0.5, "Q90": 0.9}


def fit_mean_models(X_tr, y_tr, excess_tr, vol_col: int, params: Optional[dict] = None) -> Dict[str, object]:
    return {
        "MSE": fit_rf(X_tr, y_tr, params=params),
        "MAE": fit_rf(X_tr, y_tr, criterion="absolute_error", params=params),
        "Huber": HuberReweightedRF(params).fit(X_tr, y_tr),
        "Merton-structural": MertonStructuralRF(vol_col, params).fit(X_tr, y_tr),
        "SDF-weighted": SDFWeightedRF(params).fit(X_tr, y_tr, excess_tr),
    }


def run_premium_experiment(gen: Callable[..., PremiumSeries], n_reps: int = 20, monthly: bool = False,
                           short: Optional[int] = None, long: Optional[int] = None, T: Optional[int] = None,
                           train_frac: float = 0.7, params: Optional[dict] = None, seed_step: int = 100,
                           gamma: float = GAMMA_MERTON) -> Dict[str, object]:
    """Loss-function comparison for premium recovery, replicated over independent series.

    Per repetition: generate, (optionally) aggregate to months, build features, chronological
    70/30 split, fit every model, score on the test window. Mean models: bias, RMSE, MAE, R^2.
    Quantile models: pinball loss and coverage. Returns per-rep tables and pooled predictions.
    """
    if monthly:
        short, long, T = short or 3, long or 12, T or TRADING_DAYS * 60     # 60 years -> 720 months
    else:
        short, long, T = short or 20, long or 60, T or 5000
    rows, qrows, pooled = [], [], {k: ([], []) for k in MEAN_MODELS + list(QUANT_MODELS)}
    for rep in range(n_reps):
        s = gen(T=T, seed=rep * seed_step)
        if monthly:
            s = aggregate_monthly(s)
        feats, prem, sig, excess = prepare_premium_data(s, short, long)
        X, y = feats.values, prem.values
        vol_col = list(feats.columns).index(MERTON_VOL_COL)
        split = int(train_frac * len(X))
        X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]
        preds = {k: m.predict(X_te) for k, m in fit_mean_models(X_tr, y_tr, excess.values[:split], vol_col, params).items()}
        preds["Merton (rolling vol)"] = merton_from_vol(X_te[:, vol_col], gamma)
        preds["Merton (oracle sigma)"] = merton_from_vol(sig.values[split:], gamma)
        for k, p in preds.items():
            rows.append(dict(rep=rep, model=k, bias=float(np.mean(p - y_te)), rmse=float(np.sqrt(np.mean((p - y_te) ** 2))),
                             mae=float(np.mean(np.abs(p - y_te))), r2=float(r2_score(y_te, p)) if y_te.std() > 0 else np.nan))
            pooled[k][0].append(p)
            pooled[k][1].append(y_te)
        for k, tau in QUANT_MODELS.items():
            p = QuantileGB(tau, (params or RF_PARAMS)["min_samples_leaf"]).fit(X_tr, y_tr).predict(X_te)
            qrows.append(dict(rep=rep, model=k, tau=tau, pinball=pinball_loss(y_te, p, tau), coverage=empirical_coverage(y_te, p)))
            pooled[k][0].append(p)
            pooled[k][1].append(y_te)
    per_rep, per_rep_q = pd.DataFrame(rows), pd.DataFrame(qrows)
    summary = per_rep.groupby("model")[["bias", "rmse", "mae", "r2"]].agg(["mean", "median"])
    summary.columns = ["_".join(c) for c in summary.columns]
    pooled_pred = {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in pooled.items()}
    summary["r2_pooled"] = [float(r2_score(pooled_pred[k][1], pooled_pred[k][0])) for k in summary.index]
    q_summary = per_rep_q.groupby("model")[["pinball", "coverage"]].mean()
    q_summary["target"] = [QUANT_MODELS[k] for k in q_summary.index]
    return dict(per_rep=per_rep, per_rep_quantile=per_rep_q, summary=summary.loc[MEAN_MODELS],
                quantile_summary=q_summary, pooled=pooled_pred, feature_names=list(feats.columns), vol_col=vol_col,
                signal_to_noise=float(prem.std() / pd.Series(s.returns).std()))

# %% [section] helpers
def section_sources(path: str) -> Dict[str, str]:
    """Split this module into its ``# %% [section]`` blocks (developer helper used to embed the code in the notebook)."""
    text = open(path, encoding="utf-8").read()
    marker = "# %% " + "[section] "          # built in two pieces so this line does not match itself
    parts = text.split(marker)
    out = {}
    for p in parts[1:]:
        name, _, body = p.partition("\n")
        out[name.strip()] = body.rstrip() + "\n"
    return out


def fmt_p(p: float) -> str:
    return "<1e-16" if p < 1e-16 else f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"
