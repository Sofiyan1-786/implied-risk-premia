"""Unit tests for mmlab (the unified notebook's library).

Run:  python -m pytest tests -q
The suite is deterministic (fixed seeds) and finishes in a few minutes ;)
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import mmlab as m  # noqa: E402

SMALL_RF = dict(m.RF_PARAMS, n_estimators=25, n_jobs=1)
REGIMES = [(0.5, 1.0), (0.0, 1.0), (None, None), (0.25, 0.4), (1.0, None)]


def kkt_violation(r, Sigma, sol):
    """Largest violation of the first-order conditions of the forward QP (0 at the exact optimum)."""
    g = r - 2 * sol.eta * (Sigma @ sol.weights) - sol.budget_dual
    st = m.kkt_status(sol.weights, sol.short_limit, sol.max_weight)
    v = 0.0
    if (st == m.STATUS_SLACK).any():
        v = max(v, np.abs(g[st == m.STATUS_SLACK]).max())
    if (st == m.STATUS_LOWER).any():
        v = max(v, g[st == m.STATUS_LOWER].max())          # need g <= 0 at the floor
    if (st == m.STATUS_UPPER).any():
        v = max(v, (-g[st == m.STATUS_UPPER]).max())       # need g >= 0 at the cap
    return v


# - DGPs
class TestGBM:
    def test_shapes_prices_and_return_identity(self):
        p = m.simulate_gbm(10, 100, seed=1)
        assert p.returns.shape == (10, 100)
        assert p.prices.shape == (10, 101)
        assert (p.prices > 0).all()
        np.testing.assert_allclose(p.returns, np.diff(p.prices, axis=1) / p.prices[:, :-1])
        np.testing.assert_allclose(p.prices[:, 0], p.params["s0"])

    def test_reproducible_and_seed_sensitive(self):
        a, b, c = m.simulate_gbm(seed=3), m.simulate_gbm(seed=3), m.simulate_gbm(seed=4)
        np.testing.assert_array_equal(a.returns, b.returns)
        assert not np.allclose(a.returns, c.returns)

    def test_true_mean_is_expected_simple_return(self):
        p = m.simulate_gbm(3, 60_000, seed=11)
        mu, sigma = p.params["mu"], p.params["sigma"]
        sample_mean = p.returns.mean(axis=1)
        se = p.returns.std(axis=1) / math.sqrt(p.num_days)
        assert np.all(np.abs(sample_mean - p.true_mean) < 4 * se)
        log_mean = np.log1p(p.returns).mean(axis=1)
        assert np.all(np.abs(log_mean - (mu - 0.5 * sigma**2) * m.DT) < 4 * se)
        # the log drift used by NB1 is strictly below the true expected simple return (audit I-4)
        assert np.all(p.true_mean > (mu - 0.5 * sigma**2) * m.DT)

    def test_drift_and_diffusion_share_one_clock(self):
        p = m.simulate_gbm(2, 50, seed=0)
        mu, sigma = p.params["mu"], p.params["sigma"]
        log_incr = np.diff(np.log(p.prices), axis=1)
        z = (log_incr - ((mu - 0.5 * sigma**2) * m.DT)[:, None]) / (sigma * math.sqrt(m.DT))[:, None]
        assert abs(z.mean()) < 0.5 and abs(z.std() - 1) < 0.25


class TestOtherDGPs:
    def test_factor_returns_are_correlated_gbm_is_not(self):
        f = m.panel_shape_stats(m.simulate_factor, n_universes=20)
        g = m.panel_shape_stats(m.simulate_gbm, n_universes=20)
        assert f["mean_pairwise_corr"] > 0.15
        assert abs(g["mean_pairwise_corr"]) < 0.05

    @pytest.mark.parametrize("fn", [m.innov_gaussian, m.innov_student_t(5), m.innov_student_t(3),
                                    m.innov_skew_normal(-6), m.innov_jump_mixture()])
    def test_innovations_are_standardised(self, fn):
        x = fn(20_000, np.random.default_rng(0))
        assert abs(x.mean()) < 1e-10 and abs(x.std() - 1) < 1e-10

    def test_innovation_shapes(self):
        rng = np.random.default_rng(1)
        n = 200_000
        k_gauss = stats.kurtosis(m.innov_gaussian(n, rng))
        k_t5 = stats.kurtosis(m.innov_student_t(5)(n, rng))
        k_t3 = stats.kurtosis(m.innov_student_t(3)(n, rng))
        assert k_gauss < 0.2 < k_t5 < k_t3
        assert stats.skew(m.innov_skew_normal(-6)(n, rng)) < -0.7
        mix = m.innov_jump_mixture()(n, rng)
        assert stats.skew(mix) < -0.5 and stats.kurtosis(mix) > 1.0

    def test_innovation_dgps_share_mean_and_variance_parameters(self):
        d = m.standard_dgps()
        a = d["Gaussian (arith.)"](10, 100, seed=5)
        b = d["Skew-normal"](10, 100, seed=5)
        np.testing.assert_array_equal(a.params["mu"], b.params["mu"])
        np.testing.assert_array_equal(a.params["sigma"], b.params["sigma"])
        np.testing.assert_allclose(a.true_mean, b.true_mean)

    @pytest.mark.parametrize("name", list(m.standard_dgps()))
    def test_every_dgp_returns_num_days_columns(self, name):
        p = m.standard_dgps()[name](7, 123, seed=2)
        assert p.returns.shape == (7, 123)
        assert p.true_mean.shape == (7,)
        assert np.isfinite(p.returns).all()
        assert p.name == name or p.name.startswith(name.split(" ")[0])

    def test_log_panel_and_window(self):
        p = m.simulate_gbm(4, 60, seed=0)
        lp = p.log_returns()
        np.testing.assert_allclose(lp.returns, np.log1p(p.returns))
        w = p.window(10, 30)
        assert w.returns.shape == (4, 20)
        np.testing.assert_array_equal(w.returns, p.returns[:, 10:30])


# - utilities
class TestUtilities:
    def test_mv_utility_is_negative_objective(self):
        rng = np.random.default_rng(0)
        w, r = rng.normal(size=5), rng.normal(size=5)
        A = rng.normal(size=(5, 5))
        Sigma = A @ A.T
        assert m.mv_utility(w, r, Sigma, 3.0) == pytest.approx(-m.mean_variance_objective(w @ r, 3.0, w @ Sigma @ w))

    def test_crra_utility_forms(self):
        rng = np.random.default_rng(0)
        R = rng.normal(0.001, 0.01, size=(3, 500))
        w = np.array([0.2, 0.3, 0.5])
        gross = 1 + w @ R
        assert m.crra_utility(w, R, 1.0) == pytest.approx(np.mean(np.log(gross)))
        assert m.crra_utility(w, R, 2.0) == pytest.approx(np.mean(-1 / gross))
        assert m.crra_utility(w, R, 5.0) < m.crra_utility(w, R, 1.0)   # more curvature, lower level
        assert m.crra_utility(np.array([50.0, -30.0, -19.0]), R, 2.0) == -np.inf  # gross <= 0 somewhere


# - forward problem
class TestMeanVariance:
    @pytest.mark.parametrize("sl,mw", REGIMES)
    def test_feasible_and_kkt_exact(self, sl, mw):
        for seed in range(6):
            for eta in (1.0, 5.0, 20.0):
                r, S = m.simulate_gbm(10, 100, seed=seed).sample_moments()
                sol = m.solve_mean_variance(r, S, eta, sl, mw)
                assert sol.ok
                assert sol.weights.sum() == pytest.approx(1.0, abs=1e-9)
                if sl is not None:
                    assert sol.weights.min() >= -sl - 1e-9
                if mw is not None:
                    assert sol.weights.max() <= mw + 1e-9
                assert kkt_violation(r, S, sol) < 1e-9

    def test_unconstrained_matches_closed_form(self):
        r, S = m.simulate_gbm(8, 300, seed=7).sample_moments()
        eta = 4.0
        sol = m.solve_mean_variance(r, S, eta, None, None)
        Si, e = np.linalg.inv(S), np.ones(8)
        nu = (1 - (e @ Si @ r) / (2 * eta)) / ((e @ Si @ e) / (2 * eta))
        w = Si @ (r + nu * e) / (2 * eta)
        np.testing.assert_allclose(sol.weights, w, atol=1e-10)
        assert sol.budget_dual == pytest.approx(-nu, abs=1e-10)     # r = 2 eta Sigma w + nu_cvx
        assert sol.lower_dual is None and sol.upper_dual is None

    def test_duals_nonnegative_and_complementary(self):
        r, S = m.simulate_gbm(10, 100, seed=9).sample_moments()
        sol = m.solve_mean_variance(r, S, 5.0, 0.5, 1.0)
        assert (sol.lower_dual >= -1e-10).all() and (sol.upper_dual >= -1e-10).all()
        assert np.abs(sol.lower_dual * (sol.weights + 0.5)).max() < 1e-9
        assert np.abs(sol.upper_dual * (1.0 - sol.weights)).max() < 1e-9

    def test_more_risk_aversion_means_less_variance(self):
        r, S = m.simulate_gbm(10, 100, seed=2).sample_moments()
        var = [m.solve_mean_variance(r, S, eta).weights @ S @ m.solve_mean_variance(r, S, eta).weights
               for eta in (0.5, 1, 2, 5, 10, 50)]
        assert all(a >= b - 1e-12 for a, b in zip(var[:-1], var[1:]))

    def test_scaling_invariance(self):
        r, S = m.simulate_gbm(10, 100, seed=4).sample_moments()
        a = m.solve_mean_variance(r, S, 5.0)
        b = m.solve_mean_variance(1000 * r, 1000 * S, 5.0)
        np.testing.assert_allclose(a.weights, b.weights, atol=1e-8)
        assert b.budget_dual == pytest.approx(1000 * a.budget_dual, rel=1e-6)

    def test_all_assets_pinned_is_handled(self):
        r = np.array([0.05, 0.05, -0.05, -0.05])
        S = 1e-6 * np.eye(4)
        sol = m.solve_mean_variance(r, S, 1.0, 0.5, 1.0)
        np.testing.assert_allclose(sol.weights, [1, 1, -0.5, -0.5], atol=1e-8)
        assert np.isfinite(sol.budget_dual)
        assert kkt_violation(r, S, sol) < 1e-9
        r_imp = m.kkt_implied_returns(S, sol.weights, 1.0, sol.budget_dual)
        np.testing.assert_allclose(m.solve_mean_variance(r_imp, S, 1.0, 0.5, 1.0).weights, sol.weights, atol=1e-8)

    def test_polish_can_be_disabled(self):
        r, S = m.simulate_gbm(10, 100, seed=4).sample_moments()
        a = m.solve_mean_variance(r, S, 5.0, polish=False)
        b = m.solve_mean_variance(r, S, 5.0, polish=True)
        assert np.abs(a.weights - b.weights).max() < 1e-4       # same optimum ...
        assert kkt_violation(r, S, b) <= kkt_violation(r, S, a) + 1e-14   # ... but polished is at least as exact

    def test_corner_fraction_counts_floor_and_cap(self):
        W = np.array([[-0.5, 1.0, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4]])
        assert m.corner_fraction(W, 0.5, 1.0) == pytest.approx(2 / 8)
        assert m.corner_fraction(W, 0.5, None) == pytest.approx(1 / 8)
        assert m.corner_fraction(W, None, None) == 0.0


class TestCRRA:
    def test_feasible_and_better_than_perturbations(self):
        p = m.simulate_gbm(10, 252, seed=1)
        w, st = m.solve_crra(p.returns, 5.0, 0.5, 1.0)
        assert st in ("optimal", "optimal_inaccurate")
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert w.min() >= -0.5 - 1e-6 and w.max() <= 1.0 + 1e-6
        u_star = m.crra_utility(w, p.returns, 5.0)
        interior = np.where((w > -0.5 + 1e-3) & (w < 1.0 - 1e-3))[0]
        assert len(interior) >= 2
        rng = np.random.default_rng(0)
        for _ in range(20):
            i, j = rng.choice(interior, 2, replace=False)
            d = np.zeros(10)
            d[i], d[j] = 1e-3, -1e-3
            assert m.crra_utility(w + d, p.returns, 5.0) <= u_star + 1e-10

    def test_solver_errors_are_reported_not_raised(self, monkeypatch):
        """Both solvers raising must yield status 'solver_error' and weights None (the bug that crashed the full run)."""
        import cvxpy as cp

        def boom(self, *a, **k):
            raise cp.error.SolverError("simulated failure")
        monkeypatch.setattr(cp.Problem, "solve", boom)
        p = m.simulate_gbm(5, 60, seed=0)
        w, st = m.solve_crra(p.returns, 5.0)
        assert st == "solver_error" and w is None
        r, S = p.sample_moments()
        sol = m.solve_mean_variance(r, S, 5.0)
        assert not sol.ok and sol.weights is None

    def test_log_utility_case_runs(self):
        p = m.simulate_gbm(6, 120, seed=2)
        w, st = m.solve_crra(p.returns, 1.0)
        assert st in ("optimal", "optimal_inaccurate") and w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_gamma_equals_two_eta_is_locally_mean_variance(self):
        """CRRA with gamma = 2 eta should pick nearly the same portfolio as MV with eta."""
        diffs = []
        for seed in range(4):
            p = m.simulate_gbm(10, 252, seed=seed)
            r, S = p.sample_moments()
            w_mv = m.solve_mean_variance(r, S, 5.0).weights
            w_cr, _ = m.solve_crra(p.returns, 10.0)
            diffs.append(np.abs(w_mv - w_cr).max())
        assert np.median(diffs) < 0.05


# - KKT inverse
class TestKKTInverse:
    def test_three_way_status(self):
        w = np.array([-0.5, 0.2, 1.0, -0.4999999, 0.99999999])
        st = m.kkt_status(w, 0.5, 1.0)
        assert list(st) == [m.STATUS_LOWER, m.STATUS_SLACK, m.STATUS_UPPER, m.STATUS_LOWER, m.STATUS_UPPER]
        assert (m.kkt_status(w, None, None) == m.STATUS_SLACK).all()

    @pytest.mark.parametrize("sl,mw", REGIMES)
    def test_exact_for_slack_bound_for_binding_and_reoptimises(self, sl, mw):
        for seed in range(8):
            r, S = m.simulate_gbm(10, 100, seed=seed).sample_moments()
            sol = m.solve_mean_variance(r, S, 5.0, sl, mw)
            r_imp, st = m.invert_solution(S, sol)
            slack, lower, upper = st == m.STATUS_SLACK, st == m.STATUS_LOWER, st == m.STATUS_UPPER
            if slack.any():
                np.testing.assert_allclose(r_imp[slack], r[slack], atol=1e-9)
            assert np.all(r[lower] <= r_imp[lower] + 1e-9)     # floor: implied is an upper bound
            assert np.all(r[upper] >= r_imp[upper] - 1e-9)     # cap: implied is a lower bound
            w2 = m.solve_mean_variance(r_imp, S, 5.0, sl, mw).weights
            np.testing.assert_allclose(w2, sol.weights, atol=1e-8)

    def test_upper_cap_is_labelled_binding(self):
        r, S = m.simulate_gbm(10, 100, seed=3).sample_moments()
        sol = m.solve_mean_variance(r, S, 0.2, 0.5, 0.3)     # tight cap forces many upper-binding weights
        _, st = m.invert_solution(S, sol)
        assert (st == m.STATUS_UPPER).sum() >= 1

    def test_level_shift_family_all_reoptimise_to_same_weights(self):
        r, S = m.simulate_gbm(10, 100, seed=7).sample_moments()
        sol = m.solve_mean_variance(r, S, 5.0, 0.5, 1.0)
        r_imp, _ = m.invert_solution(S, sol)
        fam = m.level_shift_family(r_imp, [-0.01, -0.005, 0, 0.005, 0.01])
        assert fam.shape == (5, 10)
        np.testing.assert_allclose(fam[2], r_imp)
        for row in fam:
            np.testing.assert_allclose(m.solve_mean_variance(row, S, 5.0, 0.5, 1.0).weights, sol.weights, atol=1e-8)


# - surrogate
class TestSurrogate:
    def test_features(self):
        r, S = m.simulate_gbm(5, 50, seed=0).sample_moments()
        w = np.full(5, 0.2)
        X = m.surrogate_features(w, S, r, 3.0)
        assert X.shape == (5, 5) and len(m.FEATURE_NAMES) == 5
        np.testing.assert_allclose(X[:, 0], w)
        np.testing.assert_allclose(X[:, 1], np.diag(S))
        np.testing.assert_allclose(X[:, 2], S @ w)
        np.testing.assert_allclose(X[:, 3], r)
        assert (X[:, 4] == 3.0).all()

    def test_dataset_shape_and_labels(self):
        X, y, b = m.build_kkt_dataset(m.simulate_gbm, num_universes=6, etas=(1.0, 5.0), num_stocks=8)
        assert X.shape == (6 * 2 * 8, 5) and y.shape == (96,) and b.shape == (96,)
        assert set(np.unique(b)) <= {0, 1, 2}
        assert 0 < (b != 0).mean() < 1
        # the code is a deterministic function of the weight column
        assert ((b == 1) == (X[:, 0] <= -0.5 + m.BINDING_TOL)).all()
        assert ((b == 2) == (X[:, 0] >= 1.0 - m.BINDING_TOL)).all()

    def test_dataset_labels_include_upper_binding(self):
        X, y, b = m.build_kkt_dataset(m.simulate_gbm, num_universes=6, etas=(0.5,), max_weight=0.3)
        assert (b == 2).mean() > 0.2

    def test_train_surrogate_learns(self):
        X, y, b = m.build_kkt_dataset(m.simulate_gbm, num_universes=40)
        s = m.train_surrogate(X, y, b)
        assert set(s.metrics) >= {"clf_acc", "r2", "rmse", "trivial", "binding_frac", "n_train"}
        assert s.metrics["r2"] > 0.5 and s.metrics["clf_acc"] > 0.85 and not s.metrics["trivial"]
        assert s.predict_returns(X[:3]).shape == (3,) and set(s.predict_binding(X)) <= {0, 1, 2}

    def test_unbounded_regime_is_trivial(self):
        X, y, b = m.build_kkt_dataset(m.simulate_gbm, num_universes=10, short_limit=None, max_weight=None)
        s = m.train_surrogate(X, y, b)
        assert b.sum() == 0 and s.metrics["trivial"] and s.metrics["clf_acc"] == 1.0

    def test_deterministic(self):
        X, y, b = m.build_kkt_dataset(m.simulate_gbm, num_universes=10)
        a, c = m.train_surrogate(X, y, b), m.train_surrogate(X, y, b)
        np.testing.assert_allclose(a.predict_returns(X), c.predict_returns(X))


# - evaluation
class TestEvaluation:
    def test_paired_test_matches_scipy_and_edge_cases(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(0.1, 1, 50), rng.normal(0, 1, 50)
        res = m.paired_test(a, b)
        t, p = stats.ttest_rel(a, b)
        assert res["t"] == pytest.approx(t) and res["p_two"] == pytest.approx(p)
        assert res["p_one"] == pytest.approx(p / 2 if t > 0 else 1 - p / 2)
        assert res["ci_lo"] < res["mean"] < res["ci_hi"] and 0 <= res["win"] <= 1 and res["n"] == 50
        same = m.paired_test(a, a)
        assert same["t"] == 0.0 and same["p_one"] == 0.5 and same["mean"] == 0.0
        const = m.paired_test(b + 1.0, b)
        assert const["p_one"] == 0.0 and const["win"] == 1.0

    def test_vol_match(self):
        r, S = m.simulate_gbm(10, 100, seed=0).sample_moments()
        w = m.vol_match(np.full(10, 0.1), S, 0.2)
        assert math.sqrt(252 * w @ S @ w) == pytest.approx(0.2)

    def test_newey_west(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1, 5000)
        assert m.newey_west_mean_test(z)["p_two"] > 0.01
        pos = m.newey_west_mean_test(z + 0.2)
        assert pos["t"] > 5 and pos["p_one"] < 1e-6 and pos["lags"] == int(5000 ** 0.25)

    def test_fit_pipelines_exact_kkt_equals_two_stage(self):
        p = m.simulate_gbm(10, 130, seed=0)
        pw = m.fit_pipelines(p, None, 5.0)
        np.testing.assert_allclose(pw.exact_kkt, pw.two_stage, atol=1e-8)
        np.testing.assert_allclose(pw.surrogate, pw.two_stage)     # no surrogate -> identical
        assert pw.status.shape == (10,) and pw.r_implied.shape == (10,)

    def test_oos_battery_null_is_zero_and_structure(self):
        X, y, b = m.build_kkt_dataset(m.simulate_gbm, num_universes=15)
        s = m.train_surrogate(X, y, b)
        res = m.run_oos_battery(m.simulate_gbm, s, n_universes=12, seed0=5000)
        assert set(res) >= {"utilities", "main", "null", "vol_matched_diff", "newey_west", "eta", "n"}
        assert abs(res["null"]["mean"]) < 1e-10
        assert res["vol_matched_diff"].shape == (12 * 130,)
        assert res["main"]["n"] == 12

    def test_eta_sweep_has_bonferroni(self):
        X, y, b = m.build_kkt_dataset(m.simulate_gbm, num_universes=15)
        s = m.train_surrogate(X, y, b)
        df = m.eta_sweep(m.simulate_gbm, s, [1.0, 5.0], n_universes=8)
        assert list(df["eta"]) == [1.0, 5.0]
        assert (df["p_bonferroni"] >= df["p_one"] - 1e-15).all()
        assert df["reject_5pct"].dtype == bool


# - CRRA sweep
class TestCRRASweep:
    def test_structure_and_diagonal_rule(self):
        df, samples = m.mv_vs_crra_sweep(m.simulate_gbm, [1.0, 5.0], [2.0, 10.0], n_universes=8, num_days=120, keep_samples=True)
        assert len(df) == 4 and set(samples) == {(1.0, 2.0), (1.0, 10.0), (5.0, 2.0), (5.0, 10.0)}
        assert df["corr"].between(-1, 1).all() and (df["std_ratio"] > 0).all()
        diag = df[df["on_diagonal"]]
        assert set(zip(diag["eta"], diag["gamma"])) == {(1.0, 2.0), (5.0, 10.0)}
        assert (diag["corr"] > 0.97).all()
        assert df.loc[~df["on_diagonal"], "corr"].max() < diag["corr"].min()
        bold = df[(df["eta"] == 5.0) & (df["gamma"] == 2.0)].iloc[0]
        assert bold["std_ratio"] > 1.0          # gamma < 2 eta: CRRA is bolder

    def test_corner_bridge(self):
        df, _ = m.mv_vs_crra_sweep(m.simulate_gbm, [1.0, 5.0], [1.0, 2.0, 10.0], n_universes=6, num_days=120)
        br = m.corner_bridge(df)
        assert set(br) >= {"pearson_mismatch_logratio", "spearman_mismatch_logratio", "mismatch_pp", "abs_log_ratio"}
        assert len(br["mismatch_pp"]) == 6


# - premium part
class TestPremiumDGPs:
    @pytest.mark.parametrize("gen", [m.premium_iid_sigma, m.premium_garch, m.premium_ou_logvol])
    def test_premium_is_gamma_sigma_squared(self, gen):
        s = gen(T=1200, seed=1)
        assert s.returns.shape == s.premium.shape == s.sigma.shape == (1200,)
        np.testing.assert_allclose(s.premium, m.GAMMA_MERTON * s.sigma**2)
        assert (s.sigma > 0).all()

    def test_iid_innovation_families(self):
        for innov in ("gaussian", "student_t", "skew_normal"):
            s = m.premium_iid_sigma(T=500, seed=0, innov=innov)
            assert np.isfinite(s.returns).all()
        with pytest.raises(ValueError):
            m.premium_iid_sigma(T=10, innov="cauchy")

    def test_garch_recursion_and_stationarity(self):
        with pytest.raises(ValueError):
            m.premium_garch(T=10, alpha=0.5, beta=0.5)
        s = m.premium_garch(T=400, seed=3, omega=1e-6, alpha=0.09, beta=0.90)
        h = s.sigma**2 / m.TRADING_DAYS
        eps = (s.returns - (m.RF_ANNUAL + s.premium) * m.DT) / (s.sigma * math.sqrt(m.DT))
        np.testing.assert_allclose(h[1:], 1e-6 + 0.09 * h[:-1] * eps[:-1] ** 2 + 0.90 * h[:-1], rtol=1e-8)

    def test_ou_logvol_calibration(self):
        s = m.premium_ou_logvol(T=200_000, seed=0, kappa=2.0, sigma_base=0.2, stationary_std=0.35)
        x = np.log(s.sigma / 0.2)
        assert abs(x.std() - 0.35) / 0.35 < 0.2
        assert abs(x.mean()) < 0.1
        fast = np.log(m.premium_ou_logvol(T=50_000, seed=0, kappa=8.0).sigma / 0.2)
        slow = np.log(m.premium_ou_logvol(T=50_000, seed=0, kappa=0.5).sigma / 0.2)
        ac = lambda v: np.corrcoef(v[:-21], v[21:])[0, 1]
        assert ac(slow) > ac(fast)
        assert ac(slow) == pytest.approx(math.exp(-0.5 * 21 / 252), abs=0.05)

    def test_aggregate_monthly(self):
        s = m.premium_garch(T=21 * 10 + 5, seed=0)
        mth = m.aggregate_monthly(s)
        assert len(mth.returns) == 10 and mth.periods_per_year == 12
        assert mth.returns[0] == pytest.approx(np.prod(1 + s.returns[:21]) - 1)
        assert mth.premium[0] == pytest.approx(s.premium[:21].mean())
        assert mth.realized_vol[0] == pytest.approx(math.sqrt((s.returns[:21] ** 2).sum() * 12))


class TestPremiumModels:
    def test_features_use_only_past_and_present(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, 300)
        f1 = m.make_features(r, 20, 60, 252)
        r2 = r.copy()
        r2[150:] += 5.0                       # perturb the future
        f2 = m.make_features(r2, 20, 60, 252)
        pd.testing.assert_frame_equal(f1.iloc[:150], f2.iloc[:150])
        assert not f1.iloc[150:].equals(f2.iloc[150:])
        assert m.MERTON_VOL_COL in f1.columns
        np.testing.assert_allclose(f1[m.MERTON_VOL_COL].dropna(), f1["vol_20"].dropna() * math.sqrt(252))

    def test_features_with_realized_vol(self):
        rng = np.random.default_rng(0)
        r, rv = rng.normal(0, 0.05, 100), rng.uniform(0.1, 0.3, 100)
        f = m.make_features(r, 3, 12, 12, realized_vol=rv)
        np.testing.assert_allclose(f[m.MERTON_VOL_COL], rv)
        assert "rv_1" in f and "rv_3" in f

    def test_prepare_alignment(self):
        s = m.aggregate_monthly(m.premium_ou_logvol(T=21 * 200, seed=1))
        feats, prem, sig, excess = m.prepare_premium_data(s, 3, 12)
        assert feats.index.equals(prem.index) and feats.index.equals(sig.index) and feats.index.equals(excess.index)
        assert not feats.isna().any().any()
        np.testing.assert_allclose(excess.values, s.returns[feats.index] - m.RF_ANNUAL / 12)

    def test_pinball_and_coverage(self):
        y, p = np.array([1.0, 2.0, 3.0]), np.array([1.5, 1.5, 1.5])
        assert m.pinball_loss(y, p, 0.5) == pytest.approx(0.5 * np.mean(np.abs(y - p)))
        assert m.pinball_loss(y, y, 0.1) == 0.0
        assert m.empirical_coverage(y, np.full(3, np.inf)) == 1.0
        assert m.empirical_coverage(y, p) == pytest.approx(1 / 3)

    def test_merton_structural_stage1_targets_observable_vol(self):
        s = m.premium_garch(T=1500, seed=2)
        feats, prem, sig, _ = m.prepare_premium_data(s, 20, 60)
        X, y = feats.values, prem.values
        vol_col = list(feats.columns).index(m.MERTON_VOL_COL)
        mdl = m.MertonStructuralRF(vol_col, SMALL_RF).fit(X, y)
        stage1 = mdl.rf1.predict(X)
        target = m.GAMMA_MERTON * X[:, vol_col] ** 2
        assert stats.pearsonr(stage1, target)[0] > 0.95
        assert stats.pearsonr(stage1, target)[0] > stats.pearsonr(stage1, y)[0]   # learned the observable form, not the truth
        assert mdl.predict(X[:5]).shape == (5,)

    def test_huber_and_sdf_weights(self):
        s = m.premium_garch(T=1200, seed=4)
        feats, prem, _, excess = m.prepare_premium_data(s, 20, 60)
        X, y = feats.values, prem.values
        hub = m.HuberReweightedRF(SMALL_RF).fit(X, y)
        assert hub.delta_used_ > 0 and hub.predict(X[:3]).shape == (3,)
        sdf = m.SDFWeightedRF(SMALL_RF).fit(X, y, excess.values)
        assert (sdf.weights_ >= 0).all() and sdf.weights_.mean() == pytest.approx(1.0, abs=1e-6)
        assert sdf.predict(X[:3]).shape == (3,)

    def test_quantile_models_are_ordered(self):
        s = m.premium_garch(T=1500, seed=5)
        feats, prem, _, _ = m.prepare_premium_data(s, 20, 60)
        X, y = feats.values, prem.values
        q = {tau: m.QuantileGB(tau).fit(X[:1000], y[:1000]).predict(X[1000:]).mean() for tau in (0.1, 0.5, 0.9)}
        assert q[0.1] < q[0.5] < q[0.9]

    def test_run_premium_experiment_daily(self):
        out = m.run_premium_experiment(m.premium_garch, n_reps=1, T=1500, params=SMALL_RF)
        assert list(out["summary"].index) == m.MEAN_MODELS
        assert out["summary"].loc["Merton (oracle sigma)", "r2_pooled"] > 0.99
        assert out["summary"].loc["MSE", "r2_pooled"] > 0.5
        assert set(out["quantile_summary"].index) == set(m.QUANT_MODELS)
        assert out["signal_to_noise"] > 0
        neg = m.run_premium_experiment(m.premium_iid_sigma, n_reps=1, T=1500, params=SMALL_RF)
        assert neg["summary"].loc["MSE", "r2_pooled"] < 0.2      # unpredictable premium: negative control

    def test_run_premium_experiment_monthly(self):
        out = m.run_premium_experiment(m.premium_ou_logvol, n_reps=1, monthly=True, T=21 * 300, params=SMALL_RF)
        assert out["feature_names"][-1] == m.MERTON_VOL_COL and "rv_1" in out["feature_names"]
        assert out["summary"].loc["Merton (oracle sigma)", "r2_pooled"] > 0.99
        assert out["summary"].loc["MSE", "r2_pooled"] > 0.3


# - notebook consistency
class TestNotebookEmbedsLibrary:
    NB = os.path.join(ROOT, "unified_implied_returns_and_risk_premia.ipynb")

    def test_sections_are_parsed(self):
        secs = m.section_sources(os.path.join(ROOT, "mmlab.py"))
        assert {"imports", "panels", "utilities", "optimisers", "kkt_inverse", "surrogate", "evaluation",
                "crra_sweep", "premium_dgps", "premium_models", "helpers"} <= set(secs)

    def test_notebook_code_matches_module(self):
        if not os.path.exists(self.NB):
            pytest.skip("notebook not built yet")
        nb = json.load(open(self.NB, encoding="utf-8"))
        secs = m.section_sources(os.path.join(ROOT, "mmlab.py"))
        found = {}
        for c in nb["cells"]:
            for tag in c.get("metadata", {}).get("tags", []):
                if tag.startswith("mmlab-section:"):
                    found[tag.split(":", 1)[1]] = "".join(c["source"])
        assert set(found) == set(secs), "every library section must be embedded exactly once"
        for name, src in secs.items():
            assert found[name].strip() == src.strip(), f"embedded section {name} drifted from mmlab.py"

    def test_notebook_has_no_em_dashes_and_no_errors(self):
        if not os.path.exists(self.NB):
            pytest.skip("notebook not built yet")
        nb = json.load(open(self.NB, encoding="utf-8"))
        for c in nb["cells"]:
            assert "—" not in "".join(c["source"])
            for o in c.get("outputs", []):
                assert o.get("output_type") != "error", o.get("evalue")
