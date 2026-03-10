import numpy as np
from itertools import permutations

from local_corex import LinearCorex


def _make_fake_data(n_samples: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n_samples)
    z2 = rng.normal(size=n_samples)
    noise = 0.1 * rng.normal(size=(n_samples, 4))

    data = np.column_stack(
        [
            z1 + noise[:, 0],
            z1 + noise[:, 1],
            z2 + noise[:, 2],
            z2 + noise[:, 3],
        ]
    )
    return data.astype(np.float32)


def test_linearcorex_fit_transform_smoke():
    data = _make_fake_data()

    model = LinearCorex(n_hidden=2, seed=42, max_iter=200, tol=1e-4)
    y = model.fit_transform(data)

    assert y.shape == (data.shape[0], 2)
    assert np.isfinite(y).all()


def test_linearcorex_seed_reproducibility():
    data = _make_fake_data()

    model_a = LinearCorex(n_hidden=2, seed=42, max_iter=200, tol=1e-4)
    model_b = LinearCorex(n_hidden=2, seed=42, max_iter=200, tol=1e-4)

    y_a = model_a.fit_transform(data)
    y_b = model_b.fit_transform(data)

    corr = np.corrcoef(y_a.T, y_b.T)[: y_a.shape[1], y_a.shape[1] :]
    best_perm = max(
        permutations(range(y_b.shape[1])),
        key=lambda perm: sum(abs(corr[i, j]) for i, j in enumerate(perm)),
    )

    y_b_aligned = y_b[:, best_perm].copy()
    for i in range(y_a.shape[1]):
        if np.corrcoef(y_a[:, i], y_b_aligned[:, i])[0, 1] < 0:
            y_b_aligned[:, i] *= -1

    np.testing.assert_allclose(y_a, y_b_aligned, rtol=1e-3, atol=3e-4)


def test_linearcorex_transform_details_and_inverse_transform():
    data = _make_fake_data()

    model = LinearCorex(n_hidden=2, seed=42, max_iter=200, tol=1e-4)
    y_fit = model.fit_transform(data)

    y_details, moments = model.transform(data, details=True)
    x_recon = model.inverse_transform(y_details)

    np.testing.assert_allclose(y_fit, y_details, rtol=1e-5, atol=1e-6)
    assert isinstance(moments, dict)
    assert "TC" in moments
    assert y_details.shape == (data.shape[0], 2)
    assert x_recon.shape == data.shape
    assert np.isfinite(y_details).all()
    assert np.isfinite(x_recon).all()
