"""The full embedded strain must equal log(U), including its normal component."""

import numpy as np
import pytest

from calm.interface.refinement.strain import compute_strain_2d


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("deformed", [False, True])
def test_full_hencky_tensors_match_stretch_logarithms(alpha, deformed):
    a = np.eye(3)
    b = np.array([[1.2, 0.15, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.0]]) if deformed else a.copy()
    state = compute_strain_2d(a, b, alpha=alpha)
    for side in ("A", "B"):
        stretch = getattr(state, f"U_{side}")
        strain = getattr(state, f"E_{side}")
        values, vectors = np.linalg.eigh(stretch)
        expected = (vectors * np.log(values)) @ vectors.T
        np.testing.assert_allclose(strain, expected, atol=1e-12)
        np.testing.assert_allclose(np.trace(strain), np.log(np.linalg.det(stretch)), atol=1e-12)
        np.testing.assert_array_equal(strain[2, :], np.zeros(3))
        np.testing.assert_array_equal(strain[:, 2], np.zeros(3))
        np.testing.assert_allclose(
            np.linalg.norm(strain, ord="fro") / np.sqrt(2.0),
            getattr(state, f"E_{side}_rms"), atol=1e-12,
        )
        if not deformed:
            np.testing.assert_allclose(strain, np.zeros((3, 3)), atol=1e-12)
