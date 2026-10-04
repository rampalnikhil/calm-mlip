from types import SimpleNamespace

import numpy as np

from calm.interface.matching._orchestrator import enumerate_coupled_match_classes_core
from calm.interface.types import AffineInvariantStrain2D
from calm.math2d.spd2x2 import invsqrt_spd
from calm.math2d.sym2x2 import eigh2_sym, eigvals2_spd


def test_small_symmetric_eigenvalues_keep_relative_accuracy():
    for scale in (1e-100, 1.0, 1e100):
        for diagonal in ([1e-12, 1.0], [-1.0, -1e-12], [-1.0, 1e-12]):
            matrix = scale * np.diag(diagonal)
            expected = np.sort(scale * np.array(diagonal))
            np.testing.assert_allclose(eigvals2_spd(matrix), expected, rtol=1e-14, atol=0)
            values, vectors = eigh2_sym(matrix)
            np.testing.assert_allclose(values, expected, rtol=1e-14, atol=0)
            np.testing.assert_allclose((vectors * values) @ vectors.T, matrix, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(eigvals2_spd(np.zeros((2, 2))), [0.0, 0.0])


def test_whitening_elongated_diagonal_metrics_keeps_roundoff_small():
    for aspect in (50, 100, 1000, 1e6):
        gram = np.diag([1.0 / aspect**2, 1.0])
        inverse_root = invsqrt_spd(gram)
        np.testing.assert_allclose(inverse_root @ gram @ inverse_root, np.eye(2), atol=1e-14, rtol=0)


def test_equal_rotated_metrics_have_exactly_zero_strain():
    for angle in (0.0, 0.3, 0.7):
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        gram = rotation @ np.diag([0.0001, 1.0]) @ rotation.T
        strain = AffineInvariantStrain2D.from_grams(gram, gram)
        np.testing.assert_array_equal(strain.principal_strains, [0.0, 0.0])
        np.testing.assert_array_equal(strain.E, np.zeros((2, 2)))


def test_large_compression_preserves_small_positive_relative_eigenvalues():
    strain = AffineInvariantStrain2D.from_grams(np.eye(2), 1e-20 * np.eye(2))
    np.testing.assert_allclose(strain.mu, [1e-20, 1e-20], rtol=1e-14, atol=0)
    np.testing.assert_allclose(strain.principal_strains, [np.log(1e-10)] * 2, rtol=1e-14)


def _slab(aspect, stretch=1.0):
    return SimpleNamespace(
        atoms=SimpleNamespace(cell=SimpleNamespace(array=np.diag([stretch, aspect * stretch, 10.0]))),
        n_atoms=1,
    )


def _matches(a, b, limit):
    identity = (np.eye(2, dtype=int),)
    return enumerate_coupled_match_classes_core(
        a, b, k_max=1, eps_principal_max=limit,
        point_group_A=identity, point_group_B=identity,
    )


def test_elongated_cells_are_admitted_at_zero_and_finite_strain_limits():
    for aspect in (50, 100, 1000):
        slab = _slab(aspect)
        assert _matches(slab, slab, 0.0)
        stretched = _slab(aspect, np.exp(0.05))
        assert _matches(slab, stretched, 0.05)
        # Numerical hygiene must not turn into a wider physical tolerance.
        assert not _matches(slab, stretched, 0.05 - 1e-8)
