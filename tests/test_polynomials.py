"""Tests for polynomial wrappers."""

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from capella_reader.polynomials import Poly1D, Poly2D


class TestPoly1D:
    """Tests for Poly1D polynomial wrapper."""

    def test_creation_from_list(self):
        """Test creating a Poly1D from a list of coefficients."""
        coeffs = [1.0, 2.0, 3.0]
        poly = Poly1D(degree=2, coefficients=coeffs)

        assert poly.degree == 2
        np.testing.assert_array_equal(poly.coefficients, coeffs)

    def test_creation_from_array(self):
        """Test creating a Poly1D from a numpy array."""
        coeffs = np.array([1.0, 2.0, 3.0])
        poly = Poly1D(degree=2, coefficients=coeffs)

        np.testing.assert_array_equal(poly.coefficients, coeffs)

    def test_as_polynomial(self):
        """Test conversion to numpy.polynomial.Polynomial."""
        coeffs = [1.0, 2.0, 3.0]
        poly = Poly1D(degree=2, coefficients=coeffs)

        np_poly = poly.as_numpy_polynomial()
        assert isinstance(np_poly, Polynomial)
        np.testing.assert_array_equal(np_poly.coef, coeffs)

    def test_evaluation_scalar(self):
        """Test evaluating the polynomial at a scalar value."""
        coeffs = [1.0, 2.0, 3.0]  # 1 + 2x + 3x^2
        poly = Poly1D(degree=2, coefficients=coeffs)

        result = poly(2.0)
        expected = 1.0 + 2.0 * 2.0 + 3.0 * 2.0**2
        assert result == pytest.approx(expected)

    def test_evaluation_array(self):
        """Test evaluating the polynomial at array values."""
        coeffs = [1.0, 2.0, 3.0]
        poly = Poly1D(degree=2, coefficients=coeffs)

        x = np.array([0.0, 1.0, 2.0])
        result = poly(x)
        expected = np.array([1.0, 6.0, 17.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_invalid_coefficients_shape(self):
        """Test that tells 2D coefficients are rejected."""
        with pytest.raises(ValueError, match="coefficients must be 1D"):
            Poly1D(degree=1, coefficients=[[1.0, 2.0], [3.0, 4.0]])


class TestPoly2D:
    """Tests for Poly2D polynomial wrapper."""

    def test_creation(self):
        """Test creating a Poly2D from a 2D array."""
        coeffs = [[1.0, 2.0], [3.0, 4.0]]
        poly = Poly2D(degree=(1, 1), coefficients=coeffs)
        assert poly.degree == (1, 1)
        np.testing.assert_array_equal(poly.coefficients, coeffs)

        poly = Poly2D(degree=1, coefficients=coeffs)
        assert poly.degree == (1, 1)
        np.testing.assert_array_equal(poly.coefficients, coeffs)

    def test_evaluation_scalar(self):
        """Test evaluating 2D polynomial at scalar values."""
        coeffs = [[1.0, 2.0], [3.0, 4.0]]  # 1 + 2y + 3x + 4xy
        poly = Poly2D(degree=1, coefficients=coeffs)

        result = poly(2.0, 3.0)
        expected = 1.0 + 2.0 * 3.0 + 3.0 * 2.0 + 4.0 * 2.0 * 3.0
        assert result == pytest.approx(expected)

    def test_evaluation_array(self):
        """Test evaluating 2D polynomial at array values."""
        coeffs = [[1.0, 2.0], [3.0, 0.0]]  # 1 + 2y + 3x
        poly = Poly2D(degree=1, coefficients=coeffs)

        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        result = poly(x, y)
        expected = np.array([1.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_higher_order(self):
        """Test higher-order polynomial evaluation."""
        coeffs = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.25, 0.0, 0.0]]
        poly = Poly2D(degree=2, coefficients=coeffs)

        result = poly(2.0, 2.0)
        expected = 1.0 + 0.5 * 2.0**2 + 1.0 * 2.0 * 2.0 + 0.25 * 2.0**2
        assert result == pytest.approx(expected)

    def test_from_fit(self):
        """Test fitting a 2D polynomial to grid data."""
        # Fit to a linear surface: p(x, y) = 1 + 2*x + 3*y
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        data = np.array(
            [
                [1.0, 4.0, 7.0],  # x=0: 1+0+3*y
                [3.0, 6.0, 9.0],  # x=1: 1+2+3*y
                [5.0, 8.0, 11.0],  # x=2: 1+4+3*y
            ]
        )
        poly = Poly2D.from_fit(x, y, data, degree=1)

        # Expected coefficients: c[i,j] for x^i * y^j
        # p(x,y) = 1 + 2*x + 3*y -> c[0,0]=1, c[1,0]=2, c[0,1]=3, c[1,1]=0
        np.testing.assert_allclose(
            poly.coefficients, [[1.0, 3.0], [2.0, 0.0]], atol=1e-10
        )

        # Verify the fit by evaluating at grid points
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                assert poly(xi, yj) == pytest.approx(data[i, j])

    def test_invalid_coefficients_shape(self):
        """Test that tells 1D coefficients are rejected."""
        with pytest.raises(ValueError, match="coefficients must be 2D"):
            Poly2D(degree=1, coefficients=[1.0, 2.0, 3.0])

    def test_from_fit_with_1d_samples_no_cross_terms(self):
        """Test 1D sample fitting without cross terms."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 0.0, 1.0, 0.0])
        data = 1.0 + 2.0 * x + 3.0 * y

        poly = Poly2D.from_fit(x, y, data, degree=1, cross_terms=False, robust=False)

        np.testing.assert_allclose(poly.coefficients[0, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(poly.coefficients[1, 0], 2.0, atol=1e-10)
        np.testing.assert_allclose(poly.coefficients[0, 1], 3.0, atol=1e-10)
        np.testing.assert_allclose(poly.coefficients[1, 1], 0.0, atol=1e-10)
