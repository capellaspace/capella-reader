"""Polynomial wrappers using numpy.polynomial.Polynomial."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval2d, polyvander2d
from pydantic import BaseModel, Field, field_serializer, field_validator


class Poly1D(BaseModel, arbitrary_types_allowed=True):
    """1D polynomial p(x) = sum c[i] x^i using numpy.polynomial.Polynomial."""

    degree: int = Field(..., description="Polynomial degree (order)")
    coefficients: np.ndarray = Field(
        ...,
        description="1D array of coefficients [c0, c1, ..., cN]",
    )

    @field_validator("coefficients", mode="before")
    @classmethod
    def _coeffs_to_1d_array(cls, v: Any) -> np.ndarray:
        arr = np.asanyarray(v, dtype=float)
        if arr.ndim != 1:
            msg = f"Poly1D coefficients must be 1D, got shape {arr.shape}"
            raise ValueError(msg)
        return arr

    @field_serializer("coefficients")
    def _serialize_coefficients(self, coefficients: np.ndarray) -> list[float]:
        """Serialize numpy array to list for JSON output."""
        return coefficients.tolist()

    def as_numpy_polynomial(self) -> Polynomial:
        """Convert to numpy.polynomial.Polynomial."""
        return Polynomial(self.coefficients)

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the polynomial at x."""
        return self.as_numpy_polynomial()(x)


class Poly2D(BaseModel, arbitrary_types_allowed=True):
    """2D polynomial p(x,y) = sum c[i,j] x^i y^j."""

    degree: tuple[int, int] = Field(..., description="Polynomial degree for (x, y)")
    coefficients: np.ndarray = Field(
        ...,
        description="2D array of coefficients [i, j] -> c_ij",
    )

    # let user pass int as degree:
    @field_validator("degree", mode="before")
    @classmethod
    def _degree_to_tuple(cls, v: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(v, int):
            return v, v
        return v

    @field_validator("coefficients", mode="before")
    @classmethod
    def _coeffs_to_2d_array(cls, v: Any) -> np.ndarray:
        arr = np.asanyarray(v, dtype=float)
        if arr.ndim != 2:
            msg = f"Poly2D coefficients must be 2D, got shape {arr.shape}"
            raise ValueError(msg)
        return arr

    @field_serializer("coefficients")
    def _serialize_coefficients(self, coefficients: np.ndarray) -> list[list[float]]:
        """Serialize numpy array to nested list for JSON output."""
        return coefficients.tolist()

    def __call__(
        self, x: float | np.ndarray, y: float | np.ndarray
    ) -> float | np.ndarray:
        """Evaluate the polynomial at (x, y).

        p(x,y) = sum_{i,j} c[i,j] * x^i * y^j
        """
        x_arr = np.asanyarray(x)
        y_arr = np.asanyarray(y)
        return polyval2d(x_arr, y_arr, self.coefficients)

    @classmethod
    def from_fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        data: np.ndarray,
        degree: int | tuple[int, int] = 1,
        # TODO: this seems hacky, not sure how to organize this
        # for the coreg. polynomial
        cross_terms: bool = True,
        weights: np.ndarray | None = None,
        robust: bool = True,
    ) -> Self:
        """Fit a 2D polynomial to gridded data.

        Parameters
        ----------
        x : np.ndarray
            1D array of x coordinates.
        y : np.ndarray
            1D array of y coordinates.
        data : np.ndarray
            2D array of data values where data[i, j] = f(x[i], y[j]).
        degree : int or tuple[int, int]
            Degree of the polynomial to fit.
        cross_terms : bool
            Whether to include cross terms in the fit.
            Default is True.
        weights : np.ndarray | None
            Weights for each data point.
            If None, uniform weights are used.
        robust : bool
            Whether to perform a robust fitting iteration by reweighting
            data based on the MAD of the residuals.
            Default is True

        Returns
        -------
        Poly2D
            Fitted 2D polynomial.

        Examples
        --------
        Fit a linear surface to a 3x3 grid:

        >>> x = np.array([0., 1., 2.])
        >>> y = np.array([0., 1., 2.])
        >>> data = np.array([[1., 2., 3.],
        ...                  [2., 3., 4.],
        ...                  [3., 4., 5.]])
        >>> poly = Poly2D.from_fit(x, y, data, degree=1)

        """
        if isinstance(degree, int):
            degree = (degree, degree)

        # Flatten for fitting
        if data.shape == (len(y), len(x)):
            # Create meshgrid for all combinations of x and y
            x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
            x_flat = x_grid.ravel()
            y_flat = y_grid.ravel()
            data_flat = data.ravel()
        elif data.shape == x.shape == y.shape:
            x_flat, y_flat, data_flat = x, y, data
        else:
            msg = (
                "data must be 2D array with shape (len(y), len(x)),"
                " or 1D arrays with same length as x and y"
            )
            raise ValueError(msg)

        if weights is None:
            weights = np.ones_like(data_flat)
        for _ in range(1 + int(robust)):
            # Create Vandermonde matrix and solve
            vander = polyvander2d(x_flat, y_flat, degree)
            if not cross_terms:
                # TODO: this isn't right for degree above 1
                vander = vander[:, :3]
            coeffs_flat = np.linalg.lstsq(
                vander * weights[:, None], data_flat * weights, rcond=None
            )[0]
            if not cross_terms:
                coeffs = np.zeros((degree[0] + 1, degree[1] + 1))
                coeffs.ravel()[:3] = coeffs_flat
            else:
                coeffs = coeffs_flat.reshape(degree[0] + 1, degree[1] + 1)

            # Add robust weighting from the residuals
            r = polyval2d(x_flat, y_flat, coeffs) - data_flat
            med = np.nanmedian(np.abs(r))
            mad = 1.4286 * np.nanmedian(np.abs(r - med))
            # Weight by Tukey's biweight
            if mad > 0:
                u = (r - med) / (3 * mad)
                w_robust = (1 - u**2) ** 2
                w_robust[np.abs(u) >= 1] = 0.0
                weights *= w_robust

        return cls(degree=degree, coefficients=coeffs)
