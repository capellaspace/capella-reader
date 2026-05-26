"""Shared cross-correlation utilities for SAR coregistration.

Cross-correlation is delegated to ``skimage.registration.phase_cross_correlation``
with ``normalization=None`` (unnormalized cross-correlation, recommended for noisy
SAR amplitude imagery per Guizar et al. 2008).

References
----------
.. [1] Guizar-Sicairos, M., Thurman, S. T., & Fienup, J. R. (2008).
   Efficient subpixel image registration algorithms. Optics Letters, 33(2),
   156-158. https://doi.org/10.1364/OL.33.000156
"""

import numpy as np
from skimage.registration import phase_cross_correlation


def correlate_grid(
    ref_data: np.ndarray,
    sec_data: np.ndarray,
    *,
    chip_size: tuple[int, int] = (256, 256),
    upsample_factor: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-pixel offsets on a regular grid of chips.

    Parameters
    ----------
    ref_data : np.ndarray
        Reference image data.
    sec_data : np.ndarray
        Secondary image data (same shape as ``ref_data``, already coarsely
        resampled onto the reference grid).
    chip_size : tuple[int, int]
        Size of correlation chips (rows, cols).
    upsample_factor : int
        Sub-pixel refinement factor passed to ``phase_cross_correlation``.

    Returns
    -------
    az_off, rg_off : np.ndarray
        Azimuth (row) and range (col) offsets at each chip, in resample
        convention (value to ADD to a reference pixel index to find the
        corresponding secondary pixel). NaN where the chip was skipped.
    error : np.ndarray
        ``phase_cross_correlation`` normalized RMS error at each chip
        (0 = perfect, ~1 = no correlation).
    """
    assert ref_data.shape == sec_data.shape
    ref = np.abs(ref_data) if np.iscomplexobj(ref_data) else ref_data
    sec = np.abs(sec_data) if np.iscomplexobj(sec_data) else sec_data

    nrows, ncols = ref.shape
    ch, cw = chip_size

    row_centers = np.arange(ch // 2, nrows - ch // 2, ch)
    col_centers = np.arange(cw // 2, ncols - cw // 2, cw)
    n_points = len(row_centers) * len(col_centers)

    az_off = np.full(n_points, np.nan)
    rg_off = np.full(n_points, np.nan)
    err = np.full(n_points, np.nan)

    i = 0
    for rc in row_centers:
        for ccc in col_centers:
            r0, r1 = rc - ch // 2, rc + ch // 2
            c0, c1 = ccc - cw // 2, ccc + cw // 2
            shift, error, _ = phase_cross_correlation(
                ref[r0:r1, c0:c1],
                sec[r0:r1, c0:c1],
                upsample_factor=upsample_factor,
                normalization=None,  # type: ignore[arg-type]
            )
            # phase_cross_correlation returns the shift needed to register
            # `moving` (secondary) onto `reference`; resample wants the inverse.
            az_off[i] = -float(shift[0])
            rg_off[i] = -float(shift[1])
            err[i] = float(error)
            i += 1
            if i % 50 == 0 or i == n_points:
                print(f"  Correlated {i}/{n_points} chips", end="\r")

    print()
    return az_off, rg_off, err


def bulk_offset(
    az_off: np.ndarray,
    rg_off: np.ndarray,
    *,
    n_mad: float = 3.0,
) -> tuple[float, float, np.ndarray]:
    """Robust constant (az, rg) shift from per-chip offsets.

    Iteratively clips outliers more than ``n_mad`` MADs from the median in
    either axis, then returns the median of what remains.

    Returns
    -------
    az_median, rg_median : float
        Robust constant offsets.
    inliers : np.ndarray
        Boolean mask of which input chips were used.
    """
    inliers = np.isfinite(az_off) & np.isfinite(rg_off)
    assert inliers.any(), "No finite chip offsets to take a median over"
    for _ in range(3):
        az_med = np.median(az_off[inliers])
        rg_med = np.median(rg_off[inliers])
        az_mad = 1.4826 * np.median(np.abs(az_off[inliers] - az_med)) + 1e-6
        rg_mad = 1.4826 * np.median(np.abs(rg_off[inliers] - rg_med)) + 1e-6
        new_inliers = (
            np.isfinite(az_off)
            & np.isfinite(rg_off)
            & (np.abs(az_off - az_med) < n_mad * az_mad)
            & (np.abs(rg_off - rg_med) < n_mad * rg_mad)
        )
        if new_inliers.sum() == inliers.sum():
            break
        inliers = new_inliers
    return float(np.median(az_off[inliers])), float(np.median(rg_off[inliers])), inliers
