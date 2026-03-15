"""
Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust
Based on: Wen et al., arXiv:2305.20030 (Tree-Ring Watermarks)

Watermark is embedded in the Fourier space of the initial noise vector x_T.
Detection: DDIM inversion to recover noise, then check for key in Fourier space.
No training required; works as a plug-in for arbitrary diffusion models.
"""

import numpy as np
from typing import Optional, Tuple, Literal
from scipy import stats


def _fft2(x: np.ndarray) -> np.ndarray:
    """2D FFT (handles real or complex)."""
    return np.fft.fft2(x)


def _ifft2(x: np.ndarray) -> np.ndarray:
    """2D IFFT."""
    return np.fft.ifft2(x).real


def _get_circular_mask(h: int, w: int, r: int, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Binary mask for low-frequency circular region (radius r in frequency bins)."""
    if center is None:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = center
    y = np.arange(h)[:, None] - cy
    x = np.arange(w)[None, :] - cx
    return (y ** 2 + x ** 2 <= r ** 2).astype(np.float64)


# ---------------------------------------------------------------------------
# Key construction (Section 3.3)
# ---------------------------------------------------------------------------


def make_key_tree_ring_zeros(shape: Tuple[int, ...], mask: np.ndarray) -> np.ndarray:
    """
    Tree-RingZeros: key is zeros in circular mask.
    Invariant to shifts, crops, dilations; not rotation-invariant in pixel space
    (but rotation in pixel space = rotation in Fourier, so ring is rotation-invariant).
    """
    key = np.zeros(shape, dtype=np.complex128)
    key[mask > 0] = 0.0
    return key


def make_key_tree_ring_rand(shape: Tuple[int, ...], mask: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Tree-RingRand: key drawn from Gaussian (same stats as natural Fourier noise).
    Allows multiple keys per model; not invariant to image transforms.
    """
    rng = np.random.default_rng(seed)
    key = np.zeros(shape, dtype=np.complex128)
    n = int(mask.sum())
    key[mask > 0] = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return key


def make_key_tree_ring_rings(
    shape: Tuple[int, ...],
    mask: np.ndarray,
    num_rings: int = 4,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Tree-RingRings: constant value per ring in Fourier space; rotation-invariant.
    Ring values drawn from Gaussian to minimize distribution shift.
    """
    h, w = shape[:2]
    cy, cx = h // 2, w // 2
    y = np.arange(h)[:, None] - cy
    x = np.arange(w)[None, :] - cx
    r_dist = np.sqrt(y ** 2 + x ** 2)
    key = np.zeros(shape, dtype=np.complex128)
    rng = np.random.default_rng(seed)
    r_max = min(cy, cx)
    for i in range(num_rings):
        r_lo = (i * r_max) / num_rings
        r_hi = ((i + 1) * r_max) / num_rings
        ring_mask = (r_dist >= r_lo) & (r_dist < r_hi) & (mask > 0)
        n = int(ring_mask.sum())
        if n > 0:
            val = rng.standard_normal() + 1j * rng.standard_normal()
            key[ring_mask] = val
    return key


# ---------------------------------------------------------------------------
# Watermark injection (noise in pixel/latent space)
# ---------------------------------------------------------------------------


def inject_watermark_noise(
    noise_shape: Tuple[int, ...],
    key_type: Literal["zeros", "rand", "rings"] = "rings",
    radius: int = 10,
    seed: Optional[int] = None,
    num_rings: int = 4,
) -> np.ndarray:
    """
    Create initial noise vector with Tree-Ring key in Fourier space.
    noise_shape: (C, H, W) e.g. (4, 64, 64) for SD latent.
    Returns noise array in same shape (real-valued).
    """
    *spatial, h, w = noise_shape
    if spatial:
        # e.g. (4, 64, 64) -> treat each channel separately with same key pattern
        out = np.zeros(noise_shape, dtype=np.float64)
        for c in range(noise_shape[0]):
            out[c] = inject_watermark_noise(
                (h, w), key_type=key_type, radius=radius, seed=seed, num_rings=num_rings
            )
        return out
    # 2D
    h, w = noise_shape
    mask = _get_circular_mask(h, w, radius)
    if key_type == "zeros":
        key = make_key_tree_ring_zeros((h, w), mask)
    elif key_type == "rand":
        key = make_key_tree_ring_rand((h, w), mask, seed=seed)
    elif key_type == "rings":
        key = make_key_tree_ring_rings((h, w), mask, num_rings=num_rings, seed=seed)
    else:
        raise ValueError("key_type must be 'zeros', 'rand', or 'rings'")
    # Start from Gaussian noise, then overwrite Fourier region with key
    rng = np.random.default_rng(seed if key_type != "rand" else None)
    noise = rng.standard_normal((h, w)) + 1j * rng.standard_normal((h, w))
    noise = _fft2(noise)  # actually we want: real noise -> FFT -> replace mask -> IFFT
    # Simpler: sample real noise, FFT, replace masked coeffs with key, IFFT
    base_noise = rng.standard_normal((h, w)).astype(np.float64)
    f = _fft2(base_noise)
    f[mask > 0] = key[mask > 0]
    out = _ifft2(f)
    return out.astype(np.float32)


def inject_watermark_noise_latent(
    latent_shape: Tuple[int, int, int],
    key_type: Literal["zeros", "rand", "rings"] = "rings",
    radius: int = 10,
    seed: Optional[int] = None,
    num_rings: int = 4,
) -> np.ndarray:
    """(C, H, W) latent shape; inject same 2D key pattern in each channel."""
    c, h, w = latent_shape
    mask = _get_circular_mask(h, w, radius)
    if key_type == "zeros":
        key_2d = make_key_tree_ring_zeros((h, w), mask)
    elif key_type == "rand":
        key_2d = make_key_tree_ring_rand((h, w), mask, seed=seed)
    else:
        key_2d = make_key_tree_ring_rings((h, w), mask, num_rings=num_rings, seed=seed)
    rng = np.random.default_rng(seed if key_type != "rand" else None)
    out = np.zeros(latent_shape, dtype=np.float32)
    for ch in range(c):
        base = rng.standard_normal((h, w)).astype(np.float64)
        f = _fft2(base)
        f[mask > 0] = key_2d[mask > 0]
        out[ch] = _ifft2(f).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Detection: distance and P-value (Section 3.4)
# ---------------------------------------------------------------------------


def detection_distance(
    inverted_noise_fourier: np.ndarray,
    key: np.ndarray,
    mask: np.ndarray,
) -> float:
    """L1 distance between inverted noise (in Fourier) and key on mask region. Eq. (3)."""
    region = inverted_noise_fourier[mask > 0]
    k_region = key[mask > 0]
    return np.abs(k_region - region).mean()


def detection_score_eta(
    inverted_noise_fourier: np.ndarray,
    key: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """
    Score eta for P-value (Eq. 5). Also returns estimated sigma^2 from image.
    """
    y = inverted_noise_fourier[mask > 0]
    k = key[mask > 0]
    sigma_sq = (np.abs(y) ** 2).mean()
    if sigma_sq < 1e-12:
        return 0.0, sigma_sq
    eta = (1.0 / sigma_sq) * np.sum(np.abs(k - y) ** 2)
    return float(eta), float(sigma_sq)


def p_value_tree_ring(
    eta: float,
    mask_size: int,
    key: np.ndarray,
    mask: np.ndarray,
    sigma_sq: float,
) -> float:
    """
    P-value under H0: y ~ N(0, sigma^2 I). Eq. (6).
    Noncentral chi-squared with |M| dof and noncentrality lambda = (1/sigma^2) sum |k*_i|^2.
    We declare watermarked if eta is *small* (distance to key is small), so p = P(chi2 <= eta).
    """
    lam = (1.0 / sigma_sq) * np.sum(np.abs(key[mask > 0]) ** 2)
    # scipy: noncentral chi2 with df = 2*|M| for complex (each complex = 2 real)?
    # Paper: "|M| degrees of freedom" and eta = (1/sigma^2) sum |k*_i - y_i|^2.
    # For complex Gaussian, |k - y|^2 is scaled chi-squared. We use df = 2*|M| for complex.
    df = 2 * mask_size  # complex numbers
    nc = (1.0 / sigma_sq) * lam  # noncentrality for sum of squared differences
    p = stats.ncx2.cdf(eta, df, nc)
    return float(p)


def build_key_for_detection(
    shape: Tuple[int, int],
    key_type: Literal["zeros", "rand", "rings"],
    radius: int,
    seed: Optional[int] = None,
    num_rings: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the key and mask used at generation (must match for detection)."""
    h, w = shape
    mask = _get_circular_mask(h, w, radius)
    if key_type == "zeros":
        key = make_key_tree_ring_zeros((h, w), mask)
    elif key_type == "rand":
        key = make_key_tree_ring_rand((h, w), mask, seed=seed)
    else:
        key = make_key_tree_ring_rings((h, w), mask, num_rings=num_rings, seed=seed)
    return key, mask


def detect_tree_ring(
    inverted_noise: np.ndarray,
    key_type: Literal["zeros", "rand", "rings"] = "rings",
    radius: int = 10,
    seed: Optional[int] = None,
    num_rings: int = 4,
    return_p_value: bool = True,
) -> dict:
    """
    inverted_noise: (C, H, W) latent noise from DDIM inversion of generated image.
    Returns dict with 'distance', 'eta', 'p_value', 'is_watermarked' (p < 0.01).
    """
    c, h, w = inverted_noise.shape
    key, mask = build_key_for_detection((h, w), key_type, radius, seed=seed, num_rings=num_rings)
    # Use first channel for 2D key (or average over channels)
    inv_2d = inverted_noise[0]
    inv_f = _fft2(inv_2d)
    dist = detection_distance(inv_f, key, mask)
    eta, sigma_sq = detection_score_eta(inv_f, key, mask)
    mask_size = int(mask.sum())
    p_val = p_value_tree_ring(eta, mask_size, key, mask, sigma_sq) if return_p_value else None
    return {
        "distance": float(dist),
        "eta": float(eta),
        "sigma_sq": float(sigma_sq),
        "p_value": p_val,
        "is_watermarked": p_val is not None and p_val < 0.01,
    }
