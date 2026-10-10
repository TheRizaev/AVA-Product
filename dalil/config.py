"""All tunable parameters in one place. Modules take a `params: Params` argument."""

from dataclasses import dataclass, asdict


@dataclass
class Params:
    # working resolution / preprocessing
    work_max_side: int = 1024
    blur_sigma: float = 1.0

    # PBP window grid
    win: int = 8
    step: int = 4
    eps: float = 0.05          # epsilon-ties, fraction of window contrast
    flat_floor: float = 6.0    # window max-min below this (gray levels) => "flat"

    # comparison / zones
    tolerance: int = 1         # +-cells searched when comparing signatures
    threshold: float = 0.35    # PBP threshold; placeholder until eval/benchmark.py calibration
    threshold_pixdiff: float = 0.08   # baseline thresholds (placeholders, calibrated by eval/benchmark.py)
    threshold_ssim: float = 0.25
    threshold_census: float = 0.25
    mismatch_frac: float = 0.4        # changed area above this fraction => "mismatch" (wrong / absent part)
    min_zone_cells: int = 4
    merge_gap_px: int = 12
    max_zones: int = 12
    blob_min_count: int = 6
    water_blob_max_cells: int = 40     # a "drop" blob has at most this many cells
    water_min_roundness: float = 0.45  # min fill-ratio * (short/long side) of a drop blob
    removed_ratio: float = 0.6         # zone activity after/before below => something_removed
    added_ratio: float = 1.6           # zone activity after/before above => something_added

    # alignment
    orb_features: int = 3000
    ratio_test: float = 0.75
    ransac_thresh: float = 4.0
    min_inliers: int = 40
    min_inlier_ratio: float = 0.25
    max_persp_distortion: float = 0.25  # max |h31|*W + |h32|*H of a normalised homography
    align_erode_px: int = 4             # erosion of the warped-after valid mask
    fixed_cam_ncc: float = 0.97         # NCC above this => frame already registered (identity fast path)
    fixed_cam_max_shift: float = 1.0    # identity path also needs visible (RMS) motion <= this (px)
    ecc_min_corr: float = 0.85          # ECC fallback accepted above this correlation
    max_references: int = 5

    # capture quality / glare
    glare_v: int = 250
    glare_s: int = 40
    glare_dilate_px: int = 9
    max_glare_frac: float = 0.35   # above => retake (glare)
    min_blur: float = 30.0         # variance of Laplacian below => blurry
    dark_mean: float = 35.0        # mean gray below => too dark
    min_coverage: float = 0.5      # below => retake (low_coverage)

    def to_dict(self) -> dict:
        return asdict(self)
