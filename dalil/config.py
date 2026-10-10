"""All tunable parameters in one place. Modules take a `params: Params` argument."""

from dataclasses import dataclass, asdict


@dataclass
class Params:
    # working resolution / preprocessing
    work_max_side: int = 1024
    blur_sigma: float = 1.5

    # PBP window grid
    win: int = 8
    step: int = 4
    eps: float = 0.05          # epsilon-ties, fraction of window contrast
    flat_floor: float = 6.0    # window max-min below this (gray levels) => "flat"

    # comparison / zones
    tolerance: int = 1         # +-cells searched when comparing signatures
    # Change-score thresholds, frozen by the CALIB split of eval/benchmark.py (eval/results/calib.json):
    # smallest threshold with <= 0.05 false events per good pair under none + monotone lighting.
    threshold: float = 0.60976          # PBP
    threshold_pixdiff: float = 0.2512
    threshold_ssim: float = 0.13503
    threshold_census: float = 0.39137
    mismatch_frac: float = 0.15       # changed area above this fraction => "mismatch" (wrong / absent /
                                      # misregistered part); largest synthetic defect: 0.054 (eval/benchmark.py)
    min_zone_cells: int = 4
    contrast_floor: float = 10.0       # PBP: cells where ref and frame window contrast are both below => unchanged
    score_smooth: int = 3              # box filter (cells) on the change map before zones; <= 1 disables
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
    fixed_cam_max_shift: float = 0.25   # identity path also needs visible (RMS) motion <= this (px)
    ecc_min_corr: float = 0.85          # ECC fallback accepted above this correlation
    max_references: int = 5
    static_background: bool = True      # part moved: also compare the unregistered frame, keep the lower score
    static_bg_min_shift: float = 3.0    # ... only when the registration moves some image corner by more px
    part_roi_margin_px: int = 8         # two motions found: check only the part (hull of its keypoints + margin)
    part_min_inliers: int = 15          # static background dominant: min inliers of the part's similarity motion
    part_ncc_margin: float = 0.05       # ... accepted only if it beats identity by this NCC margin on its region

    # capture quality / glare
    glare_v: int = 250
    glare_s: int = 40
    glare_dilate_px: int = 9
    max_glare_frac: float = 0.35   # above => retake (glare)
    min_blur: float = 30.0         # contrast-normalised variance of Laplacian below => blurry ...
    blur_ratio: float = 0.15       # ... and also below this fraction of the reference's score
    dark_mean: float = 35.0        # mean gray below => too dark
    min_coverage: float = 0.5      # below => retake (low_coverage)
    border_margin_px: int = 12     # cells this close to the image border are not checked

    # Mode 1: single-photo surface check (dalil/surface.py). Thresholds frozen by the CALIB split of
    # eval/benchmark_surface.py (eval/results/surface_calib.json): smallest threshold with a
    # false-alarm rate <= 5 % on defect-free photos under none + monotone lighting.
    surface_threshold: float = 0.23764    # PBP self-reference anomaly score threshold
    surface_samples: int = 768            # windows sampled from the same image as "look-alike" pool
    surface_k: int = 5                    # anomaly = mean of the k smallest distances
    surface_exclude_cells: int = 6        # samples within this many cells of the window are skipped
    surface_candidates: int = 8           # approximate (Hellinger) neighbours re-ranked exactly
    surface_seed: int = 20261011          # fixed sampling seed (deterministic results)
    surface_noise_floor: float = 6.0      # window contrast where structure starts to count ...
    surface_noise_full: float = 14.0      # ... and where it fully counts (gray levels, after blur)
    surface_rel_floor: float = 2.5        # ... or, if higher, these multiples of the image's 25th-
    surface_rel_full: float = 5.0         # percentile window contrast (random micro-texture: gain-free)
    surface_contrast_weight: float = 0.8  # weight of the robust contrast-outlier term
    surface_contrast_z: float = 2.5       # robust z of log-contrast where the term starts ...
    surface_contrast_span: float = 3.0    # ... and the z range over which it ramps to 1
    surface_smooth: int = 2               # box mean (cells) of the score map; <= 1 disables
    surface_glare_dilate_px: int = 12     # extra margin around the glare region (core + halo)
    surface_glare_halo_px: int = 320      # halo: brighter-than-surroundings pixels connected to a
    surface_glare_elev: float = 5.0       # highlight core, within this distance, >= this many levels
    threshold_surface_gradient: float = 0.53498   # gradient baseline (robust z / 10)
    threshold_surface_census: float = 0.14604     # census kNN baseline

    def to_dict(self) -> dict:
        return asdict(self)
