#!/usr/bin/env python3
"""Generate a synthetic .grim dataset designed to showcase 3D ISAR.

The generator creates coherent monostatic RCS measurements from a set of
3D scatterers. The resulting cube (az, el, f, pol) is suitable for the
"3D ISAR" mode in the GRIM GUI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from grim_dataset import RcsGrid

C0 = 299_792_458.0


def build_demo_scatterers() -> list[tuple[float, float, float, float, float]]:
    """Return scatterers as (x_m, y_m, z_m, amplitude, phase_rad)."""
    scatterers: list[tuple[float, float, float, float, float]] = [
        # Main body (x-axis line).
        (-0.90, 0.00, 0.00, 0.95, 0.00),
        (-0.65, 0.00, 0.00, 0.90, 0.12),
        (-0.40, 0.00, 0.00, 1.00, 0.20),
        (-0.15, 0.00, 0.00, 1.10, 0.32),
        (0.10, 0.00, 0.00, 1.05, 0.42),
        (0.35, 0.00, 0.00, 0.95, 0.56),
        (0.60, 0.00, 0.00, 0.85, 0.70),
        # Wing-like lateral features.
        (-0.05, 0.70, 0.02, 0.75, 0.25),
        (0.00, 0.95, 0.03, 0.70, 0.40),
        (0.05, 0.70, 0.02, 0.75, 0.55),
        (-0.05, -0.70, 0.02, 0.75, 0.10),
        (0.00, -0.95, 0.03, 0.70, 0.28),
        (0.05, -0.70, 0.02, 0.75, 0.48),
        # Vertical structure for z-resolution.
        (-0.55, 0.05, 0.20, 0.58, 0.18),
        (-0.55, 0.05, 0.38, 0.54, 0.34),
        (-0.55, 0.05, 0.56, 0.50, 0.47),
        # Offset bright cluster to highlight separated features in 3D.
        (0.45, 0.35, -0.25, 0.92, 0.62),
        (0.58, 0.40, -0.20, 0.82, 0.79),
        (0.36, 0.26, -0.30, 0.78, 0.53),
        # Lower aft points.
        (-0.80, -0.25, -0.30, 0.55, 0.05),
        (-0.95, -0.15, -0.24, 0.52, 0.16),
    ]
    return scatterers


def build_dataset(
    *,
    az_count: int,
    el_count: int,
    f_count: int,
    az_start_deg: float,
    az_stop_deg: float,
    el_start_deg: float,
    el_stop_deg: float,
    f_start_ghz: float,
    f_stop_ghz: float,
    noise_db: float,
    seed: int,
) -> RcsGrid:
    azimuths = np.linspace(az_start_deg, az_stop_deg, az_count, endpoint=True, dtype=float)
    elevations = np.linspace(el_start_deg, el_stop_deg, el_count, endpoint=True, dtype=float)
    frequencies_ghz = np.linspace(f_start_ghz, f_stop_ghz, f_count, endpoint=True, dtype=float)
    frequencies_hz = frequencies_ghz * 1e9
    polarizations = np.asarray(["HH", "HV", "VH", "VV"])

    az_rad = np.deg2rad(azimuths)
    el_rad = np.deg2rad(elevations)
    k_mag = (4.0 * np.pi / C0) * frequencies_hz

    cos_az = np.cos(az_rad)[:, None, None]
    sin_az = np.sin(az_rad)[:, None, None]
    cos_el = np.cos(el_rad)[None, :, None]
    sin_el = np.sin(el_rad)[None, :, None]
    kx = k_mag[None, None, :] * cos_el * cos_az
    ky = k_mag[None, None, :] * cos_el * sin_az
    kz = np.ones((az_count, 1, 1), dtype=float) * k_mag[None, None, :] * sin_el

    rcs_base = np.zeros((az_count, el_count, f_count), dtype=np.complex128)
    for x_m, y_m, z_m, amp, phase0 in build_demo_scatterers():
        phase = kx * x_m + ky * y_m + kz * z_m + phase0
        rcs_base += amp * np.exp(-1j * phase)

    # Gentle frequency taper to avoid a perfectly ideal synthetic response.
    f_norm = (frequencies_ghz - frequencies_ghz.min()) / max(np.ptp(frequencies_ghz), 1e-12)
    taper = 0.92 + 0.16 * f_norm
    rcs_base *= taper[None, None, :]

    rng = np.random.default_rng(seed)
    max_mag = float(np.max(np.abs(rcs_base))) if rcs_base.size else 1.0
    noise_sigma = max_mag * (10.0 ** (noise_db / 20.0))
    noise = (noise_sigma / np.sqrt(2.0)) * (
        rng.standard_normal(rcs_base.shape) + 1j * rng.standard_normal(rcs_base.shape)
    )
    rcs_base = rcs_base + noise

    pol_scales = np.asarray([1.00, 0.78, 0.74, 0.88], dtype=float)
    pol_phases = np.asarray([0.00, 0.26, -0.21, 0.37], dtype=float)
    rcs = np.zeros((az_count, el_count, f_count, len(polarizations)), dtype=np.complex128)
    for p_idx in range(len(polarizations)):
        rcs[..., p_idx] = pol_scales[p_idx] * rcs_base * np.exp(1j * pol_phases[p_idx])

    history = (
        "Synthetic 3D ISAR demo: coherent point-scatterer scene with elevation aperture, "
        f"{len(build_demo_scatterers())} scatterers, noise {noise_db:.1f} dB amplitude."
    )
    return RcsGrid(
        azimuths=azimuths,
        elevations=elevations,
        frequencies=frequencies_ghz,
        polarizations=polarizations,
        rcs=rcs,
        history=history,
        units={
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("demo_3d_isar.grim"),
        help="Output .grim path (default: %(default)s)",
    )
    parser.add_argument("--az-count", type=int, default=81, help="Azimuth sample count.")
    parser.add_argument("--el-count", type=int, default=49, help="Elevation sample count.")
    parser.add_argument("--f-count", type=int, default=96, help="Frequency sample count.")
    parser.add_argument("--az-start-deg", type=float, default=-60.0, help="Start azimuth (deg).")
    parser.add_argument("--az-stop-deg", type=float, default=60.0, help="Stop azimuth (deg).")
    parser.add_argument("--el-start-deg", type=float, default=-25.0, help="Start elevation (deg).")
    parser.add_argument("--el-stop-deg", type=float, default=25.0, help="Stop elevation (deg).")
    parser.add_argument("--f-start-ghz", type=float, default=8.0, help="Start frequency (GHz).")
    parser.add_argument("--f-stop-ghz", type=float, default=12.5, help="Stop frequency (GHz).")
    parser.add_argument(
        "--noise-db",
        type=float,
        default=-35.0,
        help="Complex noise amplitude in dB relative to peak (negative recommended).",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatability.")
    args = parser.parse_args()

    if args.az_count < 2 or args.el_count < 2 or args.f_count < 2:
        raise SystemExit("az-count, el-count, and f-count must each be >= 2.")
    if args.az_start_deg >= args.az_stop_deg:
        raise SystemExit("az-start-deg must be < az-stop-deg.")
    if args.el_start_deg >= args.el_stop_deg:
        raise SystemExit("el-start-deg must be < el-stop-deg.")
    if args.f_start_ghz >= args.f_stop_ghz:
        raise SystemExit("f-start-ghz must be < f-stop-ghz.")

    dataset = build_dataset(
        az_count=args.az_count,
        el_count=args.el_count,
        f_count=args.f_count,
        az_start_deg=args.az_start_deg,
        az_stop_deg=args.az_stop_deg,
        el_start_deg=args.el_start_deg,
        el_stop_deg=args.el_stop_deg,
        f_start_ghz=args.f_start_ghz,
        f_stop_ghz=args.f_stop_ghz,
        noise_db=args.noise_db,
        seed=args.seed,
    )
    saved_path = dataset.save(str(args.output))

    print(f"Wrote dataset: {saved_path}")
    print(
        "Shape: "
        f"az={len(dataset.azimuths)} "
        f"el={len(dataset.elevations)} "
        f"f={len(dataset.frequencies)} "
        f"pol={len(dataset.polarizations)}"
    )
    print(f"History: {dataset.history}")


if __name__ == "__main__":
    main()
