#!/usr/bin/env python3
"""Example user script for headless_plot_jobs.py."""

from headless_plot_jobs import run_plot_jobs

plot_jobs = [
    {
        "dataset": "sample_1.grim",
        "plot_type": "azimuth_rect",
        "variables": {
            "azimuths": [-180.0, -90.0, 0.0, 90.0, 180.0],
            "frequencies": [10.0, 20.0, 30.0],
            "elevations": [0.0],
            "polarizations": ["HH"],
        },
        "plot_scale": "dbsm",
        "legend": True,
        "x_min": -180.0,
        "x_max": 180.0,
        "y_min": -40.0,
        "y_max": 20.0,
        "output": "example_azimuth_rect.png",
    },
    {
        "datasets": ["sample_1.grim", "sample_2.grim"],
        "plot_type": "frequency",
        "variables": {
            "azimuths": [-30.0, 0.0, 30.0],
            "frequencies": [10.0, 15.0, 20.0, 25.0, 30.0],
            "elevations": [0.0],
            "polarizations": ["HH"],
        },
        "plot_scale": "dbsm",
        "legend": True,
        "x_min": 10.0,
        "x_max": 30.0,
        "y_min": -50.0,
        "y_max": 10.0,
        "output": "example_frequency_compare.png",
    },
    {
        "dataset": "sample_1.grim",
        "plot_type": "waterfall",
        "variables": {
            "azimuths": [-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0],
            "frequencies": [10.0, 15.0, 20.0, 25.0, 30.0],
            "elevations": [0.0],
            "polarizations": ["HH"],
        },
        "colormap": "viridis",
        "x_min": -180.0,
        "x_max": 180.0,
        "y_min": 10.0,
        "y_max": 30.0,
        "z_min": -10.0,
        "z_max": 0.0,
        "output": "example_waterfall_clamped.png",
    },
]

run_plot_jobs(plot_jobs=plot_jobs, output_dir="headless_outputs")
