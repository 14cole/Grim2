#!/usr/bin/env python3
"""Headless GRIM plot runner.

Use as a module from a simple user script:

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
            "output": "azimuth_rect.png",
        },
    ]
    run_plot_jobs(plot_jobs=plot_jobs, output_dir="headless_outputs")

Or run this script directly:
    python3 headless_plot_jobs.py
    python3 headless_plot_jobs.py user_plot_jobs.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication, QListWidget

from grim_cut_gui import GrimCutWindow
from grim_dataset import RcsGrid


DEFAULT_DATASET_PATH = Path("sample_1.grim")
DEFAULT_OUTPUT_DIR = Path("headless_outputs")

# Available modes: azimuth_rect, azimuth_polar, frequency, waterfall, isar_image, isar_3d
DEFAULT_PLOT_JOBS: list[dict[str, Any]] = [
    {
        "mode": "azimuth_rect",
        "output": "azimuth_rect.png",
        "azimuths": [-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0],
        "frequencies": [10.0, 20.0, 30.0],
        "elevations": [-10.0, 0.0, 10.0],
        "polarizations": ["HH"],
        "plot_scale": "dbsm",
        "legend": True,
    },
    {
        "mode": "waterfall",
        "output": "waterfall.png",
        "azimuths": list(np.linspace(-180.0, 180.0, 25)),
        "frequencies": list(np.linspace(10.0, 30.0, 21)),
        "elevations": [0.0],
        "polarizations": ["HH"],
        "plot_scale": "dbsm",
        "colormap": "viridis",
        "colorbar": True,
        "shared_colorbar": True,
    },
]

MODE_TO_METHOD = {
    "azimuth_rect": GrimCutWindow._plot_azimuth_rect,
    "azimuth_polar": GrimCutWindow._plot_azimuth_polar,
    "frequency": GrimCutWindow._plot_frequency,
    "waterfall": GrimCutWindow._plot_waterfall,
    "isar_image": GrimCutWindow._plot_isar_image,
    "isar_3d": GrimCutWindow._plot_isar_3d,
}

MODE_TO_TAB = {
    "azimuth_rect": "plotting",
    "azimuth_polar": "plotting",
    "frequency": "plotting",
    "waterfall": "plotting",
    "isar_image": "isar",
    "isar_3d": "isar",
}

Z_LIMIT_MODES = {"waterfall", "isar_image", "isar_3d"}


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, os.PathLike)):
        return [value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _value_matches(item_value: Any, target: Any, tol: float = 1e-6) -> bool:
    if isinstance(item_value, (int, float, np.integer, np.floating)) and isinstance(
        target, (int, float, np.integer, np.floating)
    ):
        return bool(np.isclose(float(item_value), float(target), atol=tol, rtol=0.0))
    return item_value == target


def _normalize_path(path_value: str | os.PathLike[str]) -> Path:
    return Path(path_value).expanduser().resolve()


class _DatasetRegistry:
    def __init__(self, window: GrimCutWindow) -> None:
        self.window = window
        self._rows_by_abs_path: dict[str, int] = {}
        self._rows_by_token: dict[str, int] = {}

    @staticmethod
    def _token_key(token: str) -> str:
        return token.strip().lower()

    def _register_token(self, token: str, row: int) -> None:
        key = self._token_key(token)
        if not key:
            return
        self._rows_by_token.setdefault(key, row)

    def _register_row_tokens(self, row: int, path: Path) -> None:
        self._register_token(path.name, row)
        self._register_token(path.stem, row)
        self._register_token(str(path), row)
        self._register_token(str(path.resolve()), row)
        name_item = self.window.table.item(row, 0)
        if name_item is not None:
            self._register_token(name_item.text(), row)

    def add_dataset(self, path_value: str | os.PathLike[str]) -> int:
        path = Path(path_value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        abs_path = str(path.resolve())
        if abs_path in self._rows_by_abs_path:
            return self._rows_by_abs_path[abs_path]

        dataset = RcsGrid.load(str(path))
        self.window._add_dataset_row(dataset, path.stem, str(path), file_name=path.name)
        row = self.window.table.rowCount() - 1
        self._rows_by_abs_path[abs_path] = row
        self._register_row_tokens(row, path)
        return row

    def row_count(self) -> int:
        return self.window.table.rowCount()

    def all_rows(self) -> list[int]:
        return list(range(self.row_count()))

    def resolve_selector(self, selector: Any) -> int:
        if isinstance(selector, (int, np.integer)):
            row = int(selector)
            if row < 0 or row >= self.row_count():
                raise ValueError(f"Dataset row index out of range: {row}")
            return row

        if isinstance(selector, os.PathLike):
            selector = str(selector)

        if isinstance(selector, str):
            token_row = self._rows_by_token.get(self._token_key(selector))
            if token_row is not None:
                return token_row

            as_path = Path(selector).expanduser()
            if as_path.exists():
                return self.add_dataset(as_path)
            if not as_path.suffix:
                grim_path = as_path.with_suffix(".grim")
                if grim_path.exists():
                    return self.add_dataset(grim_path)

        raise ValueError(
            f"Unknown dataset selector '{selector}'. Use a loaded dataset name/index or a valid .grim path."
        )

    def resolve_selectors(self, selectors: Any) -> list[int]:
        selector_values = _as_list(selectors)
        if selector_values is None:
            raise ValueError("Expected dataset selector(s), got None.")
        rows: list[int] = []
        for selector in selector_values:
            row = self.resolve_selector(selector)
            if row not in rows:
                rows.append(row)
        if not rows:
            raise ValueError("No datasets selected.")
        return rows


def _select_list_values(widget: QListWidget, targets: Any, axis_name: str) -> None:
    targets = _as_list(targets)
    widget.blockSignals(True)
    widget.clearSelection()

    if targets is None:
        for row in range(widget.count()):
            widget.item(row).setSelected(True)
        widget.blockSignals(False)
        return

    missing = []
    for target in targets:
        matched = False
        for row in range(widget.count()):
            item = widget.item(row)
            if _value_matches(item.data(Qt.UserRole), target):
                item.setSelected(True)
                matched = True
                break
        if not matched:
            missing.append(target)

    widget.blockSignals(False)
    if missing:
        raise ValueError(f"{axis_name} value(s) not found in dataset: {missing}")


def _apply_optional_visual_settings(window: GrimCutWindow, job: Mapping[str, Any]) -> None:
    scale = job.get("plot_scale")
    if scale in ("dbsm", "linear"):
        idx = window.combo_plot_scale.findData(scale)
        if idx >= 0:
            window.combo_plot_scale.setCurrentIndex(idx)

    colormap = job.get("colormap")
    if isinstance(colormap, str):
        idx = window.combo_colormap.findText(colormap)
        if idx >= 0:
            window.combo_colormap.setCurrentIndex(idx)

    polar_zero = job.get("polar_zero")
    if isinstance(polar_zero, str):
        idx = window.combo_polar_zero.findData(polar_zero)
        if idx < 0:
            idx = window.combo_polar_zero.findText(polar_zero)
        if idx >= 0:
            window.combo_polar_zero.setCurrentIndex(idx)

    if "colorbar" in job:
        window.chk_colorbar.setChecked(bool(job["colorbar"]))
    if "shared_colorbar" in job:
        window.chk_colorbar_shared.setChecked(bool(job["shared_colorbar"]))
    if "legend" in job:
        window.chk_plot_legend.setChecked(bool(job["legend"]))
    if "pbp" in job and window.btn_pbp is not None:
        window.btn_pbp.setChecked(bool(job["pbp"]))
    if "hold" in job and window.btn_hold is not None:
        window.btn_hold.setChecked(bool(job["hold"]))


def _apply_axis_selections(window: GrimCutWindow, job: Mapping[str, Any]) -> None:
    _select_list_values(window.list_az, job.get("azimuths"), "azimuth")
    _select_list_values(window.list_freq, job.get("frequencies"), "frequency")
    _select_list_values(window.list_elev, job.get("elevations"), "elevation")
    _select_list_values(window.list_pol, job.get("polarizations"), "polarization")


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric value for '{field_name}', got {value!r}") from exc


def _extract_axis_limits(job: Mapping[str, Any], axis: str) -> tuple[float | None, float | None]:
    min_val = job.get(f"{axis}_min")
    max_val = job.get(f"{axis}_max")

    if min_val is None or max_val is None:
        axis_lim = job.get(f"{axis}lim")
        if axis_lim is None:
            axis_lim = job.get(f"{axis}_lim")
        if isinstance(axis_lim, Mapping):
            if min_val is None:
                min_val = axis_lim.get("min")
            if max_val is None:
                max_val = axis_lim.get("max")
        elif isinstance(axis_lim, Sequence) and not isinstance(axis_lim, (str, bytes)):
            axis_vals = list(axis_lim)
            if len(axis_vals) >= 2:
                if min_val is None:
                    min_val = axis_vals[0]
                if max_val is None:
                    max_val = axis_vals[1]

    if min_val is None or max_val is None:
        limits = job.get("limits")
        if isinstance(limits, Mapping):
            axis_lim = limits.get(axis)
            if isinstance(axis_lim, Mapping):
                if min_val is None:
                    min_val = axis_lim.get("min")
                if max_val is None:
                    max_val = axis_lim.get("max")
            elif isinstance(axis_lim, Sequence) and not isinstance(axis_lim, (str, bytes)):
                axis_vals = list(axis_lim)
                if len(axis_vals) >= 2:
                    if min_val is None:
                        min_val = axis_vals[0]
                    if max_val is None:
                        max_val = axis_vals[1]

    min_num = _coerce_float(min_val, f"{axis}_min") if min_val is not None else None
    max_num = _coerce_float(max_val, f"{axis}_max") if max_val is not None else None
    return min_num, max_num


def _set_spin_if_present(spin, value: float | None) -> bool:
    if value is None:
        return False
    current = float(spin.value())
    if np.isclose(current, float(value), atol=1e-12, rtol=0.0):
        return False
    spin.blockSignals(True)
    spin.setValue(float(value))
    spin.blockSignals(False)
    return True


def _apply_axis_limit_spins(window: GrimCutWindow, job: Mapping[str, Any], axes: Sequence[str]) -> bool:
    changed = False
    for axis in axes:
        min_val, max_val = _extract_axis_limits(job, axis)
        if axis == "x":
            changed = _set_spin_if_present(window.spin_plot_xmin, min_val) or changed
            changed = _set_spin_if_present(window.spin_plot_xmax, max_val) or changed
        elif axis == "y":
            changed = _set_spin_if_present(window.spin_plot_ymin, min_val) or changed
            changed = _set_spin_if_present(window.spin_plot_ymax, max_val) or changed
        elif axis == "z":
            changed = _set_spin_if_present(window.spin_plot_zmin, min_val) or changed
            changed = _set_spin_if_present(window.spin_plot_zmax, max_val) or changed
    return changed


def _normalize_job(job: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(job, Mapping):
        raise TypeError(f"Each plot job must be a mapping/dict, got: {type(job).__name__}")
    normalized = dict(job)
    variables = normalized.get("variables")
    if isinstance(variables, Mapping):
        for key, value in variables.items():
            normalized.setdefault(key, value)
    mode = normalized.get("mode") or normalized.get("plot_type") or normalized.get("type")
    normalized["mode"] = mode
    return normalized


def _normalize_dataset_inputs(
    dataset: str | os.PathLike[str] | None,
    datasets: Any,
) -> list[str | os.PathLike[str]]:
    if dataset is not None and datasets is not None:
        raise ValueError("Use either 'dataset' or 'datasets', not both.")

    if dataset is not None:
        return [dataset]

    if datasets is None:
        return []

    if isinstance(datasets, Mapping):
        values = list(datasets.values())
    else:
        values = _as_list(datasets) or []

    result: list[str | os.PathLike[str]] = []
    for value in values:
        if isinstance(value, (str, os.PathLike)):
            result.append(value)
    return result


def _select_dataset_rows(window: GrimCutWindow, rows: Sequence[int]) -> None:
    selection_model = window.table.selectionModel()
    if selection_model is None:
        raise RuntimeError("Dataset table is not initialized.")
    selection_model.clearSelection()
    for row in rows:
        if row < 0 or row >= window.table.rowCount():
            raise ValueError(f"Dataset row index out of range: {row}")
        idx = window.table.model().index(row, 0)
        selection_model.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
    if rows:
        window.table.setCurrentCell(int(rows[0]), 0)
    window._on_dataset_selection_changed()


def run_plot_jobs(
    *,
    plot_jobs: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    dataset: str | os.PathLike[str] | None = None,
    datasets: Any = None,
    output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Run headless plot jobs and return written image paths."""
    if isinstance(plot_jobs, Mapping):
        plot_jobs_config = dict(plot_jobs)
        nested_jobs = plot_jobs_config.get("jobs") or plot_jobs_config.get("plot_jobs")
        if nested_jobs is None:
            raise ValueError("When plot_jobs is a dict, include a 'jobs' or 'plot_jobs' list.")
        if dataset is None and datasets is None:
            dataset = plot_jobs_config.get("dataset")
            datasets = plot_jobs_config.get("datasets")
        if output_dir == DEFAULT_OUTPUT_DIR and "output_dir" in plot_jobs_config:
            output_dir = plot_jobs_config["output_dir"]
        plot_jobs = nested_jobs

    normalized_jobs = [_normalize_job(job) for job in plot_jobs]
    if not normalized_jobs:
        raise ValueError("plot_jobs is empty. Add at least one job.")

    created_app = QApplication.instance() is None
    app = QApplication.instance() or QApplication([])
    window = GrimCutWindow()
    registry = _DatasetRegistry(window)

    default_rows: list[int] = []
    for dataset_path in _normalize_dataset_inputs(dataset, datasets):
        default_rows.append(registry.add_dataset(dataset_path))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    current_rows = list(default_rows)
    total = len(normalized_jobs)
    try:
        for index, job in enumerate(normalized_jobs, start=1):
            mode = job.get("mode")
            if mode not in MODE_TO_METHOD:
                raise ValueError(f"Unknown mode '{mode}'. Expected one of: {sorted(MODE_TO_METHOD)}")

            if "datasets" in job:
                current_rows = registry.resolve_selectors(job["datasets"])
            elif "dataset" in job:
                current_rows = registry.resolve_selectors(job["dataset"])
            elif not current_rows:
                if registry.row_count() > 0:
                    current_rows = [0]
                else:
                    raise ValueError(
                        "No dataset selected for job. Set job['dataset'] / job['datasets'] "
                        "or pass dataset(s) into run_plot_jobs()."
                    )

            _select_dataset_rows(window, current_rows)
            window._activate_plot_tab(MODE_TO_TAB[mode])
            _apply_optional_visual_settings(window, job)
            _apply_axis_selections(window, job)
            _apply_axis_limit_spins(window, job, axes=("z",))

            MODE_TO_METHOD[mode](window)

            if _apply_axis_limit_spins(window, job, axes=("x", "y")):
                window._apply_plot_limits()
            z_changed_post_plot = _apply_axis_limit_spins(window, job, axes=("z",))
            if z_changed_post_plot and mode in Z_LIMIT_MODES:
                MODE_TO_METHOD[mode](window)
                if _apply_axis_limit_spins(window, job, axes=("x", "y")):
                    window._apply_plot_limits()

            output_name = str(job.get("output", f"{mode}.png"))
            image_path = Path(output_name)
            if not image_path.is_absolute():
                image_path = output_path / image_path
            image_path.parent.mkdir(parents=True, exist_ok=True)

            dpi = int(job.get("dpi", 200))
            window.plot_figure.savefig(image_path, dpi=dpi, bbox_inches="tight")
            written_files.append(image_path)

            status = window.status.currentMessage()
            dataset_note = ", ".join(str(r) for r in current_rows)
            print(f"[{index}/{total}] {mode} (rows: {dataset_note}) -> {image_path} | status: {status}")
    finally:
        if created_app:
            app.quit()

    return written_files


def _load_user_config(config_file: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = _normalize_path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config script not found: {config_path}")

    spec = importlib.util.spec_from_file_location("headless_plot_jobs_user_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config script: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _module_value(name: str) -> Any:
        for candidate in (name, name.upper()):
            if hasattr(module, candidate):
                return getattr(module, candidate)
        return None

    config: dict[str, Any] = {}
    module_config = _module_value("config")
    if isinstance(module_config, Mapping):
        config.update(module_config)

    for key in ("dataset", "datasets", "output_dir", "plot_jobs"):
        value = _module_value(key)
        if value is not None:
            config[key] = value

    jobs_value = _module_value("jobs")
    if jobs_value is not None and "plot_jobs" not in config:
        config["plot_jobs"] = jobs_value

    if "plot_jobs" not in config:
        raise ValueError(
            f"{config_path} must define plot_jobs (or PLOT_JOBS / jobs), "
            "optionally with dataset(s) and output_dir."
        )

    if isinstance(config["plot_jobs"], Mapping):
        jobs_config = dict(config["plot_jobs"])
        nested_jobs = jobs_config.get("jobs") or jobs_config.get("plot_jobs")
        if nested_jobs is not None:
            config["plot_jobs"] = nested_jobs
        for key in ("dataset", "datasets", "output_dir"):
            if key not in config and key in jobs_config:
                config[key] = jobs_config[key]

    return config


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run GRIM headless plot jobs.")
    parser.add_argument(
        "config",
        nargs="?",
        help="Optional Python config script defining plot_jobs (and optional dataset(s), output_dir).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="dataset_paths",
        help="Dataset path. Repeat to provide multiple datasets.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory to write images.")
    parser.add_argument("--list-modes", action="store_true", help="List available plot modes and exit.")
    args = parser.parse_args(argv)

    if args.list_modes:
        print("Available modes:")
        for mode in sorted(MODE_TO_METHOD):
            print(f"  - {mode}")
        return

    dataset: str | os.PathLike[str] | None = DEFAULT_DATASET_PATH
    datasets: Any = None
    output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR
    plot_jobs: Sequence[Mapping[str, Any]] = DEFAULT_PLOT_JOBS

    if args.config:
        config = _load_user_config(args.config)
        plot_jobs = config["plot_jobs"]
        dataset = config.get("dataset")
        datasets = config.get("datasets")
        output_dir = config.get("output_dir", output_dir)

    if args.dataset_paths:
        dataset = None
        datasets = args.dataset_paths
    if args.output_dir:
        output_dir = args.output_dir

    run_plot_jobs(plot_jobs=plot_jobs, dataset=dataset, datasets=datasets, output_dir=output_dir)


if __name__ == "__main__":
    main()
