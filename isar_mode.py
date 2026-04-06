from __future__ import annotations

import numpy as np


def _unit_to_hz_scale(unit: str) -> float:
    unit = unit.strip().lower()
    if unit == "hz":
        return 1.0
    if unit == "khz":
        return 1e3
    if unit == "mhz":
        return 1e6
    if unit == "ghz":
        return 1e9
    return 1e9


def render(self) -> None:
    self.last_plot_mode = "isar_image"
    if self.active_dataset is None:
        self.status.showMessage("Select a dataset before plotting.")
        return

    az_indices = sorted(self._selected_indices(self.list_az))
    if not az_indices:
        self.status.showMessage("Select one or more azimuths to plot.")
        return
    if len(az_indices) < 2:
        self.status.showMessage("Select at least 2 azimuth samples for ISAR imaging.")
        return
    freq_indices = sorted(self._selected_indices(self.list_freq))
    if not freq_indices:
        self.status.showMessage("Select one or more frequencies to plot.")
        return
    if len(freq_indices) < 2:
        self.status.showMessage("Select at least 2 frequency samples for ISAR imaging.")
        return

    pol_idx = self._single_selection_index(self.list_pol, "polarization")
    if pol_idx is None:
        return
    elev_idx = self._single_selection_index(self.list_elev, "elevation")
    if elev_idx is None:
        return

    az_values = self.active_dataset.azimuths[az_indices]
    freq_values = self.active_dataset.frequencies[freq_indices]
    az_order = np.argsort(az_values)
    freq_order = np.argsort(freq_values)
    az_values = az_values[az_order]
    freq_values = freq_values[freq_order]

    if not self._button_checked(self.btn_hold):
        # Rebuild axes on each new render to avoid cumulative shrink from colorbar layout.
        self._remove_colorbar()
        self.plot_figure.clear()
        self.plot_ax = self.plot_figure.add_subplot(111)
        self.plot_axes = None
        self._style_plot_axes()
    rcs_slice = self.active_dataset.rcs[np.ix_(az_indices, [elev_idx], freq_indices, [pol_idx])]
    rcs_slice = rcs_slice[:, 0, :, 0]
    rcs_slice = rcs_slice[np.ix_(az_order, freq_order)]
    phase_slice = self.active_dataset.rcs_phase[np.ix_(az_indices, [elev_idx], freq_indices, [pol_idx])]
    phase_slice = phase_slice[:, 0, :, 0]
    phase_slice = phase_slice[np.ix_(az_order, freq_order)]
    if not np.any(np.isfinite(phase_slice)):
        self.status.showMessage(
            "ISAR imaging requires phase-aware samples; selected data has no finite rcs_phase."
        )
        return
    rcs_slice = np.where(np.isfinite(rcs_slice), rcs_slice, 0.0)

    theta_rad = np.deg2rad(az_values.astype(float))
    if theta_rad.size < 2 or np.any(np.diff(theta_rad) <= 0):
        self.status.showMessage("Azimuth samples must be strictly increasing for ISAR imaging.")
        return

    freq_values_float = freq_values.astype(float)
    if freq_values_float.size < 2 or np.any(np.diff(freq_values_float) <= 0):
        self.status.showMessage("Frequency samples must be strictly increasing for ISAR imaging.")
        return
    if not np.all(np.isfinite(theta_rad)) or not np.all(np.isfinite(freq_values_float)):
        self.status.showMessage("ISAR imaging requires finite azimuth and frequency samples.")
        return

    freq_unit = str(self.active_dataset.units.get("frequency", "ghz"))
    freq_hz = freq_values_float * _unit_to_hz_scale(freq_unit)

    c0 = 299_792_458.0
    df = float(np.mean(np.diff(freq_hz)))
    dtheta = float(np.mean(np.diff(theta_rad)))
    if df <= 0.0 or dtheta <= 0.0:
        self.status.showMessage("ISAR imaging requires increasing azimuth and frequency samples.")
        return

    win_az = self._isar_window(theta_rad.size)
    win_freq = self._isar_window(freq_hz.size)
    rcs_windowed = rcs_slice * np.outer(win_az, win_freq)

    range_az_fft = np.fft.ifft(rcs_windowed, axis=1)
    isar_complex = np.fft.fft(range_az_fft, axis=0)
    isar_complex = np.fft.fftshift(isar_complex, axes=(0, 1))

    magnitude = np.abs(isar_complex)
    isar_linear = magnitude
    isar_dbsm = self.active_dataset.rcs_to_dbsm(magnitude)
    isar_display = isar_linear if self._plot_scale_is_linear() else isar_dbsm

    y_range = np.fft.fftshift(np.fft.fftfreq(freq_hz.size, d=df)) * (c0 / 2.0)
    cross_freq = np.fft.fftshift(np.fft.fftfreq(theta_rad.size, d=dtheta))
    center_freq_hz = float(np.mean(freq_hz))
    x_range = cross_freq * (c0 / (2.0 * max(center_freq_hz, 1.0)))

    cmap = self._effective_colormap()
    zmin = self.spin_plot_zmin.value()
    zmax = self.spin_plot_zmax.value()
    use_clamp = zmin < zmax
    mesh = self.plot_ax.pcolormesh(
        x_range,
        y_range,
        isar_display.T,
        shading="auto",
        cmap=cmap,
        vmin=zmin if use_clamp else None,
        vmax=zmax if use_clamp else None,
    )

    elev_value = self.active_dataset.elevations[elev_idx]
    pol_value = self.active_dataset.polarizations[pol_idx]
    self.plot_ax.set_title(
        f"ISAR Image | Elevation {elev_value} deg | Pol {pol_value}",
        color=self._current_plot_text(),
    )
    self.plot_ax.set_xlabel("Cross-Range (m)")
    self.plot_ax.set_ylabel("Range (m)")

    if self.chk_colorbar.isChecked():
        colorbar = self.plot_figure.colorbar(mesh, ax=self.plot_ax)
        self.plot_colorbars = [colorbar]
        self._apply_colorbar_ticks(colorbar)
        if self._plot_scale_is_linear():
            colorbar.set_label("RCS (Linear)", color=self._current_plot_text())
        else:
            colorbar.set_label("RCS (dBsm)", color=self._current_plot_text())
        colorbar.ax.tick_params(colors=self._current_plot_text())
        for label in colorbar.ax.get_yticklabels():
            label.set_color(self._current_plot_text())

    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_ymin.blockSignals(True)
    self.spin_plot_ymax.blockSignals(True)
    self.spin_plot_xmin.setValue(float(x_range.min()))
    self.spin_plot_xmax.setValue(float(x_range.max()))
    self.spin_plot_ymin.setValue(float(y_range.min()))
    self.spin_plot_ymax.setValue(float(y_range.max()))
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self.spin_plot_ymin.blockSignals(False)
    self.spin_plot_ymax.blockSignals(False)

    self._apply_plot_limits()
    self.status.showMessage("ISAR image updated.")
