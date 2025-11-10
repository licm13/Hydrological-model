"""Topmodel-inspired hydrological model skeleton.

This module provides a compact, teaching-focused implementation of the
TOPMODEL concept (Beven & Kirkby, 1979). It mirrors the API used by the
other models in HydroLearn so that instructors can easily plug it into the
existing example workflows.

The implementation purposefully favours clarity over numerical efficiency.
It exposes the key ideas of TOPMODEL: a distribution of saturation deficits
derived from the topographic index, exponential transmissivity decay, and
the partitioning of rainfall into saturation-excess runoff versus subsurface
flow. The class can therefore be used to highlight the contrast between
topography-driven runoff production and the soil moisture concepts employed
by the other models in the repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TopmodelConfig:
    """Configuration container for :class:`Topmodel`.

    Parameters
    ----------
    m : float
        Exponential decay parameter for transmissivity with deficit depth (m).
    transmissivity_sat : float
        Saturated lateral transmissivity (mm/day) at the surface.
    initial_mean_deficit : float
        Initial catchment-mean saturation deficit (mm).
    area_weights : np.ndarray | None
        Optional fractional contributing area associated with each
        topographic index entry. When ``None`` the contributing area is
        assumed uniform.
    """

    m: float = 0.05
    transmissivity_sat: float = 3.0
    initial_mean_deficit: float = 50.0
    area_weights: np.ndarray | None = None


class Topmodel:
    """Simplified TOPMODEL implementation.

    The class exposes ``run_timestep`` and ``run`` methods to maintain
    compatibility with the rest of HydroLearn's API. The pedagogical goal is
    to emphasise the link between the spatial distribution of saturation
    deficits and catchment-scale runoff response.
    """

    def __init__(
        self,
        topographic_index: np.ndarray | None = None,
        config: TopmodelConfig | None = None,
    ) -> None:
        self.topographic_index = (
            np.asarray(topographic_index, dtype=float)
            if topographic_index is not None
            else np.linspace(5.5, 8.5, num=20)
        )
        self.config = config or TopmodelConfig()

        if self.topographic_index.ndim != 1:
            raise ValueError("`topographic_index` must be one-dimensional.")

        n_cells = self.topographic_index.size
        if self.config.area_weights is None:
            self.area_weights = np.full(n_cells, 1.0 / n_cells)
        else:
            weights = np.asarray(self.config.area_weights, dtype=float)
            if weights.shape != (n_cells,):
                raise ValueError("Area weights must match the topographic index shape.")
            self.area_weights = weights / np.sum(weights)

        self._mean_deficit = float(self.config.initial_mean_deficit)
        self._cell_deficits = np.full(n_cells, self._mean_deficit, dtype=float)

    # ------------------------------------------------------------------
    # Core hydrological routines
    # ------------------------------------------------------------------
    def run_timestep(self, precipitation: float, evapotranspiration: float) -> dict[str, float]:
        """Advance the model by one time step.

        Parameters
        ----------
        precipitation : float
            Liquid water input for the current step (mm).
        evapotranspiration : float
            Actual evapotranspiration demand (mm). In this simplified
            formulation we subtract it uniformly from the saturation deficits.
        """

        net_precip = max(precipitation - evapotranspiration, 0.0)

        # Effective deficit for each index value.
        effective_deficit = self._cell_deficits - self.config.m * (
            self.topographic_index - np.mean(self.topographic_index)
        )

        # Saturation-excess runoff occurs when precipitation exceeds the deficit.
        saturation_excess = np.maximum(net_precip - effective_deficit, 0.0)
        surface_runoff = float(np.sum(saturation_excess * self.area_weights))

        # Update local deficits after the rainfall event.
        infiltration = net_precip - saturation_excess
        self._cell_deficits = np.maximum(self._cell_deficits - infiltration, 0.0)
        self._cell_deficits += evapotranspiration

        # Exponential transmissivity relationship yields baseflow response.
        baseflow = float(
            self.config.transmissivity_sat
            * np.exp(-self._mean_deficit / max(self.config.m, 1e-6))
        )

        # Update the mean deficit used in the exponent.
        self._mean_deficit = float(
            np.sum(self._cell_deficits * self.area_weights)
        )

        discharge = surface_runoff + baseflow
        return {
            "Q": discharge,
            "surface": surface_runoff,
            "baseflow": baseflow,
            "mean_deficit": self._mean_deficit,
        }

    def run(self, precipitation: np.ndarray, evapotranspiration: np.ndarray) -> dict[str, np.ndarray]:
        """Simulate a full time series of forcings."""

        precipitation = np.asarray(precipitation, dtype=float)
        evapotranspiration = np.asarray(evapotranspiration, dtype=float)

        if precipitation.shape != evapotranspiration.shape:
            raise ValueError("Precipitation and evapotranspiration arrays must share the same shape.")

        n_steps = precipitation.size
        discharge = np.zeros(n_steps, dtype=float)
        surface = np.zeros_like(discharge)
        base = np.zeros_like(discharge)
        mean_deficit = np.zeros_like(discharge)

        for idx in range(n_steps):
            outputs = self.run_timestep(precipitation[idx], evapotranspiration[idx])
            discharge[idx] = outputs["Q"]
            surface[idx] = outputs["surface"]
            base[idx] = outputs["baseflow"]
            mean_deficit[idx] = outputs["mean_deficit"]

        return {
            "Q": discharge,
            "surface": surface,
            "baseflow": base,
            "mean_deficit": mean_deficit,
        }


__all__ = ["Topmodel", "TopmodelConfig"]
