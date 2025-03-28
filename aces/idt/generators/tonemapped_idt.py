"""
IDT Generator for a ToneMapped Camera
=====================================

Define the *IDT* generator class for a *ToneMapped Camera*.
"""

from __future__ import annotations

import logging
import typing

import cv2
import numpy as np
from colour import LUT3x1D

if typing.TYPE_CHECKING:
    from colour.hints import List, NDArrayFloat, NDArrayInt

from colour.utilities import as_float_array, as_int_array, optional
from scipy.interpolate import CubicHermiteSpline

from aces.idt import DirectoryStructure
from aces.idt.core.common import create_colour_checker_image, interpolate_nan_values
from aces.idt.core.constants import EXPOSURE_CLIPPING_THRESHOLD

if typing.TYPE_CHECKING:
    from aces.idt.framework import IDTProjectSettings

from aces.idt.generators.log_camera import IDTGeneratorLogCamera

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal, Nick Shaw, Adam Davis"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "IDTGeneratorToneMappedCamera",
]

LOGGER = logging.getLogger(__name__)


class IDTGeneratorToneMappedCamera(IDTGeneratorLogCamera):
    """
    Define an *IDT* generator for a *ToneMapped Camera*.

    Parameters
    ----------
    project_settings
        *IDT* generator settings.

    Attributes
    ----------
    -   :attr:`~aces.idt.IDTGeneratorToneMappedCamera.GENERATOR_NAME`
    -   :attr:`~aces.idt.IDTGeneratorToneMappedCamera.exposure_times`

    Methods
    -------
    -   :meth:`~aces.idt.IDTGeneratorToneMappedCamera.sort`
    -   :meth:`~aces.idt.IDTGeneratorToneMappedCamera.remove_clipped_samples`
    -   :meth:`~aces.idt.IDTGeneratorToneMappedCamera.generate_LUT`
    -   :meth:`~aces.idt.IDTGeneratorToneMappedCamera.filter_LUT`
    """

    GENERATOR_NAME = "IDTGeneratorToneMappedCamera"
    """*IDT* generator name."""

    def __init__(self, project_settings: IDTProjectSettings) -> None:
        super().__init__(project_settings)
        self._exposure_times = []

    @property
    def exposure_times(self) -> List[float]:
        """Return the exposure times for the samples."""

        return self._exposure_times

    def sort(self, start_index: int | None = None) -> NDArrayFloat:
        """
        Sort the samples produced by the image sampling process.

        This override not only sorts, collects and stacks the samples, but also
        computes the exposure times for each sample. All the swatches from
        the colour checker are used for sorting not just the last 6.

        Parameters
        ----------
        start_index
            The index to start sorting from, default to 0, so that all the
            samples from the colour checker are used.

        Returns
        -------
        :class:`np.ndarray`
            Sorting indices.
        """

        start_index = optional(start_index, 0)

        LOGGER.info("Sorting camera and reference samples...")

        reference_colour_checker_samples = (
            self.project_settings.get_reference_colour_checker_samples()
        )

        samples_camera = []
        samples_reference = []
        exposure_times = []
        for EV, images in self._samples_analysis[
            DirectoryStructure.COLOUR_CHECKER
        ].items():
            samples_reference.append(
                reference_colour_checker_samples[start_index:, ...] * pow(2, EV)
            )
            samples_EV = as_float_array(images["samples_median"])[start_index:, ...]
            samples_camera.append(samples_EV)

            num_samples = len(images["samples_median"][start_index:])
            shutter = 100 / (2**EV) if EV >= 0 else 100 * (2 ** abs(EV))
            emulated_shutter_time = np.full((num_samples, 3), shutter)
            exposure_times.append(emulated_shutter_time)

        self._samples_camera = np.vstack(samples_camera)
        self._samples_reference = np.vstack(samples_reference)
        self._exposure_times = np.vstack(exposure_times)

        return np.array([])

    def remove_clipped_samples(
        self,
        threshold: float = EXPOSURE_CLIPPING_THRESHOLD,  # noqa: ARG002
    ) -> NDArrayInt:
        """
        Remove clipped camera samples and their corresponding reference samples.

        This override by pass the process entirely as this is handled during
        subsequent processing stages.

        Returns
        -------
        :class:`np.ndarray`
            Clipped samples indices.
        """

        return as_int_array([])

    def generate_LUT(self) -> LUT3x1D:
        """
        Generate an unfiltered linearisation *LUT* for the camera samples.

        The *LUT* generation process is as follows: The camera samples are
        unlikely to cover the [0, 1] domain and thus need to be extrapolated.
        The camera samples are then grouped by exposure time and a faux colour
        checker image is created, with a subtle blur. Debevec (1997) algorithm
        is finally then used to generate the camera response curves. Any NaNs
        are removed via linear interpolation.

        Returns
        -------
        :class:`LUT3x1D`
            Unfiltered linearisation *LUT* for the camera samples.
        """

        samples_per_exposure = {}
        for idx, sample in enumerate(self.samples_camera):
            exposure_time = self._exposure_times[idx][0]
            pixel = sample.tolist()
            samples_per_exposure.setdefault(exposure_time, []).append(pixel)

        image_stack = []
        exposures = []
        keys = sorted(samples_per_exposure.keys())
        for exposure_time in keys:
            pixels = samples_per_exposure[exposure_time]
            image = create_colour_checker_image(pixels)
            kernel_size = (121, 121)

            # Apply Gaussian blur to the image
            blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
            normalized_array = np.clip(blurred_image * 255, 0, 255)
            img = normalized_array.astype(np.uint8)
            image_stack.append(img)
            exposures.append(1.0 / exposure_time)

        calibrator = cv2.createCalibrateDebevec()
        response = calibrator.process(
            image_stack, np.array(exposures, dtype=np.float32)
        )
        response = response.reshape(256, 3)
        response = interpolate_nan_values(response)

        size = self.project_settings.lut_size

        self._LUT_unfiltered = LUT3x1D(size=size, name="LUT - Unfiltered")
        self._LUT_unfiltered.table = response

        return self._LUT_unfiltered

    def filter_LUT(self) -> LUT3x1D:
        """
        Filter the unfiltered linearisation *LUT* for the camera samples.

        A weighted average is used to bias the green channel increasingly,
        before a hermite spline is applied to interpolate the values rather
        than filter them.

        Returns
        -------
        :class:`LUT3x1D`
            Filtered linearisation *LUT* for the camera samples.
        """

        self._LUT_filtered = self._LUT_unfiltered.copy()
        self._LUT_filtered.name = "LUT - Filtered"

        size = self.project_settings.lut_size

        weights = np.array([1, 2, 1])  # More weight to Green
        weighted_average = np.average(self._LUT_filtered.table, axis=1, weights=weights)

        weighted_average = np.power(weighted_average, 1 / 2.0)

        x_values = np.arange(len(weighted_average))
        derivatives = np.gradient(weighted_average, x_values)
        hermite_spline = CubicHermiteSpline(x_values, weighted_average, derivatives)

        # Generate new x values for a smoother curve
        x_new = np.linspace(0, len(weighted_average) - 1, size)

        # More points for smooth curve
        y_new = hermite_spline(x_new)
        y_new = np.power(y_new, 2.0)

        lut_rgb = np.tile(y_new[:, np.newaxis], (1, 3))
        self._LUT_filtered.table = lut_rgb

        return self._LUT_filtered
