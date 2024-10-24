"""
IDT ToneMapped Camera Generator
=============================

Define the *IDT* generator class for a *ToneMapped Camera*.
"""

import logging

import cv2
import matplotlib as mpl
import numpy as np
from colour import LUT3x1D
from colour.utilities import as_float_array
from scipy.interpolate import CubicHermiteSpline

from aces.idt import DirectoryStructure
from aces.idt.core.common import create_samples_macbeth_image, interpolate_nan_values
from aces.idt.generators.prosumer_camera import IDTGeneratorProsumerCamera

# TODO are the mpl.use things needed in every file?
mpl.use("Agg")

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


class IDTGeneratorToneMappedCamera(IDTGeneratorProsumerCamera):
    """
    Define an *IDT* generator for a *ToneMapped Camera*.

    Parameters
    ----------
    project_settings : IDTProjectSettings, optional
        *IDT* generator settings.

    Attributes
    ----------
    -   :attr:`~aces.idt.IDTBaseGenerator.GENERATOR_NAME`
    -   :attr:`~IDTGeneratorToneMappedCamera.exposure_times`

    Methods
    -------
    -   :meth:`~aces.idt.IDTBaseGenerator.sort`
    -   :meth:`~aces.idt.IDTBaseGenerator.remove_clipping`
    -   :meth:`~IDTGeneratorToneMappedCamera.generate_LUT`
    -   :meth:`~IDTGeneratorToneMappedCamera.filter_LUT`
    """

    GENERATOR_NAME = "IDTGeneratorToneMappedCamera"
    """*IDT* generator name."""

    def __init__(self, project_settings):
        super().__init__(project_settings)
        self._exposure_times = []

    @property
    def exposure_times(self):
        """Return the exposure times for the samples."""
        return self._exposure_times

    def sort(self, start_index: int = 0) -> np.ndarray:
        """
        Sort the samples produced by the image sampling process.

        We override the sort method to collect and stack the samples, but also to
        calculate the exposure times for each sample.

        We also sample all the samples from the macbeth chart not just the last 6

        Parameters
        ----------
        start_index : int, optional
            The index to start sorting from, default is -6 so we only use the last
                6 samples from the macbeth chart

        Returns
        -------
        :class:`ndarray`
            Indices of the sorted samples which can be used for additional sorting
        """
        LOGGER.info("Sorting camera and reference samples...")
        ref_col_checker = self.project_settings.get_reference_colour_checker_samples()
        samples_camera = []
        samples_reference = []
        exposure_times = []

        for EV, images in self._samples_analysis[
            DirectoryStructure.COLOUR_CHECKER
        ].items():
            samples_reference.append(ref_col_checker[start_index:, ...] * pow(2, EV))
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

    def remove_clipping(self):
        """We override the remove_clipping method as this is handled within
            the Debevec and hermite fitting later

        Returns
        -------
        :class:`ndarray`
            Indices of the clipped samples, in this case an empty list as we do not
            want to remove any samples

        """
        return []

    def generate_LUT(self) -> LUT3x1D:
        """
        Generate an unfiltered linearisation *LUT* for the camera samples.

        The *LUT* generation process is worth describing, the camera samples are
        unlikely to cover the [0, 1] domain and thus need to be extrapolated.

        The camera samples are grouped by exposure time and a faux macbeth chart image
        is created, with a subtle blur.

        The Debevec algorithm is then used to generate a response curve for the camera

        Any nans which occur are removed via interpolation

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
            image = create_samples_macbeth_image(pixels)
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

        A weighted average is used to weight the green channel more heavily, before a
        hermite spline is used to interpolate the values rather than filter them.

        Returns
        -------
        :class:`LUT3x1D`
            Filtered linearisation *LUT* for the camera

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
