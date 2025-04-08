"""
IDT Generator for Log Camera
=================================

Define the *IDT* generator class for a *Log Camera*.
"""

from __future__ import annotations

import base64
import io
import logging
import typing

import colour
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.ndimage
import scipy.optimize
import scipy.stats
from colour import (
    LUT1D,
    Extrapolator,
    LinearInterpolator,
    LUT3x1D,
)
from colour.algebra import smoothstep_function, vecmul

if typing.TYPE_CHECKING:
    from colour.hints import NDArrayFloat, Tuple

from colour.io import LUT_to_LUT
from colour.models import RGB_COLOURSPACE_ACES2065_1, RGB_luminance
from colour.utilities import (
    as_float_array,
    multiline_str,
    validate_method,
)
from scipy.optimize import minimize

from aces.idt.core import DecodingMethods, DirectoryStructure, common
from aces.idt.core.constants import EXPOSURE_CLIPPING_THRESHOLD

if typing.TYPE_CHECKING:
    from aces.idt.framework import IDTProjectSettings

from aces.idt.generators.base_generator import IDTBaseGenerator

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal, Nick Shaw, Adam Davis"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "IDTGeneratorLogCamera",
]

LOGGER = logging.getLogger(__name__)


class IDTGeneratorLogCamera(IDTBaseGenerator):
    """
    Define an *IDT* generator for a *Log Camera*.

    Parameters
    ----------
    project_settings
        *IDT* generator settings.

    Attributes
    ----------
    -   :attr:`~aces.idt.IDTGeneratorLogCamera.GENERATOR_NAME`
    -   :attr:`~aces.idt.IDTGeneratorLogCamera.samples_decoded`
    -   :attr:`~aces.idt.IDTGeneratorLogCamera.samples_weighted`

    Methods
    -------
    -   :meth:`~aces.idt.IDTGeneratorLogCamera.__str__`
    -   :meth:`~aces.idt.IDTGeneratorLogCamera.generate_LUT`
    -   :meth:`~aces.idt.IDTGeneratorLogCamera.filter_LUT`
    -   :meth:`~aces.idt.IDTGeneratorLogCamera.decode`
    -   :meth:`~aces.idt.IDTGeneratorLogCamera.optimise`
    -   :meth:`~aces.idt.IDTGeneratorLogCamera.png_measured_camera_samples`
    -   :meth:`~aces.idt.IDTGeneratorLogCamera.png_extrapolated_camera_samples`
    """

    GENERATOR_NAME = "IDTGeneratorLogCamera"
    """*IDT* generator name."""

    def __init__(self, project_settings: IDTProjectSettings) -> None:
        super().__init__(project_settings)

        self._lut_blending_edge_left = None
        self._lut_blending_edge_right = None

        self._samples_decoded = None
        self._samples_weighted = None

    @property
    def samples_decoded(self) -> NDArrayFloat | None:
        """
        Getter property for the samples of the camera decoded by applying the
        filtered *LUT*.

        Returns
        -------
        :class:`NDArray` or :py:data:`None`
            Samples of the camera decoded by applying the filtered *LUT*.
        """

        return self._samples_decoded

    @property
    def samples_weighted(self) -> NDArrayFloat | None:
        """
        Getter property for the decoded samples of the camera weighted across
        multiple exposures.

        Returns
        -------
        :class:`NDArray` or :py:data:`None`
            Samples of the decoded samples of the camera weighted across
            multiple exposures.
        """

        return self._samples_weighted

    def __str__(self) -> str:
        """
        Return a formatted string representation of the *IDT* generator.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        samples_analysis = None
        if (
            self._samples_analysis is not None
            and self._samples_analysis.get("data") is not None
        ):
            samples_analysis = self._samples_analysis["data"]["colour_checker"].get(
                self._baseline_exposure
            )
            if samples_analysis is not None:
                samples_analysis = as_float_array(samples_analysis["samples_median"])

        return multiline_str(
            self,
            [
                {
                    "label": super().__repr__(),
                    "header": True,
                },
                {"line_break": True},
                {"name": "_baseline_exposure", "label": "Baseline Exposure"},
                {
                    "formatter": lambda x: str(samples_analysis),  # noqa: ARG005
                    "label": "Samples Analysis",
                },
                {
                    "formatter": lambda x: str(self._LUT_decoding),  # noqa: ARG005
                    "label": "LUT Decoding",
                },
                {"name": "_M", "label": "M"},
                {"name": "_RGB_w", "label": "RGB_w"},
                {"name": "_k", "label": "k"},
                {"name": "_npm", "label": "NPM"},
                {"name": "_primaries", "label": "Primaries"},
                {"name": "_whitepoint", "label": "Whitepoint"},
            ],
        )

    def generate_LUT(self) -> LUT3x1D:
        """
        Generate an unfiltered linearisation *LUT* for the camera samples.

        The *LUT* generation process is as follows: The camera samples are
        unlikely to cover the [0, 1] domain and thus need to be extrapolated.

        Two extrapolated datasets are generated:

            -   Linearly extrapolated for the left edge missing data whose
                clipping likelihood is low and thus can be extrapolated safely.
            -   Constant extrapolated for the right edge missing data whose
                clipping likelihood is high and thus cannot be extrapolated
                safely

        Because the camera encoded data response is logarithmic, the slope of
        the center portion of the data is computed and fitted. The fitted line
        is used to extrapolate the right edge missing data. It is blended
        through a smoothstep with the constant extrapolated samples. The blend
        is fully achieved at the right edge of the camera samples.

        Returns
        -------
        :class:`LUT3x1D`
            Unfiltered linearisation *LUT* for the camera samples.
        """

        size = self.project_settings.lut_size
        LOGGER.info('Generating unfiltered "LUT3x1D" with "%s" size...', size)

        self._LUT_unfiltered = LUT3x1D(size=size, name="LUT - Unfiltered")

        for i in range(3):
            x = self._samples_camera[..., i] * (size - 1)
            y = self._samples_reference[..., i]

            samples = np.arange(0, size, 1)

            samples_linear = Extrapolator(LinearInterpolator(x, y))(samples)
            samples_constant = Extrapolator(
                LinearInterpolator(x, y), method="Constant"
            )(samples)

            # Searching for the index of ~middle camera code value * 125%
            # We are trying to find the logarithmic slope of the camera middle
            # range.
            index_middle = np.searchsorted(
                samples / size, np.max(self._samples_camera) / 2 * 1.25
            )
            padding = index_middle // 2
            samples_middle = np.log(np.copy(samples_linear))
            samples_middle[: index_middle - padding] = samples_middle[
                index_middle - padding
            ]
            samples_middle[index_middle + padding :] = samples_middle[
                index_middle + padding
            ]

            a, b = np.polyfit(
                samples[index_middle - padding : index_middle + padding],
                samples_middle[index_middle - padding : index_middle + padding],
                1,
            )

            # Preparing the mask to blend the logarithmic slope with the
            # extrapolated data.
            edge_left = index_middle - padding
            edge_right = np.searchsorted(samples / size, np.max(self._samples_camera))
            mask_samples = smoothstep_function(
                samples, edge_left, edge_right, clip=True
            )

            self._LUT_unfiltered.table[..., i] = samples_linear
            self._LUT_unfiltered.table[index_middle - padding :, i] = (
                np.exp(a * samples + b) * mask_samples
                + samples_constant * (1 - mask_samples)
            )[index_middle - padding :]

            self._lut_blending_edge_left, self._lut_blending_edge_right = (
                edge_left / size,
                edge_right / size,
            )

        return self._LUT_unfiltered

    def filter_LUT(self) -> LUT3x1D:
        """
        Filter/smooth the linearisation *LUT* for the camera samples.

        The *LUT* filtering is performed with a gaussian convolution, the sigma
        value represents the window size. To prevent that the edges of the
        *LUT* are affected by the convolution, the *LUT* is extended, i.e.
        extrapolated at a safe two sigmas in both directions. The left edge is
        linearly extrapolated while the right edge is logarithmically
        extrapolated.

        Returns
        -------
        :class:`LUT3x1D`
            Filtered linearisation *LUT* for the camera samples.
        """

        # Standard deviation of the gaussian convolution kernel.

        sigma = self.project_settings.lut_smoothing
        LOGGER.info('Filtering unfiltered "LUT3x1D" with "%s" sigma...', sigma)

        filter = scipy.ndimage.gaussian_filter1d  # noqa: A001
        filter_kwargs = {"sigma": sigma}

        self._LUT_filtered = self._LUT_unfiltered.copy()
        self._LUT_filtered.name = "LUT - Filtered"

        sigma_x2 = int(sigma * 2)
        x, step = np.linspace(0, 1, self._LUT_unfiltered.size, retstep=True)
        padding = np.arange(step, sigma_x2 * step + step, step)
        for i in range(3):
            y = self._LUT_filtered.table[..., i]
            x_extended = np.concatenate([-padding[::-1], x, padding + 1])

            # Filtering is performed on extrapolated data.
            y_linear_extended = Extrapolator(LinearInterpolator(x, y))(x_extended)
            y_log_extended = np.exp(
                Extrapolator(LinearInterpolator(x, np.log(y)))(x_extended)
            )

            y_linear_filtered = filter(y_linear_extended, **filter_kwargs)
            y_log_filtered = filter(y_log_extended, **filter_kwargs)

            index_middle = len(x_extended) // 2
            self._LUT_filtered.table[..., i] = np.concatenate(
                [
                    y_linear_filtered[sigma_x2:index_middle],
                    y_log_filtered[index_middle:-sigma_x2],
                ]
            )

        return self._LUT_filtered

    def decode(self) -> None:
        """
        Decode the samples produced by the image sampling process.
        """

        decoding_method = self.project_settings.decoding_method
        grey_card_reflectance = self.project_settings.grey_card_reference

        LOGGER.info(
            'Decoding analysis samples using "%s" method and "%s" grey '
            "card reflectance...",
            decoding_method,
            grey_card_reflectance,
        )

        decoding_method = validate_method(
            decoding_method,
            tuple(DecodingMethods.ALL),
        )

        grey_card_reflectance = as_float_array(grey_card_reflectance)

        if decoding_method.title() == DecodingMethods.MEDIAN:
            self._LUT_decoding = LUT1D(np.median(self._LUT_filtered.table, axis=-1))
        elif decoding_method.title() == DecodingMethods.AVERAGE:
            self._LUT_decoding = LUT_to_LUT(
                self._LUT_filtered, LUT1D, force_conversion=True
            )
        elif decoding_method.title() == DecodingMethods.PER_CHANNEL:
            self._LUT_decoding = self._LUT_filtered.copy()
        elif decoding_method.title() == DecodingMethods.ACES:
            self._LUT_decoding = LUT_to_LUT(
                self._LUT_filtered,
                LUT1D,
                force_conversion=True,
                channel_weights=RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ[1, ...],
            )

        self._LUT_decoding.name = "LUT - Decoding"
        if self._samples_analysis[DirectoryStructure.GREY_CARD]:
            sampled_grey_card_reflectance = self._samples_analysis[
                DirectoryStructure.GREY_CARD
            ]["samples_median"]

            linear_gain = grey_card_reflectance / self._LUT_decoding.apply(
                sampled_grey_card_reflectance
            )
            if decoding_method.title() == DecodingMethods.MEDIAN:
                linear_gain = np.median(linear_gain)
            elif decoding_method.title() == DecodingMethods.AVERAGE:
                linear_gain = np.average(linear_gain)
            elif decoding_method.title() == DecodingMethods.PER_CHANNEL:
                pass
            elif decoding_method.title() == DecodingMethods.ACES:
                linear_gain = RGB_luminance(
                    linear_gain,
                    RGB_COLOURSPACE_ACES2065_1.primaries,
                    RGB_COLOURSPACE_ACES2065_1.whitepoint,
                )

            # NOTE: We apply the scaling factor / linear gain obtained from the
            # grey card measurement. A secondary scaling factor is computed
            # during the IDT matrix optimization using the neutral 5 (.70 D)
            # patch as per *P-2013-001* procedure, but it has been decided to
            # by the working group to not include it in the generated *CLF* file.
            self._LUT_decoding.table *= linear_gain

        self._samples_decoded = {}
        for EV in sorted(self._samples_analysis[DirectoryStructure.COLOUR_CHECKER]):
            self._samples_decoded[EV] = self._LUT_decoding.apply(
                as_float_array(
                    self._samples_analysis[DirectoryStructure.COLOUR_CHECKER][EV][
                        "samples_median"
                    ]
                )
            )

    def optimise(self) -> Tuple[NDArrayFloat]:
        """
        Compute the *IDT* matrix.

        Returns
        -------
        :class:`tuple`
            Tuple of *IDT* matrix :math:`M`, white balance multipliers
            :math:`RGB_w` and exposure factor :math:`k` that results in a
            nominally "18% gray" object in the scene producing ACES values
            [0.18, 0.18, 0.18].
        """

        EV_range = tuple(self.project_settings.ev_range)

        # Normalised weights used to sum the exposure values. If not given, the
        # median of the exposure values is used.
        EV_weights = as_float_array(self.project_settings.ev_weights)

        # Training data multi-spectral distributions, defaults to using the *RAW to
        # ACES* v1 190 patches but can be overridden in the project settings.
        training_data = self.project_settings.get_reference_colour_checker_samples()

        optimisation_factory = self.project_settings.get_optimization_factory()

        optimisation_kwargs = self.project_settings.optimization_kwargs

        LOGGER.info(
            'Optimising the "IDT" matrix using "%s" EV range, "%s" EV '
            'weights, "%s" training data, and "%s" optimisation factory...',
            EV_range,
            EV_weights,
            training_data,
            optimisation_factory,
        )

        EV_range = [EV for EV in EV_range if EV in self._samples_decoded]
        if not EV_range:
            LOGGER.warning(
                'Given "EV range" does not contain any existing exposure values, '
                "falling back to center exposure value!"
            )

            EV_range = sorted(self._samples_decoded)
            EV_range = [EV_range[len(EV_range) // 2]]

        LOGGER.info('"EV range": %s"', EV_range)

        # NOTE: We need to check for clipping as even within the selected EV
        # range clipping might occur, thus, any of the clipped exposures from
        # the decoded samples present in the EV range is ignored,
        clipped_exposures = common.find_clipped_exposures(
            self._samples_decoded, EXPOSURE_CLIPPING_THRESHOLD
        )
        EV_range = [value for value in EV_range if value not in clipped_exposures]
        if not EV_range:
            exception = "All exposures in EV range are clipped!"

            raise ValueError(exception)

        samples_normalised = as_float_array(
            [
                self._samples_decoded[EV] * (1 / pow(2, EV))
                for EV in np.atleast_1d(EV_range)
            ]
        )

        if EV_weights.size == 0:
            self._samples_weighted = np.median(samples_normalised, axis=0)
        else:
            self._samples_weighted = np.sum(
                samples_normalised
                * as_float_array(EV_weights)[..., np.newaxis, np.newaxis],
                axis=0,
            )

        self._RGB_w = training_data[21] / self._samples_weighted[21]
        self._RGB_w /= self._RGB_w[1]
        self._samples_weighted *= self._RGB_w

        self._k = np.mean(training_data[21]) / np.mean(self._samples_weighted[21])
        self._samples_weighted *= self._k

        XYZ = vecmul(RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ, training_data)

        (
            x_0,
            objective_function,
            XYZ_to_optimization_colour_model,
            finaliser_function,
        ) = optimisation_factory()

        optimisation_settings = {
            "method": "BFGS",
            "jac": "2-point",
        }
        if not optimisation_kwargs:
            optimisation_settings.update(optimisation_kwargs)

        self._M = minimize(
            objective_function,
            x_0,
            (self._samples_weighted, XYZ_to_optimization_colour_model(XYZ)),
            **optimisation_settings,
        ).x

        self._M = finaliser_function(self._M)

        # Calculate and store the camera npm, the primaries and the whitepoint
        (
            # TODO: Investigate for better attribute names or maybe a dedicated
            # struct.
            self._npm,
            self._primaries,
            self._whitepoint,
        ) = common.calculate_camera_npm_and_primaries_wp(
            self._M,
            target_white_point=self.project_settings.illuminant,
            chromatic_adaptation_transform=self.project_settings.cat,
        )

        return self._M, self._RGB_w, self._k

    def png_measured_camera_samples(self) -> str | None:
        """
        Return the measured camera samples as *PNG* data.

        Returns
        -------
        :class:`str` or :py:data:`None`
            *PNG* data.
        """

        if self._samples_camera is None or self._samples_reference is None:
            return None

        figure, axes = colour.plotting.artist()
        axes.plot(self._samples_camera, np.log(self._samples_reference))
        colour.plotting.render(
            show=False, x_label="Camera Code Value", y_label="Log(ACES Reference)"
        )
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        data_png = base64.b64encode(buffer.getbuffer()).decode("utf8")
        plt.close()

        return data_png

    def png_extrapolated_camera_samples(self) -> str | None:
        """
        Return the extrapolated camera samples as *PNG* data.

        Returns
        -------
        :class:`str` or :py:data:`None`
            *PNG* data.
        """

        if (
            self._samples_camera is None
            or self._samples_reference is None
            or self._LUT_filtered is None
        ):
            return None

        samples = np.linspace(0, 1, self._LUT_filtered.size)
        figure, axes = colour.plotting.artist()
        for i, RGB in enumerate(("r", "g", "b")):
            axes.plot(
                self._samples_camera[..., i],
                np.log(self._samples_reference)[..., i],
                "o",
                color=RGB,
                alpha=0.25,
            )
            axes.plot(samples, np.log(self._LUT_filtered.table[..., i]), color=RGB)
            if self._lut_blending_edge_left:
                axes.axvline(self._lut_blending_edge_left, color="r", alpha=0.25)
            if self._lut_blending_edge_right:
                axes.axvline(self._lut_blending_edge_right, color="r", alpha=0.25)
        colour.plotting.render(
            show=False, x_label="Camera Code Value", y_label="Log(ACES Reference)"
        )
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        data_png = base64.b64encode(buffer.getbuffer()).decode("utf8")
        plt.close()

        return data_png
