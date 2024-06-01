"""
Input Device Transform (IDT) Prosumer Camera Utilities
======================================================
"""

import base64
import io
import json
import logging
from pathlib import Path

import colour
import matplotlib as mpl
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
from colour.algebra import smoothstep_function, vector_dot
from colour.characterisation import optimisation_factory_rawtoaces_v1
from colour.io import LUT_to_LUT
from colour.models import RGB_COLOURSPACE_ACES2065_1, RGB_luminance
from colour.utilities import (
    as_float_array,
    multiline_str,
    validate_method,
)
from scipy.optimize import minimize

from aces.idt.core.constants import DataFolderStructure, DecodingMethods
from aces.idt.generators.base_generator import IDTBaseGenerator

# TODO are the mpl.use things needed in every file?
mpl.use("Agg")
import matplotlib.pyplot as plt

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "IDTGeneratorProsumerCamera",
]

logger = logging.getLogger(__name__)


class IDTGeneratorProsumerCamera(IDTBaseGenerator):
    """
    Define an *IDT* generator for a *Prosumer Camera*.

    Parameters
    ----------
    archive : str, optional
        *IDT* archive path, i.e. a zip file path.
    image_format : str, optional
        Image format to filter.
    directory : str, optional
        Working directory.
    specification : dict, optional
        *IDT* archive specification.
    cleanup : bool, optional
        Whether to cleanup, i.e. remove, the working directory.

    Attributes
    ----------
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.image_colour_checker_segmentation`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.image_grey_card_sampling`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.baseline_exposure`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.samples_analysis`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.samples_camera`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.samples_reference`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.samples_decoded`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.samples_weighted`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.LUT_unfiltered`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.LUT_filtered`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.LUT_decoding`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.M`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.RGB_w`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.k`

    Methods
    -------
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.__str__`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.from_specification`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.sample`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.sort`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.generate_LUT`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.filter_LUT`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.decode`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.optimise`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.to_clf`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.zip`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.png_colour_checker_segmentation`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.png_grey_card_sampling`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.png_measured_camera_samples`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.png_extrapolated_camera_samples`
    """

    generator_name = "IDTGeneratorProsumerCamera"

    def __init__(self, application):
        super().__init__(application)

        self._lut_blending_edge_left = None
        self._lut_blending_edge_right = None

        self._samples_decoded = None
        self._samples_weighted = None

    @property
    def samples_decoded(self):
        """
        Getter property for the samples of the camera decoded by applying the
        filtered *LUT*.

        Returns
        -------
        :class:`NDArray` or None
            Samples of the camera decoded by applying the filtered *LUT*.
        """

        return self._samples_decoded

    @property
    def samples_weighted(self):
        """
        Getter property for the decoded samples of the camera weighted across
        multiple exposures.

        Returns
        -------
        :class:`NDArray` or None
            Samples of the decoded samples of the camera weighted across
            multiple exposures.
        """

        return self._samples_weighted

    def __str__(self):
        """
        Return a formatted string representation of the *IDT* generator.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """
        # TODO Can't do this can't have the __str__ method be dependent on an execution
        #  state of the instance
        #  _samples_analysis may not be populated yet
        samples_analysis = self._samples_analysis["data"]["colour_checker"].get(
            self._baseline_exposure
        )
        if samples_analysis is not None:
            samples_analysis = as_float_array(samples_analysis["samples_median"])

        return multiline_str(
            self,
            [
                {
                    "label": "IDT Generator",
                    "header": True,
                },
                {"line_break": True},
                {"name": "_archive", "label": "Archive"},
                {"name": "_image_format", "label": "Image Format"},
                {"name": "_directory", "label": "Directory"},
                {"name": "_specification", "label": "Specification"},
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
            ],
        )

    def generate_LUT(self):
        """
        Generate an unfiltered linearisation *LUT* for the camera samples.

        The *LUT* generation process is worth describing, the camera samples are
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
        logger.info('Generating unfiltered "LUT3x1D" with "%s" size...', size)

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

    def filter_LUT(self):
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

        sigma = self.project_settings.sigma
        logger.info('Filtering unfiltered "LUT3x1D" with "%s" sigma...', sigma)

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

    def decode(self):
        """
        Decode the samples produced by the image sampling process.
        """

        decoding_method = self.project_settings.decoding_method
        grey_card_reflectance = self.project_settings.grey_card_reference

        logger.info(
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
        if self._samples_analysis[DataFolderStructure.GREY_CARD]:
            sampled_grey_card_reflectance = self._samples_analysis[
                DataFolderStructure.GREY_CARD
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

            self._LUT_decoding.table *= linear_gain

        self._samples_decoded = {}
        for EV in sorted(self._samples_analysis[DataFolderStructure.COLOUR_CHECKER]):
            self._samples_decoded[EV] = self._LUT_decoding.apply(
                as_float_array(
                    self._samples_analysis[DataFolderStructure.COLOUR_CHECKER][EV][
                        "samples_median"
                    ]
                )
            )

    def optimise(self):
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

        # Exposure values to use when computing the *IDT* matrix.
        EV_range = tuple(self.project_settings.ev_range)
        # Normalised weights used to sum the exposure values. If not given, the median
        # of the exposure values is used.
        # TODO Where do these params live? project? generator? currently only exist
        #  in the ui
        #  TODO Needs to be a generator specific params? We use the same meta data
        #   properties on the generators and these
        # TODO can also display in the ui dynamically
        # TODO Project settings should be serialized and restored without the need for
        #  a ui so i anything generic
        #  should go there
        EV_weights = self.project_settings.ev_weights

        # Training data multi-spectral distributions, defaults to using the *RAW to
        # ACES* v1 190 patches but can be overridden in the project settings.
        training_data = self.project_settings.get_reference_colour_checker_samples()

        # Callable producing the objective function and the *CIE XYZ* to optimisation
        # colour model function.
        optimisation_factory = optimisation_factory_rawtoaces_v1

        # Parameters for :func:`scipy.optimize.minimize` definition.
        optimisation_kwargs = None

        logger.info(
            'Optimising the "IDT" matrix using "%s" EV range, "%s" EV '
            'weights, "%s" training data, and "%s" optimisation factory...',
            EV_range,
            EV_weights,
            training_data,
            optimisation_factory,
        )

        EV_range = as_float_array(EV_range)
        EV_range = [EV for EV in EV_range if EV in self._samples_decoded]
        if not EV_range:
            logger.warning(
                'Given "EV range" does not contain any existing exposure values, '
                "falling back to center exposure value!"
            )

            EV_range = sorted(self._samples_decoded)
            EV_range = [EV_range[len(EV_range) // 2]]

        logger.info('"EV range": %s"', EV_range)

        samples_normalised = as_float_array(
            [
                self._samples_decoded[EV] * (1 / pow(2, EV))
                for EV in np.atleast_1d(EV_range)
            ]
        )

        if not EV_weights:
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

        XYZ = vector_dot(RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ, training_data)

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
        if optimisation_kwargs is not None:
            optimisation_settings.update(optimisation_kwargs)

        self._M = minimize(
            objective_function,
            x_0,
            (self._samples_weighted, XYZ_to_optimization_colour_model(XYZ)),
            **optimisation_settings,
        ).x

        self._M = finaliser_function(self._M)

        return self._M, self._RGB_w, self._k

    def png_measured_camera_samples(self):
        """
        Return the measured camera samples as *PNG* data.

        Returns
        -------
        :class:`str` or None
            *PNG* data.
        """

        if self._samples_camera is None or self._samples_reference is None:
            return None

        figure, axes = colour.plotting.artist()
        axes.plot(self._samples_camera, np.log(self._samples_reference))
        colour.plotting.render(
            **{
                "show": False,
                "x_label": "Camera Code Value",
                "y_label": "Log(ACES Reference)",
            }
        )
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        data_png = base64.b64encode(buffer.getbuffer()).decode("utf8")
        plt.close()

        return data_png

    def png_extrapolated_camera_samples(self):
        """
        Return the extrapolated camera samples as *PNG* data.

        Returns
        -------
        :class:`str` or None
            *PNG* data.
        """
        # TODO This looks specific to the prosumer camera as its using the blending
        #  edges, so will keep this as an override of the base class
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
            axes.axvline(self._lut_blending_edge_left, color="r", alpha=0.25)
            axes.axvline(self._lut_blending_edge_right, color="r", alpha=0.25)
        colour.plotting.render(
            **{
                "show": False,
                "x_label": "Camera Code Value",
                "y_label": "Log(ACES Reference)",
            }
        )
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        data_png = base64.b64encode(buffer.getbuffer()).decode("utf8")
        plt.close()

        return data_png


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    resources_directory = Path(__file__).parent / "tests" / "resources"

    idt_generator = IDTGeneratorProsumerCamera.from_archive(
        resources_directory / "synthetic_001.zip", cleanup=True
    )

    logger.info(idt_generator)

    with open(
        resources_directory / "synthetic_001" / "synthetic_001.json"
    ) as json_file:
        specification = json.load(json_file)

    idt_generator = IDTGeneratorProsumerCamera.from_specification(
        specification, resources_directory / "synthetic_001"
    )

    logger.info(idt_generator)
