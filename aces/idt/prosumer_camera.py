"""
Input Device Transform (IDT) Prosumer Camera Utilities
======================================================
"""

import base64
import io
import json
import logging
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as Et
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import colour
import cv2
import jsonpickle
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
    read_image,
)
from colour.algebra import smoothstep_function, vector_dot
from colour.characterisation import optimisation_factory_rawtoaces_v1
from colour.io import LUT_to_LUT
from colour.models import RGB_COLOURSPACE_ACES2065_1, RGB_luminance
from colour.utilities import Structure, as_float_array, attest, validate_method, zeros
from colour_checker_detection import segmenter_default
from colour_checker_detection.detection import (
    as_int32_array,
    reformat_image,
    sample_colour_checker,
)
from scipy.optimize import minimize

from aces.idt.common import (
    RGB_COLORCHECKER_CLASSIC_ACES,
    SAMPLES_COUNT_DEFAULT,
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
    clf_processing_elements,
)
from aces.idt.utilities import (
    list_sub_directories,
    mask_outliers,
    working_directory,
)

mpl.use("Agg")
import matplotlib.pyplot as plt

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "DATA_SPECIFICATION",
    "DATA_SAMPLES_ANALYSIS",
    "IDTGeneratorProsumerCamera",
]

logger = logging.getLogger(__name__)

DATA_SPECIFICATION = {
    "header": {
        "schema_version": "0.1.0",
        "aces_transform_id": None,
        "aces_user_name": None,
        "camera_make": None,
        "camera_model": None,
        "iso": 800,
        "temperature": 5600,
        "additional_camera_settings": None,
        "lighting_setup_description": None,
        "debayering_platform": None,
        "debayering_settings": None,
        "encoding_colourspace": None,
    },
    "data": {
        "colour_checker": {},
        "flatfield": [],
        "grey_card": [],
    },
}
"""
Template specification for the *IDT* archive.

DATA_SPECIFICATION : dict
"""

DATA_SAMPLES_ANALYSIS = deepcopy(DATA_SPECIFICATION)
"""
Template specification for the image sampling process.

DATA_SAMPLES_ANALYSIS : dict
"""


class IDTGeneratorProsumerCamera:
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
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.archive`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.image_format`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.directory`
    -   :attr:`~aces.idt.IDTGeneratorProsumerCamera.specification`
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
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.from_archive`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.from_specification`
    -   :meth:`~aces.idt.IDTGeneratorProsumerCamera.extract`
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

    def __init__(
        self,
        archive=None,
        image_format="tif",
        directory=None,
        specification=None,
        cleanup=False,
    ):
        self._archive = str(archive)
        self._image_format = image_format
        self._directory = (
            tempfile.TemporaryDirectory().name if directory is None else directory
        )
        self._cleanup = cleanup
        self._specification = specification

        self._image_colour_checker_segmentation = None
        self._image_grey_card_sampling = None
        self._lut_blending_edge_left = None
        self._lut_blending_edge_right = None

        self._baseline_exposure = 0
        self._samples_analysis = None
        self._samples_camera = None
        self._samples_reference = None
        self._samples_decoded = None
        self._samples_weighted = None

        self._LUT_unfiltered = None
        self._LUT_filtered = None
        self._LUT_decoding = None

        self._M = None
        self._RGB_w = None
        self._k = None

    @property
    def archive(self):
        """
        Getter property for the *IDT* archive path.

        Returns
        -------
        :class:`str` or None
            *IDT* archive path, i.e. a zip file path.
        """

        return self._archive

    @property
    def image_format(self):
        """
        Getter property for image format to filter.

        Returns
        -------
        :class:`str` or None
            Image format to filter.
        """

        return self._image_format

    @property
    def directory(self):
        """
        Getter property for the working directory to extract the *IDT* archive.

        Returns
        -------
        :class:`str` or None
            Working directory to extract the *IDT* archive.
        """

        return self._directory

    @property
    def specification(self):
        """
        Getter property for the *IDT* archive specification.

        Returns
        -------
        :class:`str` or None
            *IDT* archive specification.
        """

        return self._specification

    @property
    def image_colour_checker_segmentation(self):
        """
        Getter property for the image of the colour checker with segmentation
        contours.

        Returns
        -------
        :class:`NDArray` or None
            Image of the colour checker with segmentation contours.
        """

        return self._image_colour_checker_segmentation

    @property
    def image_grey_card_sampling(self):
        """
        Getter property for the image the grey card with sampling contours.
        contours.

        Returns
        -------
        :class:`NDArray` or None
            Image of the grey card with sampling contours.
        """

        return self._image_grey_card_sampling

    @property
    def baseline_exposure(self):
        """
        Getter property for the baseline exposure.

        Returns
        -------
        :class:`float`
            Baseline exposure.
        """

        return self._baseline_exposure

    @property
    def samples_analysis(self):
        """
        Getter property for the samples produced by the colour checker sampling
        process.

        Returns
        -------
        :class:`NDArray` or None
            Samples produced by the colour checker sampling process.
        """

        return self._samples_analysis

    @property
    def samples_camera(self):
        """
        Getter property for the samples of the camera produced by the sorting
        process.

        Returns
        -------
        :class:`NDArray` or None
            Samples of the camera produced by the sorting process.
        """

        return self._samples_camera

    @property
    def samples_reference(self):
        """
        Getter property for the reference samples produced by the sorting
        process.

        Returns
        -------
        :class:`NDArray` or None
            Reference samples produced by the sorting process.
        """

        return self._samples_reference

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

    @property
    def LUT_unfiltered(self):
        """
        Getter property for the unfiltered *LUT*.

        Returns
        -------
        :class:`LUT3x1D` or None
            Unfiltered *LUT*.
        """

        return self._LUT_unfiltered

    @property
    def LUT_filtered(self):
        """
        Getter property for the filtered *LUT*.

        Returns
        -------
        :class:`LUT3x1D` or None
            Filtered *LUT*.
        """

        return self._LUT_filtered

    @property
    def LUT_decoding(self):
        """
        Getter property for the (final) decoding *LUT*.

        Returns
        -------
        :class:`LUT1D` or :class:`LUT3x1D` or None
            Decoding *LUT*.
        """

        return self._LUT_decoding

    @property
    def M(self):
        """
        Getter property for the *IDT* matrix :math:`M`,

        Returns
        -------
        :class:`NDArray` or None
           *IDT* matrix :math:`M`.
        """

        return self._M

    @property
    def RGB_w(self):
        """
        Getter property for the white balance multipliers :math:`RGB_w`.

        Returns
        -------
        :class:`NDArray` or None
            White balance multipliers :math:`RGB_w`.
        """

        return self._RGB_w

    @property
    def k(self):
        """
        Getter property for the exposure factor :math:`k` that results in a
        nominally "18% gray" object in the scene producing ACES values
        [0.18, 0.18, 0.18].

        Returns
        -------
        :class:`NDArray` or None
            Exposure factor :math:`k`
        """

        return self._k

    @staticmethod
    def from_archive(
        archive,
        image_format="tif",
        directory=None,
        cleanup=False,
        reference_colour_checker=RGB_COLORCHECKER_CLASSIC_ACES,
        size=1024,
        sigma=16,
        decoding_method="Median",
        grey_card_reflectance=(0.18, 0.18, 0.18),
        EV_range=(-1, 0, 1),
        EV_weights=None,
        training_data=RGB_COLORCHECKER_CLASSIC_ACES,
        optimisation_factory=optimisation_factory_rawtoaces_v1,
        optimisation_kwargs=None,
    ):
        """
        Return an *IDT* generator for a *Prosumer Camera* for given
        *IDT* archive path.

        Parameters
        ----------
        archive : str
            *IDT* archive path, i.e. a zip file path.
        image_format : str, optional
            Image format to filter.
        directory : str, optional
            Working directory.
        cleanup : bool, optional
            Whether to cleanup, i.e. remove, the working directory.
        reference_colour_checker : NDArray
            Reference *ACES* *RGB* values for the *ColorChecker Classic*.
        size : integer, optional
            *LUT* size.
        sigma : numeric
            Standard deviation of the gaussian convolution kernel.
        decoding_method : str, optional
            {"Median", "Average", "Per Channel", "ACES"},
            Decoding method.
        grey_card_reflectance : array_like, optional
            Measured grey card reflectance.
        EV_range : array_like, optional
            Exposure values to use when computing the *IDT* matrix.
        EV_weights : array_like, optional
            Normalised weights used to sum the exposure values. If not given,
            the median of the exposure values is used.
        training_data : NDArray, optional
            Training data multi-spectral distributions, defaults to using the
            *RAW to ACES* v1 190 patches.
        optimisation_factory : callable, optional
            Callable producing the objective function and the *CIE XYZ* to
            optimisation colour model function.
        optimisation_kwargs : dict, optional
            Parameters for :func:`scipy.optimize.minimize` definition.

        Returns
        -------
        :class:`IDTGeneratorProsumerCamera`
            *IDT* generator for a *Prosumer Camera*.
        """

        idt_generator = IDTGeneratorProsumerCamera(
            archive, image_format, directory, cleanup
        )

        idt_generator.extract()
        idt_generator.sample()
        idt_generator.sort(reference_colour_checker)
        idt_generator.generate_LUT(size)
        idt_generator.filter_LUT(sigma)
        idt_generator.decode(decoding_method, grey_card_reflectance)
        idt_generator.optimise(
            EV_range,
            EV_weights,
            training_data,
            optimisation_factory,
            optimisation_kwargs,
        )

        return idt_generator

    @staticmethod
    def from_specification(
        specification,
        directory,
        reference_colour_checker=RGB_COLORCHECKER_CLASSIC_ACES,
        size=1024,
        sigma=16,
        decoding_method="Median",
        grey_card_reflectance=(0.18, 0.18, 0.18),
        EV_range=(-1, 0, 1),
        EV_weights=None,
        training_data=RGB_COLORCHECKER_CLASSIC_ACES,
        optimisation_factory=optimisation_factory_rawtoaces_v1,
        optimisation_kwargs=None,
    ):
        """
        Return an *IDT* generator for a *Prosumer Camera* for given
        *IDT* specification.

        Parameters
        ----------
        specification : dict
            *IDT* specification.
        directory : str, optional
            Working directory.
        reference_colour_checker : NDArray
            Reference *ACES* *RGB* values for the *ColorChecker Classic*.
        size : integer, optional
            *LUT* size.
        sigma : numeric
            Standard deviation of the gaussian convolution kernel.
        decoding_method : str, optional
            {"Median", "Average", "Per Channel", "ACES"},
            Decoding method.
        grey_card_reflectance : array_like, optional
            Measured grey card reflectance.
        EV_range : array_like, optional
            Exposure values to use when computing the *IDT* matrix.
        EV_weights : array_like, optional
            Normalised weights used to sum the exposure values. If not given,
            the median of the exposure values is used.
        training_data : NDArray, optional
            Training data multi-spectral distributions, defaults to using the
            *RAW to ACES* v1 190 patches.
        optimisation_factory : callable, optional
            Callable producing the objective function and the *CIE XYZ* to
            optimisation colour model function.
        optimisation_kwargs : dict, optional
            Parameters for :func:`scipy.optimize.minimize` definition.

        Returns
        -------
        :class:`IDTGeneratorProsumerCamera`
            *IDT* generator for a *Prosumer Camera*.
        """

        # Ensuring that exposure values in the specification are floating
        # point numbers.
        for exposure in list(specification["data"]["colour_checker"].keys()):
            images = [  # noqa: C416
                image for image in specification["data"]["colour_checker"].pop(exposure)
            ]

            specification["data"]["colour_checker"][float(exposure)] = images

        idt_generator = IDTGeneratorProsumerCamera(
            directory=directory, cleanup=False, specification=specification
        )

        idt_generator.sample()
        idt_generator.sort(reference_colour_checker)
        idt_generator.generate_LUT(size)
        idt_generator.filter_LUT(sigma)
        idt_generator.decode(decoding_method, grey_card_reflectance)
        idt_generator.optimise(
            EV_range,
            EV_weights,
            training_data,
            optimisation_factory,
            optimisation_kwargs,
        )

        return idt_generator

    def extract(self):
        """
        Extract the specification from the *IDT* archive.
        """

        logger.info(
            'Extracting "%s" archive to "%s"...',
            self._archive,
            self._directory,
        )

        shutil.unpack_archive(self._archive, self._directory)

        extracted_directories = list_sub_directories(self._directory)

        attest(len(extracted_directories) == 1)

        root_directory = extracted_directories[0]

        json_files = list(root_directory.glob("*.json"))
        if len(json_files) == 1:
            specification = json_files[0]

            logger.info('Found explicit "%s" "IDT" specification file.', specification)

            with open(specification) as json_file:
                self._specification = json.load(json_file)
        else:
            logger.info('Assuming implicit "IDT" specification...')
            self._specification = deepcopy(DATA_SPECIFICATION)

            self._specification["header"]["camera_model"] = Path(self._archive).stem

            colour_checker_directory = root_directory / "data" / "colour_checker"

            attest(colour_checker_directory.exists())

            for exposure_directory in colour_checker_directory.iterdir():
                if re.match(r"-?\d", exposure_directory.name):
                    EV = exposure_directory.name

                    self._specification["data"]["colour_checker"][EV] = list(
                        (colour_checker_directory / exposure_directory).glob(
                            f"*.{self._image_format}"
                        )
                    )

            flatfield_directory = root_directory / "data" / "flatfield"
            if flatfield_directory.exists():
                self._specification["data"]["flatfield"] = list(
                    flatfield_directory.glob(f"*.{self._image_format}")
                )

            grey_card_directory = root_directory / "data" / "grey_card"
            if grey_card_directory.exists():
                self._specification["data"]["grey_card"] = list(
                    flatfield_directory.glob(f"*.{self._image_format}")
                )

        for exposure in list(self._specification["data"]["colour_checker"].keys()):
            images = [
                Path(root_directory) / image
                for image in self._specification["data"]["colour_checker"].pop(exposure)
            ]

            for image in images:
                attest(image.exists())

            self._specification["data"]["colour_checker"][float(exposure)] = images

        if self._specification["data"].get("flatfield") is not None:
            images = [
                Path(root_directory) / image
                for image in self._specification["data"]["flatfield"]
            ]
            for image in images:
                attest(image.exists())

            self._specification["data"]["flatfield"] = images

        if self._specification["data"].get("grey_card") is not None:
            images = [
                Path(root_directory) / image
                for image in self._specification["data"]["grey_card"]
            ]
            for image in images:
                attest(image.exists())

            self._specification["data"]["grey_card"] = images

    def sample(self):
        """
        Sample the images from the *IDT* specification.
        """

        logger.info('Sampling "IDT" specification images...')

        settings = Structure(**SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC)
        working_width = settings.working_width
        working_height = settings.working_height

        def _reformat_image(image):
            """Reformat given image."""

            return reformat_image(
                image, settings.working_width, settings.interpolation_method
            )

        rectangle = as_int32_array(
            [
                [working_width, 0],
                [working_width, working_height],
                [0, working_height],
                [0, 0],
            ]
        )

        self._samples_analysis = deepcopy(DATA_SAMPLES_ANALYSIS)

        # Baseline exposure value, it can be different from zero.
        if 0 not in self._specification["data"]["colour_checker"]:
            EVs = sorted(self._specification["data"]["colour_checker"].keys())
            self._baseline_exposure = EVs[len(EVs) // 2]
            logger.warning(
                "Baseline exposure is different from zero: %s", self._baseline_exposure
            )

        paths = self._specification["data"]["colour_checker"][self._baseline_exposure]

        with working_directory(self._directory):
            logger.info(
                'Reading EV "%s" baseline exposure "ColourChecker" from "%s"...',
                self._baseline_exposure,
                paths[0],
            )
            image = _reformat_image(read_image(paths[0]))

        (
            rectangles,
            clusters,
            swatches,
            segmented_image,
        ) = segmenter_default(
            image,
            additional_data=True,
            **SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
        ).values

        quadrilateral = rectangles[0]

        self._image_colour_checker_segmentation = np.copy(image)
        cv2.drawContours(
            self._image_colour_checker_segmentation, swatches, -1, (1, 0, 1), 3
        )
        cv2.drawContours(
            self._image_colour_checker_segmentation, clusters, -1, (0, 1, 1), 3
        )

        data_detection_colour_checker_EV0 = sample_colour_checker(
            image, quadrilateral, rectangle, SAMPLES_COUNT_DEFAULT, **settings
        )

        # Disabling orientation as we now have an oriented quadrilateral
        settings.reference_values = None

        # Flatfield
        if self._specification["data"].get("flatfield"):
            self._samples_analysis["data"]["flatfield"] = {"samples_sequence": []}
            for path in self._specification["data"]["flatfield"]:
                with working_directory(self._directory):
                    logger.info('Reading flatfield image from "%s"...', path)
                    image = _reformat_image(read_image(path))

                data_detection_flatfield = sample_colour_checker(
                    image,
                    data_detection_colour_checker_EV0.quadrilateral,
                    rectangle,
                    SAMPLES_COUNT_DEFAULT,
                    **settings,
                )

                self._samples_analysis["data"]["flatfield"]["samples_sequence"].append(
                    data_detection_flatfield.swatch_colours.tolist()
                )

            samples_sequence = as_float_array(
                [
                    samples[0]
                    for samples in self._samples_analysis["data"]["flatfield"][
                        "samples_sequence"
                    ]
                ]
            )
            mask = np.all(~mask_outliers(samples_sequence), axis=-1)

            self._samples_analysis["data"]["flatfield"]["samples_median"] = np.median(
                as_float_array(
                    self._samples_analysis["data"]["flatfield"]["samples_sequence"]
                )[mask],
                (0, 1),
            ).tolist()

        # Grey Card
        if self._specification["data"].get("grey_card"):
            self._samples_analysis["data"]["grey_card"] = {"samples_sequence": []}

            settings_grey_card = Structure(**settings)
            settings_grey_card.swatches_horizontal = 1
            settings_grey_card.swatches_vertical = 1

            for path in self._specification["data"]["grey_card"]:
                with working_directory(self._directory):
                    logger.info('Reading grey card image from "%s"...', path)
                    image = _reformat_image(read_image(path))

                data_detection_grey_card = sample_colour_checker(
                    image,
                    data_detection_colour_checker_EV0.quadrilateral,
                    rectangle,
                    SAMPLES_COUNT_DEFAULT,
                    **settings_grey_card,
                )

                grey_card_colour = np.ravel(data_detection_grey_card.swatch_colours)

                self._samples_analysis["data"]["grey_card"]["samples_sequence"].append(
                    grey_card_colour.tolist()
                )

            samples_sequence = as_float_array(
                [
                    samples[0]
                    for samples in self._samples_analysis["data"]["grey_card"][
                        "samples_sequence"
                    ]
                ]
            )
            mask = np.all(~mask_outliers(samples_sequence), axis=-1)

            self._samples_analysis["data"]["grey_card"]["samples_median"] = np.median(
                as_float_array(
                    self._samples_analysis["data"]["grey_card"]["samples_sequence"]
                )[mask],
                (0, 1),
            ).tolist()

            self._image_grey_card_sampling = np.copy(image)
            image_grey_card_contour = zeros(
                (working_height, working_width), dtype=np.uint8
            )
            image_grey_card_contour[
                data_detection_grey_card.swatch_masks[0][
                    0
                ] : data_detection_grey_card.swatch_masks[0][1],
                data_detection_grey_card.swatch_masks[0][
                    2
                ] : data_detection_grey_card.swatch_masks[0][3],
                ...,
            ] = 255
            contours, _hierarchy = cv2.findContours(
                image_grey_card_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(
                self._image_grey_card_sampling,
                contours,
                -1,
                (1, 0, 1),
                3,
            )

        # ColourChecker Classic Samples per EV
        self._samples_analysis["data"]["colour_checker"] = {}
        for EV in self._specification["data"]["colour_checker"]:
            self._samples_analysis["data"]["colour_checker"][EV] = {}
            self._samples_analysis["data"]["colour_checker"][EV][
                "samples_sequence"
            ] = []
            for path in self._specification["data"]["colour_checker"][EV]:
                with working_directory(self._directory):
                    logger.info(
                        'Reading EV "%s" "ColourChecker" from "%s"...',
                        EV,
                        path,
                    )

                    image = _reformat_image(read_image(path))

                data_detection_colour_checker = sample_colour_checker(
                    image,
                    data_detection_colour_checker_EV0.quadrilateral,
                    rectangle,
                    SAMPLES_COUNT_DEFAULT,
                    **settings,
                )

                self._samples_analysis["data"]["colour_checker"][EV][
                    "samples_sequence"
                ].append(data_detection_colour_checker.swatch_colours.tolist())

            sequence_neutral_5 = as_float_array(
                [
                    samples[21]
                    for samples in self._samples_analysis["data"]["colour_checker"][EV][
                        "samples_sequence"
                    ]
                ]
            )
            mask = np.all(~mask_outliers(sequence_neutral_5), axis=-1)

            self._samples_analysis["data"]["colour_checker"][EV][
                "samples_median"
            ] = np.median(
                as_float_array(
                    self._samples_analysis["data"]["colour_checker"][EV][
                        "samples_sequence"
                    ]
                )[mask],
                0,
            ).tolist()

        if self._cleanup:
            shutil.rmtree(self._directory)

    def sort(self, reference_colour_checker=RGB_COLORCHECKER_CLASSIC_ACES):
        """
        Sort the samples produced by the image sampling process.

        The *ACES* reference samples are sorted and indexed as a function of the
        camera samples ordering. This ensures that the camera samples are
        monotonically increasing.

        Parameters
        ----------
        reference_colour_checker : NDArray
            Reference *ACES* *RGB* values for the *ColorChecker Classic*.
        """

        logger.info("Sorting camera and reference samples...")

        samples_camera = []
        samples_reference = []
        for EV, images in self._samples_analysis["data"]["colour_checker"].items():
            samples_reference.append(reference_colour_checker[-6:, ...] * pow(2, EV))
            samples_EV = as_float_array(images["samples_median"])[-6:, ...]
            samples_camera.append(samples_EV)

        self._samples_camera = np.vstack(samples_camera)
        self._samples_reference = np.vstack(samples_reference)

        indices = np.argsort(np.median(self._samples_camera, axis=-1), axis=0)

        self._samples_camera = self._samples_camera[indices]
        self._samples_reference = self._samples_reference[indices]

    def generate_LUT(self, size=1024):
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

        Parameters
        ----------
        size : integer, optional
            *LUT* size.

        Returns
        -------
        :class:`LUT3x1D`
            Unfiltered linearisation *LUT* for the camera samples.
        """

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

    def filter_LUT(self, sigma=16):
        """
        Filter/smooth the linearisation *LUT* for the camera samples.

        The *LUT* filtering is performed with a gaussian convolution, the sigma
        value represents the window size. To prevent that the edges of the
        *LUT* are affected by the convolution, the *LUT* is extended, i.e.
        extrapolated at a safe two sigmas in both directions. The left edge is
        linearly extrapolated while the right edge is logarithmically
        extrapolated.

        Parameters
        ----------
        sigma : numeric
            Standard deviation of the gaussian convolution kernel.

        Returns
        -------
        :class:`LUT3x1D`
            Filtered linearisation *LUT* for the camera samples.
        """

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

    def decode(
        self,
        decoding_method="Median",
        grey_card_reflectance=(0.18, 0.18, 0.18),
    ):
        """
        Decode the samples produced by the image sampling process.

        Parameters
        ----------
        decoding_method : str, optional
            {"Median", "Average", "Per Channel", "ACES"},
            Decoding method.
        grey_card_reflectance : array_like, optional
            Measured grey card reflectance.
        """

        logger.info(
            'Decoding analysis samples using "%s" method and "%s" grey '
            "card reflectance...",
            decoding_method,
            grey_card_reflectance,
        )

        decoding_method = validate_method(
            decoding_method,
            ("Median", "Average", "Per Channel", "ACES"),
        )

        grey_card_reflectance = as_float_array(grey_card_reflectance)

        if decoding_method == "median":
            self._LUT_decoding = LUT1D(np.median(self._LUT_filtered.table, axis=-1))
        elif decoding_method == "average":
            self._LUT_decoding = LUT_to_LUT(
                self._LUT_filtered, LUT1D, force_conversion=True
            )
        elif decoding_method == "per channel":
            self._LUT_decoding = self._LUT_filtered.copy()
        elif decoding_method == "aces":
            self._LUT_decoding = LUT_to_LUT(
                self._LUT_filtered,
                LUT1D,
                force_conversion=True,
                channel_weights=RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ[1, ...],
            )

        self._LUT_decoding.name = "LUT - Decoding"

        if self._samples_analysis["data"]["grey_card"]:
            sampled_grey_card_reflectance = self._samples_analysis["data"]["grey_card"][
                "samples_median"
            ]
            linear_gain = grey_card_reflectance / self._LUT_decoding.apply(
                sampled_grey_card_reflectance
            )
            if decoding_method == "median":
                linear_gain = np.median(linear_gain)
            elif decoding_method == "average":
                linear_gain = np.average(linear_gain)
            elif decoding_method == "per channel":
                pass
            elif decoding_method == "aces":
                linear_gain = RGB_luminance(
                    linear_gain,
                    RGB_COLOURSPACE_ACES2065_1.primaries,
                    RGB_COLOURSPACE_ACES2065_1.whitepoint,
                )

            self._LUT_decoding.table *= linear_gain

        self._samples_decoded = {}
        for EV in sorted(self._samples_analysis["data"]["colour_checker"]):
            self._samples_decoded[EV] = self._LUT_decoding.apply(
                as_float_array(
                    self._samples_analysis["data"]["colour_checker"][EV][
                        "samples_median"
                    ]
                )
            )

    def optimise(
        self,
        EV_range=(-1, 0, 1),
        EV_weights=None,
        training_data=RGB_COLORCHECKER_CLASSIC_ACES,
        optimisation_factory=optimisation_factory_rawtoaces_v1,
        optimisation_kwargs=None,
    ):
        """
        Compute the *IDT* matrix.

        Parameters
        ----------
        EV_range : array_like, optional
            Exposure values to use when computing the *IDT* matrix.
        EV_weights : array_like, optional
            Normalised weights used to sum the exposure values. If not given,
            the median of the exposure values is used.
        training_data : NDArray, optional
            Training data multi-spectral distributions, defaults to using the
            *RAW to ACES* v1 190 patches.
        optimisation_factory : callable, optional
            Callable producing the objective function and the *CIE XYZ* to
            optimisation colour model function.
        optimisation_kwargs : dict, optional
            Parameters for :func:`scipy.optimize.minimize` definition.

        Returns
        -------
        :class:`tuple`
            Tuple of *IDT* matrix :math:`M`, white balance multipliers
            :math:`RGB_w` and exposure factor :math:`k` that results in a
            nominally "18% gray" object in the scene producing ACES values
            [0.18, 0.18, 0.18].
        """

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

        if EV_weights is None:
            self._samples_weighted = np.median(samples_normalised, axis=0)
        else:
            self._samples_weighted = np.sum(
                samples_normalised
                * as_float_array(EV_weights)[..., np.newaxis, np.newaxis],
                axis=0,
            )

        XYZ = vector_dot(RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ, training_data)

        self._RGB_w = 1 / np.mean(self._samples_weighted[18:], axis=0)
        self._k = self._RGB_w * np.mean(XYZ[18:], axis=0)
        self._samples_weighted *= self._k

        self._RGB_w /= self._RGB_w[1]
        self._k = np.mean(self._k)

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

    def to_clf(self, output_directory, information):
        """
        Convert the *IDT* generation process data to *Common LUT Format* (CLF).

        Parameters
        ----------
        output_directory : str
            Output directory for the zip file.
        information : dict
            Information pertaining to the *IDT* and the computation parameters.

        Returns
        -------
        :class:`str`
            *CLF* file path.
        """

        logger.info(
            'Converting "IDT" generation process data to "CLF" in "%s"'
            'output directory using given information: "%s".',
            output_directory,
            information,
        )

        aces_transform_id = self._specification["header"]["aces_transform_id"]
        aces_user_name = self._specification["header"]["aces_user_name"]
        camera_make = self._specification["header"]["camera_make"]
        camera_model = self._specification["header"]["camera_model"]

        root = Et.Element(
            "ProcessList",
            compCLFversion="3",
            id=aces_transform_id,
            name=aces_user_name,
        )

        def format_array(a):
            """Format given array :math:`a`."""

            return re.sub(r"\[|\]|,", "", "\n".join(map(str, a.tolist())))

        et_input_descriptor = Et.SubElement(root, "InputDescriptor")
        et_input_descriptor.text = f"{camera_make} {camera_model}"

        et_output_descriptor = Et.SubElement(root, "OutputDescriptor")
        et_output_descriptor.text = "ACES2065-1"

        et_info = Et.SubElement(root, "Info")
        et_metadata = Et.SubElement(et_info, "Archive")
        for (
            key,
            value,
        ) in self._specification["header"].items():
            if key == "schema_version":
                continue

            sub_element = Et.SubElement(
                et_metadata, key.replace("_", " ").title().replace(" ", "")
            )
            sub_element.text = str(value)
        et_academy_idt_calculator = Et.SubElement(et_info, "AcademyIDTCalculator")
        for key, value in information.items():
            sub_element = Et.SubElement(et_academy_idt_calculator, key)
            sub_element.text = str(value)

        et_lut = Et.SubElement(
            root,
            "LUT1D",
            inBitDepth="32f",
            outBitDepth="32f",
            interpolation="linear",
        )
        LUT_decoding = self._LUT_decoding
        channels = 1 if isinstance(LUT_decoding, LUT1D) else 3
        et_description = Et.SubElement(et_lut, "Description")
        et_description.text = f"Linearisation *{LUT_decoding.__class__.__name__}*."
        et_array = Et.SubElement(et_lut, "Array", dim=f"{LUT_decoding.size} {channels}")
        et_array.text = f"\n{format_array(LUT_decoding.table)}"

        root = clf_processing_elements(root, self._M, self._RGB_w, self._k, False)

        clf_path = (
            f"{output_directory}/"
            f"{camera_make}.Input.{camera_model}_to_ACES2065-1.clf"
        )
        Et.indent(root)

        with open(clf_path, "w") as clf_file:
            clf_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            clf_file.write(Et.tostring(root, encoding="UTF-8").decode("utf8"))

        return clf_path

    def zip(self, output_directory, information, archive_serialised_generator=False):
        """
        Zip the *Common LUT Format* (CLF) resulting from the *IDT* generation
        process.

        Parameters
        ----------
        output_directory : str
            Output directory for the *zip* file.
        information : dict
            Information pertaining to the *IDT* and the computation parameters.
        archive_serialised_generator : bool
            Whether to serialise and archive the *IDT* generator.

        Returns
        -------
        :class:`str`
            *Zip* file path.
        """

        logger.info(
            'Zipping the "CLF" resulting from the "IDT" generation '
            'process in "%s" output directory using given information: "%s".',
            output_directory,
            information,
        )

        camera_make = self._specification["header"]["camera_make"]
        camera_model = self._specification["header"]["camera_model"]

        clf_path = self.to_clf(output_directory, information)

        json_path = f"{output_directory}/{camera_make}.{camera_model}.json"
        with open(json_path, "w") as json_file:
            json_file.write(jsonpickle.encode(self, indent=2))

        zip_file = Path(output_directory) / f"IDT_{camera_make}_{camera_model}.zip"

        current_working_directory = os.getcwd()
        try:
            os.chdir(output_directory)
            with ZipFile(zip_file, "w") as zip_archive:
                zip_archive.write(clf_path.replace(output_directory, "")[1:])
                if archive_serialised_generator:
                    zip_archive.write(json_path.replace(output_directory, "")[1:])
        finally:
            os.chdir(current_working_directory)

        return zip_file

    def png_colour_checker_segmentation(self):
        """
        Return the colour checker segmentation image as *PNG* data.

        Returns
        -------
        :class:`str` or None
            *PNG* data.
        """

        if self._image_colour_checker_segmentation is None:
            return None

        colour.plotting.plot_image(
            self._image_colour_checker_segmentation,
            show=False,
        )

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        data_png = base64.b64encode(buffer.getbuffer()).decode("utf8")
        plt.close()

        return data_png

    def png_grey_card_sampling(self):
        """
        Return the grey card image sampling as *PNG* data.

        Returns
        -------
        :class:`str` or None
            *PNG* data.
        """

        if self._image_grey_card_sampling is None:
            return None

        colour.plotting.plot_image(
            self._image_grey_card_sampling,
            show=False,
        )
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        data_png = base64.b64encode(buffer.getbuffer()).decode("utf8")
        plt.close()

        return data_png

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
        resources_directory / "synthetic_002.zip", cleanup=True
    )

    logger.info("LUT :\n%s", idt_generator.LUT_decoding)
    logger.info("M :\n%s", idt_generator.M)
    logger.info("White Balance Multipliers :\n%s", idt_generator.RGB_w)
    logger.info("k :\n%s", idt_generator.k)

    with open(
        resources_directory / "synthetic_001" / "synthetic_001.json"
    ) as json_file:
        specification = json.load(json_file)

    idt_generator = IDTGeneratorProsumerCamera.from_specification(
        specification, resources_directory / "synthetic_001"
    )

    logger.info("LUT :\n%s", idt_generator.LUT_decoding)
    logger.info("M :\n%s", idt_generator.M)
    logger.info("White Balance Multipliers :\n%s", idt_generator.RGB_w)
    logger.info("k :\n%s", idt_generator.k)
