"""
Input Device Transform (IDT) Prosumer Camera Utilities
======================================================
"""

import base64
import colour
import colour_checker_detection
import cv2
import json
import jsonpickle
import io
import matplotlib
import numpy as np
import os
import re
import scipy.misc
import scipy.ndimage
import scipy.optimize
import scipy.stats
import shutil
import tempfile
from copy import deepcopy
from colour import (
    Extrapolator,
    LinearInterpolator,
    LUT1D,
    LUT3x1D,
    SDS_COLOURCHECKERS,
    SDS_ILLUMINANTS,
    read_image,
    sd_to_aces_relative_exposure_values,
)
from colour.algebra import smoothstep_function, vector_dot
from colour.characterisation import optimisation_factory_rawtoaces_v1
from colour.models import RGB_COLOURSPACE_ACES2065_1
from colour.io import LUT_to_LUT
from colour.hints import Dict, NDArray, Union
from colour.utilities import (
    CACHE_REGISTRY,
    MixinDataclassIterable,
    as_float_array,
    attest,
    optional,
    orient,
    validate_method,
)
from dataclasses import dataclass
from pathlib import Path
from scipy.optimize import minimize
from zipfile import ZipFile

from aces.idt import slugify

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "SDS_COLORCHECKER_CLASSIC",
    "SD_ILLUMINANT_ACES",
    "generate_reference_colour_checker",
    "RGB_COLORCHECKER_CLASSIC_ACES",
    "DATA_SPECIFICATION",
    "DATA_SAMPLES_ANALYSIS",
    "EXTENSION_DEFAULT",
    "SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC",
    "mask_outliers",
    "swatch_colours_from_image",
    "is_colour_checker_flipped",
    "flip_image",
    "list_sub_directories",
    "DataSampleColourCheckers",
    "sample_colour_checkers",
    "sort_samples",
    "DataGenerateLUT3x1D",
    "generate_LUT3x1D",
    "filter_LUT3x1D",
    "DataDecodeSamples",
    "decode_samples",
    "DataMatrixIdt",
    "matrix_idt",
    "archive_to_specification",
    "DataArchiveToSamples",
    "archive_to_samples",
    "DataArchiveToIdt",
    "archive_to_idt",
    "apply_idt",
    "zip_idt",
    "png_segmented_image",
    "png_measured_camera_samples",
    "png_extrapolated_camera_samples",
]

SDS_COLORCHECKER_CLASSIC = tuple(SDS_COLOURCHECKERS["ISO 17321-1"].values())
"""
Reference reflectances for the *ColorChecker Classic*.

SDS_COLORCHECKER_CLASSIC : tuple
"""


SD_ILLUMINANT_ACES = SDS_ILLUMINANTS["D60"]
"""
*ACES* reference illuminant spectral distribution,
i.e. ~*CIE Illuminant D Series D60*.

SD_ILLUMINANT_ACES : SpectralDistribution
"""


def generate_reference_colour_checker(
    sds=SDS_COLORCHECKER_CLASSIC,
    illuminant=SD_ILLUMINANT_ACES,
    chromatic_adaptation_transform="CAT02",
):
    """
    Generate the reference *ACES* *RGB* values for the *ColorChecker Classic*.

    Parameters
    ----------
    sds : tuple, optional
        *ColorChecker Classic* reflectances.
    illuminant : SpectralDistribution, optional
        Spectral distribution of the illuminant to compute the reference
        *ACES* *RGB* values.
    chromatic_adaptation_transform : str
        *Chromatic adaptation* transform.

    Returns
    -------
    NDArray
        Reference *ACES* *RGB* values.
    """

    return as_float_array(
        [
            sd_to_aces_relative_exposure_values(
                sd,
                illuminant,
                apply_chromatic_adaptation=True,
                chromatic_adaptation_transform=chromatic_adaptation_transform,
            )
            for sd in sds
        ]
    )


RGB_COLORCHECKER_CLASSIC_ACES = generate_reference_colour_checker()
"""
Reference *ACES* *RGB* values for the *ColorChecker Classic*.

RGB_COLORCHECKER_CLASSIC_ACES : NDArray
"""


DATA_SPECIFICATION = {
    "header": {"schema_version": "0.1.0", "camera": None},
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
Template specification for the colour checker sampling process.

DATA_SAMPLES_ANALYSIS : dict
"""


EXTENSION_DEFAULT = "tif"
"""
Default file format extension to search for in the *IDT* archive.

EXTENSION_DEFAULT : str
"""

SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC = (
    colour_checker_detection.SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.copy()
)
"""
Settings for the segmentation of the *X-Rite* *ColorChecker Classic* and
*X-Rite* *ColorChecker Passport* for a typical *Prosumer Camera* shoot.

SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC : dict
"""

SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC.update(
    {
        "working_width": 1600,
        "swatches_count_minimum": 24 / 2,
        "fast_non_local_means_denoising_kwargs": {
            "h": 3,
            "templateWindowSize": 5,
            "searchWindowSize": 11,
        },
        "adaptive_threshold_kwargs": {
            "maxValue": 255,
            "adaptiveMethod": cv2.ADAPTIVE_THRESH_MEAN_C,
            "thresholdType": cv2.THRESH_BINARY,
            "blockSize": int(1600 * 0.015) - int(1600 * 0.015) % 2 + 1,
            "C": 2,
        },
    }
)


def mask_outliers(a, axis=None, z_score=3):
    """
    Return the mask for the outliers of given array :math:`a` using the
    z-score.

    Parameters
    ----------
    a : array_like
        Array :math:`a` to return the outliers mask of.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.
    z_score : numeric
        z-score threshold to mask the outliers.

    Returns
    -------
    NDArray
        Mask for the outliers of given array :math:`a`.
    """

    return np.abs(scipy.stats.zscore(a, axis=axis)) > z_score


def swatch_colours_from_image(
    image,
    colour_checker_rectangle,
    samples=24,
    swatches_h=SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC[
        "swatches_horizontal"
    ],
    swatches_v=SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC["swatches_vertical"],
):
    """
    Extract the swatch colours for given image using given rectifying rectangle.

    Parameters
    ----------
    image : array_like
        Image to extract the swatch colours of.
    colour_checker_rectangle : array_like
        Rectifying rectangle.
    samples : integer, optional
        Samples count to use to compute the swatches colours. The effective
        samples count is :math:`samples_analysis^2`.
    swatches_h : int, optional
        Horizontal swatches.
    swatches_v : int, optional
        Vertical swatches.

    Returns
    -------
    NDArray
        Swatch colours.
    """

    image = colour_checker_detection.detection.segmentation.adjust_image(
        image, SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC["working_width"]
    )

    colour_checker = (
        colour_checker_detection.detection.segmentation
    ).crop_and_level_image_with_rectangle(
        image,
        cv2.minAreaRect(colour_checker_rectangle),
        SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC["interpolation_method"],
    )
    # TODO: Release new "colour-checker-detection" package.
    if colour_checker.shape[0] > colour_checker.shape[1]:
        colour_checker = orient(colour_checker, "90 CW")

    width, height = colour_checker.shape[1], colour_checker.shape[0]
    masks = colour_checker_detection.detection.segmentation.swatch_masks(
        width, height, swatches_h, swatches_v, samples
    )

    swatch_colours = []
    masks_i = np.zeros(colour_checker.shape)
    for mask in masks:
        swatch_colours.append(
            np.mean(
                colour_checker[mask[0] : mask[1], mask[2] : mask[3], ...],
                axis=(0, 1),
            )
        )
        masks_i[mask[0] : mask[1], mask[2] : mask[3], ...] = 1

    swatch_colours = as_float_array(swatch_colours)

    return swatch_colours


def is_colour_checker_flipped(swatch_colours):
    """
    Return whether the colour checker is flipped.

    The colour checker might be flipped: The mean standard deviation
    of some expected normalised chromatic and achromatic neutral
    swatches is computed. If the chromatic mean is lesser than the
    achromatic mean, it means that the colour checker is flipped.

    Parameters
    ----------
    swatch_colours : array_like
        Swatch colours.

    Returns
    -------
    bool
        Whether the colour checker is flipped.
    """

    std_means = []
    for slice_ in [
        SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC["swatches_chromatic_slice"],
        SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC[
            "swatches_achromatic_slice"
        ],
    ]:
        swatch_std_mean = as_float_array(swatch_colours[slice_])
        swatch_std_mean /= swatch_std_mean[..., 1][..., np.newaxis]
        std_means.append(np.mean(np.std(swatch_std_mean, 0)))

    is_flipped = True if std_means[0] < std_means[1] else False

    if is_flipped:
        print("Colour checker was seemingly flipped!")
        return True
    else:
        False


def flip_image(image):
    """
    Flip given image by rotating it 180 degrees.

    Parameters
    ----------
    image : array_like
        Image to rotate 180 degrees

    Returns
    -------
    NDArray
        Flipped image.
    """

    return orient(image, "180")


def list_sub_directories(
    directory,
    filterers=(
        lambda path: False if "__MACOSX" in path.name else True,
        lambda path: path.is_dir(),
    ),
):
    """
    List the sub-directories in given directory.

    Parameters
    ----------
    directory : str
        Directory to list the sub-directories from.
    filterers : array_like, optional
        List of callables used to filter the sub-directories, each callable
        takes a :class:`Path` class instance as argument and returns whether to
        include or exclude the sub-directory as a bool.
        include or exclude the sub-directory as a bool.

    Returns
    -------
    list
        Sub-directories in given directory.
    """

    sub_directories = [
        path
        for path in Path(directory).iterdir()
        if all([filterer(path) for filterer in filterers])
    ]

    print(directory, sub_directories)

    return sub_directories


@dataclass
class DataSampleColourCheckers(MixinDataclassIterable):
    """
    Analysis data from the colour checker sampling process.

    Parameters
    ----------
    samples_analysis : dict
        Samples produced by the colour checker sampling process.
    image_segmentation : NDArray
        Image with segmentation contours.
    """

    samples_analysis: dict
    image_segmentation: NDArray


def sample_colour_checkers(specification, additional_data=False):
    """
    Sample the colour checkers for given *IDT* archive specification.

    Parameters
    ----------
    specification : dict
        *IDT* archive specification.
    additional_data : bool, optional
        Whether to return additional data.

    Returns
    -------
    dict or DataSampleColourCheckers
        Samples produced by the colour checker sampling process or data from
        the colour checker sampling process.
    """

    samples_analysis = deepcopy(DATA_SAMPLES_ANALYSIS)

    # Segmentation occurs on EV 0 and is reused on all brackets.
    paths = specification["data"]["colour_checker"][0]

    # Detecting the colour checker and whether it is flipped.
    is_flipped, should_flip = False, False
    while True:
        should_flip = is_flipped
        image = read_image(paths[0])
        image = flip_image(image) if is_flipped else image

        (
            colour_checkers,
            clusters,
            swatches,
            segmented_image,
        ) = colour_checker_detection.colour_checkers_coordinates_segmentation(
            image,
            additional_data=True,
            **SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
        ).values

        assert len(colour_checkers), "Colour checker was not detected at EV 0!"

        colour_checker_rectangle = colour_checkers[0]

        swatch_colours = swatch_colours_from_image(
            image, colour_checker_rectangle
        )

        is_flipped = is_colour_checker_flipped(swatch_colours)

        if not is_flipped:
            break

    is_flipped = should_flip

    image_segmentation = (
        colour_checker_detection.detection.segmentation.adjust_image(
            image, SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC["working_width"]
        )
    )
    cv2.drawContours(image_segmentation, swatches, -1, (1, 0, 1), 3)
    cv2.drawContours(image_segmentation, clusters, -1, (0, 1, 1), 3)

    # Flatfield
    if specification["data"].get("flatfield") is not None:
        samples_analysis["data"]["flatfield"] = {"samples_sequence": []}
        for path in specification["data"]["flatfield"]:
            image = read_image(path)
            image = flip_image(image) if is_flipped else image
            swatch_colours = swatch_colours_from_image(
                image, colour_checker_rectangle
            )

            samples_analysis["data"]["flatfield"]["samples_sequence"].append(
                swatch_colours.tolist()
            )

        samples_sequence = as_float_array(
            [
                samples[0]
                for samples in samples_analysis["data"]["flatfield"][
                    "samples_sequence"
                ]
            ]
        )
        mask = np.all(~mask_outliers(samples_sequence), axis=-1)

        samples_analysis["data"]["flatfield"]["samples_median"] = np.median(
            as_float_array(
                samples_analysis["data"]["flatfield"]["samples_sequence"]
            )[mask],
            0,
        ).tolist()

    # ColourChecker Classic Samples per EV
    for EV in specification["data"]["colour_checker"].keys():
        samples_analysis["data"]["colour_checker"][EV] = {}
        samples_analysis["data"]["colour_checker"][EV]["samples_sequence"] = []
        for path in specification["data"]["colour_checker"][EV]:
            image = read_image(path)
            image = flip_image(image) if is_flipped else image
            swatch_colours = swatch_colours_from_image(
                image, colour_checker_rectangle
            )

            samples_analysis["data"]["colour_checker"][EV][
                "samples_sequence"
            ].append(swatch_colours.tolist())

        sequence_neutral_5 = as_float_array(
            [
                samples[21]
                for samples in samples_analysis["data"]["colour_checker"][EV][
                    "samples_sequence"
                ]
            ]
        )
        mask = np.all(~mask_outliers(sequence_neutral_5), axis=-1)

        samples_analysis["data"]["colour_checker"][EV][
            "samples_median"
        ] = np.median(
            as_float_array(
                samples_analysis["data"]["colour_checker"][EV][
                    "samples_sequence"
                ]
            )[mask],
            0,
        ).tolist()

    if additional_data:
        return DataSampleColourCheckers(samples_analysis, image_segmentation)
    else:
        return samples_analysis


def sort_samples(
    samples_analysis, reference_colour_checker=RGB_COLORCHECKER_CLASSIC_ACES
):
    """
    Sort the samples produced by the colour checker sampling process.

    The *ACES* reference samples are sorted and indexed as a function of the
    camera samples ordering. This ensures that the camera samples are
    monotonically increasing.

    Parameters
    ----------
    samples_analysis : dict
        Samples produced by the colour checker sampling process.
    reference_colour_checker : NDArray
        Reference *ACES* *RGB* values for the *ColorChecker Classic*.

    Returns
    -------
    tuple
        Tuple of camera and reference *ACES* *RGB* samples.
    """

    EV_reference_colour_checker = {
        i: reference_colour_checker[-6:, ...] * pow(2, i)
        for i in range(-20, 20, 1)
    }

    samples_camera = []
    samples_reference = []
    for EV, images in samples_analysis["data"]["colour_checker"].items():
        samples_reference.append(EV_reference_colour_checker[EV])
        samples_EV = as_float_array(images["samples_median"])[-6:, ...]
        samples_camera.append(samples_EV)
    samples_camera = np.vstack(samples_camera)
    samples_reference = np.vstack(samples_reference)

    indices = np.argsort(np.median(samples_camera, axis=-1), axis=0)

    samples_camera = samples_camera[indices]
    samples_reference = samples_reference[indices]

    return samples_camera, samples_reference


@dataclass
class DataGenerateLUT3x1D(MixinDataclassIterable):
    """
    Data from the linearisation *LUT3x1D* generation process.

    Parameters
    ----------
    LUT_unfiltered : LUT3x1D
        Unfiltered linearisation *LUT* for the camera samples.
    samples_linear : NDArray
        Samples generated using linear extrapolation.
    samples_constant : NDArray
        Samples generated using constant extrapolation.
    samples_middle : NDArray
        Samples used for linearly fitting the central slope in logarithmic
        space.
    mask_samples : NDArray
        Mask for blending ``samples_constant`` and the linearly fitted central
        slope.
    coefficients : NDArray
        Coefficients of the line that linearly fits the central slope.
    edges : NDArray
        Left and right edges that ``mask_samples`` spans.
    """

    LUT_unfiltered: LUT3x1D
    samples_linear: NDArray
    samples_constant: NDArray
    samples_middle: NDArray
    coefficients: NDArray
    mask_samples: NDArray
    edges: NDArray


def generate_LUT3x1D(
    samples_camera, samples_reference, size=1024, additional_data=False
):
    """
    Generate an unfiltered linearisation *LUT* for the camera samples.

    The *LUT* generation process is worth describing, the camera samples are
    unlikely to cover the [0, 1] domain and thus need to be extrapolated.

    Two extrapolated datasets are generated:

        -   Linearly extrapolated for the left edge missing data whose clipping
            likelihood is low and thus can be extrapolated safely.
        -   Constant extrapolated for the right edge missing data whose clipping
            likelihood is high and thus cannot be extrapolated safely

    Because the camera encoded data response is logarithmic, the slope of the
    center portion of the data is computed and fitted. The fitted line will be
    used to extrapolate the right edge missing data. It is blended through a
    smoothstep with the constant extrapolated samples. The blend is fully
    achieved at the right edge of the camera samples.

    Parameters
    ----------
    samples_camera : array_like
        Samples for the camera.
    samples_reference : array_like
        Reference *ACES* *RGB* samples.
    size : integer, optional
        *LUT* size.
    additional_data : bool, optional
        Whether to return additional data.

    Returns
    -------
    LUT3x1D or DataGenerateLUT3x1D
        Unfiltered linearisation *LUT* for the camera samples or data
        from the linearisation *LUT3x1D* generation process.
    """

    samples_camera = as_float_array(samples_camera)
    samples_reference = as_float_array(samples_reference)

    LUT_unfiltered = LUT3x1D(size=size, name="LUT - Unfiltered")

    for i in range(3):
        x = samples_camera[..., i] * (size - 1)
        y = samples_reference[..., i]

        samples = np.arange(0, size, 1)

        samples_linear = Extrapolator(LinearInterpolator(x, y))(samples)
        samples_constant = Extrapolator(
            LinearInterpolator(x, y), method="Constant"
        )(samples)

        # Searching for the index of ~middle camera code value * 125%
        # We are trying to find the logarithmic slope of the camera middle
        # range.
        index_middle = np.searchsorted(
            samples / size, np.max(samples_camera) / 2 * 1.25
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
        edge_right = np.searchsorted(samples / size, np.max(samples_camera))
        mask_samples = smoothstep_function(
            samples, edge_left, edge_right, clip=True
        )

        LUT_unfiltered.table[..., i] = samples_linear
        LUT_unfiltered.table[index_middle - padding :, i] = (
            np.exp(a * samples + b) * mask_samples
            + samples_constant * (1 - mask_samples)
        )[index_middle - padding :]

    if additional_data:
        return DataGenerateLUT3x1D(
            LUT_unfiltered,
            samples_linear,
            samples_constant,
            samples_middle,
            as_float_array([a, b]),
            mask_samples,
            as_float_array([edge_left / size, edge_right / size]),
        )
    else:
        return LUT_unfiltered


def filter_LUT3x1D(LUT, sigma=16):
    """
    Filter/smooth the linearisation *LUT* for the camera samples.

    The LUT filtering is performed with a gaussian convolution, the sigma value
    represents the window size. To prevent that the edges of the LUT are
    affected by the convolution, the LUT is extended, i.e. extrapolated at a
    safe two sigmas in both directions. The left edge is linearly extrapolated
    while the right edge is logarithmically extrapolated.

    Parameters
    ----------
    LUT : LUT3x1D
        Linearisation *LUT* for the camera samples.
    sigma : numeric
        Standard deviation of the gaussian convolution kernel.

    Returns
    -------
    LUT3x1D
        Filtered linearisation *LUT* for the camera samples.
    """

    filter = scipy.ndimage.gaussian_filter1d
    filter_kwargs = {"sigma": sigma}

    LUT_filtered = LUT.copy()
    LUT_filtered.name = "LUT - Filtered"

    sigma_x2 = int(sigma * 2)
    x, step = np.linspace(0, 1, LUT.size, retstep=True)
    padding = np.arange(step, sigma_x2 * step + step, step)
    for i in range(3):
        y = LUT_filtered.table[..., i]
        x_extended = np.concatenate([-padding[::-1], x, padding + 1])

        # Filtering is performed on extrapolated data.
        y_linear_extended = Extrapolator(LinearInterpolator(x, y))(x_extended)
        y_log_extended = np.exp(
            Extrapolator(LinearInterpolator(x, np.log(y)))(x_extended)
        )

        y_linear_filtered = filter(y_linear_extended, **filter_kwargs)
        y_log_filtered = filter(y_log_extended, **filter_kwargs)

        index_middle = len(x_extended) // 2
        LUT_filtered.table[..., i] = np.concatenate(
            [
                y_linear_filtered[sigma_x2:index_middle],
                y_log_filtered[index_middle:-sigma_x2],
            ]
        )

    return LUT_filtered


@dataclass
class DataDecodeSamples(MixinDataclassIterable):
    """
    Data from the decoding samples process.

    Parameters
    ----------
    samples_decoded : NDArray
        Decoded samples.
    LUT_decoding : LUT1D or LUT3x1D
        Decoding *LUT* for the camera samples, the difference with the
        linearisation *LUT* is that the former is the final *LUT* used for the
        camera samples and is the result of transforming the channels of the
        linearisation *LUT* through a median or averaging operation for
        example.
    """

    samples_decoded: NDArray
    LUT_decoding: Union[LUT1D, LUT3x1D]


def decode_samples(
    samples_analysis, LUT, decoding_method="Median", additional_data=False
):
    """
    Decode the samples produced by the colour checker sampling process.

    Parameters
    ----------
    samples_analysis : dict
        Samples produced by the colour checker sampling process.
    LUT : LUT1D or LUT3x1D
        Linearisation *LUT* for the camera samples.
    decoding_method : str
        {"Median", "Average", "Per Channel", "ACES"},
        Decoding method.
    additional_data : bool, optional
        Whether to return additional data.

    Returns
    -------
    dict or DataDecodeSamples
        Decoded samples or data from the decoding samples process.
    """

    decoding_method = validate_method(
        decoding_method,
        ["Median", "Average", "Per Channel", "ACES"],
    )

    if decoding_method == "median":
        LUT_decoding = LUT1D(np.median(LUT.table, axis=-1))
    elif decoding_method == "average":
        LUT_decoding = LUT_to_LUT(LUT, LUT1D, force_conversion=True)
    elif decoding_method == "per channel":
        LUT_decoding = LUT.copy()
    elif decoding_method == "aces":
        channel_weights = RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ[1, ...]
        LUT_decoding = LUT_to_LUT(
            LUT,
            LUT1D,
            force_conversion=True,
            channel_weights=channel_weights,
        )

    LUT_decoding.name = "LUT - Decoding"

    samples_decoded = {}
    for EV in sorted(samples_analysis["data"]["colour_checker"]):
        samples_decoded[EV] = LUT_decoding.apply(
            as_float_array(
                samples_analysis["data"]["colour_checker"][EV][
                    "samples_median"
                ]
            )
        )

    if additional_data:
        return DataDecodeSamples(samples_decoded, LUT_decoding)
    else:
        return samples_decoded


@dataclass
class DataMatrixIdt(MixinDataclassIterable):
    """
    Data from the *Input Device Transform* (IDT) matrix generation process.

    Parameters
    ----------
    M : NDArray
        *Input Device Transform* (IDT) matrix.
    samples_weighted : NDArray
        Weighted samples according to their median or the given weights.
    """

    M: NDArray
    samples_weighted: NDArray


def matrix_idt(
    samples,
    EV_range=(-1, 0, 1),
    EV_weights=None,
    training_data=RGB_COLORCHECKER_CLASSIC_ACES,
    optimisation_factory=optimisation_factory_rawtoaces_v1,
    optimisation_kwargs=None,
    additional_data=False,
):
    """
    Compute the *IDT* matrix.

    Parameters
    ----------
    samples : NDArray
        Camera samples.
    EV_range : array_like, optional
        Exposure values to use when computing the *IDT* matrix.
    EV_weights : array_like, optional
        Normalised weights used to sum the exposure values. If not given, the
        median of the exposure values is used.
    training_data : NDArray, optional
        Training data multi-spectral distributions, defaults to using the
        *RAW to ACES* v1 190 patches.
    optimisation_factory : callable, optional
        Callable producing the objective function and the *CIE XYZ* to
        optimisation colour model function.
    optimisation_kwargs : dict, optional
        Parameters for :func:`scipy.optimize.minimize` definition.
    additional_data : bool, optional
        Whether to return additional data.

    Returns
    -------
    NDArray or DataMatrixIdt
        *IDT* matrix or data from the *Input Device Transform* (IDT) matrix
        generation process.
    """

    samples_normalised = as_float_array(
        [
            samples[EV] * (1 / pow(2, EV))
            for EV in np.atleast_1d(EV_range)
            if EV in samples
        ]
    )

    if EV_weights is None:
        samples_weighted = np.median(samples_normalised, axis=0)
    else:
        samples_weighted = np.sum(
            samples_normalised
            * as_float_array(EV_weights)[..., np.newaxis, np.newaxis],
            axis=0,
        )

    XYZ = vector_dot(
        RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ, training_data
    )
    (
        objective_function,
        XYZ_to_optimization_colour_model,
    ) = optimisation_factory()
    optimisation_settings = {
        "method": "BFGS",
        "jac": "2-point",
    }
    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    M = minimize(
        objective_function,
        np.ravel(np.identity(3)),
        (samples_weighted, XYZ_to_optimization_colour_model(XYZ)),
        **optimisation_settings,
    ).x.reshape([3, 3])

    if additional_data:
        return DataMatrixIdt(M, samples_weighted)
    else:
        return M


def archive_to_specification(
    archive, directory, image_format=EXTENSION_DEFAULT
):
    """
    Extract the specification from given *IDT* archive.

    Parameters
    ----------
    archive : str
        *IDT* archive path, i.e. a zip file path.
    directory : str
        Directory to extract the *IDT* archive.
    image_format : str, optional
        Image format to filter.

    Returns
    -------
    dict
        *IDT* archive specification.
    """

    shutil.unpack_archive(archive, directory)

    extracted_directories = list_sub_directories(directory)

    attest(len(extracted_directories) == 1)

    root_directory = extracted_directories[0]

    json_files = list(root_directory.glob("*.json"))
    if len(json_files) == 1:
        with open(json_files[0]) as json_file:
            specification = json.load(json_file)
    else:
        specification = deepcopy(DATA_SPECIFICATION)

        specification["header"]["camera"] = Path(archive).stem

        colour_checker_directory = root_directory / "data" / "colour_checker"

        attest(colour_checker_directory.exists())

        for exposure_directory in colour_checker_directory.iterdir():
            if re.match(r"-?\d", exposure_directory.name):
                EV = exposure_directory.name
                specification["data"]["colour_checker"][EV] = list(
                    (colour_checker_directory / exposure_directory).glob(
                        f"*.{image_format}"
                    )
                )

        flatfield_directory = root_directory / "data" / "flatfield"
        if flatfield_directory.exists():
            specification["data"]["flatfield"] = list(
                flatfield_directory.glob(f"*.{image_format}")
            )

        grey_card_directory = root_directory / "data" / "grey_card"
        if grey_card_directory.exists():
            specification["data"]["grey_card"] = list(
                flatfield_directory.glob(f"*.{image_format}")
            )

    for exposure in list(specification["data"]["colour_checker"].keys()):
        images = [
            Path(root_directory) / image
            for image in specification["data"]["colour_checker"].pop(exposure)
        ]

        for image in images:
            attest(image.exists())

        specification["data"]["colour_checker"][int(exposure)] = images

    if specification["data"].get("flatfield") is not None:
        images = [
            Path(root_directory) / image
            for image in specification["data"]["flatfield"]
        ]
        for image in images:
            attest(image.exists())

        specification["data"]["flatfield"] = images

    if specification["data"].get("grey_card") is not None:
        images = [
            Path(root_directory) / image
            for image in specification["data"]["grey_card"]
        ]
        for image in images:
            attest(image.exists())

        specification["data"]["grey_card"] = images

    return specification


_CACHE_DATA_ARCHIVE_TO_SAMPLES = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_DATA_ARCHIVE_TO_SAMPLES"
)


@dataclass
class DataArchiveToSamples(MixinDataclassIterable):
    """
    Data from an *Input Device Transform* (IDT) archive to colour checker
    sampling process.

    Parameters
    ----------
    specification : dict
        Archive specification.
    data_sample_colour_checkers : DataSampleColourCheckers
        Analysis data from the colour checker sampling process.
    """

    specification: Dict
    data_sample_colour_checkers: DataSampleColourCheckers


def archive_to_samples(
    archive,
    image_format=EXTENSION_DEFAULT,
    additional_data=False,
    cleanup=True,
):
    """
    Extract the samples from given *IDT* archive.

    Parameters
    ----------
    archive : str
        *IDT* archive path, i.e. a zip file path.
    image_format : str, optional
        Image format to filter.
    additional_data : bool, optional
        Whether to return additional data.
    cleanup : bool, optional
        Whether to cleanup the temporary directory.

    Returns
    -------
    DataSampleColourCheckers or DataArchiveToSamples
        Data from the colour checker sampling process or data from an
        *Input Device Transform* (IDT) archive to colour checker sampling
        process.
    """

    key = (archive, image_format, additional_data)

    data_archive_to_samples = _CACHE_DATA_ARCHIVE_TO_SAMPLES.get(key)
    if data_archive_to_samples is None:
        temporary_directory = tempfile.TemporaryDirectory()
        specification = archive_to_specification(
            archive, temporary_directory.name, image_format
        )

        data_sample_colour_checkers = sample_colour_checkers(
            specification, additional_data=True
        )

        data_archive_to_samples = DataArchiveToSamples(
            specification, data_sample_colour_checkers
        )

        if cleanup:
            temporary_directory.cleanup()

    _CACHE_DATA_ARCHIVE_TO_SAMPLES[key] = data_archive_to_samples

    if additional_data:
        return data_archive_to_samples
    else:
        return data_archive_to_samples.data_sample_colour_checkers


@dataclass
class DataArchiveToIdt(MixinDataclassIterable):
    """
    Data from an *Input Device Transform* (IDT) archive to *IDT* matrix
    generation process.

    Parameters
    ----------
    data_archive_to_samples : DataArchiveToSamples
        Data from an *Input Device Transform* (IDT) archive to colour checker
        sampling process.
    samples_camera : NDArray
        Samples from the camera.
    samples_reference : NDArray
        Reference samples from the *ACES* colour checker.
    data_generate_LUT3x1D : DataGenerateLUT3x1D
        Data from the *LUT3x1D* generation process.
    LUT_filtered : LUT3x1D
        Filtered *LUT3x1D*.
    data_decode_samples : DataDecodeSamples
        Data from the decoding samples process.
    data_matrix_idt : DataMatrixIdt
        Data from the *Input Device Transform* (IDT) matrix generation process.
    """

    data_archive_to_samples: DataArchiveToSamples
    samples_camera: NDArray
    samples_reference: NDArray
    data_generate_LUT3x1D: DataGenerateLUT3x1D
    LUT_filtered: LUT3x1D
    data_decode_samples: DataDecodeSamples
    data_matrix_idt: DataMatrixIdt


def archive_to_idt(
    archive,
    image_format=EXTENSION_DEFAULT,
    archive_to_samples_kwargs=None,
    sort_samples_kwargs=None,
    generate_LUT3x1D_kwargs=None,
    filter_LUT3x1D_kwargs=None,
    decode_samples_kwargs=None,
    matrix_idt_kwargs=None,
    additional_data=False,
):
    """
    Generate the *Input Device Transform (IDT)* from given *IDT* archive.

    Parameters
    ----------
    archive : str
        *IDT* archive path, i.e. a zip file path.
    image_format : str, optional
        Image format to filter.
    archive_to_samples_kwargs : dict, optional
        Keyword arguments for the :func:`archive_to_samples` definition.
    sort_samples_kwargs : dict, optional
        Keyword arguments for the :func:`sort_samples` definition.
    generate_LUT3x1D_kwargs : dict, optional
        Keyword arguments for the :func:`generate_LUT3x1D` definition.
    filter_LUT3x1D_kwargs : dict, optional
        Keyword arguments for the :func:`filter_LUT3x1D` definition.
    decode_samples_kwargs : dict, optional
        Keyword arguments for the :func:`decode_samples` definition.
    matrix_idt_kwargs : dict, optional
        Keyword arguments for the :func:`matrix_idt` definition.
    additional_data : bool, optional
        Whether to return additional data.

    Returns
    -------
    tuple or DataArchiveToIdt
        Tuple of decoding *LUT* for the camera samples and *IDT* matrix or data
        from an *Input Device Transform* (IDT) archive to *IDT* matrix
        generation process.
    """

    archive_to_samples_kwargs = optional(archive_to_samples_kwargs, {})
    sort_samples_kwargs = optional(sort_samples_kwargs, {})
    generate_LUT3x1D_kwargs = optional(generate_LUT3x1D_kwargs, {})
    filter_LUT3x1D_kwargs = optional(filter_LUT3x1D_kwargs, {})
    decode_samples_kwargs = optional(decode_samples_kwargs, {})
    matrix_idt_kwargs = optional(matrix_idt_kwargs, {})

    data_archive_to_samples = archive_to_samples(
        archive,
        image_format,
        additional_data=True,
        **archive_to_samples_kwargs,
    )
    samples_camera, samples_reference = sort_samples(
        data_archive_to_samples.data_sample_colour_checkers.samples_analysis,
        **sort_samples_kwargs,
    )

    data_generate_LUT3x1D = generate_LUT3x1D(
        samples_camera,
        samples_reference,
        additional_data=True,
        **generate_LUT3x1D_kwargs,
    )

    LUT_filtered = filter_LUT3x1D(
        data_generate_LUT3x1D.LUT_unfiltered, **filter_LUT3x1D_kwargs
    )

    data_decode_samples = decode_samples(
        data_archive_to_samples.data_sample_colour_checkers.samples_analysis,
        LUT_filtered,
        additional_data=True,
        **decode_samples_kwargs,
    )

    data_matrix_idt = matrix_idt(
        data_decode_samples.samples_decoded,
        additional_data=True,
        **matrix_idt_kwargs,
    )

    if additional_data:
        return DataArchiveToIdt(
            data_archive_to_samples,
            samples_camera,
            samples_reference,
            data_generate_LUT3x1D,
            LUT_filtered,
            data_decode_samples,
            data_matrix_idt,
        )
    else:
        return data_matrix_idt.LUT_decoding, data_matrix_idt.M


def apply_idt(RGB, LUT, M):
    """
    Apply the given linearisation *LUT* for the camera samples and *IDT*
    matrix on given *RGB* colourspace array.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    LUT : LUT1D or LUT3x1D
        Linearisation *LUT* for the camera samples.
    M : array_like
        *IDT* matrix.

    Returns
    -------
    NDArray
        *RGB* colourspace array with *IDT* applied.
    """

    return vector_dot(M, LUT.apply(RGB))


def zip_idt(data_archive_to_idt, output_directory):
    """
    Zip the *IDT*.

    Parameters
    ----------
    data_archive_to_idt : DataArchiveToIdt
        Data from an *Input Device Transform* (IDT) archive to *IDT* matrix
        generation process.
    output_directory : str
        Output directory for the zip file.

    Returns
    -------
    str
        Zip file path.
    """

    camera_name = data_archive_to_idt.data_archive_to_samples.specification[
        "header"
    ]["camera"]

    camera_name = f"IDT_{slugify(camera_name)}"

    spi1d_file = f"{output_directory}/{camera_name}.spi1d"
    colour.write_LUT(
        data_archive_to_idt.data_decode_samples.LUT_decoding,
        spi1d_file,
    )

    spimtx_file = f"{output_directory}/{camera_name}.spimtx"
    colour.write_LUT(
        colour.LUTOperatorMatrix(data_archive_to_idt.data_matrix_idt.M),
        spimtx_file,
    )

    json_path = f"{output_directory}/{camera_name}.json"
    with open(json_path, "w") as json_file:
        json_file.write(jsonpickle.encode(data_archive_to_idt, indent=2))

    zip_file = Path(output_directory) / f"{camera_name}.zip"

    os.chdir(output_directory)
    with ZipFile(zip_file, "w") as zip_archive:
        zip_archive.write(spi1d_file.replace(output_directory, "")[1:])
        zip_archive.write(spimtx_file.replace(output_directory, "")[1:])
        zip_archive.write(json_path.replace(output_directory, "")[1:])

    return zip_file


def png_segmented_image(data_archive_to_idt):
    """
    Return the segmentation image as *PNG* data.

    Parameters
    ----------
    data_archive_to_idt : DataArchiveToIdt
        Data from an *Input Device Transform* (IDT) archive to *IDT* matrix
        generation process.

    Returns
    -------
    str
        *PNG* data.
    """

    data_sample_colour_checkers = (
        data_archive_to_idt.data_archive_to_samples.data_sample_colour_checkers
    )
    colour.plotting.plot_image(
        data_sample_colour_checkers.image_segmentation,
        standalone=False,
    )
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    data_png = base64.b64encode(buffer.getbuffer()).decode("utf8")
    plt.close()

    return data_png


def png_measured_camera_samples(data_archive_to_idt):
    """
    Return the measured camera samples as *PNG* data.

    Parameters
    ----------
    data_archive_to_idt : DataArchiveToIdt
        Data from an *Input Device Transform* (IDT) archive to *IDT* matrix
        generation process.

    Returns
    -------
    str
        *PNG* data.
    """

    figure, axes = colour.plotting.artist()
    axes.plot(
        data_archive_to_idt.samples_camera,
        np.log(data_archive_to_idt.samples_reference),
    )
    colour.plotting.render(
        **{
            "standalone": False,
            "x_label": "Camera Code Value",
            "y_label": "Log(ACES Reference)",
        }
    )
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    data_png = base64.b64encode(buffer.getbuffer()).decode("utf8")
    plt.close()

    return data_png


def png_extrapolated_camera_samples(data_archive_to_idt):
    """
    Return the extrapolated camera samples as *PNG* data.

    Parameters
    ----------
    data_archive_to_idt : DataArchiveToIdt
        Data from an *Input Device Transform* (IDT) archive to *IDT* matrix
        generation process.

    Returns
    -------
    str
        *PNG* data.
    """

    samples_camera = data_archive_to_idt.samples_camera
    samples_reference = data_archive_to_idt.samples_reference
    LUT_filtered = data_archive_to_idt.LUT_filtered
    edge_left, edge_right = data_archive_to_idt.data_generate_LUT3x1D.edges
    samples = np.linspace(0, 1, LUT_filtered.size)
    figure, axes = colour.plotting.artist()
    for i, RGB in enumerate(("r", "g", "b")):
        axes.plot(
            samples_camera[..., i],
            np.log(samples_reference)[..., i],
            "o",
            color=RGB,
            alpha=0.25,
        )
        axes.plot(samples, np.log(LUT_filtered.table[..., i]), color=RGB)
        axes.axvline(edge_left, color="r", alpha=0.25)
        axes.axvline(edge_right, color="r", alpha=0.25)
    colour.plotting.render(
        **{
            "standalone": False,
            "x_label": "Camera Code Value",
            "y_label": "Log(ACES Reference)",
        }
    )
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    data_png = base64.b64encode(buffer.getbuffer()).decode("utf8")
    plt.close()

    return data_png
