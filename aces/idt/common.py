"""
Common IDT Utilities
====================
"""

import base64
import colour
import colour_checker_detection
import cv2
import io
import matplotlib as mpl
import numpy as np
from colour import (
    SDS_COLOURCHECKERS,
    SDS_ILLUMINANTS,
    sd_to_aces_relative_exposure_values,
)
from colour.algebra import euclidean_distance, vector_dot
from colour.characterisation import whitepoint_preserving_matrix
from colour.models import RGB_COLOURSPACE_ACES2065_1, XYZ_to_Oklab, XYZ_to_IPT
from colour.utilities import as_float_array, zeros, orient

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "SDS_COLORCHECKER_CLASSIC",
    "SD_ILLUMINANT_ACES",
    "SAMPLES_COUNT_DEFAULT",
    "SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC",
    "swatch_colours_from_image",
    "generate_reference_colour_checker",
    "RGB_COLORCHECKER_CLASSIC_ACES",
    "is_colour_checker_flipped",
    "optimisation_factory_Oklab",
    "optimisation_factory_IPT",
    "error_delta_E",
    "png_compare_colour_checkers",
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

SAMPLES_COUNT_DEFAULT = 24
"""
Default samples count.

SAMPLES_COUNT_DEFAULT : int
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


def swatch_colours_from_image(
    image,
    colour_checker_rectangle,
    samples=SAMPLES_COUNT_DEFAULT,
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
        samples count is :math:`samples^2`.
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

    is_flipped = bool(std_means[0] < std_means[1])

    if is_flipped:
        print("Colour checker was seemingly flipped!")  # noqa: T201
        return True
    else:
        return False


def optimisation_factory_Oklab():
    """
    Produce the objective function and *CIE XYZ* colourspace to optimisation
    colourspace/colour model function based on the *Oklab* colourspace.

    The objective function returns the euclidean distance between the training
    data *RGB* tristimulus values and the training data *CIE XYZ* tristimulus
    values** in the *Oklab* colourspace.

    Returns
    -------
    :class:`tuple`
        :math:`x_0` initial values, objective function, *CIE XYZ* colourspace
        to *Oklab* colourspace function and finaliser function.

    Examples
    --------
    >>> optimisation_factory_Oklab()  # doctest: +SKIP
    (array([ 1.,  0.,  0.,  1.,  0.,  0.]), \
<function optimisation_factory_Oklab.<locals>\
.objective_function at 0x...>, \
<function optimisation_factory_Oklab.<locals>\
.XYZ_to_optimization_colour_model at 0x...>, \
<function optimisation_factory_Oklab.<locals>.\
finaliser_function at 0x...>)
    """

    x_0 = as_float_array([1, 0, 0, 1, 0, 0])

    def objective_function(M, RGB, Jab):
        """*Oklab* colourspace based objective function."""

        M = whitepoint_preserving_matrix(
            np.hstack([np.reshape(M, (3, 2)), zeros((3, 1))])
        )

        XYZ_t = vector_dot(
            RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ, vector_dot(M, RGB)
        )
        Jab_t = XYZ_to_Oklab(XYZ_t)

        return np.sum(euclidean_distance(Jab, Jab_t))

    def XYZ_to_optimization_colour_model(XYZ):
        """*CIE XYZ* colourspace to *Oklab* colourspace function."""

        return XYZ_to_Oklab(XYZ)

    def finaliser_function(M):
        """Finaliser function."""

        return whitepoint_preserving_matrix(
            np.hstack([np.reshape(M, (3, 2)), zeros((3, 1))])
        )

    return (
        x_0,
        objective_function,
        XYZ_to_optimization_colour_model,
        finaliser_function,
    )


def optimisation_factory_IPT():
    """
    Produce the objective function and *CIE XYZ* colourspace to optimisation
    colourspace/colour model function based on the *IPT* colourspace.

    The objective function returns the euclidean distance between the training
    data *RGB* tristimulus values and the training data *CIE XYZ* tristimulus
    values** in the *IPT* colourspace.

    Returns
    -------
    :class:`tuple`
        :math:`x_0` initial values, objective function, *CIE XYZ* colourspace
        to *IPT* colourspace function and finaliser function.

    Examples
    --------
    >>> optimisation_factory_IPT()  # doctest: +SKIP
    (array([ 1.,  0.,  0.,  1.,  0.,  0.]), \
<function optimisation_factory_IPT.<locals>\
.objective_function at 0x...>, \
<function optimisation_factory_IPT.<locals>\
.XYZ_to_optimization_colour_model at 0x...>, \
<function optimisation_factory_IPT.<locals>.\
finaliser_function at 0x...>)
    """

    x_0 = as_float_array([1, 0, 0, 1, 0, 0])

    def objective_function(M, RGB, Jab):
        """*IPT* colourspace based objective function."""

        M = whitepoint_preserving_matrix(
            np.hstack([np.reshape(M, (3, 2)), zeros((3, 1))])
        )

        XYZ_t = vector_dot(
            RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ, vector_dot(M, RGB)
        )
        Jab_t = XYZ_to_IPT(XYZ_t)

        return np.sum(euclidean_distance(Jab, Jab_t))

    def XYZ_to_optimization_colour_model(XYZ):
        """*CIE XYZ* colourspace to *IPT* colourspace function."""

        return XYZ_to_IPT(XYZ)

    def finaliser_function(M):
        """Finaliser function."""

        return whitepoint_preserving_matrix(
            np.hstack([np.reshape(M, (3, 2)), zeros((3, 1))])
        )

    return (
        x_0,
        objective_function,
        XYZ_to_optimization_colour_model,
        finaliser_function,
    )


def error_delta_E(samples_test, samples_reference):
    """
    Compute the difference :math:`\\Delta E_{00}` between two given *RGB*
    colourspace arrays.

    Parameters
    ----------
    samples_test : array_like
        Test samples.
    samples_reference : array_like
        Reference samples.

    Returns
    -------
    NDArray
        :math:`\\Delta E_{00}`.
    """

    XYZ_to_RGB_kargs = {
        "illuminant_XYZ": RGB_COLOURSPACE_ACES2065_1.whitepoint,
        "illuminant_RGB": RGB_COLOURSPACE_ACES2065_1.whitepoint,
        "matrix_XYZ_to_RGB": RGB_COLOURSPACE_ACES2065_1.matrix_XYZ_to_RGB,
    }

    Lab_test = (
        colour.convert(
            samples_test, "RGB", "CIE Lab", XYZ_to_RGB=XYZ_to_RGB_kargs
        )
        * 100
    )
    Lab_reference = (
        colour.convert(
            samples_reference, "RGB", "CIE Lab", XYZ_to_RGB=XYZ_to_RGB_kargs
        )
        * 100
    )

    return colour.delta_E(Lab_test, Lab_reference)


def png_compare_colour_checkers(samples_test, samples_reference, columns=6):
    """
    Return the colour checkers comparison as *PNG* data.

    Parameters
    ----------
    samples_test : array_like
        Test samples.
    samples_reference : array_like
        Reference samples.
    columns : integer, optional
        Number of columns for the colour checkers comparison.

    Returns
    -------
    str
        *PNG* data.
    """

    colour.plotting.plot_multi_colour_swatches(
        list(zip(samples_reference, samples_test)),
        columns=columns,
        compare_swatches="Stacked",
        direction="-y",
    )
    colour.plotting.render(
        **{
            "show": False,
        }
    )
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    data_png = base64.b64encode(buffer.getbuffer()).decode("utf8")
    plt.close()

    return data_png
