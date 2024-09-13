"""
IDT Prosumer Camera Generator
=============================

Define the *IDT* generator class for a *Prosumer Camera*.
"""

import logging

import matplotlib as mpl
from colour import LUT3x1D
from colour.utilities import as_float_array

from aces.idt import DirectoryStructure
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
    "IDTGeneratorPreLinearizedCamera",
]

LOGGER = logging.getLogger(__name__)


class IDTGeneratorPreLinearizedCamera(IDTGeneratorProsumerCamera):
    """
    Define an *IDT* generator for a *PreLinearized Camera*.

    Parameters
    ----------
    project_settings : IDTProjectSettings, optional
        *IDT* generator settings.

    Attributes
    ----------
    -   :attr:`~aces.idt.IDTBaseGenerator.GENERATOR_NAME`

    Methods
    -------
    -   :meth:`~aces.idt.IDTBaseGenerator.generate_LUT`
    -   :meth:`~aces.idt.IDTBaseGenerator.filter_LUT`
    """

    GENERATOR_NAME = "IDTGeneratorPreLinearizedCamera"
    """*IDT* generator name."""

    def __init__(self, project_settings):
        super().__init__(project_settings)

    def generate_LUT(self) -> LUT3x1D:
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
        # size = self.project_settings.lut_size
        # LOGGER.info('Generating unfiltered "LUT3x1D" with "%s" size...', size)
        # self._LUT_unfiltered = LUT3x1D(size=size, name="LUT - Unfiltered")
        # return self._LUT_unfiltered
        return None

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
        # LOGGER.info('No Filtering Simply Copying "LUT3x1D"')
        #
        # self._LUT_filtered = self._LUT_unfiltered.copy()
        # self._LUT_filtered.name = "LUT - Filtered"
        # return self._LUT_filtered
        return None

    def decode(self) -> None:
        """
        Decode the camera samples.

        The camera samples are decoded using the camera's *IDT* and the
        *IDT* generator settings.

        Returns
        -------
        None
        """
        LOGGER.info("Decoding camera samples...")
        self._samples_decoded = {}
        for EV in sorted(self._samples_analysis[DirectoryStructure.COLOUR_CHECKER]):
            self._samples_decoded[EV] = as_float_array(
                self._samples_analysis[DirectoryStructure.COLOUR_CHECKER][EV][
                    "samples_median"
                ]
            )
