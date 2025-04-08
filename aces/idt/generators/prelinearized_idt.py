"""
IDT Generator for Pre-Linearized Camera
=======================================

Define the *IDT* generator class for a *Pre-Linearized Camera*.
"""

from __future__ import annotations

import logging
import typing

from colour import LUT3x1D

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
    "IDTGeneratorPreLinearizedCamera",
]

LOGGER = logging.getLogger(__name__)


class IDTGeneratorPreLinearizedCamera(IDTGeneratorLogCamera):
    """
    Define an *IDT* generator for a *Pre-Linearized Camera*.

    Parameters
    ----------
    project_settings
        *IDT* generator settings.

    Attributes
    ----------
    -   :attr:`~aces.idt.IDTGeneratorPreLinearizedCamera.GENERATOR_NAME`

    Methods
    -------
    -   :meth:`~aces.idt.IDTGeneratorPreLinearizedCamera.generate_LUT`
    -   :meth:`~aces.idt.IDTGeneratorPreLinearizedCamera.filter_LUT`
    """

    GENERATOR_NAME = "IDTGeneratorPreLinearizedCamera"
    """*IDT* generator name."""

    def __init__(self, project_settings: IDTProjectSettings) -> None:
        super().__init__(project_settings)

    def generate_LUT(self) -> LUT3x1D:
        """Do not generate a LUT; The pre linearized camera generator does not
        generate a LUT, it assumes pre linearized exr files are provided so we only
        want a linear gain to be applied during the decode phase

        We provide an identity LUT to ensure the function fulfills its requirements
        without affecting any calculations

        Returns
        -------
        :class:`LUT3x1D`
            Unfiltered linearisation *LUT* for the camera samples.
        """
        size = self.project_settings.lut_size
        LOGGER.info('Generating unfiltered "LUT3x1D" with "%s" size...', size)
        self._LUT_unfiltered = LUT3x1D(size=size, name="LUT - Unfiltered")
        return self._LUT_unfiltered

    def filter_LUT(self) -> LUT3x1D:
        """Do not filter a LUT; The pre linearized camera generator does not
        generate a LUT, it assumes pre linearized exr files are provided so we only
        want a linear gain to be applied during the decode phase

        As the unfiltered LUT is an identity LUT we can simply copy it

        Returns
        -------
        :class:`LUT3x1D`
            Filtered linearisation *LUT* for the camera samples.
        """
        LOGGER.info('No Filtering Simply Copying "LUT3x1D"')

        self._LUT_filtered = self._LUT_unfiltered.copy()
        self._LUT_filtered.name = "LUT - Filtered"
        return self._LUT_filtered
