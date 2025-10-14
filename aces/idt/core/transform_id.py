"""
Transform ID
============

Defines functions to generate and validate URNs for ACES transforms.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from enum import Enum

ACES_URN_PREFIX: str = "URN:{aces_transform_type}"

ACES_MAJOR_VERSION: int = 2


def generate_truncated_hash(length: int = 6) -> str:
    """
    Generate a truncated SHA-256 hash from a UUID.

    Parameters
    ----------
    length
        The length of the truncated hash, default to 6.

    Returns
    -------
    :class:`str`
        Truncated hexadecimal hash.
    """

    unique_id = uuid.uuid4()

    hash_object = hashlib.sha256(str(unique_id).encode())

    hash_hex = hash_object.hexdigest()

    return hash_hex[:length]


class AcesTransformType(Enum):
    """
    Enum for the type of ACES transform.

    Attributes
    ----------
    CSC
        Color space conversion.
    OUTPUT
        Output transform.
    INV_OUTPUT
        Inverse output transform.
    LOOK
        Look transform.
    INV_LOOK
        Inverse look transform.
    LIB
        Library transform.
    UTIL
        Utility transform.
    """

    CSC: str = "CSC"
    OUTPUT: str = "Output"
    INV_OUTPUT: str = "InvOutput"
    LOOK: str = "Look"
    INV_LOOK: str = "InvLook"
    LIB: str = "Lib"
    UTIL: str = "Util"


def generate_idt_urn(
    colourspace_vendor: str,
    encoding_colourspace: str,
    encoding_transfer_function: str,
    version_number: int,
) -> str:
    """
    Generate a URN for an ACES IDT (Input Device Transform) using the provided
    parameters.

    The URN is formatted as:
    `URN:<transform_type>.<colorspace_vendor>.<encoding_colourspace>_
        <encoding_transfer_function>_<hash_id>.a<ACES_MAJOR_VERSION>.v<version_number>`

    Parameters
    ----------
    colourspace_vendor
        The vendor for the colorspace, e.g., the device or software provider.
    encoding_colourspace
        The name of the encoding color space.
    encoding_transfer_function
        The transfer function used to encode the colorspace.
    version_number
        The version number of the transform.

    Returns
    -------
    :class:`str`
        The generated URN in the format described.

    Notes
    -----
    -   Dots in the colorspace_vendor, encoding_colourspace, and
        encoding_transfer_function strings are replaced with underscores to
        ensure valid URN format, as dots are used as delimiters in the URN.
    """

    # Sanitize input strings by replacing dots with underscores
    colorspace_vendor_clean = colourspace_vendor.replace(".", "_")
    encoding_colourspace_clean = encoding_colourspace.replace(".", "_")
    encoding_transfer_function_clean = encoding_transfer_function.replace(".", "_")

    urn_prefix = ACES_URN_PREFIX.format(aces_transform_type=AcesTransformType.CSC.value)
    hash_id = generate_truncated_hash()
    colorspace_name = (
        f"{encoding_colourspace_clean}_{encoding_transfer_function_clean}_{hash_id}"
    )
    return (
        f"{urn_prefix}.{colorspace_vendor_clean}.{colorspace_name}.a{ACES_MAJOR_VERSION}"
        f".v{version_number}"
    )


def is_valid_csc_urn(urn: str) -> bool:
    """
    Check if the given URN string is valid according to the expected format,
    specifically for the CSC (Color Space Conversion) transform type.

    Parameters
    ----------
    urn
        The URN string to validate.

    Returns
    -------
    :class:`bool`
        Whether the URN is valid and its transform type is CSC.
    """

    urn_regex = (
        rf"^URN:{AcesTransformType.CSC.value}\.[a-zA-Z0-9_-]+"
        rf"\.[a-zA-Z0-9_]+_[a-zA-Z0-9_]+_[a-f0-9]{{6}}\.a{ACES_MAJOR_VERSION}\.v\d+$"
    )

    return bool(re.match(urn_regex, urn))
