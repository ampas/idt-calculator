"""
Transform ID
============

Defines functions to generate and validate URNs for ACES transforms.
"""

import hashlib
import re
import uuid
from enum import Enum

ACES_URN_PREFIX = "URN:{aces_transform_type}"
ACES_MAJOR_VERSION = 2


def generate_truncated_hash(length: int = 6) -> str:
    """
    Generate a truncated SHA-256 hash from a UUID.

    Parameters
    ----------
    length : int, optional
        The length of the truncated hash, by default 4.

    Returns
    -------
    str
        The truncated hexadecimal hash.
    """
    # Generate a UUID
    unique_id = uuid.uuid4()

    # Hash the UUID using SHA-256
    hash_object = hashlib.sha256(str(unique_id).encode())

    # Convert the hash to a hexadecimal string
    hash_hex = hash_object.hexdigest()

    # Truncate the hash to the desired length
    return hash_hex[:length]


class AcesTransformType(Enum):
    """
    Enum for the type of ACES transform.

    Attributes
    ----------
    CSC : str
        Color space conversion.
    OUTPUT : str
        Output transform.
    INV_OUTPUT : str
        Inverse output transform.
    LOOK : str
        Look transform.
    INV_LOOK : str
        Inverse look transform.
    LIB : str
        Library transform.
    UTIL : str
        Utility transform.
    """

    CSC = "CSC"
    OUTPUT = "Output"
    INV_OUTPUT = "InvOutput"
    LOOK = "Look"
    INV_LOOK = "InvLook"
    LIB = "Lib"
    UTIL = "Util"


def generate_idt_urn(
    colorspace_vendor: str,
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
    colorspace_vendor : str
        The vendor for the colorspace, e.g., the device or software provider.
    encoding_colourspace : str
        The name of the encoding color space.
    encoding_transfer_function : str
        The transfer function used to encode the colorspace.
    version_number : int
        The version number of the transform.

    Returns
    -------
    str
        The generated URN in the format described.
    """
    urn_prefix = ACES_URN_PREFIX.format(aces_transform_type=AcesTransformType.CSC.value)
    hash_id = generate_truncated_hash()
    colorspace_name = f"{encoding_colourspace}_{encoding_transfer_function}_{hash_id}"
    return (
        f"{urn_prefix}.{colorspace_vendor}.{colorspace_name}.a{ACES_MAJOR_VERSION}"
        f".v{version_number}"
    )


def is_valid_idt_urn(urn: str) -> bool:
    """
    Check if the given URN string is valid according to the expected format,
    specifically for the CSC (Color Space Conversion) transform type.

    Parameters
    ----------
    urn : str
        The URN string to validate.

    Returns
    -------
    bool
        True if the URN is valid and the transform type is CSC, False otherwise.
    """
    urn_regex = (
        rf"^URN:{AcesTransformType.CSC.value}\.[a-zA-Z0-9_-]+"
        rf"\.[a-zA-Z0-9_]+_[a-zA-Z0-9_]+_[a-f0-9]{{6}}\.a{ACES_MAJOR_VERSION}\.v\d+$"
    )

    return bool(re.match(urn_regex, urn))
