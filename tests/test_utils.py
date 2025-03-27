"""Module for generic unit testing helpers"""

from __future__ import annotations

import hashlib
import json
import os
import unittest
from typing import TypeVar

import requests

T = TypeVar("T", bound="TestIDTBase")

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal, Nick Shaw, Adam Davis"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "TestIDTBase",
]


class TestIDTBase(unittest.TestCase):
    """Define a base class for other unit testing classes."""

    @classmethod
    def setUpClass(cls) -> None:
        """Download any missing test resources from the manifest."""
        file_path = cls.get_download_manifest()
        with open(file_path) as f:
            manifest = json.load(f)

        for file_path, data in manifest.items():
            url = data.get("URL")
            expected_sha256 = data.get("SHA256")
            filename = os.path.basename(file_path)
            local_folder = cls.get_test_resources_folder()
            expected_file = os.path.join(local_folder, filename)
            download = False
            if not os.path.exists(expected_file):
                download = True
            else:
                sha256 = cls.sha256_checksum(expected_file)
                if sha256 != expected_sha256:
                    download = True
            if download:
                cls.download_file_from_dropbox(url, local_folder, file_path)

    @classmethod
    def get_unit_test_folder(cls: type[T]) -> T:
        """
        Get the folder the unit test file is stored within.

        Returns
        -------
        :class:`str`
            The folder path holding the file which is running.
        """

        script_file_path = os.path.abspath(__file__)

        return os.path.dirname(script_file_path)

    @classmethod
    def get_test_output_folder(cls: type[T]) -> T:
        """
        Get the unit test output folder.

        Returns
        -------
        :class:`str`
            The unit test output folder.
        """

        return os.path.join(cls.get_unit_test_folder(), "output")

    @classmethod
    def get_test_resources_folder(cls: type[T]) -> T:
        """Get the unit test resource folder.

        Returns
        -------
        :class:`str`
            The unit test resource folder
        """

        return os.path.join(cls.get_unit_test_folder(), "resources")

    @classmethod
    def get_download_manifest(cls) -> str:
        """Get the download manifest for this test."""
        return os.path.join(cls.get_test_resources_folder(), "download_manifest.json")

    @classmethod
    def download_file_from_dropbox(
        cls, sharing_link: str, local_folder: str, local_filename: str | None = None
    ) -> None:
        """
        Download a file from Dropbox using the sharing link.

        Parameters
        ----------
        sharing_link
            the Dropbox sharing link to the file
        local_folder
            the local folder to save the file to
        local_filename
            the local filename to save the file as (optional)
        """
        # Ensure the local folder exists
        os.makedirs(local_folder, exist_ok=True)

        # Convert the sharing link to a direct download link
        direct_link = sharing_link.replace("?dl=0", "?dl=1")

        # Ensure the agent is a linux wget to ensure we get the file not the html page
        headers = {"user-agent": "Wget/1.16 (linux-gnu)"}
        response = requests.get(
            direct_link, stream=True, headers=headers, timeout=(5, 15)
        )
        if response.status_code == 200:
            # If no local filename is provided, extract one from the URL
            if local_filename is None:
                local_filename = os.path.basename(sharing_link.split("?")[0])

            # Write the content to a file
            file_path = os.path.join(local_folder, local_filename)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        else:
            msg = f"Failed to download file. HTTP status code: {response.status_code}"
            raise OSError(msg)

    @classmethod
    def sha256_checksum(cls, file_path: str, chunk_size: int = 8192) -> str:
        """
        Calculate the SHA256 checksum of a file.

        Parameters
        ----------
        file_path
            the file path
        chunk_size
            the chunk size to read the file in bytes

        Returns
        -------
            a sha256 checksum of the file

        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
