"""
Structures
==========

Define various helper classes, especially to serialize properties to and from
*JSON*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from colour.hints import Any, Callable, Tuple

__author__ = "Alex Forsythe, Joshua Pines, Thomas Mansencal, Nick Shaw, Adam Davis"
__copyright__ = "Copyright 2022 Academy of Motion Picture Arts and Sciences"
__license__ = "Academy of Motion Picture Arts and Sciences License Terms"
__maintainer__ = "Academy of Motion Picture Arts and Sciences"
__email__ = "acessupport@oscars.org"
__status__ = "Production"

__all__ = [
    "PathEncoder",
    "SerializableConstants",
    "Metadata",
    "MetadataProperty",
    "metadata_property",
    "MixinSerializableProperties",
]


class PathEncoder(json.JSONEncoder):
    """
    Define a custom :class:`json.JSONEncoder` sub-class that can encode
    :class:`Path` class instance objects.
    """

    def default(self, obj: Any) -> Any:
        """
        Convert the object to a *JSON* serializable object.

        Parameters
        ----------
        obj
            Object to serialize.

        Returns
        -------
        :class:`object`
            *JSON* serializable object.
        """

        if isinstance(obj, Path):
            # Convert the PosixPath to a string
            return str(obj)

        if isinstance(obj, np.ndarray):
            return list(obj)

        return super().default(obj)


class SerializableConstants:
    """Constants for the serializable."""

    HEADER = "header"
    DATA = "data"


@dataclass
class Metadata:
    """
    Store the metadata information for a serializable property.

    This metadata is primarily designed to store data associated with how the
    property might be displayed in a UI.
    """

    default_value: Any = field(default=None)
    description: str = field(default="")
    display_name: str = field(default="")
    name: str = field(default="")
    serialize_group: str = field(default=SerializableConstants.HEADER)
    ui_category: str = field(default="")
    ui_type: str = field(default="")
    options: Any = field(default=None)


class MetadataProperty:
    """
    Define a property storing a :class:`Metadata` class instance with support
    for getter and setter functionality.
    """

    def __init__(
        self,
        getter: Callable,
        setter: Callable | None = None,
        metadata: Metadata | None = None,
    ):
        self.getter = getter
        self.setter = setter
        self.metadata = metadata

    def __get__(self, instance: Any, owner: Any) -> Any:
        """Get the value of the property."""

        if instance is None:
            # Return the object itself when accessed from the class.
            return self

        return self.getter(instance)

    def __set__(self, instance: Any, value: Any) -> Any:
        """Set the value of the property."""

        if self.setter:
            self.setter(instance, value)
        else:
            raise AttributeError("No setter defined for this property")


def metadata_property(metadata: Metadata | None = None) -> Any:
    """
    Decorate a class attribute to create a property using given :class`Metadata`
    class instance, supporting both getter and setter functionality.
    """

    def wrapper(getter):
        def setter(instance, value):
            setattr(instance, f"_{getter.__name__}", value)

        return MetadataProperty(getter, setter, metadata)

    return wrapper


class MixinSerializableProperties:
    """
    Define a mixin class for serializable objects containing
    :class:`IDTMetadataProperty` class instances that can be converted to and
    from *JSON*.
    """

    @property
    def properties(self) -> Tuple[str, MetadataProperty]:
        """
        Generator for the properties of the object getting the name and
        property as a tuple.

        Yields
        ------
        :class:`tuple`
        """

        for name, object_ in self.__class__.__dict__.items():
            if isinstance(object_, MetadataProperty):
                yield name, object_

    def to_json(self) -> str:
        """
        Convert the object to a *JSON* string representation.

        Returns
        -------
        :class:`str`:
            *JSON* string representation.
        """

        output = {SerializableConstants.HEADER: {}, SerializableConstants.DATA: {}}
        for name, prop in self.properties:
            if prop.metadata.serialize_group:
                output[prop.metadata.serialize_group][name] = prop.getter(self)
            else:
                output[SerializableConstants.DATA] = prop.getter(self)
        return json.dumps(output, indent=4, cls=PathEncoder)

    @classmethod
    def from_json(cls, data: Any) -> MixinSerializableProperties:
        """
        Create a new instance of the class from the given object or *JSON*
        string.

        Parameters
        ----------
        data:
            Object or string we want to load the *JSON* from.

        Returns
        -------
        :class:`MixinSerializableProperties`
            Loaded object.
        """

        item = cls()
        if isinstance(data, str):
            data = json.loads(data)

        for name, prop in item.properties:
            if prop.metadata.serialize_group == SerializableConstants.HEADER:
                if name in data[SerializableConstants.HEADER]:
                    prop.setter(item, data[SerializableConstants.HEADER][name])
            else:
                prop.setter(item, data[SerializableConstants.DATA])

        return item

    def to_file(self, filepath: str):
        """
        Serialize the object to a *JSON* file.

        Parameters
        ----------
        filepath
            Path to the *JSON* file.
        """

        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, filepath: str) -> MixinSerializableProperties:
        """
        Load the object from a *JSON* file.

        Parameters
        ----------
        filepath
            Path to the *JSON* file.

        Returns
        -------
        :class:`MixinSerializableProperties`
            Loaded object.
        """

        with open(filepath) as f:
            data = json.load(f)

        return cls.from_json(data)
