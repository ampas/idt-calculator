""" Module contains a selection of helper classes used throughout the IDT

"""
import json


class IDTMetaData:
    """ Simple class to store metadata information for a property within an IDT project, this data is not used in any
        calculations or computation and is primarily designed to store data which would be associated with how the
        property should be displayed in a UI.

    """

    def __init__(self, default_value=None, description="", display_name="", ui_category="", ui_type="", options=None):
        self.default_value = default_value
        self.description = description
        self.display_name = display_name
        self.ui_category = ui_category
        self.ui_type = ui_type
        self.options = options or []


class IDTMetadataProperty:
    """A property descriptor that stores IDTMetaData and supports both getter and setter functionality."""

    def __init__(self, getter, setter=None, metadata=None):
        self.getter = getter
        self.setter = setter
        self.metadata = metadata

    def __get__(self, instance, owner):
        if instance is None:
            # Return the descriptor itself when accessed from the class
            return self
        return self.getter(instance)

    def __set__(self, instance, value):
        if self.setter:
            self.setter(instance, value)
        else:
            raise AttributeError("No setter defined for this property")


def idt_metadata_property(metadata=None):
    """A decorator for creating a property with IDTMetaData and supporting both getter and setter functionality."""

    def wrapper(getter):
        # Define a setter function to handle setting the value
        def setter(instance, value):
            setattr(instance, f"_{getter.__name__}", value)

        # Create a IDTMetadataProperty descriptor with both getter and setter
        return IDTMetadataProperty(getter, setter, metadata)

    return wrapper


class BaseSerializable:
    @property
    def properties(self):
        for name, descriptor in self.__class__.__dict__.items():
            if isinstance(descriptor, IDTMetadataProperty):
                yield name, descriptor

    def to_json(self):
        data = {}
        for name, prop in self.properties:
            data[name] = prop.getter(self)

        return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, data):
        item = cls()
        if isinstance(data, str):
            data = json.loads(data)

        for name, prop in item.properties:
            prop.setter(item, data[name])
        return item

    def to_file(self, filepath):
        with open(filepath, 'w') as f:
            f.write(self.to_json())

    def from_file(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)

        return self.from_json(data)