""" Module which contains the ProjectSettings for the IDT maker acts as the data model, serialization, loading and saving of a project.

"""
from idt.core.constants import ProjectSettingsMetaDataConstants
from idt.core.structures import BaseSerializable, idt_metadata_property


class IDTProjectSettings(BaseSerializable):
    """ Class which holds the project settings for the IDT maker. The order the properties are created is the order
        they are serialized and deserialized.
    """

    def __init__(self):
        super().__init__()
        self._schema_version = IDTProjectSettings.schema_version.metadata.default_value
        self._aces_transform_id = IDTProjectSettings.aces_transform_id.metadata.default_value
        self._aces_user_name = IDTProjectSettings.aces_user_name.metadata.default_value
        self._camera_make = IDTProjectSettings.camera_make.metadata.default_value
        self._camera_model = IDTProjectSettings.camera_model.metadata.default_value
        self._iso = IDTProjectSettings.iso.metadata.default_value
        self._temperature = IDTProjectSettings.temperature.metadata.default_value
        self._additional_camera_settings = IDTProjectSettings.additional_camera_settings.metadata.default_value
        self._lighting_setup_description = IDTProjectSettings.lighting_setup_description.metadata.default_value
        self._debayering_platform = IDTProjectSettings.debayering_platform.metadata.default_value
        self._debayering_settings = IDTProjectSettings.debayering_settings.metadata.default_value
        self._encoding_colourspace = IDTProjectSettings.encoding_colourspace.metadata.default_value
        self._rgb_display_colourspace = IDTProjectSettings.rgb_display_colourspace.metadata.default_value
        self._cat = IDTProjectSettings.cat.metadata.default_value
        self._optimisation_space = IDTProjectSettings.optimisation_space.metadata.default_value
        self._illuminant_interpolator = IDTProjectSettings.illuminant_interpolator.metadata.default_value
        self._decoding_method = IDTProjectSettings.decoding_method.metadata.default_value
        self._ev_range = IDTProjectSettings.ev_range.metadata.default_value
        self._grey_card_reference = IDTProjectSettings.grey_card_reference.metadata.default_value
        self._lut_size = IDTProjectSettings.lut_size.metadata.default_value
        self._lut_smoothing = IDTProjectSettings.lut_smoothing.metadata.default_value
        self._data = IDTProjectSettings.data.metadata.default_value

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.SCHEMA_VERSION)
    def schema_version(self):
        return self._schema_version

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.ACES_TRANSFORM_ID)
    def aces_transform_id(self):
        return self._aces_transform_id

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.ACES_USER_NAME)
    def aces_user_name(self):
        return self._aces_user_name

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.CAMERA_MAKE)
    def camera_make(self):
        return self._camera_make

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.CAMERA_MODEL)
    def camera_model(self):
        return self._camera_model

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.ISO)
    def iso(self):
        return self._iso

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.TEMPERATURE)
    def temperature(self):
        return self._temperature

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.ADDITIONAL_CAMERA_SETTINGS)
    def additional_camera_settings(self):
        return self._additional_camera_settings

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.LIGHTING_SETUP_DESCRIPTION)
    def lighting_setup_description(self):
        return self._lighting_setup_description

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.DEBAYERING_PLATFORM)
    def debayering_platform(self):
        return self._debayering_platform

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.DEBAYERING_SETTINGS)
    def debayering_settings(self):
        return self._debayering_settings

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.ENCODING_COLOUR_SPACE)
    def encoding_colourspace(self):
        return self._encoding_colourspace

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.RGB_DISPLAY_COLOURSPACE)
    def rgb_display_colourspace(self):
        return self._rgb_display_colourspace

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.CAT)
    def cat(self):
        return self._cat

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.OPTIMISATION_SPACE)
    def optimisation_space(self):
        return self._optimisation_space

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.ILLUMINANT_INTERPOLATOR)
    def illuminant_interpolator(self):
        return self._illuminant_interpolator

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.DECODING_METHOD)
    def decoding_method(self):
        return self._decoding_method

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.EV_RANGE)
    def ev_range(self):
        return self._ev_range

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.GREY_CARD_REFERENCE)
    def grey_card_reference(self):
        return self._grey_card_reference

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.LUT_SIZE)
    def lut_size(self):
        return self._lut_size

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.LUT_SMOOTHING)
    def lut_smoothing(self):
        return self._lut_smoothing

    @idt_metadata_property(metadata=ProjectSettingsMetaDataConstants.DATA)
    def data(self):
        return self._data


