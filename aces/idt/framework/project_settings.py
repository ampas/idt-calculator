"""Module which contains the ProjectSettings for the IDT maker acts as the data model,
serialization, loading and saving of a project.

"""
from aces.idt.core.constants import ProjectSettingsMetaDataConstants as PsMdC
from aces.idt.core.structures import BaseSerializable, idt_metadata_property


class IDTProjectSettings(BaseSerializable):
    """Class which holds the project settings for the IDT maker. The order the
    properties are created is the order they are serialized and deserialized.
    """

    def __init__(self):
        super().__init__()
        self._schema_version = IDTProjectSettings.schema_version.metadata.default_value
        self._aces_transform_id = (
            IDTProjectSettings.aces_transform_id.metadata.default_value
        )
        self._aces_user_name = IDTProjectSettings.aces_user_name.metadata.default_value
        self._camera_make = IDTProjectSettings.camera_make.metadata.default_value
        self._camera_model = IDTProjectSettings.camera_model.metadata.default_value
        self._iso = IDTProjectSettings.iso.metadata.default_value
        self._temperature = IDTProjectSettings.temperature.metadata.default_value
        self._additional_camera_settings = (
            IDTProjectSettings.additional_camera_settings.metadata.default_value
        )
        self._lighting_setup_description = (
            IDTProjectSettings.lighting_setup_description.metadata.default_value
        )
        self._debayering_platform = (
            IDTProjectSettings.debayering_platform.metadata.default_value
        )
        self._debayering_settings = (
            IDTProjectSettings.debayering_settings.metadata.default_value
        )
        self._encoding_colourspace = (
            IDTProjectSettings.encoding_colourspace.metadata.default_value
        )
        self._rgb_display_colourspace = (
            IDTProjectSettings.rgb_display_colourspace.metadata.default_value
        )
        self._cat = IDTProjectSettings.cat.metadata.default_value
        self._optimisation_space = (
            IDTProjectSettings.optimisation_space.metadata.default_value
        )
        self._illuminant_interpolator = (
            IDTProjectSettings.illuminant_interpolator.metadata.default_value
        )
        self._decoding_method = (
            IDTProjectSettings.decoding_method.metadata.default_value
        )
        self._ev_range = IDTProjectSettings.ev_range.metadata.default_value
        self._grey_card_reference = (
            IDTProjectSettings.grey_card_reference.metadata.default_value
        )
        self._lut_size = IDTProjectSettings.lut_size.metadata.default_value
        self._lut_smoothing = IDTProjectSettings.lut_smoothing.metadata.default_value
        self._data = IDTProjectSettings.data.metadata.default_value

    @idt_metadata_property(metadata=PsMdC.SCHEMA_VERSION)
    def schema_version(self):
        """Return the schema version

        Returns
        -------
        str
            The project settings schema version

        """
        return self._schema_version

    @idt_metadata_property(metadata=PsMdC.ACES_TRANSFORM_ID)
    def aces_transform_id(self):
        """Return the aces transform id

        Returns
        -------
        str
            The ACES transform ID

        """
        return self._aces_transform_id

    @idt_metadata_property(metadata=PsMdC.ACES_USER_NAME)
    def aces_user_name(self):
        """Return the aces user name

        Returns
        -------
        str
            The ACES user name

        """
        return self._aces_user_name

    @idt_metadata_property(metadata=PsMdC.CAMERA_MAKE)
    def camera_make(self):
        """Return the camera make

        Returns
        -------
        str
            The camera make

        """
        return self._camera_make

    @idt_metadata_property(metadata=PsMdC.CAMERA_MODEL)
    def camera_model(self):
        """Return the camera model

        Returns
        -------
        str
            The camera model

        """
        return self._camera_model

    @idt_metadata_property(metadata=PsMdC.ISO)
    def iso(self):
        """Return the ISO

        Returns
        -------
        int
            The ISO

        """
        return self._iso

    @idt_metadata_property(metadata=PsMdC.TEMPERATURE)
    def temperature(self):
        """Return the colour temperature in kelvin

        Returns
        -------
        int
            The temperature

        """
        return self._temperature

    @idt_metadata_property(metadata=PsMdC.ADDITIONAL_CAMERA_SETTINGS)
    def additional_camera_settings(self):
        """Return the additional camera settings

        Returns
        -------
        str
            The additional camera settings

        """
        return self._additional_camera_settings

    @idt_metadata_property(metadata=PsMdC.LIGHTING_SETUP_DESCRIPTION)
    def lighting_setup_description(self):
        """Return the lighting setup description

        Returns
        -------
        str
            The lighting setup description

        """
        return self._lighting_setup_description

    @idt_metadata_property(metadata=PsMdC.DEBAYERING_PLATFORM)
    def debayering_platform(self):
        """Return the debayering platform

        Returns
        -------
        str
            The debayering platform

        """
        return self._debayering_platform

    @idt_metadata_property(metadata=PsMdC.DEBAYERING_SETTINGS)
    def debayering_settings(self):
        """Return the debayering settings settings

        Returns
        -------
        str
            The debayering settings

        """
        return self._debayering_settings

    @idt_metadata_property(metadata=PsMdC.ENCODING_COLOUR_SPACE)
    def encoding_colourspace(self):
        """Return the encoding colour space

        Returns
        -------
        str
            The encoding colour space

        """
        return self._encoding_colourspace

    @idt_metadata_property(metadata=PsMdC.RGB_DISPLAY_COLOURSPACE)
    def rgb_display_colourspace(self):
        """Return the rgb display colour space

        Returns
        -------
        str
            The RGB display colourspace

        """
        return self._rgb_display_colourspace

    @idt_metadata_property(metadata=PsMdC.CAT)
    def cat(self):
        """Return the name of the CAT

        Returns
        -------
        str
            The CAT

        """
        return self._cat

    @idt_metadata_property(metadata=PsMdC.OPTIMISATION_SPACE)
    def optimisation_space(self):
        """Return the optimisation space

        Returns
        -------
        str
            The optimisation space

        """
        return self._optimisation_space

    @idt_metadata_property(metadata=PsMdC.ILLUMINANT_INTERPOLATOR)
    def illuminant_interpolator(self):
        """Return the illuminant interpolator

        Returns
        -------
        str
            The illuminant interpolator

        """
        return self._illuminant_interpolator

    @idt_metadata_property(metadata=PsMdC.DECODING_METHOD)
    def decoding_method(self):
        """Return the decoding method

        Returns
        -------
        str
            The decoding method

        """
        return self._decoding_method

    @idt_metadata_property(metadata=PsMdC.EV_RANGE)
    def ev_range(self):
        """Return the ev range

        Returns
        -------
        list
            The EV range

        """
        return self._ev_range

    @idt_metadata_property(metadata=PsMdC.GREY_CARD_REFERENCE)
    def grey_card_reference(self):
        """Return the grey card reference values

        Returns
        -------
        list
            The grey card reference

        """
        return self._grey_card_reference

    @idt_metadata_property(metadata=PsMdC.LUT_SIZE)
    def lut_size(self):
        """Return the lut size

        Returns
        -------
        int
            The size of the lut

        """
        return self._lut_size

    @idt_metadata_property(metadata=PsMdC.LUT_SMOOTHING)
    def lut_smoothing(self):
        """Return the lut smoothing

        Returns
        -------
        int
            The smoothing amount for the lut

        """
        return self._lut_smoothing

    @idt_metadata_property(metadata=PsMdC.DATA)
    def data(self):
        """Return the data structure to hold the image sequences

        Returns
        -------
        dict
            The data structure to hold the image sequences

        """
        return self._data
