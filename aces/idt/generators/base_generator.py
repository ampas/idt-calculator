"""
Module holds a common base generator class for which all other IDT generators inherit from
"""


class BaseGenerator:
    def __init__(self, application):
        self._application = application
        self._projectSettings = application.project_settings
        self._samples_analysis = None

    def generate_LUT(self):
        raise NotImplementedError("generate_LUT method not implemented")

    def filter_LUT(self):
        raise NotImplementedError("filter_LUT method not implemented")

    def decode(self):
        raise NotImplementedError("decode method not implemented")

    def optimise(self):
        raise NotImplementedError("optimise method not implemented")
