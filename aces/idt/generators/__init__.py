from .base_generator import IDTBaseGenerator
from .prosumer_camera import IDTGeneratorProsumerCamera

GENERATORS = {IDTGeneratorProsumerCamera.GENERATOR_NAME: IDTGeneratorProsumerCamera}

__all__ = ["IDTBaseGenerator"]
__all__ += ["IDTGeneratorProsumerCamera"]
__all__ += ["GENERATORS"]
