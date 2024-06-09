from .base_generator import IDTBaseGenerator
from .prosumer_camera import IDTGeneratorProsumerCamera

GENERATORS = {IDTGeneratorProsumerCamera.generator_name: IDTGeneratorProsumerCamera}

__all__ = ["IDTBaseGenerator"]
__all__ += ["IDTGeneratorProsumerCamera"]
__all__ += ["GENERATORS"]
