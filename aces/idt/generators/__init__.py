from .base_generator import IDTBaseGenerator
from .prelinearized_idt import IDTGeneratorPreLinearizedCamera
from .prosumer_camera import IDTGeneratorProsumerCamera

GENERATORS = {
    IDTGeneratorProsumerCamera.GENERATOR_NAME: IDTGeneratorProsumerCamera,
    IDTGeneratorPreLinearizedCamera.GENERATOR_NAME: IDTGeneratorPreLinearizedCamera,
}

__all__ = ["IDTBaseGenerator"]
__all__ += ["IDTGeneratorProsumerCamera"]
__all__ += ["IDTGeneratorPreLinearizedCamera"]
__all__ += ["GENERATORS"]
