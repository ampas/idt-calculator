from .base_generator import IDTBaseGenerator
from .prelinearized_idt import IDTGeneratorPreLinearizedCamera
from .prosumer_camera import IDTGeneratorProsumerCamera
from .tonemapped_idt import IDTGeneratorToneMappedCamera

GENERATORS = {
    IDTGeneratorProsumerCamera.GENERATOR_NAME: IDTGeneratorProsumerCamera,
    IDTGeneratorPreLinearizedCamera.GENERATOR_NAME: IDTGeneratorPreLinearizedCamera,
    IDTGeneratorToneMappedCamera.GENERATOR_NAME: IDTGeneratorToneMappedCamera,
}

__all__ = ["IDTBaseGenerator"]
__all__ += ["IDTGeneratorProsumerCamera"]
__all__ += ["IDTGeneratorPreLinearizedCamera"]
__all__ += ["IDTGeneratorToneMappedCamera"]
__all__ += ["GENERATORS"]
