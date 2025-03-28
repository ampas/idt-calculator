from .base_generator import IDTBaseGenerator
from .log_camera import IDTGeneratorLogCamera
from .prelinearized_idt import IDTGeneratorPreLinearizedCamera
from .tonemapped_idt import IDTGeneratorToneMappedCamera

GENERATORS = {
    IDTGeneratorLogCamera.GENERATOR_NAME: IDTGeneratorLogCamera,
    IDTGeneratorPreLinearizedCamera.GENERATOR_NAME: IDTGeneratorPreLinearizedCamera,
    IDTGeneratorToneMappedCamera.GENERATOR_NAME: IDTGeneratorToneMappedCamera,
}

__all__ = ["IDTBaseGenerator"]
__all__ += ["IDTGeneratorLogCamera"]
__all__ += ["IDTGeneratorPreLinearizedCamera"]
__all__ += ["IDTGeneratorToneMappedCamera"]
__all__ += ["GENERATORS"]
