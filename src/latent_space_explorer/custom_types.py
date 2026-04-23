from enum import Enum


class DimReductionType(Enum):
    __order__ = " tSNE MDS "
    tSNE = 0
    MDS = 1


class CodeSpace(Enum):
    LATENT = 0
    PCA = 1


class VisualizeShapeMode(Enum):
    SINGLE = 0
    DOUBLE = 1


class ShapeType(Enum):
    SELECT_1 = 0
    SELECT_2 = 1
    INTERPOLATE = 2
