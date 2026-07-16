import os
import sys

import cv2


TESTES_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

HOMOLOGACAO_DIR = os.path.dirname(
    TESTES_DIR
)

PROJETO_DIR = os.path.dirname(
    HOMOLOGACAO_DIR
)

sys.path.insert(
    0,
    PROJETO_DIR
)


from src.hand_detector import (
    detectar_mao,
    desenhar_mao,
    pegar_pontos
)