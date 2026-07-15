import os
import sys

import cv2


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJETO_DIR = os.path.dirname(
    BASE_DIR
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


camera = cv2.VideoCapture(0)


while True:

    sucesso, frame = camera.read()

    if not sucesso:
        break


    resultado = detectar_mao(
        frame
    )


    pontos = pegar_pontos(
        resultado
    )


    frame = desenhar_mao(
        frame,
        resultado
    )


    cv2.putText(
        frame,
        f"Pontos: {len(pontos)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Teste LibrAI - 225 pontos",
        frame
    )


    if cv2.waitKey(1) & 0xFF == 27:
        break


camera.release()
cv2.destroyAllWindows()