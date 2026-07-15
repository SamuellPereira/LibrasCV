import cv2
import mediapipe as mp
import json
import os


video_path = "dataset/obrigado/obrigado.mp4"

saida = "dataset/obrigado/obrigado.json"


mp_holistic = mp.solutions.holistic


holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1
)


cap = cv2.VideoCapture(video_path)


dados = []


while True:

    sucesso, frame = cap.read()

    if not sucesso:
        break


    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    resultado = holistic.process(
        frame_rgb
    )


    frame_dados = {}


    if resultado.left_hand_landmarks:

        frame_dados["mao_esquerda"] = []

        for ponto in resultado.left_hand_landmarks.landmark:

            frame_dados["mao_esquerda"].append(
                [
                    ponto.x,
                    ponto.y,
                    ponto.z
                ]
            )


    if resultado.right_hand_landmarks:

        frame_dados["mao_direita"] = []

        for ponto in resultado.right_hand_landmarks.landmark:

            frame_dados["mao_direita"].append(
                [
                    ponto.x,
                    ponto.y,
                    ponto.z
                ]
            )


    dados.append(frame_dados)



cap.release()

os.makedirs(
    os.path.dirname(saida),
    exist_ok=True
)

with open(
    saida,
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        dados,
        arquivo,
        indent=4
    )


print("Dataset criado!")
print("Frames:", len(dados))