import cv2
import mediapipe as mp

camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

while True:
    sucesso, frame = camera.read()

    if not sucesso:
        print("Erro ao acessar a câmera.")
        break

    # Espelha a câmera
    frame = cv2.flip(frame, 1)

    # Converter BGR para RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar mãos
    resultado = hands.process(rgb)

    # Se encontrou alguma mão
    if resultado.multi_hand_landmarks:

        for mao in resultado.multi_hand_landmarks:

            # Mostra os pontos no terminal
            for id, ponto in enumerate(mao.landmark):
                print(
                    "Ponto:", id,
                    "X:", ponto.x,
                    "Y:", ponto.y
                )

            # Desenha a mão
            mp_draw.draw_landmarks(
                frame,
                mao,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Inicio do projeto teste", frame)

    if cv2.waitKey(1) == 27:
        break


camera.release()
cv2.destroyAllWindows()