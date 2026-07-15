import mediapipe as mp


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def detectar_mao(frame):

    frame_rgb = mp.solutions.hands.Hands

    imagem = frame.copy()

    resultado = hands.process(
        imagem
    )

    return resultado



def desenhar_mao(frame, resultado):

    if resultado.multi_hand_landmarks:

        for mao in resultado.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                mao,
                mp_hands.HAND_CONNECTIONS
            )

    return frame



def pegar_pontos(resultado):

    pontos = []


    if resultado.multi_hand_landmarks:

        for mao in resultado.multi_hand_landmarks:

            for ponto in mao.landmark:

                pontos.append(
                    ponto.x
                )

                pontos.append(
                    ponto.y
                )

                pontos.append(
                    ponto.z
                )


    return pontos