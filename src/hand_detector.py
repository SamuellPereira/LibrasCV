import cv2
import mediapipe as mp


# ============================================================
# MEDIAPIPE HOLISTIC
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# DETECÇÃO
# ============================================================

def detectar_mao(frame):

    if frame is None:
        return None

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Melhora um pouco o desempenho durante o processamento
    frame_rgb.flags.writeable = False

    resultado = holistic.process(
        frame_rgb
    )

    frame_rgb.flags.writeable = True

    return resultado


# ============================================================
# DESENHO
# ============================================================

def desenhar_mao(frame, resultado):

    if frame is None or resultado is None:
        return frame

    # Mão esquerda
    if resultado.left_hand_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            resultado.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # Mão direita
    if resultado.right_hand_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            resultado.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # Corpo
    if resultado.pose_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            resultado.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

    return frame


# ============================================================
# EXTRAÇÃO DOS 225 VALORES
# ============================================================

def pegar_pontos(resultado):

    if resultado is None:
        return []

    pontos = []


    # --------------------------------------------------------
    # MÃO ESQUERDA: 21 pontos × 3 = 63
    # --------------------------------------------------------

    if resultado.left_hand_landmarks:

        for ponto in resultado.left_hand_landmarks.landmark:

            pontos.extend([
                ponto.x,
                ponto.y,
                ponto.z
            ])

    else:

        pontos.extend(
            [0.0] * 63
        )


    # --------------------------------------------------------
    # MÃO DIREITA: 21 pontos × 3 = 63
    # --------------------------------------------------------

    if resultado.right_hand_landmarks:

        for ponto in resultado.right_hand_landmarks.landmark:

            pontos.extend([
                ponto.x,
                ponto.y,
                ponto.z
            ])

    else:

        pontos.extend(
            [0.0] * 63
        )


    # --------------------------------------------------------
    # CORPO: 33 pontos × 3 = 99
    # --------------------------------------------------------

    if resultado.pose_landmarks:

        for ponto in resultado.pose_landmarks.landmark:

            pontos.extend([
                ponto.x,
                ponto.y,
                ponto.z
            ])

    else:

        pontos.extend(
            [0.0] * 99
        )


    # Segurança: o modelo exige exatamente 225 valores
    if len(pontos) != 225:

        print(
            f"⚠ Quantidade inválida de pontos: {len(pontos)}"
        )

        return []


    return pontos