import cv2
import mediapipe as mp
import json
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

os.makedirs(
    DATASET_DIR,
    exist_ok=True
)


mp_holistic = mp.solutions.holistic


def extrair_landmarks(video_path, nome_sinal):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Não conseguiu abrir o vídeo")
        return


    sequencia = []

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(
        f"🎥 Processando: {video_path}"
    )

    print(
        f"Frames no vídeo: {total_frames}"
    )


    with mp_holistic.Holistic(

        static_image_mode=False,

        model_complexity=1,

        min_detection_confidence=0.3,

        min_tracking_confidence=0.3

    ) as modelo:


        contador = 0


        while True:


            sucesso, frame = cap.read()


            if not sucesso:
                break



            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            resultado = modelo.process(
                rgb
            )


            pontos = []



            # =====================
            # MÃO ESQUERDA
            # =====================

            if resultado.left_hand_landmarks:


                for p in resultado.left_hand_landmarks.landmark:

                    pontos.extend([
                        p.x,
                        p.y,
                        p.z
                    ])


            else:

                pontos.extend(
                    [0] * 63
                )



            # =====================
            # MÃO DIREITA
            # =====================

            if resultado.right_hand_landmarks:


                for p in resultado.right_hand_landmarks.landmark:

                    pontos.extend([
                        p.x,
                        p.y,
                        p.z
                    ])


            else:

                pontos.extend(
                    [0] * 63
                )



            # =====================
            # CORPO (POSE)
            # =====================

            if resultado.pose_landmarks:


                for p in resultado.pose_landmarks.landmark:

                    pontos.extend([
                        p.x,
                        p.y,
                        p.z
                    ])


            else:

                pontos.extend(
                    [0] * 99
                )



            sequencia.append(
                pontos
            )


            contador += 1


            if contador % 30 == 0:

                print(
                    f"✅ Frames processados: {contador} | Pontos: {len(pontos)}"
                )



    cap.release()



    caminho_saida = os.path.join(

        DATASET_DIR,

        nome_sinal + ".json"

    )



    dados = {

        "sinal": nome_sinal,

        "total_frames": len(sequencia),

        "frames": sequencia

    }



    with open(

        caminho_saida,

        "w",

        encoding="utf-8"

    ) as arquivo:


        json.dump(

            dados,

            arquivo,

            ensure_ascii=False,

            indent=2

        )



    print()
    print(
        "🚀 Dataset criado:"
    )

    print(
        caminho_saida
    )

    print(
        "Total frames:",
        len(sequencia)
    )