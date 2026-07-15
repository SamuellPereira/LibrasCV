import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# CAMINHOS
# ============================================================

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

from src.hand_detector import (  # noqa: E402
    detectar_mao,
    desenhar_mao,
    pegar_pontos
)


MODELO_PATH = os.path.join(
    BASE_DIR,
    "trainer",
    "modelos",
    "librai_v1_melhor.keras"
)

CLASSES_PATH = os.path.join(
    BASE_DIR,
    "trainer",
    "datasets",
    "classes.json"
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TOTAL_FRAMES = 60
DIMENSAO_FRAME = 225

CONFIANCA_MINIMA = 80.0

PREDICOES_PARA_CONFIRMAR = 5

INTERVALO_PREDICAO = 5

MOSTRAR_TOP_5 = True


# ============================================================
# CARREGAMENTO
# ============================================================

def carregar_modelo():

    if not os.path.exists(MODELO_PATH):

        raise FileNotFoundError(
            f"Modelo não encontrado:\n{MODELO_PATH}"
        )

    print("🧠 Carregando modelo...")

    modelo = tf.keras.models.load_model(
        MODELO_PATH
    )

    print("✅ Modelo carregado!")

    return modelo


def carregar_classes():

    if not os.path.exists(CLASSES_PATH):

        raise FileNotFoundError(
            f"Arquivo de classes não encontrado:\n{CLASSES_PATH}"
        )

    with open(
        CLASSES_PATH,
        "r",
        encoding="utf-8"
    ) as arquivo:

        dados = json.load(
            arquivo
        )

    classes = dados["classes"]

    print(
        f"✅ {len(classes)} classes carregadas!"
    )

    return classes


# ============================================================
# DESENHO
# ============================================================

def desenhar_texto(
    frame,
    texto,
    posicao,
    escala=0.7,
    espessura=2
):

    cv2.putText(
        frame,
        texto,
        posicao,
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        (0, 0, 0),
        espessura + 2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        texto,
        posicao,
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        (255, 255, 255),
        espessura,
        cv2.LINE_AA
    )


def desenhar_painel(
    frame,
    palavra,
    confianca,
    quantidade_frames,
    fps,
    top_5
):

    altura_painel = 230 if MOSTRAR_TOP_5 else 135

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (500, altura_painel),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
        frame
    )

    desenhar_texto(
        frame,
        f"Frames: {quantidade_frames}/{TOTAL_FRAMES}",
        (25, 40),
        0.65
    )

    desenhar_texto(
        frame,
        f"FPS: {fps:.1f}",
        (25, 70),
        0.65
    )

    if palavra:

        desenhar_texto(
            frame,
            f"Palavra: {palavra.upper()}",
            (25, 110),
            0.9,
            2
        )

        desenhar_texto(
            frame,
            f"Confianca: {confianca:.2f}%",
            (25, 145),
            0.7
        )

    else:

        desenhar_texto(
            frame,
            "Aguardando movimento...",
            (25, 110),
            0.75
        )

    if MOSTRAR_TOP_5 and top_5:

        desenhar_texto(
            frame,
            "Top 5:",
            (25, 180),
            0.6
        )

        texto_top = " | ".join(
            [
                f"{nome}: {valor:.1f}%"
                for nome, valor in top_5
            ]
        )

        desenhar_texto(
            frame,
            texto_top[:75],
            (25, 210),
            0.45,
            1
        )


# ============================================================
# PREDIÇÃO
# ============================================================

def prever(
    modelo,
    classes,
    sequencia
):

    entrada = np.asarray(
        sequencia,
        dtype=np.float32
    )

    if entrada.shape != (
        TOTAL_FRAMES,
        DIMENSAO_FRAME
    ):

        return None, 0.0, []

    entrada = np.expand_dims(
        entrada,
        axis=0
    )

    probabilidades = modelo.predict(
        entrada,
        verbose=0
    )[0]

    indice = int(
        np.argmax(probabilidades)
    )

    palavra = classes[indice]

    confianca = float(
        probabilidades[indice] * 100
    )

    indices_top = np.argsort(
        probabilidades
    )[-5:][::-1]

    top_5 = [
        (
            classes[int(indice_classe)],
            float(
                probabilidades[indice_classe] * 100
            )
        )
        for indice_classe in indices_top
    ]

    return palavra, confianca, top_5


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    print("=" * 60)
    print("🤟 LibrAI - Teste com Webcam")
    print("=" * 60)

    modelo = carregar_modelo()

    classes = carregar_classes()

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        camera = cv2.VideoCapture(
            0
        )

    if not camera.isOpened():

        raise RuntimeError(
            "Não foi possível abrir a câmera."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    sequencia = deque(
        maxlen=TOTAL_FRAMES
    )

    historico_predicoes = deque(
        maxlen=PREDICOES_PARA_CONFIRMAR
    )

    palavra_confirmada = None
    confianca_confirmada = 0.0

    top_5_atual = []

    contador_frames = 0

    tempo_anterior = time.time()

    fps = 0.0

    print()
    print("✅ Webcam iniciada.")
    print("ESC = sair")
    print("R = limpar buffer")
    print()

    while True:

        sucesso, frame = camera.read()

        if not sucesso:

            print("❌ Erro ao ler a câmera.")
            break

        frame = cv2.flip(
            frame,
            1
        )

        resultado = detectar_mao(
            frame
        )

        frame = desenhar_mao(
            frame,
            resultado
        )

        pontos = pegar_pontos(
            resultado
        )

        if len(pontos) == DIMENSAO_FRAME:

            sequencia.append(
                pontos
            )

        contador_frames += 1

        if (
            len(sequencia) == TOTAL_FRAMES
            and contador_frames % INTERVALO_PREDICAO == 0
        ):

            palavra, confianca, top_5 = prever(
                modelo,
                classes,
                list(sequencia)
            )

            top_5_atual = top_5

            if confianca >= CONFIANCA_MINIMA:

                historico_predicoes.append(
                    palavra
                )

                if (
                    len(historico_predicoes)
                    == PREDICOES_PARA_CONFIRMAR
                    and len(
                        set(historico_predicoes)
                    ) == 1
                ):

                    palavra_confirmada = palavra

                    confianca_confirmada = confianca

                    print(
                        "Reconhecido:",
                        palavra_confirmada.upper(),
                        f"- {confianca_confirmada:.2f}%"
                    )

            else:

                historico_predicoes.clear()

        tempo_atual = time.time()

        intervalo = tempo_atual - tempo_anterior

        if intervalo > 0:

            fps = 1.0 / intervalo

        tempo_anterior = tempo_atual

        desenhar_painel(
            frame=frame,
            palavra=palavra_confirmada,
            confianca=confianca_confirmada,
            quantidade_frames=len(sequencia),
            fps=fps,
            top_5=top_5_atual
        )

        cv2.imshow(
            "LibrAI - Webcam IA",
            frame
        )

        tecla = cv2.waitKey(
            1
        ) & 0xFF

        if tecla == 27:

            break

        if tecla in (
            ord("r"),
            ord("R")
        ):

            sequencia.clear()

            historico_predicoes.clear()

            palavra_confirmada = None

            confianca_confirmada = 0.0

            top_5_atual = []

            print("🔄 Buffer limpo.")


    camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    try:

        main()

    except Exception as erro:

        print()
        print("❌ Erro:")
        print(erro)