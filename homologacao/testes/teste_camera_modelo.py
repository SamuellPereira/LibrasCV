import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# CAMINHOS DO PROJETO
# ============================================================

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

sys.path.insert(
    0,
    HOMOLOGACAO_DIR
)


from config_paths import (  # noqa: E402
    MODELO_MELHOR_PATH,
    CLASSES_PATH
)

from src.hand_detector import (  # noqa: E402
    detectar_mao,
    desenhar_mao,
    pegar_pontos
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODELO_PATH = MODELO_MELHOR_PATH

TOTAL_FRAMES = 60
DIMENSAO_FRAME = 225

CONFIANCA_MINIMA = 80.0

PREDICOES_PARA_CONFIRMAR = 5

INTERVALO_PREDICAO = 15

MOSTRAR_TOP_5 = True


# ============================================================
# CARREGAMENTO DO MODELO
# ============================================================

def carregar_modelo():

    if not os.path.exists(MODELO_PATH):

        raise FileNotFoundError(
            "Modelo não encontrado:\n"
            f"{MODELO_PATH}"
        )

    print("🧠 Carregando modelo...")

    modelo = tf.keras.models.load_model(
        MODELO_PATH
    )

    print("✅ Modelo carregado!")

    return modelo


# ============================================================
# CARREGAMENTO DAS CLASSES
# ============================================================

def carregar_classes():

    if not os.path.exists(CLASSES_PATH):

        raise FileNotFoundError(
            "Arquivo de classes não encontrado:\n"
            f"{CLASSES_PATH}"
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
# ABERTURA DA CÂMERA
# ============================================================

def abrir_camera():

    print("📷 Procurando uma câmera disponível...")

    # Testa alguns índices automaticamente.
    for indice in range(4):

        print(
            f"   Tentando câmera {indice}..."
        )

        camera = cv2.VideoCapture(
            indice,
            cv2.CAP_DSHOW
        )

        if not camera.isOpened():

            camera.release()

            camera = cv2.VideoCapture(
                indice
            )

        if camera.isOpened():

            sucesso, frame = camera.read()

            if sucesso and frame is not None:

                print(
                    f"✅ Câmera {indice} aberta!"
                )

                camera.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    640
                )

                camera.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    480
                )

                return camera

        camera.release()

    raise RuntimeError(
        "Nenhuma câmera disponível foi encontrada.\n"
        "Feche OBS, Teams, Discord, navegador e o aplicativo "
        "Câmera do Windows e tente novamente."
    )


# ============================================================
# DESENHO DE TEXTO
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


# ============================================================
# PAINEL DA INTERFACE
# ============================================================

def desenhar_painel(
    frame,
    palavra,
    confianca,
    quantidade_frames,
    fps,
    top_5
):

    altura_painel = 245 if MOSTRAR_TOP_5 else 155

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (620, altura_painel),
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
        f"Buffer: {quantidade_frames}/{TOTAL_FRAMES}",
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
            (25, 115),
            0.9
        )

        desenhar_texto(
            frame,
            f"Confianca: {confianca:.2f}%",
            (25, 150),
            0.7
        )

    else:

        desenhar_texto(
            frame,
            "Aguardando reconhecimento...",
            (25, 115),
            0.7
        )

    if MOSTRAR_TOP_5 and top_5:

        desenhar_texto(
            frame,
            "Top 5:",
            (25, 185),
            0.6
        )

        for posicao, (
            nome,
            valor
        ) in enumerate(
            top_5,
            start=1
        ):

            linha = 185 + (
                posicao * 11
            )

            desenhar_texto(
                frame,
                f"{posicao}. {nome}: {valor:.1f}%",
                (120, linha),
                0.38,
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

    formato_esperado = (
        TOTAL_FRAMES,
        DIMENSAO_FRAME
    )

    if entrada.shape != formato_esperado:

        print(
            "⚠ Formato inválido:",
            entrada.shape
        )

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
        np.argmax(
            probabilidades
        )
    )

    palavra = classes[
        indice
    ]

    confianca = float(
        probabilidades[indice]
        * 100
    )

    indices_top = np.argsort(
        probabilidades
    )[-5:][::-1]

    top_5 = []

    for indice_classe in indices_top:

        top_5.append(
            (
                classes[
                    int(indice_classe)
                ],
                float(
                    probabilidades[
                        indice_classe
                    ] * 100
                )
            )
        )

    return (
        palavra,
        confianca,
        top_5
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print("🤟 LibrAI - Teste com Webcam")
    print("=" * 60)

    print("\n1️⃣ Carregando modelo...")

    modelo = carregar_modelo()

    print("\n2️⃣ Carregando classes...")

    classes = carregar_classes()

    print("\n3️⃣ Abrindo câmera...")

    camera = abrir_camera()

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
    print("✅ LibrAI iniciado!")
    print("ESC = sair")
    print("R = limpar o buffer")
    print()

    while True:

        sucesso, frame = camera.read()

        if not sucesso:

            print(
                "❌ Não foi possível ler "
                "o frame da câmera."
            )

            break

        frame = cv2.flip(
            frame,
            1
        )

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

        mao_esquerda = (
            resultado is not None
            and resultado.left_hand_landmarks is not None
        )

        mao_direita = (
            resultado is not None
            and resultado.right_hand_landmarks is not None
        )

        tem_mao = mao_esquerda or mao_direita


        if tem_mao and len(pontos) == DIMENSAO_FRAME:

            sequencia.append(
                pontos
            )

        else:

            sequencia.clear()
            historico_predicoes.clear()

            palavra_confirmada = None
            confianca_confirmada = 0.0
            top_5_atual = []

        contador_frames =2

        if (
            len(sequencia)
            == TOTAL_FRAMES
            and contador_frames
            % INTERVALO_PREDICAO
            == 0
        ):

            (
                palavra,
                confianca,
                top_5
            ) = prever(
                modelo,
                classes,
                list(sequencia)
            )

            top_5_atual = top_5

            if (
                palavra is not None
                and confianca
                >= CONFIANCA_MINIMA
            ):

                historico_predicoes.append(
                    palavra
                )

                if (
                    len(
                        historico_predicoes
                    )
                    == PREDICOES_PARA_CONFIRMAR
                    and len(
                        set(
                            historico_predicoes
                        )
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

        intervalo = (
            tempo_atual
            - tempo_anterior
        )

        if intervalo > 0:

            fps = 1.0 / intervalo

        tempo_anterior = tempo_atual

        desenhar_painel(
            frame=frame,
            palavra=palavra_confirmada,
            confianca=confianca_confirmada,
            quantidade_frames=len(
                sequencia
            ),
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

            print(
                "🔄 Buffer limpo."
            )

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# INICIALIZAÇÃO
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as erro:

        print()
        print("=" * 60)
        print("❌ ERRO NO LIBRAI")
        print("=" * 60)
        print(erro)