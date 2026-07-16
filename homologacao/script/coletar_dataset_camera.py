import json
import os
import sys
import time

import cv2


# ============================================================
# CAMINHOS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

HOMOLOGACAO_DIR = os.path.dirname(
    SCRIPT_DIR
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


from config_paths import DATASET_CAMERA_DIR  # noqa: E402

from src.hand_detector import (  # noqa: E402
    detectar_mao,
    desenhar_mao,
    pegar_pontos
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TOTAL_FRAMES = 60
DIMENSAO_FRAME = 225

REPETICOES_POR_SINAL = 10

# Começaremos com poucas palavras.
# Troque pelos sinais que você realmente sabe executar.
SINAIS = [
    "a",
    "casa",
    "chuveiro",
    "obrigado",
    "tudo"
]


# ============================================================
# FUNÇÕES
# ============================================================

def abrir_camera():

    print("📷 Procurando câmera...")

    for indice in range(4):

        camera = cv2.VideoCapture(
            indice,
            cv2.CAP_DSHOW
        )

        if not camera.isOpened():
            camera.release()
            continue

        sucesso, frame = camera.read()

        if sucesso and frame is not None:

            camera.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                640
            )

            camera.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                480
            )

            print(
                f"✅ Câmera {indice} aberta!"
            )

            return camera

        camera.release()

    raise RuntimeError(
        "Nenhuma câmera foi encontrada."
    )


def escrever_texto(
    frame,
    texto,
    posicao,
    escala=0.7
):

    cv2.putText(
        frame,
        texto,
        posicao,
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        (0, 0, 0),
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        texto,
        posicao,
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def proximo_numero(pasta_sinal):

    arquivos = [
        arquivo
        for arquivo in os.listdir(pasta_sinal)
        if arquivo.endswith(".json")
    ]

    numeros = []

    for arquivo in arquivos:

        nome = os.path.splitext(
            arquivo
        )[0]

        parte_final = nome.split("_")[-1]

        if parte_final.isdigit():
            numeros.append(
                int(parte_final)
            )

    if not numeros:
        return 1

    return max(numeros) + 1


def salvar_amostra(
    sinal,
    sequencia
):

    pasta_sinal = os.path.join(
        DATASET_CAMERA_DIR,
        sinal
    )

    os.makedirs(
        pasta_sinal,
        exist_ok=True
    )

    numero = proximo_numero(
        pasta_sinal
    )

    nome_arquivo = (
        f"{sinal}_{numero:03d}.json"
    )

    caminho = os.path.join(
        pasta_sinal,
        nome_arquivo
    )

    dados = {
        "sinal": sinal,
        "origem": "camera",
        "total_frames": len(sequencia),
        "dimensao_frame": DIMENSAO_FRAME,
        "frames": sequencia
    }

    with open(
        caminho,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            dados,
            arquivo,
            ensure_ascii=False
        )

    print(
        f"✅ Salvo: {caminho}"
    )


def contagem_regressiva(
    camera,
    sinal,
    repeticao
):

    for numero in range(3, 0, -1):

        inicio = time.time()

        while time.time() - inicio < 1:

            sucesso, frame = camera.read()

            if not sucesso:
                return False

            frame = cv2.flip(
                frame,
                1
            )

            escrever_texto(
                frame,
                f"Sinal: {sinal.upper()}",
                (20, 40),
                0.8
            )

            escrever_texto(
                frame,
                (
                    f"Repeticao: "
                    f"{repeticao}/"
                    f"{REPETICOES_POR_SINAL}"
                ),
                (20, 75),
                0.65
            )

            escrever_texto(
                frame,
                str(numero),
                (285, 250),
                3
            )

            cv2.imshow(
                "LibrAI - Coleta",
                frame
            )

            tecla = cv2.waitKey(
                1
            ) & 0xFF

            if tecla == 27:
                return False

    return True


def gravar_sinal(
    camera,
    sinal,
    repeticao
):

    sequencia = []

    while len(sequencia) < TOTAL_FRAMES:

        sucesso, frame = camera.read()

        if not sucesso:
            return None

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

        mao_esquerda = (
            resultado is not None
            and resultado.left_hand_landmarks
            is not None
        )

        mao_direita = (
            resultado is not None
            and resultado.right_hand_landmarks
            is not None
        )

        tem_mao = (
            mao_esquerda
            or mao_direita
        )

        pontos = pegar_pontos(
            resultado
        )

        if (
            tem_mao
            and len(pontos)
            == DIMENSAO_FRAME
        ):

            sequencia.append(
                pontos
            )

        escrever_texto(
            frame,
            f"Sinal: {sinal.upper()}",
            (20, 40),
            0.8
        )

        escrever_texto(
            frame,
            (
                f"Repeticao: "
                f"{repeticao}/"
                f"{REPETICOES_POR_SINAL}"
            ),
            (20, 75),
            0.65
        )

        escrever_texto(
            frame,
            (
                f"Gravando: "
                f"{len(sequencia)}/"
                f"{TOTAL_FRAMES}"
            ),
            (20, 110),
            0.65
        )

        if not tem_mao:

            escrever_texto(
                frame,
                "Mostre pelo menos uma mao",
                (20, 145),
                0.6
            )

        cv2.imshow(
            "LibrAI - Coleta",
            frame
        )

        tecla = cv2.waitKey(
            1
        ) & 0xFF

        if tecla == 27:
            return None

    return sequencia


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    print("=" * 60)
    print("🤟 LibrAI - Coleta pela câmera")
    print("=" * 60)

    print()
    print("Sinais que serão coletados:")

    for sinal in SINAIS:
        print(" -", sinal)

    print()
    print(
        f"Repetições por sinal: "
        f"{REPETICOES_POR_SINAL}"
    )

    print(
        f"Frames por repetição: "
        f"{TOTAL_FRAMES}"
    )

    print()
    print("ESC = encerrar")
    print()

    camera = abrir_camera()

    try:

        for sinal in SINAIS:

            print()
            print("=" * 60)
            print(
                f"🤟 Prepare o sinal: "
                f"{sinal.upper()}"
            )
            print("=" * 60)

            for repeticao in range(
                1,
                REPETICOES_POR_SINAL + 1
            ):

                continuar = contagem_regressiva(
                    camera,
                    sinal,
                    repeticao
                )

                if not continuar:
                    return

                sequencia = gravar_sinal(
                    camera,
                    sinal,
                    repeticao
                )

                if sequencia is None:
                    return

                salvar_amostra(
                    sinal,
                    sequencia
                )

                print(
                    f"✔ {sinal}: "
                    f"{repeticao}/"
                    f"{REPETICOES_POR_SINAL}"
                )

        print()
        print("=" * 60)
        print("✅ COLETA FINALIZADA!")
        print("=" * 60)

    finally:

        camera.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    try:
        main()

    except Exception as erro:

        print()
        print("❌ Erro:")
        print(erro)