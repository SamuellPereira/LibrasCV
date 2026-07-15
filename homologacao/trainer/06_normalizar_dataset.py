import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_ORIGINAL = os.path.join(
    BASE_DIR,
    "dataset"
)

DATASET_NOVO = os.path.join(
    BASE_DIR,
    "dataset_normalizado"
)

os.makedirs(
    DATASET_NOVO,
    exist_ok=True
)

FRAMES_DESEJADOS = 60

total = 0

print("=" * 60)
print("🚀 NORMALIZANDO DATASET")
print("=" * 60)


for pasta in os.listdir(DATASET_ORIGINAL):

    origem = os.path.join(
        DATASET_ORIGINAL,
        pasta
    )

    if not os.path.isdir(origem):
        continue

    destino = os.path.join(
        DATASET_NOVO,
        pasta
    )

    os.makedirs(
        destino,
        exist_ok=True
    )


    arquivos = [
        a
        for a in os.listdir(origem)
        if a.endswith(".json")
    ]


    for arquivo in arquivos:

        total += 1

        caminho = os.path.join(
            origem,
            arquivo
        )

        with open(
            caminho,
            "r",
            encoding="utf-8"
        ) as f:

            dados = json.load(f)


        frames = dados["frames"]


        indices = np.linspace(
            0,
            len(frames) - 1,
            FRAMES_DESEJADOS,
            dtype=int
        )


        novos_frames = [
            frames[i]
            for i in indices
        ]


        dados["frames"] = novos_frames
        dados["total_frames"] = FRAMES_DESEJADOS


        novo_json = os.path.join(
            destino,
            arquivo
        )


        with open(
            novo_json,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                dados,
                f
            )


        print(
            f"[{total}] {arquivo} ✔"
        )


print()
print("=" * 60)
print("✅ DATASET NORMALIZADO!")
print("=" * 60)
print("Arquivos processados:", total)
print("Frames por vídeo:", FRAMES_DESEJADOS)