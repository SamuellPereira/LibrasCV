import os
import json
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(
    BASE_DIR,
    "dataset_normalizado"
)

DATASET_AUG = os.path.join(
    BASE_DIR,
    "dataset_augmented"
)

os.makedirs(
    DATASET_AUG,
    exist_ok=True
)

COPIAS = 20

print("=" * 60)
print("🚀 GERANDO DATA AUGMENTATION")
print("=" * 60)

total = 0

for classe in sorted(os.listdir(DATASET)):

    origem = os.path.join(
        DATASET,
        classe
    )

    if not os.path.isdir(origem):
        continue

    destino = os.path.join(
        DATASET_AUG,
        classe
    )

    os.makedirs(
        destino,
        exist_ok=True
    )

    for arquivo in os.listdir(origem):

        if not arquivo.endswith(".json"):
            continue

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

        # salva o original
        with open(
            os.path.join(destino, "0.json"),
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                dados,
                f
            )

        # gera cópias
        for copia in range(1, COPIAS):

            novo = {
                "sinal": dados["sinal"],
                "frames": []
            }

            for frame in dados["frames"]:

                novo_frame = []

                for valor in frame:

                    ruido = random.uniform(
                        -0.005,
                        0.005
                    )

                    novo_frame.append(
                        valor + ruido
                    )

                novo["frames"].append(
                    novo_frame
                )

            with open(
                os.path.join(
                    destino,
                    f"{copia}.json"
                ),
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    novo,
                    f
                )

        total += COPIAS

        print(
            f"✔ {classe}"
        )

print()
print("=" * 60)
print("✅ FINALIZADO")
print("=" * 60)
print("Arquivos criados:", total)