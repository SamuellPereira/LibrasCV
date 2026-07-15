import os
import json
import statistics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset_normalizado"
)

total_classes = 0
total_amostras = 0

print("=" * 70)
print("🧠 LIBRAI DATASET INSPECTOR")
print("=" * 70)

for pasta in sorted(os.listdir(DATASET_DIR)):

    caminho_pasta = os.path.join(
        DATASET_DIR,
        pasta
    )

    if not os.path.isdir(caminho_pasta):
        continue

    total_classes += 1

    arquivos = [
        a for a in os.listdir(caminho_pasta)
        if a.endswith(".json")
    ]

    print(f"\n📂 {pasta}")
    print("-" * 50)

    for arquivo in arquivos:

        total_amostras += 1

        caminho = os.path.join(
            caminho_pasta,
            arquivo
        )

        with open(
            caminho,
            "r",
            encoding="utf-8"
        ) as f:

            dados = json.load(f)

        frames = dados["frames"]

        quantidade_frames = len(frames)

        dimensao = len(frames[0])

        valores = []

        frames_zerados = 0

        for frame in frames:

            valores.extend(frame)

            if sum(frame) == 0:
                frames_zerados += 1

        media = statistics.mean(valores)
        desvio = statistics.pstdev(valores)

        print(f"Arquivo........... {arquivo}")
        print(f"Frames............ {quantidade_frames}")
        print(f"Dimensão.......... {dimensao}")
        print(f"Frames zerados.... {frames_zerados}")
        print(f"Média............. {media:.4f}")
        print(f"Desvio............ {desvio:.4f}")
        print()

print("=" * 70)
print("📊 RESUMO")
print("=" * 70)
print(f"Classes........... {total_classes}")
print(f"Amostras.......... {total_amostras}")