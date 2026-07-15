import os
import json
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

total = 0
corretos = 0
erros = 0

menor_frame = 999999
maior_frame = 0

dimensao = None

print("=" * 60)
print("🔍 VALIDANDO DATASET")
print("=" * 60)


for pasta in os.listdir(DATASET_DIR):

    caminho_pasta = os.path.join(
        DATASET_DIR,
        pasta
    )

    if not os.path.isdir(caminho_pasta):
        continue


    for arquivo in os.listdir(caminho_pasta):

        if not arquivo.endswith(".json"):
            continue


        total += 1

        caminho = os.path.join(
            caminho_pasta,
            arquivo
        )

        try:

            with open(
                caminho,
                "r",
                encoding="utf-8"
            ) as f:

                dados = json.load(f)

        except Exception as e:

            print(f"❌ {arquivo}")
            print(e)
            erros += 1
            continue


        if "frames" not in dados:

            print(f"❌ {arquivo} sem 'frames'")
            erros += 1
            continue


        frames = dados["frames"]

        if len(frames) == 0:

            print(f"❌ {arquivo} vazio")
            erros += 1
            continue


        tamanho = len(frames[0])

        if dimensao is None:
            dimensao = tamanho


        arquivo_ok = True


        for frame in frames:

            if len(frame) != tamanho:

                print(f"❌ {arquivo} possui frames de tamanhos diferentes")

                arquivo_ok = False
                break


            for valor in frame:

                if math.isnan(valor):

                    print(f"❌ {arquivo} possui NaN")

                    arquivo_ok = False
                    break

                if math.isinf(valor):

                    print(f"❌ {arquivo} possui Infinity")

                    arquivo_ok = False
                    break


            if not arquivo_ok:
                break


        if not arquivo_ok:

            erros += 1
            continue


        menor_frame = min(
            menor_frame,
            len(frames)
        )

        maior_frame = max(
            maior_frame,
            len(frames)
        )

        corretos += 1


print()
print("=" * 60)
print("📊 RELATÓRIO")
print("=" * 60)

print("Arquivos encontrados :", total)
print("Arquivos corretos    :", corretos)
print("Arquivos com erro    :", erros)
print("Menor sequência      :", menor_frame)
print("Maior sequência      :", maior_frame)
print("Dimensão dos frames  :", dimensao)