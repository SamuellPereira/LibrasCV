import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "dataset")

print("=" * 60)
print("📂 Organizando dataset...")
print("=" * 60)

movidos = 0

for arquivo in os.listdir(DATASET):

    caminho = os.path.join(DATASET, arquivo)

    if not os.path.isfile(caminho):
        continue

    if not arquivo.endswith(".json"):
        continue

    nome = os.path.splitext(arquivo)[0].strip()

    # ignora nomes inválidos
    if nome == "":
        print("⚠ Ignorado:", arquivo)
        continue

    pasta = os.path.join(DATASET, nome)

    os.makedirs(pasta, exist_ok=True)

    shutil.move(
        caminho,
        os.path.join(pasta, arquivo)
    )

    movidos += 1
    print(f"✔ {nome}")

print("\n==============================")
print("Arquivos movidos:", movidos)
print("==============================")