import json
import os

import numpy as np


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

HOMOLOGACAO_DIR = os.path.dirname(
    BASE_DIR
)

DATASET_AUGMENTED = os.path.join(
    HOMOLOGACAO_DIR,
    "dataset_augmented"
)

DATASETS_DIR = os.path.join(
    BASE_DIR,
    "datasets"
)

os.makedirs(
    DATASETS_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

SEMENTE = 42

PERCENTUAL_TREINO = 0.80
PERCENTUAL_VALIDACAO = 0.10
PERCENTUAL_TESTE = 0.10

FRAMES_ESPERADOS = 60
DIMENSAO_ESPERADA = 225


# ============================================================
# FUNÇÕES
# ============================================================

def carregar_amostras():

    classes = sorted([
        nome
        for nome in os.listdir(DATASET_AUGMENTED)
        if os.path.isdir(
            os.path.join(
                DATASET_AUGMENTED,
                nome
            )
        )
    ])

    mapa_classes = {
        nome: indice
        for indice, nome in enumerate(classes)
    }

    x = []
    y = []

    ignorados = 0
    total_lidos = 0

    print("=" * 70)
    print("🤟 LIBRAI - CRIANDO DATASET NUMPY")
    print("=" * 70)
    print(f"Classes encontradas: {len(classes)}")
    print()

    for numero_classe, classe in enumerate(classes, start=1):

        pasta_classe = os.path.join(
            DATASET_AUGMENTED,
            classe
        )

        arquivos = sorted([
            arquivo
            for arquivo in os.listdir(pasta_classe)
            if arquivo.endswith(".json")
        ])

        print(
            f"[{numero_classe}/{len(classes)}] "
            f"{classe}: {len(arquivos)} exemplos"
        )

        for arquivo in arquivos:

            caminho = os.path.join(
                pasta_classe,
                arquivo
            )

            try:

                with open(
                    caminho,
                    "r",
                    encoding="utf-8"
                ) as arquivo_json:

                    dados = json.load(
                        arquivo_json
                    )

                frames = dados.get(
                    "frames"
                )

                if not frames:
                    print(
                        f"  ⚠ Ignorado sem frames: {arquivo}"
                    )
                    ignorados += 1
                    continue

                array_frames = np.asarray(
                    frames,
                    dtype=np.float32
                )

                formato_esperado = (
                    FRAMES_ESPERADOS,
                    DIMENSAO_ESPERADA
                )

                if array_frames.shape != formato_esperado:

                    print(
                        f"  ⚠ Formato inválido em {arquivo}: "
                        f"{array_frames.shape}"
                    )

                    ignorados += 1
                    continue

                if not np.isfinite(array_frames).all():

                    print(
                        f"  ⚠ NaN ou Infinity em: {arquivo}"
                    )

                    ignorados += 1
                    continue

                x.append(
                    array_frames
                )

                y.append(
                    mapa_classes[classe]
                )

                total_lidos += 1

            except Exception as erro:

                print(
                    f"  ❌ Erro em {arquivo}: {erro}"
                )

                ignorados += 1

    if not x:

        raise RuntimeError(
            "Nenhuma amostra válida foi carregada."
        )

    x = np.asarray(
        x,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    print()
    print("Amostras válidas:", total_lidos)
    print("Amostras ignoradas:", ignorados)
    print("Formato de X:", x.shape)
    print("Formato de y:", y.shape)

    return x, y, classes, mapa_classes


def embaralhar_dados(x, y):

    rng = np.random.default_rng(
        SEMENTE
    )

    indices = rng.permutation(
        len(x)
    )

    return x[indices], y[indices]


def dividir_dataset(x, y):

    total = len(x)

    fim_treino = int(
        total * PERCENTUAL_TREINO
    )

    fim_validacao = fim_treino + int(
        total * PERCENTUAL_VALIDACAO
    )

    x_train = x[:fim_treino]
    y_train = y[:fim_treino]

    x_val = x[fim_treino:fim_validacao]
    y_val = y[fim_treino:fim_validacao]

    x_test = x[fim_validacao:]
    y_test = y[fim_validacao:]

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test
    )


def salvar_arrays(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    classes,
    mapa_classes
):

    np.save(
        os.path.join(
            DATASETS_DIR,
            "X_train.npy"
        ),
        x_train
    )

    np.save(
        os.path.join(
            DATASETS_DIR,
            "y_train.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            DATASETS_DIR,
            "X_val.npy"
        ),
        x_val
    )

    np.save(
        os.path.join(
            DATASETS_DIR,
            "y_val.npy"
        ),
        y_val
    )

    np.save(
        os.path.join(
            DATASETS_DIR,
            "X_test.npy"
        ),
        x_test
    )

    np.save(
        os.path.join(
            DATASETS_DIR,
            "y_test.npy"
        ),
        y_test
    )

    with open(
        os.path.join(
            DATASETS_DIR,
            "classes.json"
        ),
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            {
                "total_classes": len(classes),
                "classes": classes,
                "mapa_classes": mapa_classes,
                "frames_por_amostra": FRAMES_ESPERADOS,
                "dimensao_por_frame": DIMENSAO_ESPERADA
            },
            arquivo,
            ensure_ascii=False,
            indent=4
        )


def exibir_resumo(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    classes
):

    print()
    print("=" * 70)
    print("📊 DATASET FINAL")
    print("=" * 70)

    print(
        "Classes:",
        len(classes)
    )

    print(
        "Treino:",
        len(x_train),
        x_train.shape,
        y_train.shape
    )

    print(
        "Validação:",
        len(x_val),
        x_val.shape,
        y_val.shape
    )

    print(
        "Teste:",
        len(x_test),
        x_test.shape,
        y_test.shape
    )

    print()
    print(
        "Arquivos salvos em:",
        DATASETS_DIR
    )


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    x, y, classes, mapa_classes = carregar_amostras()

    x, y = embaralhar_dados(
        x,
        y
    )

    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test
    ) = dividir_dataset(
        x,
        y
    )

    salvar_arrays(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        classes,
        mapa_classes
    )

    exibir_resumo(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        classes
    )


if __name__ == "__main__":

    main()