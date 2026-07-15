import json
import os
import random

import numpy as np
import tensorflow as tf


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "datasets"
)

MODELO = os.path.join(
    BASE_DIR,
    "modelos",
    "librai_v1_melhor.keras"
)


print("=" * 70)
print("🤟 LIBRAI - TESTE DO MODELO")
print("=" * 70)


modelo = tf.keras.models.load_model(
    MODELO
)


X = np.load(
    os.path.join(
        DATASET_DIR,
        "X_test.npy"
    )
)

y = np.load(
    os.path.join(
        DATASET_DIR,
        "y_test.npy"
    )
)


with open(
    os.path.join(
        DATASET_DIR,
        "classes.json"
    ),
    "r",
    encoding="utf-8"
) as arquivo:

    info = json.load(
        arquivo
    )


classes = info["classes"]


print()

print("Amostras disponíveis:", len(X))

print()


while True:

    indice = random.randint(
        0,
        len(X)-1
    )


    entrada = np.expand_dims(
        X[indice],
        axis=0
    )


    previsao = modelo.predict(
        entrada,
        verbose=0
    )[0]


    classe_prevista = np.argmax(
        previsao
    )


    confianca = previsao[
        classe_prevista
    ] * 100


    real = classes[
        y[indice]
    ]


    prevista = classes[
        classe_prevista
    ]


    print("=" * 70)

    print("REAL:")
    print(real)

    print()

    print("IA RESPONDEU:")
    print(prevista)

    print()

    print(f"Confiança: {confianca:.2f}%")

    print()

    top5 = np.argsort(
        previsao
    )[-5:][::-1]


    print("TOP 5")

    for posicao, indice_classe in enumerate(top5, start=1):

        print(
            f"{posicao}.",
            classes[indice_classe],
            "-",
            f"{previsao[indice_classe]*100:.2f}%"
        )

    print("=" * 70)

    print()

    continuar = input(
        "Enter = próximo | q = sair: "
    )

    if continuar.lower() == "q":
        break