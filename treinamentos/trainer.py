import os
import csv
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


DATASET = "dataset"
MODELO = "models/libras_model.keras"


def carregar_dataset():

    X = []
    y = []

    sinais = os.listdir(DATASET)

    print("Sinais encontrados:", sinais)


    for indice, sinal in enumerate(sinais):

        pasta = os.path.join(DATASET, sinal)

        if not os.path.isdir(pasta):
            continue


        for arquivo in os.listdir(pasta):

            caminho = os.path.join(
                pasta,
                arquivo
            )

            dados = []


            with open(caminho, "r") as csv_file:

                leitor = csv.reader(csv_file)

                for linha in leitor:

                    dados.extend(
                        [
                            float(valor)
                            for valor in linha
                        ]
                    )


            X.append(dados)

            y.append(indice)



    return np.array(X), np.array(y)



print("Carregando dataset...")


X, y = carregar_dataset()


print("Exemplos:", len(X))
print("Formato:", X.shape)



y = to_categorical(y)



modelo = Sequential()


modelo.add(
    Dense(
        128,
        activation="relu",
        input_shape=(X.shape[1],)
    )
)


modelo.add(
    Dropout(0.3)
)


modelo.add(
    Dense(
        64,
        activation="relu"
    )
)


modelo.add(
    Dense(
        y.shape[1],
        activation="softmax"
    )
)



modelo.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)



modelo.summary()



modelo.fit(
    X,
    y,
    epochs=100
)



os.makedirs(
    "models",
    exist_ok=True
)


modelo.save(MODELO)


print("Modelo criado!")

