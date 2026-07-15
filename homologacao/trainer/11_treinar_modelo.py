import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GRU,
    Input,
    Masking
)
from tensorflow.keras.models import Sequential


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASETS_DIR = os.path.join(
    BASE_DIR,
    "datasets"
)

MODELOS_DIR = os.path.join(
    BASE_DIR,
    "modelos"
)

os.makedirs(
    MODELOS_DIR,
    exist_ok=True
)

MODELO_FINAL = os.path.join(
    MODELOS_DIR,
    "librai_v1.keras"
)

MELHOR_MODELO = os.path.join(
    MODELOS_DIR,
    "librai_v1_melhor.keras"
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

EPOCAS = 50
BATCH_SIZE = 32
SEMENTE = 42

np.random.seed(SEMENTE)
tf.random.set_seed(SEMENTE)


# ============================================================
# CARREGAMENTO
# ============================================================

def carregar_dados():

    print("=" * 70)
    print("🤟 LIBRAI — TREINAMENTO V1")
    print("=" * 70)
    print("\nCarregando arrays...")

    x_train = np.load(
        os.path.join(
            DATASETS_DIR,
            "X_train.npy"
        )
    )

    y_train = np.load(
        os.path.join(
            DATASETS_DIR,
            "y_train.npy"
        )
    )

    x_val = np.load(
        os.path.join(
            DATASETS_DIR,
            "X_val.npy"
        )
    )

    y_val = np.load(
        os.path.join(
            DATASETS_DIR,
            "y_val.npy"
        )
    )

    x_test = np.load(
        os.path.join(
            DATASETS_DIR,
            "X_test.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            DATASETS_DIR,
            "y_test.npy"
        )
    )

    with open(
        os.path.join(
            DATASETS_DIR,
            "classes.json"
        ),
        "r",
        encoding="utf-8"
    ) as arquivo:

        informacoes = json.load(arquivo)

    total_classes = informacoes["total_classes"]

    print("Treino:", x_train.shape)
    print("Validação:", x_val.shape)
    print("Teste:", x_test.shape)
    print("Classes:", total_classes)

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        total_classes
    )


# ============================================================
# MODELO
# ============================================================

def criar_modelo(
    total_classes,
    frames,
    dimensoes
):

    modelo = Sequential([
        Input(
            shape=(frames, dimensoes)
        ),

        Masking(
            mask_value=0.0
        ),

        GRU(
            128,
            return_sequences=True
        ),

        Dropout(
            0.30
        ),

        GRU(
            64
        ),

        BatchNormalization(),

        Dropout(
            0.30
        ),

        Dense(
            256,
            activation="relu"
        ),

        Dropout(
            0.30
        ),

        Dense(
            total_classes,
            activation="softmax"
        )
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo


# ============================================================
# TREINAMENTO
# ============================================================

def main():

    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        total_classes
    ) = carregar_dados()

    modelo = criar_modelo(
        total_classes=total_classes,
        frames=x_train.shape[1],
        dimensoes=x_train.shape[2]
    )

    print()
    modelo.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        ),

        ModelCheckpoint(
            MELHOR_MODELO,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    print("\n🚀 Iniciando treinamento...\n")

    modelo.fit(
        x_train,
        y_train,
        validation_data=(
            x_val,
            y_val
        ),
        epochs=EPOCAS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print("\n🧪 Avaliando no conjunto de teste...")

    perda, acuracia = modelo.evaluate(
        x_test,
        y_test,
        verbose=1
    )

    print()
    print("=" * 70)
    print("📊 RESULTADO FINAL")
    print("=" * 70)
    print(f"Loss de teste: {perda:.4f}")
    print(f"Acurácia de teste: {acuracia * 100:.2f}%")

    modelo.save(
        MODELO_FINAL
    )

    print("\n✅ Modelo final salvo em:")
    print(MODELO_FINAL)

    print("\n✅ Melhor modelo salvo em:")
    print(MELHOR_MODELO)


if __name__ == "__main__":
    main()