import os


# Pasta homologacao
HOMOLOGACAO_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Raiz do projeto LibrAI
PROJETO_DIR = os.path.dirname(
    HOMOLOGACAO_DIR
)


# ============================================================
# DADOS
# ============================================================

DADOS_DIR = os.path.join(
    HOMOLOGACAO_DIR,
    "dados"
)

DATASET_DIR = os.path.join(
    DADOS_DIR,
    "dataset"
)

DATASET_NORMALIZADO_DIR = os.path.join(
    DADOS_DIR,
    "dataset_normalizado"
)

DATASET_AUGMENTED_DIR = os.path.join(
    DADOS_DIR,
    "dataset_augmented"
)

DATASET_CAMERA_DIR = os.path.join(
    DADOS_DIR,
    "dataset_camera"
)


# ============================================================
# HOMOLOGAÇÃO
# ============================================================

TEMP_DIR = os.path.join(
    HOMOLOGACAO_DIR,
    "temp"
)

SCRIPTS_DIR = os.path.join(
    HOMOLOGACAO_DIR,
    "script"
)

TESTES_DIR = os.path.join(
    HOMOLOGACAO_DIR,
    "testes"
)

TRAINER_DIR = os.path.join(
    HOMOLOGACAO_DIR,
    "trainer"
)


# ============================================================
# TREINAMENTO
# ============================================================

TRAINER_DATASETS_DIR = os.path.join(
    TRAINER_DIR,
    "datasets"
)

MODELOS_DIR = os.path.join(
    TRAINER_DIR,
    "modelos"
)

LOGS_DIR = os.path.join(
    TRAINER_DIR,
    "logs"
)

MODELO_MELHOR_PATH = os.path.join(
    MODELOS_DIR,
    "librai_v1_melhor.keras"
)

MODELO_FINAL_PATH = os.path.join(
    MODELOS_DIR,
    "librai_v1.keras"
)

CLASSES_PATH = os.path.join(
    TRAINER_DATASETS_DIR,
    "classes.json"
)


# ============================================================
# CRIA PASTAS NECESSÁRIAS
# ============================================================

PASTAS = [
    DADOS_DIR,
    DATASET_DIR,
    DATASET_NORMALIZADO_DIR,
    DATASET_AUGMENTED_DIR,
    DATASET_CAMERA_DIR,
    TEMP_DIR,
    TRAINER_DATASETS_DIR,
    MODELOS_DIR,
    LOGS_DIR
]

for pasta in PASTAS:
    os.makedirs(
        pasta,
        exist_ok=True
    )