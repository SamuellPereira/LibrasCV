import os
from extract_landmarks import extrair_landmarks


LIMITE = 160


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


TEMP_DIR = os.path.join(
    BASE_DIR,
    "temp"
)


DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)


def extrair_nome_sinal(nome):

    nome = nome.lower()


    remover = [
        "sinal",
        "de",
        "em",
        "libras",
        "com",
        "libraslab"
    ]


    palavras = nome.split("_")


    resultado = []


    for palavra in palavras:

        if palavra not in remover:
            resultado.append(palavra)


    return "_".join(resultado)



print("=" * 60)
print("🤟 LibrAI - Processador de Dataset")
print("=" * 60)



videos = [
    arquivo
    for arquivo in os.listdir(TEMP_DIR)
    if arquivo.endswith(".mp4")
]

if LIMITE:
    videos = videos[:LIMITE]



print(f"\nEncontrados {len(videos)} vídeos.\n")



for i, video in enumerate(videos):


    caminho_video = os.path.join(
        TEMP_DIR,
        video
    )


    nome_original = os.path.splitext(
        video
    )[0]


    nome_sinal = extrair_nome_sinal(
        nome_original
    )



    print("\n" + "-" * 60)


    print(
        f"[{i+1}/{len(videos)}]"
    )


    print(
        "Nome original:",
        nome_original
    )


    print(
        "Nome do sinal:",
        nome_sinal
    )



    try:


        extrair_landmarks(
            caminho_video,
            nome_sinal
        )



        arquivo_json = os.path.join(
            DATASET_DIR,
            nome_sinal + ".json"
        )



        if os.path.exists(arquivo_json):

            os.remove(caminho_video)


            print(
                "🗑️ Vídeo removido:",
                video
            )



        print(
            "✅ Finalizado:",
            nome_sinal
        )



    except Exception as erro:


        print(
            "❌ Erro:",
            nome_sinal
        )


        print(
            erro
        )



print(
    "\n🎉 Todos os vídeos foram processados!"
)