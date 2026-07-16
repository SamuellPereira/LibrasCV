import re
import os
from yt_dlp import YoutubeDL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(TEMP_DIR, exist_ok=True)


def limpar_nome(nome):

    nome = nome.lower()

    nome = nome.replace(" ", "_")

    nome = re.sub(r'[^a-z0-9_]', '', nome)

    return nome


def baixar_video(video):

    nome = limpar_nome(video["titulo"])

    caminho = os.path.join(
        TEMP_DIR,
        f"{nome}.mp4"
    )

    opcoes = {

        "format": "bestvideo+bestaudio/best",

        "outtmpl": caminho,

        "quiet": False,

        "noplaylist": True,

        "merge_output_format": "mp4",

        "extractor_args": {
            "youtube": {
                "player_client": ["android"]
            }
        }
    }

    print("⬇", nome)

    with YoutubeDL(opcoes) as ydl:
        ydl.download([video["url"]])

    return caminho