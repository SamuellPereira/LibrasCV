from youtube_importer import pegar_videos_do_canal
from youtube_downloader import baixar_video

CANAL = "https://www.youtube.com/@LibrasLab"

print("="*60)
print("🤟 LibrAI Importador")
print("="*60)

videos = pegar_videos_do_canal(CANAL)

print(f"\nEncontrados {len(videos)} vídeos.\n")

for i, video in enumerate(videos):

    print(f"\n[{i+1}/{len(videos)}]")
    print(video["titulo"])

    try:

        caminho = baixar_video(video)

        print("✔ Salvo em:", caminho)

    except Exception as erro:

        print("❌ Erro ao baixar:")
        print(erro)

        print("➡ Pulando para o próximo vídeo...")