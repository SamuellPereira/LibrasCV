from yt_dlp import YoutubeDL


def pegar_videos_do_canal(url):

    opcoes = {
        "extract_flat": True,
        "quiet": True,
        "skip_download": True
    }

    videos = []

    print("\n🔎 Procurando vídeos...")

    with YoutubeDL(opcoes) as ydl:

        info = ydl.extract_info(url, download=False)

        for video in info["entries"]:

            videos.append(
                {
                    "titulo": video["title"],
                    "url": f"https://www.youtube.com/watch?v={video['id']}"
                }
            )

    print(f"\n✅ {len(videos)} vídeos encontrados!\n")

    return videos