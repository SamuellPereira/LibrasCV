import edge_tts
import asyncio
import os
import pygame


async def gerar_audio(texto):

    voz = "pt-BR-FranciscaNeural"

    comunicacao = edge_tts.Communicate(
        texto,
        voz
    )

    await comunicacao.save(
        "fala.mp3"
    )


def falar(texto):

    asyncio.run(
        gerar_audio(texto)
    )


    pygame.mixer.init()

    pygame.mixer.music.load(
        "fala.mp3"
    )

    pygame.mixer.music.play()


    while pygame.mixer.music.get_busy():
        pass


    pygame.mixer.quit()


    os.remove(
        "fala.mp3"
    )