import os
import json


def carregar_dataset():

    dataset = {}

    pasta_dataset = "dataset"


    for sinal in os.listdir(pasta_dataset):

        caminho_sinal = os.path.join(
            pasta_dataset,
            sinal
        )


        if not os.path.isdir(caminho_sinal):
            continue


        dataset[sinal] = []


        for arquivo in os.listdir(caminho_sinal):

            if not arquivo.endswith(".json"):
                continue


            caminho_arquivo = os.path.join(
                caminho_sinal,
                arquivo
            )


            with open(
                caminho_arquivo,
                "r",
                encoding="utf-8"
            ) as f:

                dados = json.load(f)


            frames = []


            for frame in dados:

                pontos = []


                # mão esquerda
                if "mao_esquerda" in frame:

                    for ponto in frame["mao_esquerda"]:

                        pontos.extend(ponto)


                # mão direita
                if "mao_direita" in frame:

                    for ponto in frame["mao_direita"]:

                        pontos.extend(ponto)


                frames.append(pontos)


            dataset[sinal].append(frames)


    print("Dataset carregado:")
    
    for item in dataset:
        print(" -", item)


    return dataset



def reconhecer(pontos, dados):

    # Sem mão = sem reconhecimento
    if not pontos:
        return None


    menor_distancia = float("inf")

    sinal_encontrado = None


    for sinal, exemplos in dados.items():

        for exemplo in exemplos:

            for frame_salvo in exemplo:


                # ignora exemplos vazios
                if not frame_salvo:
                    continue


                distancia = calcular_distancia(
                    pontos,
                    frame_salvo
                )


                if distancia < menor_distancia:

                    menor_distancia = distancia
                    sinal_encontrado = sinal



    print(
        "Melhor:",
        sinal_encontrado,
        "Distância:",
        menor_distancia
    )


    # limite de confiança
    if menor_distancia < 2.5:

        return sinal_encontrado


    return None




def calcular_distancia(pontos1, pontos2):

    distancia = 0


    for a, b in zip(
        pontos1,
        pontos2
    ):

        distancia += abs(a - b)


    return distancia