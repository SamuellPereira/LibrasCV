import csv
import os

print("DATA COLLECTOR FOI IMPORTADO")


def salvar_sequencia(nome_sinal, sequencia):

    pasta = f"dataset/{nome_sinal}"

    os.makedirs(pasta, exist_ok=True)

    arquivos = os.listdir(pasta)

    numero = len(arquivos) + 1

    arquivo = f"{pasta}/exemplo_{numero:03}.csv"


    with open(arquivo, "w", newline="") as arquivo_csv:

        escritor = csv.writer(arquivo_csv)

        for frame in sequencia:
            escritor.writerow(frame)


    print("Salvo em:", arquivo)
    print("Quantidade de frames:", len(sequencia))