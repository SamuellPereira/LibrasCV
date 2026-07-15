from recognizer import carregar_dataset
from ui import LibrAI


def iniciar():

    print("=" * 40)
    print("🤟 LibrAI iniciando...")
    print("=" * 40)


    # =========================
    # CARREGAR DATASET
    # =========================

    print("\n📚 Carregando dataset...")


    try:

        dados = carregar_dataset()


        print("\n🧠 Sinais carregados:")


        if dados:

            for sinal in dados.keys():

                print("  ✔", sinal)

        else:

            print("  ⚠ Nenhum sinal encontrado")


    except Exception as erro:

        print("\n❌ Erro carregando dataset:")
        print(erro)

        return



    print("\n🚀 IA pronta!")
    print("=" * 40)



    # =========================
    # INICIAR INTERFACE
    # =========================


    try:

        app = LibrAI(dados)

        app.mainloop()


    except Exception as erro:

        print("\n❌ Erro na interface:")
        print(erro)



# =========================
# EXECUÇÃO
# =========================

if __name__ == "__main__":

    iniciar()