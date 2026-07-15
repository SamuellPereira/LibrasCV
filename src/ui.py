import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

from hand_detector import detectar_mao, pegar_pontos, desenhar_mao
from recognizer import carregar_dataset, reconhecer
from voice import falar


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class LibrAI(ctk.CTk):

    def __init__(self, dados):

        super().__init__()


        self.dados = dados

        self.frase = []
        self.ultimo_sinal = None


        self.title("🤟 LibrAI")
        self.geometry("1300x800")
        self.minsize(1200,700)


        self.configure(
            fg_color="#0F172A"
        )


        self.criar_interface()


        self.camera = cv2.VideoCapture(0)


        if not self.camera.isOpened():

            print("Erro: câmera não encontrada")


        self.atualizar_camera()



    def criar_interface(self):


        topo = ctk.CTkFrame(
            self,
            height=70,
            fg_color="#1E293B"
        )

        topo.pack(
            fill="x"
        )


        titulo = ctk.CTkLabel(
            topo,
            text="🤟 LibrAI",
            font=("Segoe UI",28,"bold")
        )


        titulo.pack(
            side="left",
            padx=25,
            pady=15
        )


        status = ctk.CTkLabel(
            topo,
            text="🟢 Online",
            font=("Segoe UI",18)
        )


        status.pack(
            side="right",
            padx=25
        )



        corpo = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )


        corpo.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20
        )



        # CAMERA

        self.video = ctk.CTkFrame(
            corpo,
            fg_color="#1E293B",
            corner_radius=15
        )


        self.video.pack(
            side="left",
            fill="both",
            expand=True
        )



        self.lbl_video = ctk.CTkLabel(
            self.video,
            text=""
        )


        self.lbl_video.pack(
            expand=True,
            fill="both",
            padx=10,
            pady=10
        )



        # LATERAL


        lateral = ctk.CTkFrame(
            corpo,
            width=360,
            fg_color="#1E293B",
            corner_radius=15
        )


        lateral.pack(
            side="right",
            fill="y",
            padx=(20,0)
        )



        ctk.CTkLabel(
            lateral,
            text="Palavra",
            font=("Segoe UI",20,"bold")
        ).pack(
            anchor="w",
            padx=20,
            pady=(20,5)
        )



        self.lbl_palavra = ctk.CTkLabel(
            lateral,
            text="-",
            font=("Segoe UI",34)
        )


        self.lbl_palavra.pack(
            anchor="w",
            padx=20
        )



        ctk.CTkLabel(
            lateral,
            text="Frase",
            font=("Segoe UI",20,"bold")
        ).pack(
            anchor="w",
            padx=20,
            pady=(40,5)
        )



        self.lbl_frase = ctk.CTkTextbox(
            lateral,
            width=300,
            height=180
        )


        self.lbl_frase.pack(
            padx=20
        )


        self.lbl_frase.insert(
            "0.0",
            "Aguardando..."
        )



        self.btn_audio = ctk.CTkButton(
            lateral,
            text="🔊 Reproduzir",
            command=self.reproduzir_audio
        )


        self.btn_audio.pack(
            fill="x",
            padx=20,
            pady=(30,10)
        )



        self.btn_limpar = ctk.CTkButton(
            lateral,
            text="🗑 Limpar",
            command=self.limpar_frase
        )


        self.btn_limpar.pack(
            fill="x",
            padx=20,
            pady=10
        )



        self.btn_treinar = ctk.CTkButton(
            lateral,
            text="🧠 Treinar IA"
        )


        self.btn_treinar.pack(
            fill="x",
            padx=20,
            pady=10
        )



    def reproduzir_audio(self):

        texto = self.lbl_frase.get(
            "0.0",
            "end"
        )


        if texto.strip():

            falar(texto)



    def limpar_frase(self):

        self.frase = []

        self.ultimo_sinal = None


        self.lbl_palavra.configure(
            text="-"
        )


        self.lbl_frase.delete(
            "0.0",
            "end"
        )



    def atualizar_camera(self):


        sucesso, frame = self.camera.read()



        if sucesso:


            resultado = detectar_mao(frame)



            frame = desenhar_mao(
                frame,
                resultado
            )



            pontos = pegar_pontos(resultado)



            if pontos:


                sinal = reconhecer(
                    pontos,
                    self.dados
                )



                if sinal:


                    palavra = sinal.upper()


                    print(
                        "Reconhecido:",
                        palavra
                    )



                    self.lbl_palavra.configure(
                        text=palavra
                    )



                    if self.ultimo_sinal != palavra:


                        self.frase.append(
                            palavra
                        )


                        self.ultimo_sinal = palavra



                        texto = " ".join(
                            self.frase
                        )



                        self.lbl_frase.delete(
                            "0.0",
                            "end"
                        )



                        self.lbl_frase.insert(
                            "0.0",
                            texto
                        )



            else:

                self.ultimo_sinal = None




            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )



            imagem = Image.fromarray(
                frame_rgb
            )



            imagem = imagem.resize(
                (700,500)
            )



            imagem = ImageTk.PhotoImage(
                imagem
            )



            self.lbl_video.configure(
                image=imagem
            )


            self.lbl_video.image = imagem




        self.after(
            30,
            self.atualizar_camera
        )




if __name__ == "__main__":

    dados = carregar_dataset()

    app = LibrAI(dados)

    app.mainloop()