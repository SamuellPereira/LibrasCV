import cv2
import os


video = r"D:\Desktop\libras ia\homologacao\temp\irma.mp4"


print("Existe?", os.path.exists(video))


cap = cv2.VideoCapture(video)


print("Abriu?", cap.isOpened())


frames = 0

while True:

    sucesso, frame = cap.read()

    if not sucesso:
        break

    frames += 1


cap.release()


print("Frames encontrados:", frames)