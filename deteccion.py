from PyQt5 import QtCore,QtWidgets,QtGui
import sys

''' 
Facial Landmark Detection in Python with OpenCV

Detection from web cam
'''

# Import Packages
import cv2
import os
import urllib.request as urlreq
import numpy as np


os.listdir()

# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"
haarcascade_clf = "data/" + haarcascade

# check if data folder is in working directory
if (os.path.isdir('data')):
    # check if haarcascade is in data directory
    if (haarcascade in os.listdir('data')):
        print("File exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
        print("File downloaded")
else:
    # create data folder in current directory
    os.mkdir('data')
    # download haarcascade to data folder
    urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
    print("File downloaded")

# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier(haarcascade_clf)

# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "LFBmodel.yaml"
LBFmodel_file = "data/" + LBFmodel

# check if data folder is in working directory
if (os.path.isdir('data')):
    # check if Landmark detection model is in data directory
    if (LBFmodel in os.listdir('data')):
        print("File exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
        print("File downloaded")
else:
    # create data folder in current directory
    os.mkdir('data')
    # download Landmark detection model to data folder
    urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
    print("File downloaded")

# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel_file)

# get image from webcam

sujetos = {}
archivos = []

""" 
for archivo in os.listdir():
    if(archivo.endswith(".jpeg") or archivo.endswith(".jpg")):
        persona = archivo.split(" ")[0]
        if(persona not in sujetos):
            sujetos[persona] = {}
        sujetos[persona][archivo] = []
 """
class App:
    def __init__(self):
        for archivo in os.listdir():
            if(archivo.endswith(".jpeg") or archivo.endswith(".jpg")):
                archivos.append(archivo)
        #CREACION DE INTERFAZ

        app = QtWidgets.QApplication([])
        self.ventana = QtWidgets.QFrame()
        self.ventana.setWindowTitle("Aplicacion")
        self.ventana.setFixedSize(700,500)
        self.ventana.setStyleSheet("background-color:white")
        ventanaLayout = QtWidgets.QGridLayout()


        lista = QtWidgets.QListWidget()
        lista.setStyleSheet("font-size:16px;background-color:#EFEFEF")
        lista.addItems(archivos)
        lista.setFixedWidth(200)
        lista.itemClicked.connect(self.itemChanged)
        ventanaLayout.addWidget(lista,0,0)

        panelVista = QtWidgets.QWidget()
        panelVista.setStyleSheet("background-color:#EFEFEF")
        panelVistaLayout = QtWidgets.QGridLayout()
        panelVista.setLayout(panelVistaLayout)
        ventanaLayout.addWidget(panelVista,0,1)

        self.imgLabel = QtWidgets.QLabel()
        img = QtGui.QPixmap("Harry 1.jpeg").scaled(300,300,aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.imgLabel.setPixmap(img)
        panelVistaLayout.addWidget(self.imgLabel,0,0,QtCore.Qt.AlignmentFlag.AlignCenter)
        
        botonClasificar = QtWidgets.QPushButton("Clasificar")
        botonClasificar.clicked.connect(self.clasificar)
        botonClasificar.setStyleSheet("background-color:#CCCCCC")
        panelVistaLayout.addWidget(botonClasificar,1,0)
        
        botonAdivinar = QtWidgets.QPushButton("Adivinar")
        botonAdivinar.clicked.connect(self.adivinar)
        botonAdivinar.setStyleSheet("background-color:#CCCCCC")
        panelVistaLayout.addWidget(botonAdivinar,2,0)
        
        
        self.ventana.setLayout(ventanaLayout)
        self.ventana.show()
        sys.exit(app.exec())
    
    def itemChanged(self,item:QtWidgets.QListWidgetItem):
        img = QtGui.QPixmap(item.text()).scaled(300,300,aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.imgLabel.setPixmap(img)
        self.imgLabel.repaint()
    
    def clasificar(self):
        text,ok = QtWidgets.QInputDialog().getText(self.ventana,"Clasificacion","como se llama esta persona ?")
        if text and ok:
            print(text)
    
    def adivinar(self):
        print("Es una Persona! creo...")
    
        
App()

# read webcam
frame = cv2.imread("Harry 1.jpeg")

# convert frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces using the haarcascade classifier on the "grayscale image"
faces = detector.detectMultiScale(gray)
    

for (x,y,w,d) in faces:
    # Detect landmarks on "gray"
    _, landmarks = landmark_detector.fit(gray, np.array(faces))

    for landmark in landmarks:
        for x,y in landmark[0]:
            # display landmarks on "frame/image,"
            # with blue colour in BGR and thickness 2
            cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), 2)
            


# save last instance of detected image
cv2.imwrite('result/face-detect.jpg', frame)    
    
# Show image
cv2.imshow("frame", frame)

# terminate the capture window
while(True):
    if cv2.waitKey(20) & 0xFF  == ord('q'):
        cv2.destroyAllWindows()