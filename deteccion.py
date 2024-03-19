from PyQt5 import QtCore,QtWidgets,QtGui
import sys
import math

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


        self.lista = QtWidgets.QListWidget()
        self.lista.setStyleSheet("font-size:16px;background-color:#EFEFEF")
        self.lista.addItems(archivos)
        self.lista.setFixedWidth(200)
        self.lista.itemClicked.connect(self.itemChanged)
        self.lista.setCurrentRow(0)
        ventanaLayout.addWidget(self.lista,0,0)

        panelVista = QtWidgets.QWidget()
        panelVista.setStyleSheet("background-color:#EFEFEF")
        panelVistaLayout = QtWidgets.QGridLayout()
        panelVista.setLayout(panelVistaLayout)
        ventanaLayout.addWidget(panelVista,0,1)

        self.imgLabel = QtWidgets.QLabel()
        img = QtGui.QPixmap(archivos[0]).scaled(300,300,aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio)
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
            if text not in sujetos:
                sujetos[text] = []
                
            if len(sujetos[text]) == 2:
                mensaje = QtWidgets.QMessageBox(self.ventana)
                mensaje.setText("el sujeto ya tiene dos imagenes guia")
                mensaje.exec_()
                return
            
            frame = cv2.imread(self.lista.currentItem().text())

            # convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using the haarcascade classifier on the "grayscale image"
            faces = detector.detectMultiScale(gray)
                
            if(len(faces) == 0):
                mensaje = QtWidgets.QMessageBox(self.ventana)
                mensaje.setText("No se encuentran caras en la img")
                mensaje.exec_()
                return

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
            sujetos[text].append(landmarks)
            self.lista.takeItem(self.lista.currentRow())
            self.itemChanged(self.lista.currentItem())
                
    
    def adivinar(self):
        #obtener alpha y beta para la img
        similitud = {}
        for sujeto in sujetos:
            #verifica que el sujeto tiene suficientes imagenes prueba
            if len(sujetos[sujeto]) != 2:
                continue
            suma = 0
            img1 = sujetos[sujeto][0][0][0]
            img2 = sujetos[sujeto][1][0][0]
            
            print(img1)
            xp1i1 = img1[0][0]
            xp2i1 = img1[5][0] 
            xp1i2 = img2[0][0]
            xp2i2 = img2[5][0]
            
            #obtiene los puntos de la imagen siendo analizada
            frame = cv2.imread(self.lista.currentItem().text())
            # convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces using the haarcascade classifier on the "grayscale image"
            faces = detector.detectMultiScale(gray)
                
            if(len(faces) == 0):
                mensaje = QtWidgets.QMessageBox(self.ventana)
                mensaje.setText("No se encuentran caras en la img")
                mensaje.exec_()
                return
                
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
            img0 = landmarks[0][0]
            xp1i0 = img0[0][0]
            xp2i0 = img0[5][0]
            #obtiene los componentes alpha y beta
            alpha = ((xp1i0*xp2i2) - (xp2i0*xp1i2))/((xp1i1*xp2i2)-(xp2i1*xp1i2))
            beta = ((xp2i0*xp1i1)-(xp1i0*xp2i1))/((xp1i1*xp2i2)-(xp2i1*xp1i2))
            
            
            for punto in range(len(img0)):
                puntoEstimado = alpha*img1[punto][0] + beta*img2[punto][0]
                parecido = (1 - math.fabs((img0[punto][0]-puntoEstimado)/img0[punto][0]))
                suma = suma + parecido
            similitud[sujeto] = suma/len(img0)
            
            
        if len(similitud) == 0:
            mensaje = QtWidgets.QMessageBox(self.ventana)
            mensaje.setText("No se tiene suficientes datos para adivinar")
            mensaje.exec_()
            return
        
        print(similitud)
        persona = max(similitud,key=similitud.get)
        if(similitud[persona] < 0.89):
            mensaje = QtWidgets.QMessageBox(self.ventana)
            mensaje.setText("No se ha reconocido a nadie")
            mensaje.exec_()
            return
        else:
            mensaje = QtWidgets.QMessageBox(self.ventana)
            mensaje.setText("Esta persona es "+persona+" ! con una confianza del "+"{:5.2f}".format(similitud[persona]*100)+"%")
            mensaje.exec_()
            return
            
    
        
App()
