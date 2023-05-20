
#Importamos todas las librerías necesarias
from tensorflow.keras.utils import to_categorical 
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np
from sklearn.utils import shuffle
from sklearn import model_selection as ms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#Cargamos el modelo y los datos de test para obtener las métricas.
modelo = keras.models.load_model('D:/VIU/TFM/Codigo/TFM/1D/Difference/10ModeloMobileNetD.tf')


datos1d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/ATTM/Imagenes/10/ImagenesEnDatos.npy')
datos2d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/CTRW/Imagenes/10/ImagenesEnDatos.npy')
datos3d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/FBM/Imagenes/10.2/ImagenesEnDatos.npy')
datos4d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/LW/Imagenes/10/ImagenesEnDatos.npy')
datos5d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/SBM/Imagenes/10/ImagenesEnDatos.npy')

x_test1d = datos1d[8000:,:]
x_test2d = datos2d[8000:,:]
x_test3d = datos3d[8000:,:]
x_test4d = datos4d[8000:,:]
x_test5d = datos5d[8000:,:]





datos1s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/ATTM/Imagenes/10/ImagenesEnDatos.npy')


datos2s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/CTRW/Imagenes/10/ImagenesEnDatos.npy')
datos3s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/FBM/Imagenes/10.2/ImagenesEnDatos.npy')
datos4s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/LW/Imagenes/10/ImagenesEnDatos.npy')
datos5s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/SBM/Imagenes/10/ImagenesEnDatos.npy')

x_test1s = datos1s[8000:,:]
x_test2s = datos2s[8000:,:]
x_test3s = datos3s[8000:,:]
x_test4s = datos4s[8000:,:]
x_test5s = datos5s[8000:,:]

datos1m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/ATTM/Imagenes/10/ImagenesEnDatos.npy')


datos2m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/CTRW/Imagenes/10/ImagenesEnDatos.npy')
datos3m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/FBM/Imagenes/10/ImagenesEnDatos.npy')
datos4m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/LW/Imagenes/10/ImagenesEnDatos.npy')
datos5m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/SBM/Imagenes/10/ImagenesEnDatos.npy')

x_test1m = datos1m[8000:,:]
x_test2m = datos2m[8000:,:]
x_test3m = datos3m[8000:,:]
x_test4m = datos4m[8000:,:]
x_test5m = datos5m[8000:,:]


x_test = np.concatenate((x_test1d,x_test2d,x_test3d,x_test4d,x_test5d))


y_test1d = [[0 for _ in range(1)] for _ in range(2000)]
y_test2d = [[1 for _ in range(1)] for _ in range(2000)]
y_test3d = [[2 for _ in range(1)] for _ in range(2000)]
y_test4d = [[3 for _ in range(1)] for _ in range(2000)]
y_test5d = [[4 for _ in range(1)] for _ in range(2000)]


y_test = np.concatenate((y_test1d,y_test2d,y_test3d,y_test4d,y_test5d))


x_test, y_test = shuffle(x_test, y_test)


y_test= to_categorical(y_test)


#Generamos la predicción
y_predD=modelo.predict(x_test)
y_predD=np.argmax(y_predD,axis=1)

y_verdadero=np.argmax(y_test,axis=1)

#Definimos la matriz de confusión
def dibujar_matriz_confusion(y_verdadero, y_pred, clases,
                          normalizar=False,
                          titulo=None,
                          cmap=plt.cm.Blues):


    if not titulo:
        if normalizar:
            titulo = 'Matriz de confusión normalizada'
        else:
            titulo = 'Matriz de confusión sin normalizar'

    # Generamos la matriz
    cm = confusion_matrix(y_verdadero, y_pred)
    if normalizar:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión sin normalizar')

    #Dibujamos la matriz
    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=clases, yticklabels=clases,
           title=titulo,
           ylabel='Etiqueta verdadera',
           xlabel='Etiqueta predicha')


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.2f' if normalizar else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)

#Dibujamos la matriz de confusión
confusion_mtx=confusion_matrix(y_verdadero,y_predD)

#Proporcionamos los nombre de las clases.
nombres_clases=['diffATTM','diffCTRW','diffFBM','diffLW','diffSBM']

# Dibujamos las matrices
dibujar_matriz_confusion(y_verdadero, y_predD, clases=nombres_clases,
                      titulo='Matríz de confusión de 10MobileNetD sin normalizar')
dibujar_matriz_confusion(y_verdadero, y_predD, clases=nombres_clases,normalizar = True,
                      titulo='Matríz de confusión de 10MobileNetD normalizada')

#Generamos las puntuaciones.
MobileNet_acc = accuracy_score(y_verdadero, y_predD)
print('Puntuación de acierto de 10MobileNetD = ', MobileNet_acc)

MobileNet_F1 = f1_score(y_verdadero, y_predD, average = 'macro')
print('Puntuación de F1 de 10MobileNetD = ', MobileNet_F1)



#Repetimos el proceso para summation

x_test = np.concatenate((x_test1s,x_test2s,x_test3s,x_test4s,x_test5s))

y_test = np.concatenate((y_test1d,y_test2d,y_test3d,y_test4d,y_test5d))

x_test, y_test = shuffle(x_test, y_test)

y_test= to_categorical(y_test)


model = keras.models.load_model('D:/VIU/TFM/Codigo/TFM/1D/Summation/10ModeloMobileNetS.tf')


y_predS=model.predict(x_test)
y_predS=np.argmax(y_predS,axis=1)
y_verdadero=np.argmax(y_test,axis=1)

confusion_mtx=confusion_matrix(y_verdadero,y_predS)
nombres_clases=['summATTM','summCTRW','summFBM','summLW','summSBM']


dibujar_matriz_confusion(y_verdadero, y_predS, clases=nombres_clases,
                      titulo='Matríz de confusión de 10MobileNetS sin normalizar')
dibujar_matriz_confusion(y_verdadero, y_predS, clases=nombres_clases, normalizar = True,
                      titulo='Matríz de confusión de 10MobileNetS normalizada')

MobileNet_acc = accuracy_score(y_verdadero, y_predS)
print('Puntuación de acierto de 10MobileNetS = ', MobileNet_acc)

MobileNet_F1 = f1_score(y_verdadero, y_predS, average = 'macro')
print('Puntuación de F1 de 10MobileNetS = ', MobileNet_F1)





#Repetimos el proceso para Markov.
x_test = np.concatenate((x_test1m,x_test2m,x_test3m,x_test4m,x_test5m))

y_test = np.concatenate((y_test1d,y_test2d,y_test3d,y_test4d,y_test5d))

x_test, y_test = shuffle(x_test, y_test)

y_test= to_categorical(y_test)



model = keras.models.load_model('D:/VIU/TFM/Codigo/TFM/1D/Markov/10ModeloMobileNetM.tf')





y_predM=model.predict(x_test)
y_predM=np.argmax(y_predM,axis=1)
y_verdadero=np.argmax(y_test,axis=1)

confusion_mtx=confusion_matrix(y_verdadero,y_predM)
nombres_clases=['markovATTM','markovCTRW','markovFBM','markovLW','markovSBM']

dibujar_matriz_confusion(y_verdadero, y_predM, clases=nombres_clases,
                      titulo='Matríz de confusión de 10MobileNetM sin normalizar')
dibujar_matriz_confusion(y_verdadero, y_predM, clases=nombres_clases, normalizar = True,
                      titulo='Matríz de confusión de 10MobileNetM normalizada')

MobileNet_acc = accuracy_score(y_verdadero, y_predM)
print('Puntuación de acierto de 10MobileNetM = ', MobileNet_acc)

MobileNet_F1 = f1_score(y_verdadero, y_predM, average = 'macro')
print('Puntuación de F1 de 10MobileNetM = ', MobileNet_F1)


"""
Con esto, generamos las puntuaciones para los modelos de 1 longitud de trayectoria de MobileNet. Nos faltan 4 longitudes de MobileNet.
Además, para generar las puntuaciones de los modelos de VGG19, solo hay que cargar dichos modelos en vez de los de MobileNet, por lo que
el script es idéntico.
"""