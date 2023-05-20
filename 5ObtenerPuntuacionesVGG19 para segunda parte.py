# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:03:55 2023

@author: ricar
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:11:08 2023

@author: ricar
"""
#from tensorflow.keras.applications import ResNet50

from tensorflow.keras.utils import to_categorical 

from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np

modelo = keras.models.load_model('D:/VIU/TFM/Codigo/TFM/1D/Difference/40ModeloVGG19D4.tf')



datos1d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/ATTM/Imagenes/40/ImagenesEnDatos.npy')


datos2d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/CTRW/Imagenes/40/ImagenesEnDatos.npy')
datos3d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/FBM/Imagenes/40.2/ImagenesEnDatos.npy')
datos4d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/LW/Imagenes/40/ImagenesEnDatos.npy')
datos5d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/SBM/Imagenes/40/ImagenesEnDatos.npy')

x_train1d, x_test1d = datos1d[:8000,:], datos1d[8000:,:]


x_train2d, x_test2d = datos2d[:8000,:], datos2d[8000:,:]


x_train3d, x_test3d = datos3d[:8000,:], datos3d[8000:,:]


x_train4d, x_test4d = datos4d[:8000,:], datos4d[8000:,:]


x_train5d, x_test5d = datos5d[:8000,:], datos5d[8000:,:]





datos1s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/ATTM/Imagenes/40/ImagenesEnDatos.npy')


datos2s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/CTRW/Imagenes/40/ImagenesEnDatos.npy')
datos3s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/FBM/Imagenes/40.2/ImagenesEnDatos.npy')
datos4s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/LW/Imagenes/40/ImagenesEnDatos.npy')
datos5s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/SBM/Imagenes/40/ImagenesEnDatos.npy')

x_train1s, x_test1s = datos1s[:8000,:], datos1s[8000:,:]
x_train2s, x_test2s = datos2s[:8000,:], datos2s[8000:,:]
x_train3s, x_test3s = datos3s[:8000,:], datos3s[8000:,:]
x_train4s, x_test4s = datos4s[:8000,:], datos4s[8000:,:]
x_train5s, x_test5s = datos5s[:8000,:], datos5s[8000:,:]

datos1m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/ATTM/Imagenes/40/ImagenesEnDatos.npy')


datos2m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/CTRW/Imagenes/40/ImagenesEnDatos.npy')
datos3m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/FBM/Imagenes/40/ImagenesEnDatos.npy')
datos4m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/LW/Imagenes/40/ImagenesEnDatos.npy')
datos5m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/SBM/Imagenes/40/ImagenesEnDatos.npy')

x_train1m, x_test1m = datos1m[:8000,:], datos1m[8000:,:]
x_train2m, x_test2m = datos2m[:8000,:], datos2m[8000:,:]
x_train3m, x_test3m = datos3m[:8000,:], datos3m[8000:,:]
x_train4m, x_test4m = datos4m[:8000,:], datos4m[8000:,:]
x_train5m, x_test5m = datos5m[:8000,:], datos5m[8000:,:]





x_train = np.concatenate((x_train1d,x_train2d,x_train3d,x_train4d,x_train5d)) 
x_test = np.concatenate((x_test1d,x_test2d,x_test3d,x_test4d,x_test5d))
#print(x_train[1])
#arr = np.concatenate((x_train1,x_train2))



y_train1d, y_test1d = [[0 for _ in range(1)] for _ in range(8000)],[[0 for _ in range(1)] for _ in range(2000)]
#y_train1s, y_test1s = [[1 for _ in range(1)] for _ in range(8000)],[[1 for _ in range(1)] for _ in range(2000)]

y_train2d, y_test2d = [[1 for _ in range(1)] for _ in range(8000)],[[1 for _ in range(1)] for _ in range(2000)]
#y_train2s, y_test2s = [[3 for _ in range(1)] for _ in range(8000)],[[3 for _ in range(1)] for _ in range(2000)]

y_train3d, y_test3d = [[2 for _ in range(1)] for _ in range(8000)],[[2 for _ in range(1)] for _ in range(2000)]
#y_train3s, y_test3s = [[5 for _ in range(1)] for _ in range(8000)],[[5 for _ in range(1)] for _ in range(2000)]

y_train4d, y_test4d = [[3 for _ in range(1)] for _ in range(8000)],[[3 for _ in range(1)] for _ in range(2000)]
#y_train4s, y_test4s = [[7 for _ in range(1)] for _ in range(8000)],[[7 for _ in range(1)] for _ in range(2000)]

y_train5d, y_test5d = [[4 for _ in range(1)] for _ in range(8000)],[[4 for _ in range(1)] for _ in range(2000)]
#y_train5s, y_test5s = [[9 for _ in range(1)] for _ in range(8000)],[[9 for _ in range(1)] for _ in range(2000)]

y_train = np.concatenate((y_train1d,y_train2d,y_train3d,y_train4d,y_train5d))
y_test = np.concatenate((y_test1d,y_test2d,y_test3d,y_test4d,y_test5d))
from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)


from sklearn import model_selection as ms
from sklearn.metrics import confusion_matrix
xtrain,xval,ytrain,yval= ms.train_test_split(x_train,y_train,test_size=.3)






ytrain= to_categorical(ytrain)
yval= to_categorical(yval)
y_test= to_categorical(y_test)
"""
f,ax=plt.subplots(2,1) #Creates 2 subplots under 1 column

#Assigning the first subplot to graph training loss and validation loss
ax[0].plot(modelo.history.history['loss'],color='b',label='Training Loss')
ax[0].plot(modelo.history.history['val_loss'],color='r',label='Validation Loss')
ax[0].legend()
#Plotting the training accuracy and validation accuracy
ax[1].plot(modelo.history.history['accuracy'],color='b',label='Training  Accuracy')
ax[1].plot(modelo.history.history['val_accuracy'],color='r',label='Validation Accuracy')
ax[1].legend()
"""

#Making prediction
y_predD=modelo.predict(x_test)
y_predD=np.argmax(y_predD,axis=1)

y_verdadero=np.argmax(y_test,axis=1)

#Defining function for confusion matrix plot
def dibujar_matriz_confusion(y_verdadero, y_pred, clases,
                          normalizar=False,
                          titulo=None,
                          cmap=plt.cm.Blues):


    if not titulo:
        if normalizar:
            titulo = 'Normalized confusion matrix'
        else:
            titulo = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_verdadero, y_pred)
    if normalizar:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=clases, yticklabels=clases,
           title=titulo,
           ylabel='Etiqueta verdadera',
           xlabel='Etiqueta predicha')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
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

#Plotting the confusion matrix
confusion_mtx=confusion_matrix(y_verdadero,y_predD)

nombres_clases=['diffATTM','diffCTRW','diffFBM','diffLW','diffSBM']

# Plotting non-normalized confusion matrix
dibujar_matriz_confusion(y_verdadero, y_predD, clases=nombres_clases,
                      titulo='Matríz de confusión de 10VGG19D sin normalizar')
dibujar_matriz_confusion(y_verdadero, y_predD, clases=nombres_clases,normalizar = True,
                      titulo='Matríz de confusión de 10VGG19D normalizada')
from sklearn.metrics import accuracy_score
VGG19_acc = accuracy_score(y_verdadero, y_predD)
print('Puntuación de acierto de 10VGG19D = ', VGG19_acc)
from sklearn.metrics import f1_score
VGG19_F1 = f1_score(y_verdadero, y_predD, average = 'macro')
print('Puntuación de F1 de 10VGG19D = ', VGG19_F1)



x_train = np.concatenate((x_train1s,x_train2s,x_train3s,x_train4s,x_train5s)) 
x_test = np.concatenate((x_test1s,x_test2s,x_test3s,x_test4s,x_test5s))


y_train = np.concatenate((y_train1d,y_train2d,y_train3d,y_train4d,y_train5d))
y_test = np.concatenate((y_test1d,y_test2d,y_test3d,y_test4d,y_test5d))

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)

xtrain,xval,ytrain,yval= ms.train_test_split(x_train,y_train,test_size=.3)

ytrain= to_categorical(ytrain)
yval= to_categorical(yval)
y_test= to_categorical(y_test)


"""
"""

model = keras.models.load_model('D:/VIU/TFM/Codigo/TFM/1D/Summation/40ModeloVGG19S4.tf')

#Plotting the training and validation loss




#Making prediction
y_predS=model.predict(x_test)
y_predS=np.argmax(y_predS,axis=1)
y_verdadero=np.argmax(y_test,axis=1)
#Plotting the confusion matrix
confusion_mtx=confusion_matrix(y_verdadero,y_predS)
nombres_clases=['summATTM','summCTRW','summFBM','summLW','summSBM']
# Plotting non-normalized confusion matrix
dibujar_matriz_confusion(y_verdadero, y_predS, clases=nombres_clases,
                      titulo='Matríz de confusión de 10VGG19S sin normalizar')
dibujar_matriz_confusion(y_verdadero, y_predS, clases=nombres_clases, normalizar = True,
                      titulo='Matríz de confusión de 10VGG19S normalizada')
from sklearn.metrics import accuracy_score
VGG19_acc = accuracy_score(y_verdadero, y_predS)
print('Puntuación de acierto de 10VGG19S = ', VGG19_acc)
from sklearn.metrics import f1_score
VGG19_F1 = f1_score(y_verdadero, y_predS, average = 'macro')
print('Puntuación de F1 de 10VGG19S = ', VGG19_F1)




x_train = np.concatenate((x_train1m,x_train2m,x_train3m,x_train4m,x_train5m)) 
x_test = np.concatenate((x_test1m,x_test2m,x_test3m,x_test4m,x_test5m))


y_train = np.concatenate((y_train1d,y_train2d,y_train3d,y_train4d,y_train5d))
y_test = np.concatenate((y_test1d,y_test2d,y_test3d,y_test4d,y_test5d))

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)

xtrain,xval,ytrain,yval= ms.train_test_split(x_train,y_train,test_size=.3)

ytrain= to_categorical(ytrain)
yval= to_categorical(yval)
y_test= to_categorical(y_test)

""""""
"""

"""
model = keras.models.load_model('D:/VIU/TFM/Codigo/TFM/1D/Markov/40ModeloVGG19M4.tf')

#Plotting the training and validation loss




#Making prediction
y_predM=model.predict(x_test)
y_predM=np.argmax(y_predM,axis=1)
y_verdadero=np.argmax(y_test,axis=1)
#Plotting the confusion matrix
confusion_mtx=confusion_matrix(y_verdadero,y_predM)
nombres_clases=['markovATTM','markovCTRW','markovFBM','markovLW','markovSBM']
# Plotting non-normalized confusion matrix
dibujar_matriz_confusion(y_verdadero, y_predM, clases=nombres_clases,
                      titulo='Matríz de confusión de 10VGG19M sin normalizar')
dibujar_matriz_confusion(y_verdadero, y_predM, clases=nombres_clases, normalizar = True,
                      titulo='Matríz de confusión de 10VGG19M normalizada')
from sklearn.metrics import accuracy_score
VGG19_acc = accuracy_score(y_verdadero, y_predM)
print('Puntuación de acierto de 10VGG19M = ', VGG19_acc)
from sklearn.metrics import f1_score
VGG19_F1 = f1_score(y_verdadero, y_predM, average = 'macro')
print('Puntuación de F1 de 10VGG19M = ', VGG19_F1)

