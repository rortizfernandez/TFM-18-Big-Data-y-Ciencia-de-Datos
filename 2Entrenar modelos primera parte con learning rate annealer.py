#Importamos todas las librerías necesarias
import numpy as np
from sklearn.utils import shuffle
from sklearn import model_selection as ms
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.callbacks import ReduceLROnPlateau 
from tensorflow.keras.applications import VGG19,MobileNet
from tensorflow.keras.layers import Flatten, Dense, Dropout 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Sequential



#Cargamos los datos de la codificación diference para los distintos modelos de una longitud de trayectoria. Podemos ver su forma.
datos1d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/ATTM/Imagenes/10/ImagenesEnDatos.npy')
print(datos1d.shape)

datos2d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/CTRW/Imagenes/10/ImagenesEnDatos.npy')
datos3d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/FBM/Imagenes/10.2/ImagenesEnDatos.npy')
datos4d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/LW/Imagenes/10/ImagenesEnDatos.npy')
datos5d = np.load('D:/VIU/TFM/Codigo/TFM/1D/Difference/SBM/Imagenes/10/ImagenesEnDatos.npy')


#Dividimos para cada uno de los conjuntos, en entrenamiento y test.
x_train1d, x_test1d = datos1d[:8000,:], datos1d[8000:,:]
x_train2d, x_test2d = datos2d[:8000,:], datos2d[8000:,:]
x_train3d, x_test3d = datos3d[:8000,:], datos3d[8000:,:]
x_train4d, x_test4d = datos4d[:8000,:], datos4d[8000:,:]
x_train5d, x_test5d = datos5d[:8000,:], datos5d[8000:,:]




#Cargamos los datos de la codificación summation para los distintos modelos de una longitud de trayectoria. Podemos ver su forma.
datos1s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/ATTM/Imagenes/10/ImagenesEnDatos.npy')
print(datos1s.shape)

datos2s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/CTRW/Imagenes/10/ImagenesEnDatos.npy')
datos3s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/FBM/Imagenes/10.2/ImagenesEnDatos.npy')
datos4s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/LW/Imagenes/10/ImagenesEnDatos.npy')
datos5s = np.load('D:/VIU/TFM/Codigo/TFM/1D/Summation/SBM/Imagenes/10/ImagenesEnDatos.npy')

#Dividimos para cada uno de los conjuntos, en entrenamiento y test.
x_train1s, x_test1s = datos1s[:8000,:], datos1s[8000:,:]
x_train2s, x_test2s = datos2s[:8000,:], datos2s[8000:,:]
x_train3s, x_test3s = datos3s[:8000,:], datos3s[8000:,:]
x_train4s, x_test4s = datos4s[:8000,:], datos4s[8000:,:]
x_train5s, x_test5s = datos5s[:8000,:], datos5s[8000:,:]


#Cargamos los datos de la codificación markov para los distintos modelos de una longitud de trayectoria. Podemos ver su forma.
datos1m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/ATTM/Imagenes/10/ImagenesEnDatos.npy')
print(datos1m.shape)

datos2m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/CTRW/Imagenes/10/ImagenesEnDatos.npy')
datos3m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/FBM/Imagenes/10/ImagenesEnDatos.npy')
datos4m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/LW/Imagenes/10/ImagenesEnDatos.npy')
datos5m = np.load('D:/VIU/TFM/Codigo/TFM/1D/Markov/SBM/Imagenes/10/ImagenesEnDatos.npy')

#Dividimos para cada uno de los conjuntos, en entrenamiento y test.
x_train1m, x_test1m = datos1m[:8000,:], datos1m[8000:,:]
x_train2m, x_test2m = datos2m[:8000,:], datos2m[8000:,:]
x_train3m, x_test3m = datos3m[:8000,:], datos3m[8000:,:]
x_train4m, x_test4m = datos4m[:8000,:], datos4m[8000:,:]
x_train5m, x_test5m = datos5m[:8000,:], datos5m[8000:,:]





#Primero entrenaremos los modelos de difference, por lo que concatenamos los datos de entrenamiento y test.

x_train = np.concatenate((x_train1d,x_train2d,x_train3d,x_train4d,x_train5d)) 
x_test = np.concatenate((x_test1d,x_test2d,x_test3d,x_test4d,x_test5d))


"""
Creamos las variables que nos proporcionan la información de la clase a la que pertenece cada trayectoria.
Las 8000 primeras de train y las 2000 primeras de test son ATTM, las siguientes CTRW y así sucesivamente.
"""

y_train1d, y_test1d = [[0 for _ in range(1)] for _ in range(8000)],[[0 for _ in range(1)] for _ in range(2000)]

y_train2d, y_test2d = [[1 for _ in range(1)] for _ in range(8000)],[[1 for _ in range(1)] for _ in range(2000)]
y_train3d, y_test3d = [[2 for _ in range(1)] for _ in range(8000)],[[2 for _ in range(1)] for _ in range(2000)]
y_train4d, y_test4d = [[3 for _ in range(1)] for _ in range(8000)],[[3 for _ in range(1)] for _ in range(2000)]
y_train5d, y_test5d = [[4 for _ in range(1)] for _ in range(8000)],[[4 for _ in range(1)] for _ in range(2000)]

#Concatenamos.
y_train = np.concatenate((y_train1d,y_train2d,y_train3d,y_train4d,y_train5d))
y_test = np.concatenate((y_test1d,y_test2d,y_test3d,y_test4d,y_test5d))

"""
Distribuimos los datos mediante una función para barajarlos y que no esten todos los datos de la misma clase juntos. Aplicamos el mismo 
barajado a los datos de entrenamiento x e y para que mantengan su relación.
"""
x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)


#Dividimos los datos de entrenamiento en entrenamiento y validación con un 15% de validación y un 85% de entrenamiento.
xtrain,xval,ytrain,yval= ms.train_test_split(x_train,y_train,test_size=.3)

print(xtrain.shape,ytrain.shape)
print(xval.shape,yval.shape)
print(x_test.shape,y_test.shape)



#Convertimos en variables caterógiras las de clasificación, ya que no son continuas sino discretas.
ytrain= to_categorical(ytrain)
yval= to_categorical(yval)
y_test= to_categorical(y_test)
print(xtrain.shape,ytrain.shape)
print(xval.shape,yval.shape)
print(x_test.shape,y_test.shape)

#Esto son generadores de imagenes que aplican rotaciones y ampliaciones para generar nuevas imagenes a partir de las proporcionadas.
train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 )

val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 )

test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 ) 

#Ajustar el aumento definido anteriormente a los datos.
train_generator.fit(xtrain)
val_generator.fit(xval)
test_generator.fit(x_test)

#Learning Rate Annealer para reducir el lerning rate si no se aumenta la precisión
lrr= ReduceLROnPlateau(monitor='val_accuracy', factor=.1,  patience=3, min_lr=1e-5) 
#Definimos las variables.
batch_size= 100
epochs=15#30
learn_rate=.001

#Creamos nuestro modelo MobileNet y añadimos las capas comentadas en el trabajo.
base_model_mobile = MobileNet(include_top=False,weights='imagenet' ,input_shape=(36,36,3),classes=ytrain.shape[1])

model_mobileD=Sequential()
#Añadir las capas densas junto con las de activación y las normalizaciones de batch
model_mobileD.add(base_model_mobile)
model_mobileD.add(Flatten())

model_mobileD.add(Dense(1296,activation=('relu'),input_dim=648))
model_mobileD.add(BatchNormalization())
model_mobileD.add(Dense(648,activation=('relu'))) 
model_mobileD.add(Dropout(.4))#Añadimos capas de dropout para descartar pesos.
model_mobileD.add(BatchNormalization())
model_mobileD.add(Dense(324,activation=('relu'))) 
model_mobileD.add(Dropout(.3))
model_mobileD.add(BatchNormalization())
model_mobileD.add(Dense(162,activation=('relu')))
model_mobileD.add(Dropout(.2))
model_mobileD.add(Dense(5,activation=('softmax'))) #Capa de clasificación, número de clases.

#Mostramos un resumen del modelo
model_mobileD.summary()

#Creamos el compilador.
sgd=SGD(learning_rate=learn_rate,momentum=.9,nesterov=False)

#Compilar el modelo
model_mobileD.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

#Entrenamos el modelo pasandole todas las variables generadas anteriormente.
model_mobileD.fit(train_generator.flow(xtrain,ytrain,batch_size=batch_size),
      epochs=epochs, steps_per_epoch=xtrain.shape[0]//batch_size, 
      validation_data=val_generator.flow(xval,yval,batch_size=batch_size),validation_steps=120,callbacks=[lrr],verbose=1)


#Guardamos el modelo.
model_mobileD.save('D:/VIU/TFM/Codigo/TFM/1D/Difference/10ModeloMobileNetD.tf')






#Hacemos lo mismo para el modelo VGG19.
base_model_VGG = VGG19(include_top=False,weights='imagenet' ,input_shape=(36,36,3),classes=ytrain.shape[1])

model_VGGD=Sequential()

model_VGGD.add(base_model_VGG)
model_VGGD.add(Flatten())


model_VGGD.add(Dense(1296,activation=('relu'),input_dim=648))
model_VGGD.add(BatchNormalization())
model_VGGD.add(Dense(648,activation=('relu'))) 
model_VGGD.add(Dropout(.4))
model_VGGD.add(BatchNormalization())
model_VGGD.add(Dense(324,activation=('relu'))) 
model_VGGD.add(Dropout(.3))
model_VGGD.add(BatchNormalization())
model_VGGD.add(Dense(162,activation=('relu')))
model_VGGD.add(Dropout(.2))
model_VGGD.add(Dense(5,activation=('softmax'))) 

model_VGGD.summary()
sgd=SGD(learning_rate=learn_rate,momentum=.9,nesterov=False)
model_VGGD.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])


model_VGGD.fit(train_generator.flow(xtrain,ytrain,batch_size=batch_size),
      epochs=epochs, steps_per_epoch=xtrain.shape[0]//batch_size, 
      validation_data=val_generator.flow(xval,yval,batch_size=batch_size),validation_steps=120,callbacks=[lrr],verbose=1)

model_VGGD.save('D:/VIU/TFM/Codigo/TFM/1D/Difference/10ModeloVGG19D.tf')



























#Repetimos los pasos para los modelos de summation

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

print(xtrain.shape,ytrain.shape)
print(xval.shape,yval.shape)
print(x_test.shape,y_test.shape)


train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 )

val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 )

test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 ) 


train_generator.fit(xtrain)
val_generator.fit(xval)
test_generator.fit(x_test)


lrr= ReduceLROnPlateau(monitor='val_accuracy', factor=.1,  patience=3, min_lr=1e-5) 
sgd=SGD(learning_rate=learn_rate,momentum=.9,nesterov=False)

base_model_mobileS = MobileNet(include_top=False,weights='imagenet' ,input_shape=(36,36,3),classes=ytrain.shape[1])

model_mobileS=Sequential()

model_mobileS.add(base_model_mobileS)
model_mobileS.add(Flatten())


model_mobileS.add(Dense(1296,activation=('relu'),input_dim=648))
model_mobileS.add(BatchNormalization())
model_mobileS.add(Dense(648,activation=('relu'))) 
model_mobileS.add(Dropout(.4))
model_mobileS.add(BatchNormalization())
model_mobileS.add(Dense(324,activation=('relu'))) 
model_mobileS.add(Dropout(.3))
model_mobileS.add(BatchNormalization())
model_mobileS.add(Dense(162,activation=('relu')))
model_mobileS.add(Dropout(.2))
model_mobileS.add(Dense(5,activation=('softmax')))


model_mobileS.summary()




model_mobileS.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])



model_mobileS.fit(train_generator.flow(xtrain,ytrain,batch_size=batch_size),
      epochs=epochs, steps_per_epoch=xtrain.shape[0]//batch_size, 
      validation_data=val_generator.flow(xval,yval,batch_size=batch_size),validation_steps=120,callbacks=[lrr],verbose=1)



model_mobileS.save('D:/VIU/TFM/Codigo/TFM/1D/Summation/10ModeloMobileNetS.tf')










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

print(xtrain.shape,ytrain.shape)
print(xval.shape,yval.shape)
print(x_test.shape,y_test.shape)

train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 )

val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 )

test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 ) 


train_generator.fit(xtrain)
val_generator.fit(xval)
test_generator.fit(x_test)


lrr= ReduceLROnPlateau(monitor='val_accuracy', factor=.1,  patience=3, min_lr=1e-5) 
sgd=SGD(learning_rate=learn_rate,momentum=.9,nesterov=False)



base_model_VGG = VGG19(include_top=False,weights='imagenet' ,input_shape=(36,36,3),classes=ytrain.shape[1])

model_VGGS=Sequential()

model_VGGS.add(base_model_VGG)
model_VGGS.add(Flatten())


model_VGGS.add(Dense(1296,activation=('relu'),input_dim=648))
model_VGGS.add(BatchNormalization())
model_VGGS.add(Dense(648,activation=('relu'))) 
model_VGGS.add(Dropout(.4))
model_VGGS.add(BatchNormalization())
model_VGGS.add(Dense(324,activation=('relu'))) 
model_VGGS.add(Dropout(.3))
model_VGGS.add(BatchNormalization())
model_VGGS.add(Dense(162,activation=('relu')))
model_VGGS.add(Dropout(.2))
model_VGGS.add(Dense(5,activation=('softmax')))


model_VGGS.summary()

model_VGGS.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])



model_VGGS.fit(train_generator.flow(xtrain,ytrain,batch_size=batch_size),
      epochs=epochs, steps_per_epoch=xtrain.shape[0]//batch_size, 
      validation_data=val_generator.flow(xval,yval,batch_size=batch_size),validation_steps=120,callbacks=[lrr],verbose=1)

model_VGGS.save('D:/VIU/TFM/Codigo/TFM/1D/Summation/10ModeloVGG19S.tf')



















#Repetimos los pasos para los modelos de Markov.

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

print(xtrain.shape,ytrain.shape)
print(xval.shape,yval.shape)
print(x_test.shape,y_test.shape)

train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 )

val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 )

test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 ) 


train_generator.fit(xtrain)
val_generator.fit(xval)
test_generator.fit(x_test)


lrr= ReduceLROnPlateau(monitor='val_accuracy', factor=.1,  patience=3, min_lr=1e-5) 
sgd=SGD(learning_rate=learn_rate,momentum=.9,nesterov=False)

base_model_mobile = MobileNet(include_top=False,weights='imagenet' ,input_shape=(36,36,3),classes=ytrain.shape[1])

model_mobileM=Sequential()

model_mobileM.add(base_model_mobile)
model_mobileM.add(Flatten())


model_mobileM.add(Dense(1296,activation=('relu'),input_dim=648))
model_mobileM.add(BatchNormalization())
model_mobileM.add(Dense(648,activation=('relu'))) 
model_mobileM.add(Dropout(.4))
model_mobileM.add(BatchNormalization())
model_mobileM.add(Dense(324,activation=('relu'))) 
model_mobileM.add(Dropout(.3))
model_mobileM.add(BatchNormalization())
model_mobileM.add(Dense(162,activation=('relu')))
model_mobileM.add(Dropout(.2))
model_mobileM.add(Dense(5,activation=('softmax')))


model_mobileM.summary()


model_mobileM.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])


model_mobileM.fit(train_generator.flow(xtrain,ytrain,batch_size=batch_size),
      epochs=epochs, steps_per_epoch=xtrain.shape[0]//batch_size, 
      validation_data=val_generator.flow(xval,yval,batch_size=batch_size),validation_steps=120,callbacks=[lrr],verbose=1)



model_mobileM.save('D:/VIU/TFM/Codigo/TFM/1D/Markov/10ModeloMobileNetM.tf')






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

print(xtrain.shape,ytrain.shape)
print(xval.shape,yval.shape)
print(x_test.shape,y_test.shape)

train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 )

val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 )

test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1 ) 


train_generator.fit(xtrain)
val_generator.fit(xval)
test_generator.fit(x_test)


lrr= ReduceLROnPlateau(monitor='val_accuracy', factor=.1,  patience=3, min_lr=1e-5) 
sgd=SGD(learning_rate=learn_rate,momentum=.9,nesterov=False)




base_model_VGG = VGG19(include_top=False,weights='imagenet' ,input_shape=(36,36,3),classes=ytrain.shape[1])

model_VGGM=Sequential()

model_VGGM.add(base_model_VGG)
model_VGGM.add(Flatten())


model_VGGM.add(Dense(1296,activation=('relu'),input_dim=648))
model_VGGM.add(BatchNormalization())
model_VGGM.add(Dense(648,activation=('relu'))) 
model_VGGM.add(Dropout(.4))
model_VGGM.add(BatchNormalization())
model_VGGM.add(Dense(324,activation=('relu'))) 
model_VGGM.add(Dropout(.3))
model_VGGM.add(BatchNormalization())
model_VGGM.add(Dense(162,activation=('relu')))
model_VGGM.add(Dropout(.2))
model_VGGM.add(Dense(5,activation=('softmax')))


model_VGGM.summary()

model_VGGM.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

model_VGGM.fit(train_generator.flow(xtrain,ytrain,batch_size=batch_size),
      epochs=epochs, steps_per_epoch=xtrain.shape[0]//batch_size, 
      validation_data=val_generator.flow(xval,yval,batch_size=batch_size),validation_steps=120,callbacks=[lrr],verbose=1)

model_VGGM.save('D:/VIU/TFM/Codigo/TFM/1D/Markov/10ModeloVGG19M.tf')




"""
Con esto generamos los modelos para una longitud de tryactoria, hay que repetir este proceso para las 5 longitudes, lo que genera
5 scripts más cambiando los datos de entrada.
"""