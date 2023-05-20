
#Importamos las librerías necesarias.
from andi_datasets.datasets_theory import datasets_theory
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from pyts.image import MarkovTransitionField
from pyts.image import GramianAngularField
import imageio
from mpl_toolkits.axes_grid1 import ImageGrid


#Marcamos Agg para usarlo como backend de dibujo y cerramos todas las gráficas abiertas.
matplotlib.use("Agg")
plt.close('all')



#Creamos un objeto dataset theory de la librería andi_datasets y comprobamos los modelos disponibles.
AD = datasets_theory()
AD.avail_models_name
"""
Generamos 10000 trayectorias de longitud 10, con exponente anómalo 0.7 para el modelo SBM de dimensión 1.
Aquí se iran variando los parámetros para obtener las 5 longitudes y los 5 modelos como se ha comentado en el trabajo.
Guardamos los datos en crudo por si los necesitamos.
"""
trayectorias10 = AD.create_dataset(T=10, N_models=10000, exponents=[0.7], models = [4], dimension = 1)
np.savetxt('1D/Datos/10Trayec0.7Exp1DimSBMSinTratar.csv', trayectorias10)

#Los convertimos en un dataframe y lo guardamos por si queremos visualizar los datos agradablemente.
df = pd.DataFrame(data=trayectorias10.astype(float))
df.to_csv('1D/Datos/10Trayec0.7Exp1DimSBM.csv', sep=' ', header=False, float_format='%.2f', index=False)


#Transformamos las trayectorias mediante la codificación de Markov Transition Field y guardamos los datos.
mtf = MarkovTransitionField(n_bins=8)
X_mtf = mtf.fit_transform(trayectorias10[:,2:])
X_mtf1 = X_mtf.reshape(X_mtf.shape[0], -1)
np.savetxt('1D/Markov/SBM/10Trayec0.7Exp1DimSBMMarkov.csv', X_mtf1, delimiter=',')

# Dibujamos los 50 primeros Markov Transition Field
container = plt.figure(figsize=(10, 5))

grid = ImageGrid(container, 111, nrows_ncols=(5, 10), axes_pad=0.1, share_all=True,
                 cbar_mode='single')
for i, ax in enumerate(grid):
    im = ax.imshow(X_mtf[i], cmap='rainbow', origin='lower', vmin=0., vmax=1.)
grid[0].get_yaxis().set_ticks([])
grid[0].get_xaxis().set_ticks([])
plt.colorbar(im, cax=grid.cbar_axes[0])
ax.cax.toggle_label(True)

container.suptitle("Markov transitient fields para los 50 primeros lanzamientos de 10 metros en serie temporal", y=0.92)
container.savefig('1D/Markov/SBM/10Trayec0.7Exp1DimSBMMarkov.png')


#A continuación, representamos todos los Markov Transition Fields generados y guardamos la información de las imagenes en un archivo npy.
width_ratios = (2, 7, 0.4)
height_ratios = (2, 7)
data = []
gif = []
dibujar = plt.figure()

for i in range(0,10000):
        fig = plt.figure(frameon=False)
        fig.set_size_inches(0.5,0.5)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(X_mtf[i], cmap='rainbow', origin='lower', vmin=0., vmax=1.)

        if(i<5):
            gif.append(fig)
        fig.savefig('1D/Markov/SBM/Imagenes/10/'+str(i+1)+'Trayec0.7Exp1DimSBMMarkov.png')
        data.append(plt.imread(r'D:\VIU\TFM\Codigo\TFM\1D\Markov\SBM\Imagenes\10\\'+str(i+1)+'Trayec0.7Exp1DimSBMMarkov.png')[:,:,0:3].astype(np.uint8))

        print('1-'+str(i))
np.save('D:/VIU/TFM/Codigo/TFM/1D/Markov/SBM/Imagenes/10/ImagenesEnDatos', data)



#Generamos los gifs de las 5 primeras imagenes.
with imageio.get_writer('1D/Markov/SBM/10Trayec0.7Exp1DimSBMMarkov.gif', mode='I',duration =0.5) as writer:
    for fig in gif:
            # Crear una imagen a partir de la figura
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
            w, h = fig.canvas.get_width_height()
            img = imageio.core.util.Image(np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3))
            # Agregar la imagen al GIF
            writer.append_data(img)



#Repetimos el proceso para los datos con Gramian Angular Difference Field
gaf = GramianAngularField(sample_range = (-1,1),method="d")
X_mtf = gaf.fit_transform(trayectorias10[:,2:])
X_mtf1 = X_mtf.reshape(X_mtf.shape[0], -1)
np.savetxt('1D/Difference/SBM/10Trayec0.7Exp1DimSBMDiference.csv', X_mtf1, delimiter=',')

# Dibujamos los 50 primeros Gramian Angular Difference Field
container = plt.figure(figsize=(10, 5))

grid = ImageGrid(container, 111, nrows_ncols=(5, 10), axes_pad=0.1, share_all=True,
                 cbar_mode='single')
for i, ax in enumerate(grid):
    im = ax.imshow(X_mtf[i], cmap='rainbow', origin='lower', vmin=-1., vmax=1.)
grid[0].get_yaxis().set_ticks([])
grid[0].get_xaxis().set_ticks([])
plt.colorbar(im, cax=grid.cbar_axes[0])
ax.cax.toggle_label(True)

container.suptitle("Gramian angular difference fields para los 50 primeros lanzamientos de 10 metros en serie temporal", y=0.92)
container.savefig('1D/Difference/SBM/10Trayec0.7Exp1DimSBMDifference.png')
    
width_ratios = (2, 7, 0.4)
height_ratios = (2, 7)
data = []
gif = []
dibujar = plt.figure()
#A continuación, representamos todos los Gramian Angular Difference Fields generados y guardamos la información de las imagenes en un archivo npy.
for i in range(0,10000):

        fig = plt.figure(frameon=False)
        fig.set_size_inches(0.5,0.5)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(X_mtf[i], cmap='rainbow', origin='lower', vmin=-1., vmax=1.)

        if(i<5):
            gif.append(fig)
        fig.savefig('1D/Difference/SBM/Imagenes/10/'+str(i+1)+'Trayec0.7Exp1DimSBMDifference.png')
        data.append(plt.imread(r'D:\VIU\TFM\Codigo\TFM\1D\Difference\SBM\Imagenes\10\\'+str(i+1)+'Trayec0.7Exp1DimSBMDifference.png')[:,:,0:3].astype(np.uint8))


        print('2-'+str(i))
np.save('D:/VIU/TFM/Codigo/TFM/1D/Difference/SBM/Imagenes/10/ImagenesEnDatos', data)



#Generamos los gifs de las 5 primeras imagenes.
with imageio.get_writer('1D/Difference/SBM/10Trayec0.7Exp1DimSBMDiference.gif', mode='I',duration =0.5) as writer:
    for fig in gif:
            # Crear una imagen a partir de la figura
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
            w, h = fig.canvas.get_width_height()
            img = imageio.core.util.Image(np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3))
            # Agregar la imagen al GIF
            writer.append_data(img)
            

#Repetimos el proceso para los datos con Gramian Angular Summation Field
gaf = GramianAngularField(sample_range = (-1,1),method="s")
X_mtf = gaf.fit_transform(trayectorias10[:,2:])
X_mtf1 = X_mtf.reshape(X_mtf.shape[0], -1)
np.savetxt('1D/Summation/SBM/10Trayec0.7Exp1DimSBMSummation.csv', X_mtf1, delimiter=',')

# Dibujamos los 50 primeros Gramian Angular Summation Field
container = plt.figure(figsize=(10, 5))

grid = ImageGrid(container, 111, nrows_ncols=(5, 10), axes_pad=0.1, share_all=True,
                 cbar_mode='single')
for i, ax in enumerate(grid):
    im = ax.imshow(X_mtf[i], cmap='rainbow', origin='lower', vmin=-1., vmax=1.)
grid[0].get_yaxis().set_ticks([])
grid[0].get_xaxis().set_ticks([])
plt.colorbar(im, cax=grid.cbar_axes[0])
ax.cax.toggle_label(True)

container.suptitle("Gramian angular summation fields para los 50 primeros lanzamientos de 10 metros en serie temporal", y=0.92)
container.savefig('1D/Summation/SBM/10Trayec0.7Exp1DimSBMSumation.png')
    
width_ratios = (2, 7, 0.4)
height_ratios = (2, 7)
data = []
gif = []
dibujar = plt.figure()
#A continuación, representamos todos los Gramian Angular Summation Fields generados y guardamos la información de las imagenes en un archivo npy.
for i in range(0,10000):
        fig = plt.figure(frameon=False)
        fig.set_size_inches(0.5,0.5)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(X_mtf[i], cmap='rainbow', origin='lower', vmin=-1., vmax=1.)

        if(i<5):
            gif.append(fig)
        fig.savefig('1D/Summation/SBM/Imagenes/10/'+str(i+1)+'Trayec0.7Exp1DimSBMSumation.png')
        data.append(plt.imread(r'D:\VIU\TFM\Codigo\TFM\1D\Summation\SBM\Imagenes\10\\'+str(i+1)+'Trayec0.7Exp1DimSBMSumation.png')[:,:,0:3].astype(np.uint8))
        print('3-'+str(i))

np.save('D:/VIU/TFM/Codigo/TFM/1D/Summation/SBM/Imagenes/10/ImagenesEnDatos', data)



#Generamos los gifs de las 5 primeras imagenes.
with imageio.get_writer('1D/Summation/SBM/10Trayec0.7Exp1DimSBMSummation.gif', mode='I',duration =0.5) as writer:
    for fig in gif:
            # Crear una imagen a partir de la figura
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
            w, h = fig.canvas.get_width_height()
            img = imageio.core.util.Image(np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3))
            # Agregar la imagen al GIF
            writer.append_data(img)



"""
Una vez terminado el proceso, tenemos las trayectorias de 1 longitud para 1 modelo. Hay que repetir el proceso cambiando las longitudes de 
trayectoria, los modelos y el nombre de los archivos. Esto genera un total de 25 scripts.
"""