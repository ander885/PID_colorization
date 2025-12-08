# Colorización automática de imágenes en escala de grises

Proyecto de la asignatura PID sobre **colorización automática de imágenes en escala de grises** utilizando el modelo preentrenado de Zhang et al. (“Colorful Image Colorization”) integrado en `OpenCV-DNN`.

El objetivo es:

- Reutilizar un modelo CNN ya entrenado para colorear imágenes en escala de grises.
- Construir un **pipeline reproducible** de inferencia:
  - Imagen original a color → versión en escala de grises.
  - Colorización automática a partir del canal L.
- Evaluar la calidad de la colorización mediante **PSNR** y **SSIM**.
- Analizar el efecto del **parámetro de temperatura** en la etapa de decodificación de color.

## Estructura del repositorio

```text
.
├── models/
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel
│   └── pts_in_hull.npy
├── imgs/
│   ├── Imagen01.jpg ... Imagen20.jpg     # Imágenes originales a color
│   └── ansel_adams.jpg                   # Imagen de prueba en escala de grises
├── imgs_out/
│   └── ImagenXX_color.jpg                # Colorizaciones generadas
├── eval/
│   ├── tables/
│   │   └── metrics.csv                   # Métricas PSNR/SSIM
│   └── figs/
│       ├── mosaic_metrics.png
│       ├── metricas_barras.png
│       ├── psnr_vs_ssim.png
│       ├── ranking_metricas.png
│       └── temperature_analysis/         # Comparativas por temperatura
├── notebooks/
│   ├── 01_baseline_dnn.ipynb
│   └── 02_metrics.ipynb
└── README.md
```
> **Nota:** Algunas carpetas (`imgs_out/`, `eval/tables/`, `eval/figs/`) se crean o rellenan al ejecutar los notebooks.

## Requisitos

- Python 3.x
- Librerías principales:
  - `numpy`
  - `opencv-python`
  - `matplotlib`
  - `pandas`
  - `scikit-image`
  - `jupyter`

Opcionalmente se puede usar un fichero `requirements.txt` con estas dependencias:

```txt
numpy
opencv-python
matplotlib
pandas
scikit-image
jupyter
```
## Instalacion
```bash
pip install -r requirements.txt
```
## Modelos preentrenados
En la carpeta models/ deben estar los ficheros del modelo de Zhang et al.:
* colorization_deploy_v2.prototxt
* colorization_release_v2.caffemodel
* pts_in_hull.npy
Estos ficheros no se entrenan en este proyecto: se reutilizan tal cual para la inferencia.

## Cómo reproducir los experimentos
### 1. Ejecutar el Notebook 01 – baseline de colorización

**1. Lanzar Jupyter en la raíz del proyecto:**
   ```bash
   jupyter notebook
   ```

**2. Abrir notebooks/01_baseline_dnn.ipynb.**

**3. Ejecutar las primeras celdas para:**
   * Importar librerías.
   * Definir rutas (DATA_RAW, DATA_RES, etc.).
   * Cargar la red con cv.dnn.readNetFromCaffe.
   * Inyectar los 313 centros de color en la capa class8_ab y el vector de reescalado en conv8_313_rh.

**4. Ejecutar la celda de demostración con ansel_adams.jpg:**
   * Se muestra la imagen en gris y su versión colorizada.
   * Se guarda imgs_out/ansel_adams_color.jpg (solo para comprobación visual).

**5. Ejecutar la celda de procesado por lotes:**
   * Para cada ImagenXX.jpg en imgs/:
      * Se genera ImagenXX_gray.jpg (escala de grises).
      * Se genera ImagenXX_color.jpg y se guarda en imgs_out/.

**6. Ejecutar las celdas de temperatura para generar comparativas con distintos valores de T:**
   * Las imágenes resultantes se guardan en eval/figs/temperature_analysis/.

### 2. Ejecutar el Notebook 02 – métricas y visualización

**1. Abrir notebooks/02_metrics.ipynb desde Jupyter.**

**2. Ejecutar la celda que define list_pairs():**
   * Busca todas las ImagenXX.jpg en imgs/ (ignorando _gray, _color y ansel_adams).
   * Empareja cada original con su correspondiente ImagenXX_color.jpg en imgs_out/.

**3. Ejecutar la celda de cálculo de PSNR y SSIM:**
   * Para cada par (original, colorizada):
      * Se leen ambas imágenes.
      * Se ajusta el tamaño si es necesario.
      * Se convierten a RGB.
      * Se calculan PSNR y SSIM.
   * Los resultados se guardan en eval/tables/metrics.csv y se muestra una tabla ordenada por imagen.

**4. Ejecutar las celdas de visualización para generar:**
   * mosaic_metrics.png: original vs colorizada.
   * metricas_barras.png: barras de PSNR y SSIM por imagen.
   * psnr_vs_ssim.png: diagrama de dispersión PSNR vs SSIM.
   * ranking_metricas.png: rankings de imágenes según PSNR y por SSIM.

Todas estas figuras se guardan en eval/figs/.

## Resultados esperados
   * Un conjunto de imágenes colorizadas en imgs_out/.
   * Fichero metrics.csv con PSNR y SSIM por imagen.
   * Figuras con:
      * Comparación visual (antes / después).
      * Gráficas de PSNR y SSIM.
      * Dispersión PSNR vs SSIM.
      * Ranking de imágenes según el rendimiento.

## Autores
* Ander Serrano
* Adrián Muriel

Basado en el modelo “Colorful Image Colorization” de Zhang et al. (ECCV 2016).
