[:es: Español](#spanish) | [:gb: English](#english)
---
<h1 align="center"> Estimación de altura humana "in the wild" mediante calibración geométrica y aprendizaje profundo </h1>
<h4 align="center">  Curso 2025-2026, Máster Ciencia de Datos e Ingeniería de Computadores, ETSIIT UGR.</h4>
<h5 align="center"> Trabajo de Fin de Máster </h5><a id='spanish'></a>

<img style="display:block;width:100%;margin:auto;padding-bottom:25px" src="https://github.com/briansenas/TFM_SVMIW/blob/main/imgs/coco-scale-sample-crop.png?raw=True"/>

<div style="display:flex"> 
<div style="flex:50%;max-width:30%"><font size="4"> <emph><strong>Autor</strong></emph>: Brian Sena Simons</font> </div>
<div style="flex:50%;text-align:right"><font size="4"> <emph><strong>Directores</emph></strong>: Dr. Pablo Mesejo Santiago y Dr. Enrique Bermejo Nievas </font></div>
</div>

## :pushpin: Introducción 
Este TFM trata de la implementación del artículo de Zhu <emph>et al</emph>[[1]](#1), 
en el framework de visión por computador "Detectron2"[[2]](#2). En lugar del ya no disponible conjunto de datos SUN360, se utiliza Pano360[[3]](#3). Los valores obtenidos en Pano360 y COCO-Scale son del mismo orden de magnitud que los artículos 
de referencia y, aunque la comparación no es estrictamente directa debido a diferencias entre las particiones utilizadas, sugieren que la implementación reproduce razonablemente el comportamiento descrito. Finalmente, se evalúa su rendimiento en un conjunto de datos independiente IdentAI y se identifican las limitaciones. 

Para una lectura detallada de los modelos y experimentación pulsar [aquí](https://github.com/briansenas/TFM_SVMIW/blob/main/Document/proyecto.tex).


## 📚 Tabla de contenidos

- [Descripción general](#-herramientas-de-cámara)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Notas](#-notas)
- [Licencia](#-licencia)

## 🎥 Herramientas de cámara

Para más información sobre el módelo implementado sobre el trabajo de Zhu <emph>et al</emph>[[1]](#1), entrar al repositorio [d2-scale-net](https://github.com/briansenas/detectron2-scale-net/tree/feature/multiple-classes). Una vez instalado los requisitos de Detectron2[[2]](#2) se podrá hacer uso del script `train_calib.py` para entrenar el modelo. Para más información, ver Apéndice "A.5. Lanzar experimentos".

Respecto a los scripts de Python, se proporciona un conjunto modular de herramientas de línea de comandos para gestionar flujos de trabajo de cámaras basados en vídeo, como:

- Extraer fotogramas de un vídeo utilizando `ffmpeg`
- Calibrar una cámara utilizando patrones de tablero de ajedrez
- Estimar la altura de una persona utilizando YOLO y los parámetros de la cámara
- Cortar fragmentos de vídeo entre marcas de tiempo

Cada funcionalidad se implementa como un script independiente con una función `register_subparser()` para su integración en la CLI.

---

## 📁 Estructura del proyecto

```plaintext
.
├── datagg
│   ├── cam1-cut_frames         # Fotogramas extraídos mediante extract-frames
│   ├── cam1-cut.mkv            # Grabación original
│   ├── cam1.mkv                # Grabación recortada mediante cut-video.py
│   └── intrinsics              # Parámetros intrínsecos generados automáticamente mediante calibrate-camera
├── Document/                   # Documentación relacionada con los resultados y detalles del proyecto
├── LICENSE
├── pyproject.toml
├── README.md
├── scripts/                    # Carpeta para módulos cargados automáticamente
│   ├── extract_frames.py            # Extraer fotogramas de un vídeo
│   ├── calibrate_camera.py          # Calibrar la cámara utilizando fotogramas con patrón de tablero de ajedrez
│   ├── estimate_height.py           # Estimar la altura de una persona utilizando YOLO + calibración
│   ├── cut_video.py                 # Cortar segmentos de vídeo utilizando ffmpeg
|   └── extract_realsense_frames.py  # Extraer fotogramas de un vídeo a partir de un archivo .bag
|   └── ...
├── main.py                      # Punto de entrada si se desea unificar todos los comandos
├── uv.lock
└── yolov8n.pt
```

---

## ⚙️ Instalación

Recomendamos utilizar [`uv`](https://github.com/astral-sh/uv), un gestor rápido de paquetes de Python y de entornos virtuales.

### 1. Instalar `uv`

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

### 2. Configurar el proyecto

```bash
uv venv
uv pip install -r requirements.txt
```

> O, para realizar la instalación manualmente:
>
> ```bash
> pip install -r requirements.txt
> ```

---

## 🚀 Uso

Cada script puede ejecutarse desde la línea de comandos e incluye su propio subparser para facilitar su integración.

Para obtener más información, puedes ejecutar el siguiente comando o consultar [scripts/README.md](scripts/README.md).

```bash
python main.py -h
```

## 🧠 Notas

* Asegúrate de que `ffmpeg` está instalado y disponible en el `PATH` de tu sistema. En Linux: `sudo apt install ffmpeg`.
* El modelo YOLO utilizado por defecto es `yolov8n.pt`, aunque puedes utilizar otros modelos preentrenados o modelos personalizados.

---

## 📬 Licencia

GNU GENERAL PUBLIC LICENSE Versión 3, 29 de junio de 2007


[:es: Español](#spanish) | [:gb: English](#english)
---
<h1 align="center"> Human height estimation "in the wild" through geometric calibration and deep learning</h1>
<h4 align="center">  2025-2026 Course, Master in Data Science and Computer Engineering, ETSIIT UGR.</h4>
<h5 align="center"> Master's thesis </h5><a id='english'></a>

<img style="display:block;width:100%;margin:auto;padding-bottom:25px" src="https://github.com/briansenas/TFM_SVMIW/blob/main/imgs/coco-scale-sample-crop.png?raw=True"/>

<div style="display:flex"> 
<div style="flex:50%;max-width:30%"><font size="4"> <emph><strong>Author</strong></emph>: Brian Sena Simons</font> </div>
<div style="flex:50%;text-align:right"><font size="4"> <emph><strong>Directores</emph></strong>: Dr. Pablo Mesejo Santiago y Dr. Enrique Bermejo Nievas </font></div>
</div>

## :pushpin: Introduction

This thesis focuses on the implementation of the paper by Zhu <emph>et al.</emph>[[1]](#1) within the computer vision framework "Detectron2"[[2]](#2). Instead of the now-unavailable SUN360 dataset, Pano360[[3]](#3) is used. The results obtained on Pano360 and COCO-Scale are of the same order of magnitude as those reported in the reference papers and, although the comparison is not strictly direct due to differences between the partitions used, they suggest that the implementation reasonably reproduces the behavior described in the original work. Finally, its performance is evaluated on an independent dataset, IdentAI, and its limitations are identified.

For a detailed reading of the models and experimentation, click [here](https://github.com/briansenas/TFM_SVMIW/blob/main/Document/proyecto.tex).

## 📚 Table of Contents

- [Overview](#-camera-tools)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Notes](#-notes)
- [License](#-license)

## 🎥 Camera Tools

For more information about the model implemented based on the work of Zhu <emph>et al.</emph>[[1]](#1), please visit the [d2-scale-net](https://github.com/briansenas/detectron2-scale-net/tree/feature/multiple-classes) repository. Once the Detectron2[[2]](#2) requirements have been installed, the `train_calib.py` script can be used to train the model. For more information, see Appendix "A.5. Launching Experiments".

Regarding the Python scripts, a modular set of command-line tools is provided for managing video-based camera workflows, such as:

- Extracting frames from a video using `ffmpeg`
- Calibrating a camera using chessboard patterns
- Estimating human height using YOLO and camera parameters
- Cutting video clips between timestamps

Each feature is implemented as a standalone script with a `register_subparser()` function for CLI integration.

---

## 📁 Project Structure

```plaintext
.
├── datagg
│   ├── cam1-cut_frames         # Extracted frames using extract-frames
│   ├── cam1-cut.mkv            # Original footageg
│   ├── cam1.mkv                # Cut footage using cut-video.py
│   └── intrinsics              # Automatically generated intrinsics using calibrate-camera
├── Document/                   # Documentation regarding the project outcome and details
├── LICENSE
├── pyproject.toml
├── README.md
├── scripts/                     # Folder for auto-loaded modules
│   ├── extract_frames.py            # Extract frames from a video
│   ├── calibrate_camera.py          # Calibrate camera using chessboard frames
│   ├── estimate_height.py           # Estimate person height using YOLO + calibration
│   ├── cut_video.py                 # Cut video segments using ffmpeg
|   └── extract_realsense_frames.py  # Extract frames from a video from a .bag file
|   └── ...
├── main.py                      # Entry point if you wish to unify all commands
├── uv.lock
└── yolov8n.pt
````

---

## ⚙️ Installation

We recommend using [`uv`](https://github.com/astral-sh/uv), a fast Python package manager and virtual environment tool.

### 1. Install `uv`

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

### 2. Set up project

```bash
uv venv
uv pip install -r requirements.txt
```

> Or to install manually:
>
> ```bash
> pip install -r requirements.txt
> ```

---

## 🚀 Usage

Each script is executable from the command line and includes its own subparser for integration.
For more information you can run the following code or read more at [scripts/README.md](scripts/README.md).

```bash
python main.py -h
```

## 🧠 Notes

* Make sure `ffmpeg` is installed and accessible in your system path. For linux: `sudo apt install ffmpeg`.
* The YOLO model defaults to `yolov8n.pt`, but you can use other pretrained models or custom ones.

---

## 📬 License

GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

---

## :books: Referencias
<a id="1">[1]</a> R. Zhu et al., "Single View Metrology in the Wild" en Computer Vision - ECCV 2020, Springer INternation PUblishing, 2020, págs. 316-333
<a id="2">[2]</a> Y. Wu, A. Kirillov, F. Massa, W.-Y. Lo y R. Girshick, Detectron2, https://github.com/facebookresearch/detectron2, 2019. 
<a id="3">[3]</a> M. Kocabas, C.-H. P. Huang, J. Tesch, L. Müller, O. Hilliges y M. J. Black, ((SPEC: Seeing People in the Wild with an Estimated Camera,)) en International Conference on Computer Vision, 2021, págs. 11 035-11 045. 
