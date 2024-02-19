# White Mold YOLOv8 Model

## Institute of Computing (IC) at University of Campinas (Unicamp)

## Postgraduate Program in Computer Science

### Team

* Rubens de Castro Pereira - student at IC-Unicamp
* Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
* Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
* Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans

### Main purpose

This Python project aims to train and inference the YOLOv8 model in the image dataset of white mold disease and its stages.

This implementation is based on this notebook: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb?authuser=1

## Installing Python Virtual Environment

```
module load python/3.10.10-gcc-9.4.0
```
```
pip install --user virtualenv
```
```
virtualenv -p python3.10 venv-wm-model-yolo-v8
```
```
source venv-wm-model-yolo-v8/bin/activate
```
```
pip install -r requirements.txt
```

## Running Python Application

```
access specific folder 'wm-model-yolo-v8'
```
```
python my-python-modules/manage_yolo-v8_train.py
```

## Submitting Python Application at LoveLace environment

Version of CUDA module to load:
- module load cuda/11.5.0-intel-2022.0.1

```
qsub wm-model-yolo-v8.script
```
```
qstat -u rubenscp
```
```
qstat -q umagpu
```

The results of job execution can be visualizedat some files as:

* errors
* output

## Troubleshootings

- In the first execution, some files are downloaded to the CENAPAD environment, and it's possible that thgis operation can't be done automatically because the security rules. So, you must identify the files need to download in the right place and do it manually by 'wget' command. For example: 
    - cd ~/.config/Ultralytics
    - wget https://ultralytics.com/assets/Arial.ttf
    