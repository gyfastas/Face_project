

# Face_project

## Introduction

   This is a Face Recognition system application based on MTCNN Face Detection Model and Sphere net Face Recognition Model. The sphere face model code is based on  https://github.com/clcarwin/sphereface_pytorch, Thanks for his great work.



## Model Installation

Pretrained model can be achieved through the following link.

You can download it and put the model folder in the project folder

https://pan.baidu.com/s/1AbIxu066iBVeClGOshdu9w Passward: qgjd



## Prerequisite

- Python 3.6+ [3.6 verified]
- Pytorch 0.4+ [0.4 verified]
- Tensorflow 1.12+ [1.12 verified]

## Installation

1.clone the repository:

```
git clone https://github.com/gyfastas/Face_project.git
```

2.install python requirements

```
pip install -r requirements.txt
```

3.run Mainwindow.py



### Using other model

If you are going to use another Face Recognition Model:

```
python Mainwindow.py -n "your network name" -m "your model path"
```

