# csgo_yolov5
Before doing anything, you must clone this repos and install requirements
```
$ git clone https://github.com/Surayuth/csgo_yolov5
$ cd csgo_yolov5
$ pip install -r requirements.txt
```
1. Dataset for training, validation and testing can be created by   
```
$ python3 create_dataset.py 0.2 0.1 --resize 1600 900 --download -1
```
