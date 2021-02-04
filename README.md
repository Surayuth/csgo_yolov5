# csgo_yolov5
**Before doing anything, you must clone this repos and install requirements**
```
$ git clone https://github.com/Surayuth/csgo_yolov5
$ cd csgo_yolov5
$ pip install -r requirements.txt
```
**1. Create dataset for training, validation and testing by**
```
$ python3 create_dataset.py 0.2 0.1 --resize 1600 900 --download -1
```
The command above creates a directory, `csgo_dataset`, having the right structure for being trained by yolov5 model. The `val_ratio` and `test_ratio` are set to 0.2 and 0.1, respectively. You can also resize all the images to specific size by using `--resize w h`. The optional argument `--download -1` will download all datasets
available to this repos right now. You can change `-1` to the numbers of datasets you want to download, i.e., `--download 1 2`.

**2. Install yolov5 (inside csgo_yolov5)**
```
$ git clone https://github.com/ultralytics/yolov5 
$ cd yolov5
$ pip install -r requirements.txt  
```
**3. Copy YAML file of `csgo_dataset` to yolov5 directory**
```
$ cp ../csgo.yaml yolov5
```
**4. Train model**
```
$ python3 train.py --img 1600 --batch 2 --epochs 5 --data csgo.yaml --weights yolov5s.pt
```
