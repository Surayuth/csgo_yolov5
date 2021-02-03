# import libraries
import os
import cv2
import sys
import glob
import math
import yaml
import gdown
import shutil
import random
import zipfile
import argparse
import numpy as np
from tqdm import tqdm

# define function
def download_data(idxs):
    '''
    download data to the current directory.
    note that index start from 1 
    '''
    ids = [
        '1yPoJmHu6bFBr-MZbbN4NZQ_E3wFtcH29',
        '199QGp78eHt_-Oq02jTz8EPpqhush5-zO',
        '1aPTqp_tCjArhuA8m2s8omoSXTAUX4Zcl'
        ]
    if idxs[0] == -1:
        idxs = np.arange(len(ids)) + 1
    for idx in idxs:
        url = 'https://drive.google.com/uc?id=' + ids[idx-1]
        output = f'csgo_dataset_{idx}.zip'
        gdown.download(url, output, quiet=False)

def extract(src='', dst=''):
    '''
    default paths of src and dst are current
    directory
    '''
    root = os.path.join(src, 'csgo_dataset*.zip')
    zip_files = sorted(glob.glob(root))
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            name = zip_file.split('/')[-1]
            zip_ref.extractall(dst)

def resize(src, dst, w, h):
    dim = (w, h)
    root = os.path.join(src, '*.png')
    paths = glob.glob(root)
    for path in paths:
        name = path.split('/')[-1]
        path_dst = os.path.join(dst, name)
        img = cv2.imread(path)
        resized = cv2.resize(img, dim)
        cv2.imwrite(path_dst, resized)

def copy_annos(src, dst):
    root = os.path.join(src, '*.txt')
    paths = glob.glob(root)
    for path in paths:
        name = path.split('/')[-1]
        path_dst = os.path.join(dst, name)
        shutil.copy(path, path_dst)

def create_dirs(path_dataset):
    if os.path.isdir(path_dataset):
        shutil.rmtree(path_dataset)
    os.mkdir(path_dataset)
    for dir1 in ['images', 'labels']:
        path1 = os.path.join(path_dataset, dir1)
        os.makedirs(path1)
        for dir2 in ['train', 'val', 'test']:
            path2 = os.path.join(path1, dir2)
            os.makedirs(path2)
    return

def copy_files(paths, dest):
    for path in paths:
        shutil.copy(path, dest)
    return

def create_train_val_test(folder, path_dataset, val_ratio, test_ratio):
    # path of images and labels
    img_paths = sorted(glob.glob(folder + '/*.png'))
    label_paths = sorted(glob.glob(folder + '/*.txt'))
    
    # shuffle images 
    random.shuffle(img_paths)
    
    # create train
    train_ratio = 1 - val_ratio - test_ratio
    train_size = math.floor(train_ratio*len(img_paths))
    train_img_paths = img_paths[0:train_size]
    train_label_paths = label_paths[0:train_size]
    current_size = train_size
    copy_files(train_img_paths, path_dataset + '/images/train/')
    copy_files(train_label_paths, path_dataset + '/labels/train/')
    
    # create val
    val_size = math.floor(val_ratio*len(img_paths))
    val_paths = img_paths[current_size:current_size + val_size]
    val_img_paths = img_paths[current_size:current_size + val_size]
    val_label_paths = label_paths[current_size:current_size + val_size]
    current_size += val_size
    copy_files(val_img_paths, path_dataset + '/images/val/')
    copy_files(val_label_paths, path_dataset + '/labels/val/')
    
    # create test
    test_img_paths = img_paths[current_size:]
    test_label_paths = label_paths[current_size:]
    copy_files(test_img_paths, path_dataset + '/images/test/')
    copy_files(test_label_paths, path_dataset + '/labels/test/')

    return

def create_yaml(dataset):
    dict_file = {
        'train': '../' + dataset + '/images/train/',
        'val': '../' + dataset + '/images/val/',
        'test': '../' + dataset + '/images/test/',
        'nc': 2,
        'names': ['t', 'ct']        
    }

    with open('./yolov5/csgo.yaml', 'w') as file:
        documents = yaml.dump(dict_file, file, default_flow_style=None)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('val', type=float)
    parser.add_argument('test', type=float)
    parser.add_argument('-r', '--resize', nargs='+', type=int)
    parser.add_argument('-d', '--download', nargs='+', type=int)
    args = parser.parse_args()
    
    # download data
    if args.download:
        download_data(args.download)
        
    # 1) extract zip files in the current dir
    t = tqdm(total=5, unit="task", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    t.set_description('extracting data')
    extract()
    t.update()
    
    # 2) resize image
    t.set_description('resizing images')
    if args.resize:
        if os.path.isdir('resized_images'):
            shutil.rmtree('resized_images')
        os.mkdir('resized_images')
        w = args.resize[0]
        h = args.resize[1]
        resize('obj_train_data', 'resized_images', w, h)
        copy_annos('obj_train_data', 'resized_images')
    t.update()
    
    # 3) create folder for containing data named 'csgo_dataset'
    t.set_description('creating folder')
    create_dirs('csgo_dataset')
    t.update()
    
    # 4) copy files extracted in 1) to the folder create in 2)
    t.set_description('splitting data')
    if args.resize:
        create_train_val_test('resized_images', 'csgo_dataset', args.val, args.test)
    else:
        create_train_val_test('obj_train_data', 'csgo_dataset', args.val, args.test)
    t.update()
    
    # 5) create yaml file where the folder containing the dataset is 'csgo_dataset'
    t.set_description('creating YAML')
    create_yaml('csgo_dataset')
    t.update()
    
if __name__ == '__main__':
    main()
