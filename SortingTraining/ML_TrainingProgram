############################
# This code has been sourced and modified from
# https://towardsdatascience.com/how-to-build-an-image-classifier-for-waste-sorting-6d11d3c9c478
############################

from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import matplotlib as mpl
import os
import zipfile as zf
import shutil
import re
import seaborn as sns

# Initial Code to Extract and Sort Files from dataset
if not os.path.exists("dataset-resized"):
    files = zf.ZipFile("dataset-resized.zip", "r")
    files.extractall()
    files.close()

    os.listdir(os.path.join(os.getcwd(), "dataset-resized"))


## helper functions ##

## splits indices for a folder into train, validation, and test indices with random sampling
## input: folder path
## output: train, valid, and test indices
def split_indices(folder, seed1, seed2):
    n = len(os.listdir(folder))
    full_set = list(range(1, n + 1))

    ## train indices
    random.seed(seed1)
    train = random.sample(list(range(1, n + 1)), int(.5 * n))

    ## temp
    remain = list(set(full_set) - set(train))

    ## separate remaining into validation and test
    random.seed(seed2)
    valid = random.sample(remain, int(.5 * len(remain)))
    test = list(set(remain) - set(valid))

    return (train, valid, test)


## gets file names for a particular type of waste, given indices
## input: waste category and indices
## output: file names
def get_names(waste_type, indices):
    file_names = [waste_type + str(i) + ".jpg" for i in indices]
    return (file_names)


## moves group of source files to another folder
## input: list of source files and destination folder
## no output
def move_files(source_files, destination_folder):
    for file in source_files:
        shutil.move(file, destination_folder)


def main():

    ## paths will be train/cardboard, train/glass, etc...
    subsets = ['train', 'valid']
    waste_types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    ## create destination folders for data subset and waste type
    for subset in subsets:
        for waste_type in waste_types:
            folder = os.path.join('data', subset, waste_type)
            if not os.path.exists(folder):
                os.makedirs(folder)

    if not os.path.exists(os.path.join('data', 'test')):
        os.makedirs(os.path.join('data', 'test'))

    ## move files to destination folders for each waste type
    for waste_type in waste_types:
        source_folder = os.path.join('dataset-resized', waste_type)
        train_ind, valid_ind, test_ind = split_indices(source_folder, 1, 1)

        ## move source files to train
        train_names = get_names(waste_type, train_ind)
        train_source_files = [os.path.join(source_folder, name) for name in train_names]
        train_dest = "data/train/" + waste_type
        move_files(train_source_files, train_dest)

        ## move source files to valid
        valid_names = get_names(waste_type, valid_ind)
        valid_source_files = [os.path.join(source_folder, name) for name in valid_names]
        valid_dest = "data/valid/" + waste_type
        move_files(valid_source_files, valid_dest)

        ## move source files to test
        test_names = get_names(waste_type, test_ind)
        test_source_files = [os.path.join(source_folder, name) for name in test_names]
        ## I use data/test here because the images can be mixed up
        move_files(test_source_files, "data/test")

    path = Path(os.getcwd())/"data"
    tfms = get_transforms(do_flip=True, flip_vert=True)
    data = ImageDataBunch.from_folder(path, test="test", ds_tfms=tfms, bs=8)

    print(data.classes)

    learn = cnn_learner(data, models.resnet34, metrics=error_rate)

    learn.lr_find(start_lr=1e-6,end_lr=1e1)

    learn.fit_one_cycle(20, max_lr=5.13e-03)

    preds = learn.get_preds(ds_type=DatasetType.Test)

    print(preds[0].shape)
    print(preds[0])

    ## saves the index (0 to 5) of most likely (max) predicted class for each image
    max_idxs = np.asarray(np.argmax(preds[0], axis=1))

    yhat = []
    for max_idx in max_idxs:
        yhat.append(data.classes[max_idx])

    y = []

    ## convert POSIX paths to string first
    for label_path in data.test_ds.items:
        y.append(str(label_path))

    ## then extract waste type from file path
    pattern = re.compile("([a-z]+)[0-9]+")
    for i in range(len(y)):
        y[i] = pattern.search(y[i]).group(1)

    cm = confusion_matrix(y, yhat)
    print(cm)

    df_cm = pd.DataFrame(cm, waste_types, waste_types)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")

    correct = 0

    for r in range(len(cm)):
        for c in range(len(cm)):
            if r == c:
                correct += cm[r, c]

    accuracy = correct / sum(sum(cm))
    print(accuracy)

    model_path = learn.save("recyc_model", return_path=True)

if __name__ == '__main__':
    main()


