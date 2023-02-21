from pathlib import Path
from json import load
from random import shuffle

from cv2 import imread, cvtColor, COLOR_BGR2GRAY
from numpy import array, reshape, expand_dims
from sklearn.preprocessing import StandardScaler
from keras.utils import Sequence
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

from .backends.misc import name_to_parameters


def name_to_y(file, y_keys):
    file = Path(file)
    parameters = name_to_parameters(file)
    y = []
    for y_key in y_keys:
        y.append(parameters[y_key])
    return y


def generate_y_scaler(train_files, y_keys):

    all_y = []
    for file in train_files:
        all_y.append(name_to_y(file=file, y_keys=y_keys))
    
    y_scaler = StandardScaler()
    y_scaler.fit_transform(all_y)

    return y_scaler


def img_file_to_img(file):
    file = Path(file)
    x_i = imread(str(file))
    x_i = cvtColor(x_i, COLOR_BGR2GRAY)
    x_i = 255 - x_i
    x_i = x_i / 255
    x_i = expand_dims(x_i, axis=2)
    return x_i


class DataGenerator(Sequence):

    def __init__(self, files, y_scaler, y_keys, batch_size):
        self.files = files
        self.y_scaler = y_scaler
        self.y_keys = y_keys
        self.batch_size = batch_size

        self.len_files = len(self.files)

    def on_epoch_end(self):
        shuffle(self.files)

    def __len__(self):
        return self.len_files // self.batch_size

    def __getitem__(self, index):

        max_index = self.batch_size * (index + 1)
        min_index = self.batch_size * index
        batch_files = self.files[min_index:max_index]

        x = []
        y = []
        for file in batch_files:
            x.append(img_file_to_img(file))
            y.append(name_to_y(file=file, y_keys=self.y_keys))

        x = array(x)
        y = self.y_scaler.transform(y)

        return x, y


def simple_model(len_y_keys):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(len_y_keys))
    model.compile(optimizer='Adam', loss='mae', metrics=['mse', 'mae'])
    return model
    

def train(model_path, y_keys, epochs, batch_size):
    
    split = model_path / "split.json"
    with open(split, "r") as f:
        split = load(f)

    y_scaler = generate_y_scaler(train_files=split["train"], y_keys=y_keys)

    train_generator = DataGenerator(files=split["train"], y_scaler=y_scaler, y_keys=y_keys, batch_size=batch_size)
    val_generator = DataGenerator(files=split["val"], y_scaler=y_scaler, y_keys=y_keys, batch_size=batch_size)

    model = simple_model(len_y_keys=len(y_keys))
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

    history_data = dict(loss=history.history['loss'], val_loss=history.history['val_loss'])

    return model, history_data, y_scaler


def test(model, y_keys, y_scaler, y_files, batch_size):

    def _predict(b):

        x = []
        y = []
        for file in b:
            x.append(img_file_to_img(file))
            y.append(name_to_y(file=file, y_keys=y_keys))
        x = array(x)
        y_norm = y_scaler.transform(y)

        y_pred_norm = model.predict(x)
        y_pred = y_scaler.inverse_transform(y_pred_norm)

        for i, y_norm_i in enumerate(y_norm):
            y_pred_norm_i = y_pred_norm[i]
            pva_norm.append([y_norm_i, y_pred_norm_i])

        for i, y_i in enumerate(y):
            y_pred_i = y_pred[i]
            pva.append([y_i, y_pred_i])
    
    pva = []
    pva_norm = []

    batch = []
    for file in y_files:
        batch.append(file)
        if len(batch) == batch_size:
            _predict(batch)
            batch = []
    if batch:
        _predict(batch)

    return array(pva_norm), array(pva)
