from pathlib import Path
from random import shuffle

from tqdm import tqdm
from cv2 import imread, cvtColor, COLOR_BGR2GRAY
from numpy import array, expand_dims
from sklearn.preprocessing import MinMaxScaler
from keras.utils import Sequence
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import load_model
from joblib import load as joblib_load

from . import name_to_parameters


def name_to_y(file, y_key):
    file = Path(file)
    parameters = name_to_parameters(file)
    return parameters[y_key]


def generate_y_scaler(train_files, y_key):

    all_y = []
    for file in train_files:
        all_y.append(name_to_y(file=file, y_key=y_key))

    y_scaler = MinMaxScaler()
    y_scaler.fit_transform(array(all_y).reshape(-1, 1))

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

    def __init__(self, files, y_scaler, y_key, batch_size):
        self.files = files
        self.y_scaler = y_scaler
        self.y_key = y_key
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
            y.append(name_to_y(file=file, y_key=self.y_key))

        x = array(x)
        y = self.y_scaler.transform(array(y).reshape(-1, 1))

        return x, y


def simple_model():
    # Same model structure that all models use
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='Adam', loss='mae', metrics=['mse', 'mae'])
    return model
    

def train(split, y_key, epochs, batch_size):

    y_scaler = generate_y_scaler(train_files=split["train"], y_key=y_key)

    train_generator = DataGenerator(files=split["train"], y_scaler=y_scaler, 
                                    y_key=y_key, batch_size=batch_size)
    val_generator = DataGenerator(files=split["val"], y_scaler=y_scaler, 
                                  y_key=y_key, batch_size=batch_size)

    model = simple_model()
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, 
                        callbacks=Callback())

    # Only keep loss and val_loss from history
    history_data = dict(loss=history.history['loss'], 
                        val_loss=history.history['val_loss'])

    return model, history_data, y_scaler


def test(model, y_key, y_scaler, y_files, batch_size):

    def _predict(b):

        # Gather x and y data in files
        x = []
        y = []
        for file_i in b:
            x.append(img_file_to_img(file_i))
            y.append(name_to_y(file=file_i, y_key=y_key))
        x = array(x)

        # Normalize y data
        y_norm = y_scaler.transform(array(y).reshape(-1, 1))

        # Predict property
        y_predicted_norm = model.predict(x, verbose=0)
        y_predicted = y_scaler.inverse_transform(y_predicted_norm)

        # Flatten y data
        y_norm = y_norm.flatten()
        y_predicted_norm = y_predicted_norm.flatten()
        y_predicted = y_predicted.flatten()

        # Update PVA data
        for i, y_norm_i in enumerate(y_norm):
            y_predicted_norm_i = y_predicted_norm[i]
            pva_norm.append([y_norm_i, y_predicted_norm_i])
        for i, y_i in enumerate(y):
            y_predicted_i = y_predicted[i]
            pva.append([y_i, y_predicted_i])
    
    pva = []
    pva_norm = []

    # Predict on files in batches
    batch = []
    for file in tqdm(y_files, desc=f"Predicting {y_key}"):
        batch.append(file)
        if len(batch) == batch_size:
            _predict(batch)
            batch = []
    if batch:
        _predict(batch)

    return array(pva_norm), array(pva)


def default_predict(img_files: list, parameter: str):
    
    # Load scaler from control model
    scaler_path = Path(__file__).parent.parent 
    scaler_path = scaler_path / f"data/models/control/{parameter}/y_scaler.save"
    with open(scaler_path, "rb") as f:
        scaler = joblib_load(f)

    # Load control model
    model_path = Path(__file__).parent.parent / f"data/models/control/{parameter}/model"
    model = load_model(model_path)

    # Load images
    images = []
    for img in img_files:
        images.append(img_file_to_img(img))
    images = array(images)

    # Make predictions
    predicted = model.predict(images, verbose=0)
    predicted = scaler.inverse_transform(predicted).flatten()

    return predicted
