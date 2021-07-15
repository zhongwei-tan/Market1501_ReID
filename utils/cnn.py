import numpy as np
import math
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import load_img
from sklearn.utils import shuffle as shuffle_tuple

from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Model

from general import seq


def create_model(image_shape, num_person_ids):
    cnn_model = MobileNetV2(input_shape=image_shape, alpha=0.5, include_top=False, pooling="max")
    cnn_model.trainable = False
    global_pool = cnn_model.layers[-1].output  # 1 x 1 x 2048
    dense = Dense(num_person_ids)(global_pool)
    softmax_output = Activation("softmax")(dense)
    baseline_model = Model(cnn_model.input, softmax_output)
    baseline_model.summary()
    return baseline_model


# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class DataGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size, num_classes, shuffle=False, augment=False):
        self.x, self.y = x_set, y_set
        self.total_num_image = len(x_set)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):

        # curr_batch_size = self.batch_size if self.total_num_image > ((idx + 1) * self.batch_size) else (self.total_num_image - idx * self.batch_size)

        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        if self.shuffle:
            batch_x, batch_y = shuffle_tuple(batch_x, batch_y)

        if self.augment:
            batch_x = np.array([np.asarray(load_img(file_path)).astype(np.uint8) for file_path in batch_x]).astype(np.uint8)
            batch_x = seq.augment_images(batch_x)
            batch_x = batch_x / 255.
        else:
            batch_x = np.array([np.asarray(load_img(file_name)) / 255. for file_name in batch_x])

        batch_x = (batch_x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        batch_y = to_categorical(np.array(batch_y), num_classes=self.num_classes)

        return batch_x, batch_y