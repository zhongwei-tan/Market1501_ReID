import numpy as np
import math
from sklearn.utils import shuffle as shuffle_tuple
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras import backend as K

from general import seq


def create_model(image_shape, num_person_ids):
    anchor_input = Input(image_shape, name="anchor_input")
    positive_input = Input(image_shape, name="positive_input")
    negative_input = Input(image_shape, name="negative_input")

    cnn_model = MobileNetV2(input_shape=image_shape, alpha=0.5, include_top=False, pooling="max")
    cnn_model.trainable = False

    anchor_embedding = cnn_model(anchor_input)
    positive_embedding = cnn_model(positive_input)
    negative_embedding = cnn_model(negative_input)

    merged_vector = concatenate([anchor_embedding, positive_embedding, negative_embedding])

    dense_anchor = Dense(num_person_ids)(anchor_embedding)
    softmax_anchor_output = Activation("softmax")(dense_anchor)
    
    triplet_model = Model(
        input=[anchor_input, positive_input, negative_input], 
        output=[merged_vector, softmax_anchor_output]
    )

    triplet_model.summary()
    
    return triplet_model



def triplet_loss(y_true, y_pred, alpha=0.3):
    y_pred = K.l2_normalize(y_pred, axis=1)
    batch_num = y_pred.shape.as_list()[-1] / 3

    anchor = y_pred[:, :batch_num]
    positive = y_pred[:, batch_num:2*batch_num]
    negative = y_pred[:, 2*batch_num:3*batch_num]

    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    loss = K.maximum(pos_dist - neg_dist + alpha, 0.0)

    return loss


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