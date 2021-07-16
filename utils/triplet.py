import numpy as np
import math
import random
from sklearn.utils import shuffle as shuffle_tuple
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras import backend as K

from .general import seq

random.seed(2021)


def create_model(image_shape, num_person_ids, show_model_summary=False):
    anchor_input = Input(image_shape, name="anchor_input")
    positive_input = Input(image_shape, name="positive_input")
    negative_input = Input(image_shape, name="negative_input")

    cnn_model = MobileNetV2(input_shape=image_shape, alpha=0.5, include_top=False, pooling="max")
    cnn_model.trainable = False

    anchor_embedding = cnn_model(anchor_input)
    positive_embedding = cnn_model(positive_input)
    negative_embedding = cnn_model(negative_input)

    merged_vector = concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=-1, name="triplet")

    dense_anchor = Dense(num_person_ids)(anchor_embedding)
    softmax_anchor_output = Activation("softmax", name="softmax")(dense_anchor)
    
    triplet_model = Model([anchor_input, positive_input, negative_input], [merged_vector, softmax_anchor_output])

    if show_model_summary:
        triplet_model.summary()
    
    return triplet_model


def create_semi_hard_triplet_model(image_shape, num_person_ids, show_model_summary=False):
    cnn_model = MobileNetV2(input_shape=image_shape, alpha=0.5, include_top=False, pooling="max")
    cnn_model.trainable = False

    global_pool = cnn_model.layers[-1].output
    dense_normalized = Lambda(lambda x: K.l2_normalize(x, axis=1), name="triplet")(global_pool)
    
    dense = Dense(num_person_ids)(global_pool)
    softmax_output = Activation("softmax", name="softmax")(dense)
    
    triplet_model = Model(cnn_model.input, [dense_normalized, softmax_output])

    if show_model_summary:
        triplet_model.summary()
    
    return triplet_model


def triplet_loss(y_true, y_pred, alpha=0.3):
    y_pred = K.l2_normalize(y_pred, axis=1)
    batch_num = y_pred.shape.as_list()[-1] // 3

    anchor = y_pred[:, 0:batch_num]
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
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        if self.shuffle:
            batch_x, batch_y = shuffle_tuple(batch_x, batch_y)

        if self.augment:
            batch_x = np.array([np.asarray(load_img(file_path)).astype(np.uint8) for file_path in batch_x]).astype(np.uint8)
            batch_x = seq.augment_images(batch_x)
            batch_x = batch_x / 255.
        else:
            batch_x = np.array([np.asarray(load_img(file_path)) / 255. for file_path in batch_x])

        batch_x = (batch_x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        batch_y = to_categorical(np.array(batch_y), num_classes=self.num_classes)

        return batch_x, batch_y


class DataGeneratorTriplet(Sequence):
    def __init__(self, x_set, y_set, batch_size, num_classes, shuffle=False, augment=False):
        self.x, self.y = x_set, y_set

        # Make dict with key -> person_id, value -> list of associated images
        self.image_to_label = {}
        for image_path, image_label in zip(self.x, self.y):
            self.image_to_label.setdefault(image_label, []).append(image_path)

        # Get only anchor_id with more than 1 image
        self.anchor_filtered = [k for k, v in self.image_to_label.items() if len(v) > 1]

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):

        if self.shuffle:
            random.shuffle(self.anchor_filtered)
        
        # Get random sample of anchor_ids; amount: batch_size
        anchor_ids_sampled = random.sample(self.anchor_filtered, k=self.batch_size)
        # Get candidates of nagetive sample ids
        negative_id_cands = list(set(self.image_to_label.keys()) - set(anchor_ids_sampled))

        # Get anchor and positive image paths
        anchor_positive_list = [tuple(random.sample(self.image_to_label[id], k=2)) for id in anchor_ids_sampled]
        anchor_img_paths, positive_img_paths = zip(*anchor_positive_list)

        # Get negative image_paths
        negative_id_sampled = random.sample(negative_id_cands, k=self.batch_size)
        negative_img_paths = [random.choice(self.image_to_label[id]) for id in negative_id_sampled]

        if self.augment:
            anchor_X_batch = np.array([np.asarray(load_img(file_path)).astype(np.uint8) for file_path in anchor_img_paths]).astype(np.uint8)
            anchor_X_batch = seq.augment_images(anchor_X_batch)

            positive_X_batch = np.array([np.asarray(load_img(file_path)).astype(np.uint8) for file_path in positive_img_paths]).astype(np.uint8)
            positive_X_batch = seq.augment_images(positive_X_batch)

            negative_X_batch = np.array([np.asarray(load_img(file_path)).astype(np.uint8) for file_path in negative_img_paths]).astype(np.uint8)
            negative_X_batch = seq.augment_images(negative_X_batch)
            
        else:
            anchor_X_batch = np.array([np.asarray(load_img(file_path)) for file_path in anchor_img_paths])
            positive_X_batch = np.array([np.asarray(load_img(file_path)) for file_path in positive_img_paths])
            negative_X_batch = np.array([np.asarray(load_img(file_path)) for file_path in negative_img_paths])

        anchor_X_batch = anchor_X_batch / 255.
        positive_X_batch = positive_X_batch / 255.
        negative_X_batch = negative_X_batch / 255.

        # Minus mean, devide by standard_deviation
        anchor_X_batch = (anchor_X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        positive_X_batch = (positive_X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        negative_X_batch = (negative_X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        anchor_Y_batch = to_categorical(np.array(anchor_ids_sampled), num_classes=self.num_classes)

        return ([anchor_X_batch, positive_X_batch, negative_X_batch], [anchor_Y_batch, anchor_Y_batch])


class DataGeneratorHardTriplet(Sequence):
    def __init__(self, x_set, y_set, person_id_num, image_per_person_id, num_classes, shuffle=False, augment=False):
        self.x, self.y = x_set, y_set

        # Make dict with key -> person_id, value -> list of associated images
        self.image_to_label = {}
        for image_path, image_label in zip(self.x, self.y):
            self.image_to_label.setdefault(image_label, []).append(image_path)

        # Get only anchor_id with at least `image_per_person_id`
        self.y_filtered = [k for k, v in self.image_to_label.items() if len(v) >= image_per_person_id]

        self.person_id_num = person_id_num
        self.image_per_person_id = image_per_person_id
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        return math.ceil(len(self.x) / (self.person_id_num * self.image_per_person_id))

    def __getitem__(self, idx):

        if self.shuffle:
            random.shuffle(self.y_filtered)

        # Get random sample of ids; amount: `person_id_num`
        person_ids_chosen = random.sample(self.y_filtered, k=self.person_id_num)
        # For each id, get random sample of associate images; amount: `image_per_person_id`
        img_paths_sampled = [random.sample(self.image_to_label[id], k=self.image_per_person_id) for id in person_ids_chosen]
        img_paths_sampled = [path for paths in img_paths_sampled for path in paths]  # Flattening `img_paths_sampled`

        # Expand person_ids_chosen by `image_per_person_id` times to map with `img_paths_sampled`
        label_sampled = [[id] * self.image_per_person_id for id in person_ids_chosen]
        label_sampled = np.array([label for labels in label_sampled for label in labels])  # Flattening `label_sampled`

        if self.augment:
            X_batch = np.array([np.asarray(load_img(file_path)).astype(np.uint8) for file_path in img_paths_sampled]).astype(np.uint8)
            X_batch = seq.augment_images(X_batch)
        else:
            X_batch = np.array([np.asarray(load_img(file_path)) for file_path in img_paths_sampled])

        X_batch = X_batch / 255.
        X_batch = (X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        Y_batch = to_categorical(np.array(label_sampled), num_classes=self.num_classes)

        return (X_batch, [label_sampled, Y_batch])