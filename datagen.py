import os
import cv2
import random
import numpy as np
from sklearn.utils import shuffle as shuffle_tuple

import imgaug as ia
from imgaug import augmenters as iaa
ia.seed(2021)
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)


def load_img_batch(img_path_list, img_label_list, num_classes, image_shape):

    batch_size = len(img_path_list)

    X_batch = np.zeros((batch_size, image_shape[0], image_shape[1], 3))
    y_batch = np.zeros((batch_size, num_classes))

    for i in range(batch_size):
        
        img = cv2.cvtColor(cv2.imread(img_path_list[i]), cv2.COLOR_BGR2RGB)

        if img.shape != image_shape:
            img = cv2.resize(img, image_shape)
        
        X_batch[i] = img

        if img_path_list is not None:
            label = img_label_list[i]

            y_batch[i, label] = 1

    return (X_batch, y_batch) if img_label_list is not None else X_batch


def train_batch_generator(img_path_list, img_label_list, num_classes, image_shape, batch_size=32, 
                            shuffle=False, save_to_dir=None, augment=False):
    assert len(img_path_list) == len(img_label_list)

    total_num_img = len(img_path_list)

    if shuffle:
        img_path_list, img_label_list = shuffle_tuple(img_path_list, img_label_list)
    
    batch_index = 0

    while True:
        
        # curr_index will be reset from 0 when (batch_index * batch_size) > total_num_img
        curr_index = (batch_index * batch_size) % total_num_img

        if total_num_img >= (curr_index + batch_size):
            curr_batch_size = batch_size
            batch_index += 1
        else:
            curr_batch_size = total_num_img - curr_index
            batch_index = 0
        
        X_batch, y_batch = load_img_batch(
            img_path_list[curr_index: curr_index + curr_batch_size],
            img_label_list[curr_index: curr_index + curr_batch_size],
            num_classes, 
            image_shape
        )

        if augment:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)
        
        if save_to_dir:
            for img, ori_path in zip(X_batch, img_path_list[curr_index, curr_index + curr_batch_size]):
                img.imwrite(os.path.join(save_to_dir, os.path.basename(ori_path)[:-4] + f" {batch_index}.jpg"))

        X_batch = X_batch / 255.

        X_batch = (X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        yield X_batch, y_batch


def train_batch_generator_hard_triplet(img_path_list, img_label_list, num_classes, image_shape, P=16, K=4, 
                            shuffle=False, augment=False):
    assert len(img_path_list) == len(img_label_list)

    total_num_img = len(img_path_list)

    if shuffle:
        img_path_list, img_label_list = shuffle_tuple(img_path_list, img_label_list)
    
    dic = {}
    for image_path, image_label in zip(img_path_list, img_label_list):
        dic.setdefault(image_label, []).append(image_path)

    selected_labels = [k for k, v in dic.items() if len(v) >= K]

    while True:
        person_ids_sampled = random.sample(list(selected_labels), k=P)
        img_path_sampled = [random.sample(dic[person_id], k=K) for person_id in person_ids_sampled]

        img_path_sampled_list = []
        [img_path_sampled_list.extend(w) for w in img_path_sampled]

        person_ids_sampled_list = []
        tmp_sampled_list = [[w] * K for w in person_ids_sampled]
        [person_ids_sampled_list.extend(w) for w in tmp_sampled_list]

        y_batch = np.array(person_ids_sampled_list)
        X_batch, Y_batch = load_img_batch(
            img_path_sampled_list, person_ids_sampled_list, 
            num_classes=num_classes, image_shape=image_shape
        )

        if augment:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)
        
        X_batch = X_batch / 255.

        X_batch = (X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        yield (X_batch, [y_batch, Y_batch])
    

def predict_batch_generator(img_path_list, image_shape, batch_size=32):

    total_num_img = len(img_path_list)
    
    batch_index = 0

    while True:
        
        # curr_index will be reset from 0 when (batch_index * batch_size) > total_num_img
        curr_index = (batch_index * batch_size) % total_num_img

        if total_num_img >= (curr_index + batch_size):
            curr_batch_size = batch_size
            batch_index += 1
        else:
            curr_batch_size = total_num_img - curr_index
            batch_index = 0
        
        X_batch = load_img_batch(
            img_path_list[curr_index, curr_index + curr_batch_size],
            None,
            1,
            image_shape
        )

        X_batch = X_batch / 255.

        X_batch = (X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        yield X_batch