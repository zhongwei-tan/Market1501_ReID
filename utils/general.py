from tensorflow.keras.losses import categorical_crossentropy

def categorical_crossentropy_label_smoothing(y_true, y_pred):
    label_smoothing = 0.1
    return categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)

from imgaug import augmenters as iaa
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Random Erase
    iaa.Sometimes(
        0.5,
        iaa.Cutout(nb_iterations=1, size=[0.3, 0.4], squared=False)
    ),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Sometimes(
        0.3,
        iaa.Affine(
            rotate=(-10, 10),
            shear=(-8, 8)
        )
    ),
], random_order=True, random_state=2021)