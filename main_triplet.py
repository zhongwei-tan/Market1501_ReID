import os
import PIL
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa
# from datagen import train_batch_generator_hard_triplet, predict_batch_generator
from utils.triplet import create_semi_hard_triplet_model, create_model, DataGeneratorTriplet, DataGeneratorHardTriplet, triplet_loss

from utils.general import categorical_crossentropy_label_smoothing
# tf.device("/cpu:0")

# Set training parameters
image_shape = (128, 64, 3)  # h x w x c
learning_rate = 1 
# learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 10, 0.96, staircase=True)
batch_size = 128
person_id_num = 32; image_per_person_id = 4  # parameters for DataGeneratorHardTriplet()
num_epoch = 200
use_label_smoothing = True
use_semi_hard_triplet_loss = True

# Import and preprocess data
train_image_dir = "/home/opendata/PersonReID/market1501/bounding_box_train"

train_image_filenames = sorted([filename for filename in os.listdir(train_image_dir) if filename.endswith(".jpg")])
train_image_paths = [os.path.join(train_image_dir, name) for name in train_image_filenames]

train_person_ids = [name[:4] for name in train_image_filenames]
label_encoder = LabelEncoder()
label_encoder.fit(train_person_ids)
train_person_ids_encoded = label_encoder.transform(train_person_ids)
num_person_ids = len(set(train_person_ids_encoded))

train_img_paths, val_img_paths, train_person_ids, val_person_ids = train_test_split(
    train_image_paths, train_person_ids_encoded, test_size=0.4, random_state=2021, stratify=train_person_ids_encoded)
print(f"\nData info:")
print(f"# train images: {len(train_img_paths)}, # val images: {len(val_img_paths)}, # image labels: {num_person_ids}\n")

# Contruct model
model = create_semi_hard_triplet_model(image_shape, num_person_ids) if use_semi_hard_triplet_loss else create_model(image_shape, num_person_ids)

ce_loss = categorical_crossentropy_label_smoothing if use_label_smoothing else "categorical_crossentropy"
t_loss = tfa.losses.TripletSemiHardLoss(margin=0.3) if use_semi_hard_triplet_loss else triplet_loss
loss = {"triplet": t_loss, "softmax": ce_loss}
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=loss, metrics={"triplet": "accuracy", "softmax": "accuracy"}, loss_weights=None)

# Train model
checkpoint_path = "model_checkpoint"
checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_triplet_accuracy", verbose=1, save_best_only=True)
callbacks = [checkpoint]

if use_semi_hard_triplet_loss:
    train_generator = DataGeneratorHardTriplet(train_img_paths, train_person_ids, person_id_num, image_per_person_id, num_classes=num_person_ids, shuffle=True, augment=True)
    val_generator = DataGeneratorHardTriplet(val_img_paths, val_person_ids, person_id_num, image_per_person_id, num_classes=num_person_ids)
else:
    train_generator = DataGeneratorTriplet(train_img_paths, train_person_ids, batch_size=batch_size, num_classes=num_person_ids, shuffle=True, augment=True)
    val_generator = DataGeneratorTriplet(val_img_paths, val_person_ids, batch_size=batch_size, num_classes=num_person_ids)

model.fit(
    train_generator,
    epochs=num_epoch,
    validation_data=val_generator,
    callbacks=callbacks,
    shuffle=True,
)

# train_generator = train_batch_generator_hard_triplet(
#     train_img_paths, train_person_ids, num_person_ids, image_shape,
#     shuffle=True, augment=True,
# )
# val_generator = train_batch_generator_hard_triplet(
#     val_img_paths, val_person_ids, num_person_ids, image_shape,
# )

# model.fit(
#     train_generator,
#     steps_per_epoch=len(train_image_paths) // batch_size,
#     validation_data=val_generator,
#     validation_steps=len(val_img_paths) // batch_size,
#     verbose=True,
#     shuffle=True,
#     epochs=num_epoch,
#     callbacks=callbacks
# )

print("Training completed and model saved.")
model.save("mobilenetv2_triplet.h5")