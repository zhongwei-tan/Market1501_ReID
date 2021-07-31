import os
import PIL
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from utils.cnn import create_model, DataGenerator
from utils.general import categorical_crossentropy_label_smoothing


# Set training parameters
image_shape = (128, 64, 3)  # h x w x c
learning_rate = 0.0001
# learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 10, 0.96, staircase=True)
batch_size = 128
num_epoch = 200
use_label_smoothing = False

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
print(f"# train images: {len(train_img_paths)}, # val images: {len(val_img_paths)}, # image labels: {num_person_ids}")

# Contruct model
baseline_model = create_model(image_shape, num_person_ids)

loss = categorical_crossentropy_label_smoothing if use_label_smoothing else "categorical_crossentropy"
optimizer = Adam(learning_rate=learning_rate)
baseline_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

# Train model
checkpoint_path = "model_checkpoints/model_checkpoint-{epoch:02d}-{loss:.4f}.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_accuracy", verbose=1, save_best_only=True, save_freq=int(50 * len(train_image_paths) / batch_size))
callbacks = [checkpoint]

train_generator = DataGenerator(train_img_paths, train_person_ids, batch_size=batch_size, num_classes=num_person_ids, shuffle=True, augment=True)
val_generator = DataGenerator(val_img_paths, val_person_ids, batch_size=batch_size, num_classes=num_person_ids)
baseline_model.fit(
    train_generator,
    epochs=num_epoch,
    validation_data=val_generator,
    callbacks=callbacks,
    shuffle=True,
)

# train_generator = train_batch_generator(
#     train_img_paths, train_person_ids, num_person_ids, image_shape, batch_size,
#     shuffle=True, augment=True,
# )
# val_generator = train_batch_generator(
#     val_img_paths, val_person_ids, num_person_ids, image_shape, batch_size,
#     shuffle=True, augment=True,
# )

# baseline_model.fit(
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
baseline_model.save("mobilenetv2_freeze_01.h5")