
import os
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import normalize

from utils.triplet import DataGeneratorPredict


def top_n_accuracy(n, queries, galleries, sorted_distance_query_gallery):
    count = 0
    galleries = np.array(galleries)
    for query, sorted_galleries_idxs in zip(queries, sorted_distance_query_gallery):
        top_n_idxs = sorted_galleries_idxs[:n]
        if query in galleries[top_n_idxs]:
            count += 1
    
    return count / len(queries) * 100

def mAP(queries, galleries, sorted_distance_query_gallery):
    APs = 0
    for query, sorted_galleries_idxs in zip(queries, sorted_distance_query_gallery):
        AP = 0
        count = 0
        total_ground_truth_positives = galleries.count(query)
        if total_ground_truth_positives > 0:
            for i, idx in enumerate(sorted_galleries_idxs):
                if count == total_ground_truth_positives:
                    break
                if query == galleries[idx]:
                    count += 1
                    AP += count / (i + 1)
            APs += AP / total_ground_truth_positives
    
    return APs / len(queries)

####
# Define model path and test parameters
####
model = load_model("resnet50v2_triplet_added_tricks_rmbn_2.h5", compile=False)
center_loss = True  # change this according to the model used
batch_size = 128

####
# Load test datasets
####
query_dir = "/home/opendata/PersonReID/market1501/query"
gallery_dir = "/home/opendata/PersonReID/market1501/gt_bbox"

query_image_filenames = sorted([filename for filename in os.listdir(query_dir) if filename.endswith(".jpg")])
query_labels = [name[:name.index("_")] for name in query_image_filenames]
query_labels_set = set(query_labels)
query_image_paths = [os.path.join(query_dir, name) for name in query_image_filenames]

gallery_image_filenames = sorted([
    filename for filename in os.listdir(gallery_dir) 
    if filename.endswith(".jpg") and filename not in query_image_filenames and filename[:filename.index("_")] in query_labels_set
])
gallery_labels = [name[:name.index("_")] for name in gallery_image_filenames]
gallery_image_paths = [os.path.join(gallery_dir, name) for name in gallery_image_filenames]

####
# Define model with desired input output
####
if center_loss:
    dense_features = model.get_layer("features_bn").output
    model_extract_features = Model(model.input[0], dense_features)
else:
    dense_features = model.get_layer("triplet").output
    model_extract_features = Model(model.input, dense_features)

model_extract_features.compile()

####
# Get model output of queries and galleries
####
query_generator = DataGeneratorPredict(query_image_paths, batch_size)
query_features = model_extract_features.predict(query_generator, verbose=1)

gallery_generator = DataGeneratorPredict(gallery_image_paths, batch_size)
gallery_features = model_extract_features.predict(gallery_generator, verbose=1)

####
# Compute similarities of queries and galleries
####
query_features = normalize(query_features, norm="l2")
gallery_features = normalize(gallery_features, norm="l2")
similarity_list = np.dot(query_features, np.transpose(gallery_features))
distance_query_gallery = 1 - similarity_list
sorted_distance_query_gallery = np.argsort(distance_query_gallery, axis=1)

####
# Evaluate model
####
print(f"Top-1 accuracy: {top_n_accuracy(1, query_labels, gallery_labels, sorted_distance_query_gallery):.3f} %")
print(f"Top-5 accuracy: {top_n_accuracy(5, query_labels, gallery_labels, sorted_distance_query_gallery):.3f} %")
print(f"Top-10 accuracy: {top_n_accuracy(10, query_labels, gallery_labels, sorted_distance_query_gallery):.3f} %")
print(f"mAP: {mAP(query_labels, gallery_labels, sorted_distance_query_gallery):.3f}")