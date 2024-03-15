import streamlit as st
from zipfile import ZipFile
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from scipy.spatial.distance import cosine

st.cache_data
def extract_features_and_paths(data_path):
    extraction_dir = 'womenfashion'
    if not os.path.exists(extraction_dir):
        os.makedirs(extraction_dir)

    with ZipFile(data_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_dir)

    image_directory = os.path.join(extraction_dir, 'womenfashion')

    image_paths_list = [file for file in glob.glob(os.path.join(image_directory, '*.*')) if file.endswith(('.jpg', '.png', '.jpeg', 'webp'))]

    base_model = ResNet50(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)

    all_features = []
    all_image_names = []

    for img_path in image_paths_list:
        preprocessed_img = preprocess_image(img_path)
        features = extract_features(model, preprocessed_img)
        all_features.append(features)
        all_image_names.append(os.path.basename(img_path))

    return all_features, all_image_names, model, image_paths_list

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

def recommend_fashion_items_resnet(input_image_path, all_features, all_image_names, model, top_n=5):
    preprocessed_img = preprocess_image(input_image_path)
    input_features = extract_features(model, preprocessed_img)

    similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
    similar_indices = np.argsort(similarities)[-top_n:]

    similar_indices = [idx for idx in similar_indices if idx != all_image_names.index(input_image_path)]

    st.image(input_image_path, caption="Input Image", use_column_width=True)

    for i, idx in enumerate(similar_indices[:top_n], start=1):
        image_path = os.path.join('', all_image_names[idx])
        st.image(image_path, caption=f"Recommendation {i}", use_column_width=True)

def main():
    st.title("Fashion Item Recommender")

    uploaded_file = st.file_uploader("Upload a zip file", type="zip")
    if uploaded_file:
        data_path = 'uploaded.zip'
        with open(data_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        all_features, all_image_names, model, image_paths_list = extract_features_and_paths(data_path)

        st.write("Select an image for recommendation:")
        selected_image_path = st.selectbox("Select an image", image_paths_list)
        recommend_fashion_items_resnet(selected_image_path, all_features, all_image_names, model)

if __name__ == "__main__":
    main()
