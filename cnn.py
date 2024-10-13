import os
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt
import imagehash
import random

image_folder = 'images'
feature_folder = 'features'

# Usando modelo pré treinado CNN (VGG16)
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)  # usando layer 'fc1' 

# Colocando no modelo para processar as imagens no padrao do VGG16
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Extraindo e salvando as features
def extract_and_save_features(image_folder, feature_folder):
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)

    for img_name in os.listdir(image_folder):
        #print(img_name)
        img_path = os.path.join(image_folder, img_name)
        img_array = preprocess_image(img_path)
        features = model.predict(img_array)  
        feature_path = os.path.join(feature_folder, os.path.splitext(img_name)[0] + '.npy')
        np.save(feature_path, features) 

def load_all_features(feature_folder):
    features = []
    file_names = []
    for feature_file in os.listdir(feature_folder):
        feature_path = os.path.join(feature_folder, feature_file)
        feature = np.load(feature_path)
        features.append(feature)
        file_names.append(feature_file)

    return np.array(features), file_names

def get_images_similar_to(image_path, features, file_names):

    feature = features[file_names.index(image_path.split('/')[-1].split('.')[0] + '.npy')]
    feature_distances = []

    for feature_compare, file_name in zip(features, file_names):
        # Linalg = distancia euclidiana
        distances = np.linalg.norm(feature_compare - feature)
        feature_distances.append((distances, file_name))
    
    similar_indices = np.argsort(feature_distances, axis=0)[0:9]
    similar_indices = map(lambda x: x[0], similar_indices)

    lista_final = []

    for i in similar_indices:
        lista_final.append(feature_distances[i])

    return lista_final

def experimento_efetividade_operacoes_imagens_cnn(features, index):
    lista_imagens = os.listdir(image_folder)

    random_index = index

    features, file_names = features
    feature_distances = get_images_similar_to(lista_imagens[random_index], features, os.listdir(feature_folder))

    feature_distances = sorted(feature_distances, key=lambda x: x[0])

    num_img_avaliadas = 9

    imagem_selecionada = os.listdir(image_folder)[random_index]

    acc = 0
    loss = 0
    # print("imagem_selecionada:", imagem_selecionada)
    for i in range(num_img_avaliadas):
        nome_img = feature_distances[i][1].split("_")[0:2]
        nome_img = "_".join(nome_img)
        # print("nome_img:", nome_img)
        if (nome_img in imagem_selecionada):
            acc += 1
        else:
            loss += 1

    #print("acc:", acc, "loss:", loss)
    
    return {
        'accuracy': acc,
        'loss': loss
    }

def plot_images(image_dir, image_path, similar_indices):

    count = 0
    plt.subplot(1, 1, 1)
    img = image.load_img(image_path, target_size=(224, 224))
    plt.imshow(img)
    plt.axis('off')

    for i in similar_indices:
        plt.subplot(9, 9, count + 1)
        count += 1
        img = image.load_img(os.path.join(image_dir, os.listdir(image_dir)[i]), target_size=(224, 224))
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f'Images similar to {os.path.basename(image_path)}', fontsize=20)
    plt.show()

def features_to_hash(feature_folder):
    hashes = []
    for feature_file in os.listdir(feature_folder):
        feature_path = os.path.join(feature_folder, feature_file)
        feature = np.load(feature_path)
        hash = imagehash.phash(feature)
        #print(hash)
        hashes.append(hash)
    return hashes

# * Só precisa chamar uma vez para extrair e salvar as features na pasta 'features'
# extract_and_save_features(image_folder, feature_folder)

# features_to_hash(feature_folder)
# get_images_similar_to('images/monarch.jpg', feature_folder) 