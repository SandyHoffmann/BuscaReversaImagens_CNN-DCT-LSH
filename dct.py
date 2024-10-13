import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import random
import imagehash
import matplotlib.cm as cm

"""
Implementação baseada no artigo:
- Perceptual Hashing using Convolutional Neural
Networks for Large Scale Reverse Image Search
Mathieu Gaillard

- https://www.phash.org/docs/pubs/thesis_zauner.pdf

- https://www.geeksforgeeks.org/discrete-cosine-transform-algorithm-program/
"""

image_folder = 'images'
def DCT_image(img_array):
    dct_values = [[0 for i in range(img_array.shape[1])] for j in range(img_array.shape[0])]
    
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            ci = (2 ** (1/2)) / (img_array.shape[0] ** (1/2))
            cj = (2 ** (1/2)) / (img_array.shape[1] ** (1/2))
            if i == 0:
                ci = 1 / (img_array.shape[0] ** (1/2))
            if j == 0:
                cj = 1 / (img_array.shape[1] ** (1/2))

            somatorio = 0
            for k in range(img_array.shape[0]):
                for l in range(img_array.shape[1]):
                    somakl = img_array[k][l] * np.cos((2 * k + 1) * i * np.pi / (2 * img_array.shape[0])) * np.cos((2 * l + 1) * j * np.pi / (2 * img_array.shape[1]))
                    somatorio += somakl

            dct_values[i][j] = ci * cj * somatorio

    return np.array(dct_values)

def DCT_Perceptual_Hashing_Implementado(img_path, img_name, resize_size=64, dct_block_size=(8,8)):
        image = Image.open(img_path)

        # o resultado é convertido para preto e branco
        img_preto_branco = image.convert('L').resize((resize_size, resize_size), Image.Resampling.LANCZOS)
        # depois é borrado com boxblur para eliminar ruido
        img_com_blur = img_preto_branco.filter(ImageFilter.BoxBlur(radius= 7))
        # transformando para array
        img_array = np.asarray(img_com_blur, dtype = np.float32)
        # utilizando a DCT do OpenCV
        img_dct = cv.dct(img_array)

        # * Implementado manualmente dando resultados identicos a biblioteca
        # img_dct = DCT_image(img_array)
        # plt.imshow(img_dct)
        # plt.show()
        # print(img_dct)

        # cortando o bloco de 8x8
        img_dct = img_dct[0:dct_block_size[0], 0:dct_block_size[0]]
        
        # calculando a mediana
        mediana = np.median(img_dct)

        hash_binario = 0

        img_dct = img_dct.flatten()
        for i in range(img_dct.shape[0]):
            hash_binario <<= 1
            if img_dct[i].flatten()[0] > mediana:
                hash_binario |= 0x1
        hash_binario &= 0xFFFFFFFFFFFFFFFF
        # retornando o hash
        return hash_binario

def DCT_Perceptual_Hashing_LibImageHash(img_path, img_name, resize_size=32, dct_block_size=(8,8)):
    img = Image.open(img_path)
    hash = imagehash.phash(img)
    # hex to decimal
    hex_hash = str(hash)
    hash_valores = int(hex_hash, 16)   

    return hash_valores

def DCT_Perceptual_Hashing(image_folder, resize_size=32, dct_block_size=(8,8)):
    hash_list = []
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        hash_binario = DCT_Perceptual_Hashing_Implementado(img_path, img_name, resize_size, dct_block_size)
        hash_list.append(hash_binario)
    
    return hash_list

"""
Hamming Distance

Valores entre 0 e 1, mais proximo de 0 significa que as imagens sao mais similares
"""
def hamming_distance(hash_referencia, hash_list):
    hamming_list = []
    index_img = 0
    for hash in hash_list:
        hamming = bin(hash[0] ^ hash_referencia[0]).count('1')
        hamming_list.append([hamming/(len(bin(hash[0])) - 2), hash[1], hash[2]])
        index_img += 1
    return hamming_list


def hash_int_to_hex(hash_int):
    return hex(hash_int)


def experimento_efetividade_operacoes_imagens(index):
    # gera_imagens_similares_experimento()

    lista_imagens = os.listdir(image_folder)
    hash_list = []
    for img_name in lista_imagens:
        img_path = os.path.join(image_folder, img_name)
        hash_binario = DCT_Perceptual_Hashing_Implementado(img_path, img_name, resize_size=32, dct_block_size=(8,8))
        hash_list.append((hash_binario, img_name, hash_binario))

    random_index = index

    hamming_dist_list = hamming_distance(hash_list[random_index], hash_list)

    hamming_dist_list = sorted(hamming_dist_list, key=lambda x: x[0])

    imagem_selecionada = os.listdir(image_folder)[random_index]
    img = Image.open(os.path.join(image_folder, imagem_selecionada))

    num_img_avaliadas = 9

    acc = 0
    loss = 0

    for i in range(num_img_avaliadas):
        img = False
        print(f' Imagem: {hamming_dist_list[i][1]} : Distancia de Hamming: {hamming_dist_list[i][0]} | hash: {hash_int_to_hex(hamming_dist_list[i][2])}')
        nome_img = hamming_dist_list[i][1].split('_')[0:2]

        nome_img = '_'.join(nome_img)
        print(nome_img)
        print(imagem_selecionada)
        if (nome_img in imagem_selecionada):
            acc += 1
        else:
            loss += 1

    print(f'Accuracy: {acc} | Loss: {loss}')

    return {
        'accuracy': acc,
        'loss': loss
        }

def main():
    qtd_por_tipo = 5
    tipos_disponiveis = ['original', 'blur', 'preto_branco', 'resize' , 'compressao', 'rotacao', 'flip', 'crop']

    dict_resultados = {}

    data = {'original': {'accuracy': 25, 'loss': 15}, 'blur': {'accuracy': 25, 'loss': 15}, 'preto_branco': {'accuracy': 25, 'loss': 15}, 'resize': {'accuracy': 25, 'loss': 15}, 'compressao': {'accuracy': 25, 'loss': 15}, 'rotacao': {'accuracy': 6, 'loss': 34}, 'flip': {'accuracy': 5, 'loss': 35}, 'crop': {'accuracy': 5, 'loss': 35}}

    labels = list(data.keys())
    colors = plt.colormaps['tab20c'].colors
    colors = random.sample(colors, len(colors))

    print("colors:", colors)

    accuracy = [data[group]['accuracy'] for group in labels]
    loss = [data[group]['loss'] for group in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    plt.bar(x - width/2, accuracy, width, label='Accuracy', color=colors[0])
    plt.bar(x + width/2, loss, width, label='Loss', color=colors[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Quantidade')
    plt.title('Resultados')
    plt.xticks(x, labels)
    plt.legend()

    plt.show()

def plot_imagens_transformacoes():
    plt.subplot(9, 9, 1)
    img = Image.open(os.path.join(image_folder, '0_image_test.jpg'))
    plt.legend('Original')
    plt.imshow(img, )

    plt.subplot(9, 9, 2)
    img = Image.open(os.path.join(image_folder, 'image_test_blur.jpg'))
    plt.legend('Blur')
    plt.imshow(img, )

    plt.subplot(9, 9, 3)
    img = Image.open(os.path.join(image_folder, 'image_test_resize.jpg'))
    plt.legend('Resize')
    plt.imshow(img, )

    plt.subplot(9, 9, 4)
    img = Image.open(os.path.join(image_folder, 'image_test_compressao.jpg'))
    plt.legend('Reduzindo resolução')
    plt.imshow(img, )

    plt.subplot(9, 9, 5)
    img = Image.open(os.path.join(image_folder, 'image_test_rotacao.jpg'))
    plt.legend('Rotacao 90 graus')
    plt.imshow(img, )

    plt.subplot(9, 9, 6)
    img = Image.open(os.path.join(image_folder, 'image_test_flip.jpg'))
    plt.legend('Flip')
    plt.imshow(img, )

    plt.subplot(9, 9, 7)
    img = Image.open(os.path.join(image_folder, 'image_test_crop.jpg'))
    plt.legend('Crop')
    plt.imshow(img, )

    plt.show()


def experimento_busca_imagem_similar():
    hash_list = DCT_Perceptual_Hashing(image_folder, resize_size=64, dct_block_size=(16,16))
    hamming_dist_list = hamming_distance(hash_list[14], hash_list)
    hamming_dist_list.sort()

    imagem_selecionada = os.listdir(image_folder)[14]
    img = Image.open(os.path.join(image_folder, imagem_selecionada))

    plt.subplot(1, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    count = 0
    print(os.listdir(image_folder))
    for i in hamming_dist_list[:5]:
        plt.subplot(5, 5, count + 1)
        count += 1
        print(i[1])
        img = Image.open(os.path.join(image_folder, os.listdir(image_folder)[i[1]]))
        plt.imshow(img, )
        plt.axis('off')


    plt.show()


# experimento_busca_imagem_similar()