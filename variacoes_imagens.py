import os
from PIL import Image, ImageFilter, ImageDraw
from matplotlib import pyplot as plt
import random
def adicionar_elemento_grafico(imagem_teste):

    draw = ImageDraw.Draw(imagem_teste)
    
    width, height = imagem_teste.size

    square_size = random.randint(10, 50)

    top_left_x = random.randint(0, width - square_size)
    top_left_y = random.randint(0, height - square_size)

    bottom_right_x = top_left_x + square_size
    bottom_right_y = top_left_y + square_size

    random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], fill=random_color)

    return imagem_teste

def gera_imagens_similares_experimento(image_folder_imagens_originais, image_folder_perceptual):

    # imagens para testar perceptual hashing de imagens com diferentes operações
    lista_imagens = os.listdir(image_folder_imagens_originais)
    
    for imagem in lista_imagens:

        imagem_selecionada = imagem

        nome_img = imagem_selecionada.split(".")[0]
        extensao = imagem_selecionada.split(".")[1]

        imagem_teste = Image.open(os.path.join(image_folder_imagens_originais, imagem_selecionada))

        imagem_teste.save(os.path.join(image_folder_perceptual, nome_img + "_original." + extensao))

        # operacao blur

        imagem_blur = imagem_teste.filter(ImageFilter.BLUR)

        imagem_blur.save(os.path.join(image_folder_perceptual, nome_img + "_blur." + extensao))

        # operacao preto_branco

        imagem_preto_branco = imagem_teste.convert('L')

        imagem_preto_branco.save(os.path.join(image_folder_perceptual, nome_img + "_preto_branco." + extensao))

        # operacao resize

        imagem_resize = imagem_teste.resize((128, 128))

        imagem_resize.save(os.path.join(image_folder_perceptual, nome_img + "_resize." + extensao))

        # operacao compressao

        imagem_compressao = imagem_teste.filter(ImageFilter.SHARPEN)

        imagem_compressao.save(os.path.join(image_folder_perceptual, nome_img + "_compressao." + extensao))

        # operacao rotacao

        imagem_rotacao = imagem_teste.rotate(90)

        imagem_rotacao.save(os.path.join(image_folder_perceptual, nome_img + "_rotacao." + extensao))

        # operacao flip

        imagem_flip = imagem_teste.transpose(Image.FLIP_LEFT_RIGHT)

        imagem_flip.save(os.path.join(image_folder_perceptual, nome_img + "_flip." + extensao))

        # operacao crop

        imagem_crop = imagem_teste.crop((0, 0, imagem_teste.size[0] // 2, imagem_teste.size[0] // 2))

        imagem_crop.save(os.path.join(image_folder_perceptual, nome_img + "_crop." + extensao))

        imagem_com_quadrado = adicionar_elemento_grafico(imagem_teste.copy())

        imagem_com_quadrado.save(os.path.join(image_folder_perceptual, nome_img + "_with_square." + extensao))

# Colocar diretorios de imagens originais e onde quer que fiquem as imagens modificadas
gera_imagens_similares_experimento(
    r"original_images", 
    r"images"
)

print("Imagens geradas com sucesso!")