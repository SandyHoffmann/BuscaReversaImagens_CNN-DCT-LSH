from dct import experimento_efetividade_operacoes_imagens as experimento_efetividade_operacoes_imagens_dct
from cnn import experimento_efetividade_operacoes_imagens_cnn, load_all_features
from lsh import inicia_lsh, experimento_efetividade_operacoes_imagens as experimento_efetividade_operacoes_imagens_lsh
import matplotlib.pyplot as plt
import random
import numpy as np
import os
# def sorteia_imagens(tipo):
image_folder = 'images'

def main():
    qtd_por_tipo = 2
    tipos_disponiveis = ['original', 'blur', 'preto_branco', 'resize' , 'compressao', 'rotacao', 'flip', 'crop', 'with_square']
    label_tipos_disponiveis = ['Original', 'Blur Gaussiano', 'Preto e Branco', 'Mudança de Tamanho' , 'Realce de nitidez', 'Rotação (90%)', 'Espelhar', 'Corte', 'Com elemento quadrado']
    
    dict_resultados_dct = {}
    dict_resultados_lsh = {}
    dict_resultados_cnn = {}

    lista_imagens_sorteadas = {}

    lista_imagens = os.listdir(image_folder)
    print("lista_imagens:", lista_imagens)
    
    for tipo in tipos_disponiveis:
        for i in range(qtd_por_tipo):
            imagens_disponiveis = list(filter(lambda nome: tipo in nome, lista_imagens))
            print("imagens_disponiveis:", imagens_disponiveis)

            random_index_img_disponivel = random.randint(0, len(imagens_disponiveis) - 1)	

            random_index = lista_imagens.index(imagens_disponiveis[random_index_img_disponivel])
            if not lista_imagens_sorteadas.get(tipo):
                lista_imagens_sorteadas[tipo] = []
            lista_imagens_sorteadas[tipo].append(random_index)

    print("lista_imagens_sorteadas:", lista_imagens_sorteadas)


    #!para dct
    for (tipo, label) in zip(tipos_disponiveis, label_tipos_disponiveis):
        dict_resultados_dct[label] = {"accuracy": 0, "loss": 0}
        for i in range(qtd_por_tipo):
            resultados = experimento_efetividade_operacoes_imagens_dct(lista_imagens_sorteadas[tipo][i])
            dict_resultados_dct[label]['accuracy'] += resultados['accuracy']
            dict_resultados_dct[label]['loss'] += resultados['loss']

    print(f'DCT: {dict_resultados_dct}')

    #!para lsh
    lsh = inicia_lsh()

    for (tipo, label) in zip(tipos_disponiveis, label_tipos_disponiveis):
        dict_resultados_lsh[label] = {"accuracy": 0, "loss": 0}
        for i in range(qtd_por_tipo):
            resultados = experimento_efetividade_operacoes_imagens_lsh(lsh, lista_imagens_sorteadas[tipo][i])
            dict_resultados_lsh[label]['accuracy'] += resultados['accuracy']
            dict_resultados_lsh[label]['loss'] += resultados['loss']

    print(f'LSH: {dict_resultados_lsh}')

    #!para cnn
    features, file_names = load_all_features('features')
    for (tipo, label) in zip(tipos_disponiveis, label_tipos_disponiveis):
        dict_resultados_cnn[label] = {"accuracy": 0, "loss": 0}
        for i in range(qtd_por_tipo):
            resultados = experimento_efetividade_operacoes_imagens_cnn((features, file_names), lista_imagens_sorteadas[tipo][i])
            dict_resultados_cnn[label]['accuracy'] += resultados['accuracy']
            dict_resultados_cnn[label]['loss'] += resultados['loss']

    print(f'CNN: {dict_resultados_cnn}')

    fig, ax = plt.subplots()


    labels = list(dict_resultados_dct.keys())
    colors = plt.colormaps['tab20c'].colors
    colors = random.sample(colors, len(colors))

    print("colors:", colors)
    colors = ["#83c5be", "#22577a", "#ff7f50"]

    accuracy_dct = [dict_resultados_dct[group]['accuracy'] for group in labels]
    accuracy_lsh = [dict_resultados_lsh[group]['accuracy'] for group in labels]
    accuracy_cnn = [dict_resultados_cnn[group]['accuracy']  for group in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    ax.bar(x - width, accuracy_dct, width, label='DCT acertos', color=colors[0])
    ax.bar(x, accuracy_cnn, width, label='CNN acertos', color=colors[1])
    ax.bar(x + width, accuracy_lsh, width, label='LSH acertos', color=colors[2])

    ax.legend(fontsize=12)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Quantidade', fontsize=16)
    # ax.set_title('Resultados Experimento 1 - DCT', fontsize=20)
    ax.set_title('Resultados Experimento Comparativo', fontsize=20)

    ax.set_xticks(x, labels,  fontsize=12)

    plt.show()

main()