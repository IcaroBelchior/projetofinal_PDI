import cv2
import numpy as np
import argparse
import os

def exibir_salvar(nome, imagem):
    """
    Função para exibir e salvar as imagens intermediárias.
    """
    cv2.imshow(nome, imagem)
    cv2.imwrite(nome + ".png", imagem)  # Salva a imagem no disco
    cv2.waitKey(0)

def service_conversao(image_path):
    """
    Carrega a imagem e converte para HSV.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Imagem não encontrada: {image_path}")
    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    exibir_salvar("1-Imagem Original", image)
    exibir_salvar("2-Imagem HSV", image_hsv)
    
    return image, image_hsv

def service_segmentacao_vacas(image_hsv):
    """
    Segmenta as vacas na imagem usando uma faixa de cor branca no espaço HSV.
    """
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(image_hsv, lower_white, upper_white)

    exibir_salvar("3-Segmentação Vacas", mask)
    return mask

def service_segmentacao_moscas(image_hsv):
    """
    Segmenta as moscas-brancas usando limiarização Otsu no canal de valor.
    """
    v_channel = image_hsv[:, :, 2]
    _, mask = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    exibir_salvar("3-Segmentação Moscas", mask)
    return mask

def service_morphological_cleanup(mask):
    """
    Aplica operação de fechamento morfológico para remover buracos internos nos objetos.
    """
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    exibir_salvar("4-Fechamento Morfológico", cleaned)
    return cleaned

def service_erosao(mask):
    """
    Aplica erosão para remover pequenos ruídos.
    """
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    
    exibir_salvar("5-Erosão", eroded)
    return eroded

def service_bounding_box(image, mask, area_min=100, method='circle'):
    """
    Encontra os objetos na máscara e desenha um contorno na imagem original.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_min:
            count += 1
            if method == 'circle':
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(output, center, radius, (0, 255, 0), 2)
            else:  
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(output, f'Contagem: {count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    exibir_salvar("6-Resultado Final", output)
    return output, count

def experiment_vacas(image_path):
    """
    Pipeline completo para detecção de vacas.
    """
    original, hsv = service_conversao(image_path)
    mask = service_segmentacao_vacas(hsv)
    mask = service_morphological_cleanup(mask)
    mask = service_erosao(mask)
    output, count = service_bounding_box(original, mask, area_min=200, method='circle')

    print(f"Número de vacas detectadas: {count}")

def experiment_moscas(image_path):
    """
    Pipeline completo para detecção de moscas-brancas.
    """
    original, hsv = service_conversao(image_path)
    mask = service_segmentacao_moscas(hsv)
    mask = service_morphological_cleanup(mask)
    mask = service_erosao(mask)
    output, count = service_bounding_box(original, mask, area_min=50, method='rectangle')

    print(f"Número de moscas-brancas detectadas: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processamento de Imagens para Smart Farming: Contagem de Bovinos e Moscas-Brancas"
    )
    parser.add_argument("experiment", choices=["vacas", "moscas"],
                        help="Tipo de experimento a ser executado (vacas ou moscas)")
    parser.add_argument("image_path", help="Caminho da imagem a ser processada")
    args = parser.parse_args()

    if args.experiment == "vacas":
        experiment_vacas(args.image_path)
    elif args.experiment == "moscas":
        experiment_moscas(args.image_path)
    
    cv2.destroyAllWindows()  # Fecha todas as janelas quando terminar
