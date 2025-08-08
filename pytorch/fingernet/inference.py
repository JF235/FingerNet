import os
import glob
import argparse
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

# Supondo que seu projeto tenha uma estrutura onde 'fingernet' é um pacote.
from .interface import get_fingernet

def run_inference(input_path: str, output_path: str, weights_path: str, recursive: bool, batch_size: int, device: str, num_gpus: int | None, num_cores: int | None):
    """
    Executa a inferência em imagens usando o FingerNetWrapper com suporte a lotes
    e salva os resultados processados.

    Args:
        input_path (str): Caminho para um arquivo de imagem ou um diretório.
        output_path (str): Caminho para o diretório onde os resultados serão salvos.
        weights_path (str): Caminho para o arquivo de pesos do modelo (.pth).
        recursive (bool): Se True, busca por imagens recursivamente.
        batch_size (int): O tamanho do lote para processamento.
        device (str): O dispositivo a ser utilizado (ex: "cpu" ou "cuda").
        num_gpus (int | None): O número de GPUs a serem utilizadas (ou None para usar todas).
        num_cores (int | None): O número de CPUs a serem utilizadas (ou None para usar todas).
    """
    # 1. Preparação do Modelo e Dispositivo
    print("--- Inicializando FingerNet ---")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        # Set torch number of cores
        if num_cores is not None:
            torch.set_num_threads(num_cores)

    device = torch.device(device)


    try:
        model = get_fingernet(weights_path, device, log=True, num_gpus=num_gpus)
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        return

    print(f"\n--- Buscando imagens em: {input_path} ---")
    
    # 2. Descoberta de Imagens
    image_paths = []
    if os.path.isfile(input_path):
        image_paths.append(input_path)
    elif os.path.isdir(input_path):
        extensoes = ['png', 'jpg', 'jpeg', 'bmp', 'tif']
        for ext in extensoes:
            pattern = f"{input_path}/**/*.{ext.lower()}" if recursive else f"{input_path}/*.{ext.lower()}"
            image_paths.extend(glob.glob(pattern, recursive=recursive))
    
    if not image_paths:
        print("Nenhuma imagem encontrada.")
        return

    # 3. Geração dos Lotes (Batches)
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    print(f"Encontradas {len(image_paths)} imagens, divididas em {len(batches)} lotes de tamanho até {batch_size}.")

    # 4. Loop de Inferência por Lote
    for batch_idx, batch_paths in tqdm(enumerate(batches), total=len(batches), desc="Processando lotes", unit="batch"):
        #print(f"\n--- Processando Lote {batch_idx + 1}/{len(batches)} ---")
        
        # a. Carregar e pré-processar as imagens do lote
        tensor_list = []
        for img_path in batch_paths:
            img_pil = Image.open(img_path).convert('L')
            img_np = np.array(img_pil, dtype=np.float32) / 255.0
            tensor_list.append(torch.from_numpy(img_np))

        # b. Empilhar em um único tensor de lote [B, C, H, W]
        # O torch.stack criará um lote, que é então movido para o dispositivo.
        input_batch = torch.stack(tensor_list).unsqueeze(1).to(device)

        # c. Executar a inferência no lote inteiro
        results = model(input_batch)

        # d. Salvar resultados para cada imagem no lote
        for i, image_path in enumerate(batch_paths):
            #print(f"  Salvando resultados para: {os.path.basename(image_path)}")

            # Desempacotar resultado da i-ésima imagem do lote
            minutiae_tensor = results['minutiae'][i]
            enhanced_tensor = results['enhanced_image'][i]
            mask_tensor = results['segmentation_mask'][i]
            orientation_tensor = results['orientation_field'][i]
            
            # Preparar o diretório de saída
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            result_dir = os.path.join(output_path, base_name)
            os.makedirs(result_dir, exist_ok=True)

            # Mover tensores para CPU e salvar
            mnts_cpu = minutiae_tensor.cpu().numpy()
            enh_cpu = enhanced_tensor.cpu().numpy()
            mask_cpu = mask_tensor.cpu().numpy()
            ori_cpu = orientation_tensor.cpu().numpy()

            # Salvar minúcias
            minutiae_path = os.path.join(result_dir, 'minutiae.txt')
            np.savetxt(minutiae_path, mnts_cpu, fmt=['%.0f', '%.0f','%.6f','%.6f'], header='x, y, angle, score', delimiter=',')

            # Salvar imagem melhorada
            Image.fromarray(enh_cpu).save(os.path.join(result_dir, 'enhanced.png'))
            
            # Salvar máscara
            Image.fromarray(mask_cpu).save(os.path.join(result_dir, 'mask.png'))

            # Salvar campo de orientação (visual e bruto)
            ori_norm_visual = ((ori_cpu - ori_cpu.min()) / (ori_cpu.max() - ori_cpu.min() + 1e-8) * 255).astype(np.uint8)
            Image.fromarray(ori_norm_visual).save(os.path.join(result_dir, 'orientation_visual.png'))
            
            np.save(os.path.join(result_dir, 'orientation_field.npy'), ori_cpu) # SALVANDO O .NPY
            
    print("\n--- Processo de inferência concluído. ---")