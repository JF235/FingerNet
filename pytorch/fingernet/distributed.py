# fingernet/distributed.py

import os
import time
import numpy as np
import cv2
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# Reutiliza as funções de "peças" dos módulos originais e inalterados
from .inference import _prepare_batch_from_paths, _generate_log
from .interface import _postprocess, FingerNetUtils

def run_distributed_inference(
    fnet: FingerNetUtils,
    image_paths: list[str],
    output_dir: str,
    save_mode: str,
    batch_size: int,
    is_main_process: bool,
):
    """
    Orquestra a inferência em um ambiente distribuído.
    Cada processo (GPU) executa esta função.
    """
    start_time = time.time()
    
    # 1. Criar um sampler que distribui os dados entre os processos
    sampler = DistributedSampler(image_paths, shuffle=False)
    
    # 2. Criar um DataLoader para gerar lotes para este processo específico
    dataloader = DataLoader(
        image_paths,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2, # Pode ajustar conforme seu sistema
        pin_memory=True,
    )

    if output_dir and is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Resultados (modo: {save_mode}) serão salvos em: {output_dir}")

    local_results = []  # Armazena os resultados (resumo) deste processo

    # 3. Iterar sobre os dados designados para este processo
    progress_bar = tqdm(dataloader, desc=f"[Rank {fnet.device.index}] Processando", unit="batch", disable=not is_main_process)

    for batch_paths in progress_bar:
        batch_tensor, original_shapes = _prepare_batch_from_paths(list(batch_paths))
        batch_tensor = batch_tensor.to(fnet.device)

        with torch.no_grad():
            # Acessa o modelo original através de .module
            batch_outputs = fnet.model.module(batch_tensor)

        # A lógica de pós-processamento e salvamento é idêntica à original,
        # mas cada processo a executa em paralelo para seus próprios dados.
        for j in range(len(batch_paths)):
            single_output = {key: val[j].unsqueeze(0) for key, val in batch_outputs.items()}
            mnts, enh, mask, ori = _postprocess(single_output, original_shapes[j])
            
            # Cada processo salva seus próprios arquivos. Não há conflitos.
            if output_dir:
                base_name = os.path.splitext(os.path.basename(batch_paths[j]))[0]
                single_item_dir = os.path.join(output_dir, base_name)
                os.makedirs(single_item_dir, exist_ok=True)

                if save_mode in ['full', 'enhance']:
                    cv2.imwrite(os.path.join(single_item_dir, "enhanced.png"), enh)
                if save_mode in ['full', 'mnt']:
                    np.savetxt(os.path.join(single_item_dir, "minutiae.txt"), mnts, fmt=["%d", "%d", "%.6f", "%.6f"])
                if save_mode in ['full', 'ori']:
                    np.save(os.path.join(single_item_dir, "orientation.npy"), ori)
                    ori_img = cv2.normalize(ori, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imwrite(os.path.join(single_item_dir, "orientation_grayscale.png"), ori_img)
                if save_mode == 'full':
                    cv2.imwrite(os.path.join(single_item_dir, "mask.png"), mask)

            # Guarda um resumo para o log final
            local_results.append({
                "input_path": batch_paths[j],
                "minutiae": mnts, # Passa as minúcias para contagem no log
            })

    # 4. Sincronizar e coletar resultados para o log
    dist.barrier() # Espera todos os processos terminarem
    
    # Prepara uma lista para receber os objetos de todos os processos
    all_processes_results = [None] * dist.get_world_size()
    
    # Coleta a lista 'local_results' de cada processo
    dist.all_gather_object(all_processes_results, local_results)

    # 5. O processo principal gera o log consolidado
    if is_main_process:
        # Achata a lista de listas em uma única lista de resultados
        final_results = [item for sublist in all_processes_results for item in sublist]
        total_time = time.time() - start_time
        
        print(f"\n✨ Inferência distribuída para {len(final_results)} imagens concluída em {total_time:.2f} segundos.")
        
        # Reutiliza a função de log original
        _generate_log(final_results, total_time, batch_size * dist.get_world_size(), log_dir=output_dir)