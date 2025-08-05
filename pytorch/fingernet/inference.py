import torch
import os
import time
import datetime
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2  # Importado para salvar as imagens

from .interface import FingerNetUtils, _postprocess

def _preprocess_for_batch(
    image_path: str,
    target_dims: tuple[int, int]
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Pré-processa uma única imagem para um lote, aplicando padding até as dimensões alvo.
    """
    img = Image.open(image_path).convert('L')
    img_np = np.array(img, dtype=np.float32) / 255.0
    original_height, original_width = img_np.shape

    pad_height = target_dims[0] - original_height
    pad_width = target_dims[1] - original_width

    img_padded = np.pad(
        img_np,
        ((0, pad_height), (0, pad_width)),
        mode='constant',
        constant_values=0
    )

    img_tensor = torch.from_numpy(img_padded).unsqueeze(0).unsqueeze(0)
    return img_tensor, (original_height, original_width)


def _prepare_batch_from_paths(
    paths: list[str]
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """
    Converte uma lista de caminhos de imagem em um único tensor de lote (batch).
    """
    if not paths:
        raise ValueError("A lista de caminhos de imagem não pode estar vazia.")

    max_h, max_w = 0, 0
    for path in paths:
        with Image.open(path) as img:
            w, h = img.size
            if h > max_h:
                max_h = h
            if w > max_w:
                max_w = w

    target_h = (max_h + 7) & ~7
    target_w = (max_w + 7) & ~7

    tensor_list = []
    original_shapes = []
    for path in paths:
        tensor, orig_shape = _preprocess_for_batch(path, (target_h, target_w))
        tensor_list.append(tensor)
        original_shapes.append(orig_shape)

    batch_tensor = torch.cat(tensor_list, dim=0)

    return batch_tensor, original_shapes

def _generate_log(
    results: list[dict],
    total_time: float,
    batch_size: int,
    log_dir: str | None = None
):
    """Gera um log do processamento: imprime no stdout por padrão, salva em arquivo se log_dir for fornecido."""
    timestamp = datetime.datetime.now()
    num_images = len(results)
    avg_time = total_time / num_images if num_images > 0 else 0

    log_content = []
    log_content.append("="*50)
    log_content.append("        Log de Inferência - FingerNet")
    log_content.append("="*50)
    log_content.append(f"Data e Hora: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    log_content.append(f"Total de Imagens Processadas: {num_images}")
    log_content.append(f"Tamanho do Lote (Batch Size): {batch_size}")
    log_content.append(f"Tempo Total de Execução: {total_time:.2f} segundos")
    log_content.append(f"Tempo Médio por Imagem: {avg_time:.3f} segundos")
    log_content.append("\n" + "-"*50)
    log_content.append("Resumo por Imagem:")
    log_content.append("-"*50)

    for res in results:
        log_content.append(f"  - Arquivo: {os.path.basename(res['input_path'])}")
        log_content.append(f"    - Minúcias Detectadas: {len(res['minutiae'])}")

    log_content.append("\nFim do log.")

    log_str = "\n".join(log_content)

    if log_dir is None:
        print("\n" + log_str)
    else:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"fingernet_inference_log_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt")
        try:
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write(log_str)
            print(f"\n📄 Log de processamento salvo em: {log_filename}")
        except IOError as e:
            print(f"\n❌ Erro ao salvar o arquivo de log: {e}")

# =================================================================================
# SEÇÃO 2: FUNÇÃO PRINCIPAL DE INFERÊNCIA (COM ALTERAÇÕES)
# =================================================================================

def run_inference(
    fnet: FingerNetUtils,
    image_paths: str | list[str],
    output_dir: str | None = None,
    save_mode: str = 'full',
    batch_size: int = 8,
) -> list[dict[str, any]]:
    """
    Executa a inferência em lote e salva os resultados com base no modo especificado.

    Args:
        fnet: A instância do utilitário FingerNet.
        image_paths: Um único caminho de arquivo ou uma lista deles.
        output_dir: Diretório para salvar os resultados. Se None, nada é salvo.
        save_mode: O que salvar. Opções: 'full', 'enhance', 'mnt', 'ori'.
                   Só é relevante se output_dir for fornecido.
        batch_size: O número de imagens a processar por lote.

    Returns:
        Uma lista de dicionários com os resultados.
    """
    start_time = time.time()

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    all_results = []
    num_images = len(image_paths)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Resultados (modo: {save_mode}) serão salvos em: {output_dir}")

    print(f"Iniciando inferência para {num_images} imagem(ns) com batch size de {batch_size}...")

    with tqdm(total=num_images, desc="Processando Lotes", unit="img") as progress_bar:
        for i in range(0, num_images, batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_tensor, original_shapes = _prepare_batch_from_paths(batch_paths)
            batch_tensor = batch_tensor.to(fnet.device)

            with torch.no_grad():
                batch_outputs = fnet.model(batch_tensor)

            for j in range(len(batch_paths)):
                single_output = {key: val[j].unsqueeze(0) for key, val in batch_outputs.items()}
                mnts, enh, mask, ori = _postprocess(single_output, original_shapes[j])
                result_dict = {
                    "input_path": batch_paths[j], "minutiae": mnts,
                    "enhanced_image": enh, "mask": mask, "orientation_field": ori,
                }
                all_results.append(result_dict)

                # --- LÓGICA DE SALVAMENTO CENTRALIZADA ---
                if output_dir:
                    base_name = os.path.splitext(os.path.basename(result_dict['input_path']))[0]
                    single_item_dir = os.path.join(output_dir, base_name)
                    os.makedirs(single_item_dir, exist_ok=True)

                    if save_mode in ['full', 'enhance']:
                        cv2.imwrite(os.path.join(single_item_dir, "enhanced.png"), result_dict['enhanced_image'])
                    if save_mode in ['full', 'mnt']:
                        np.savetxt(os.path.join(single_item_dir, "minutiae.txt"), result_dict['minutiae'], fmt=["%d", "%d", "%.6f", "%.6f"])
                    if save_mode in ['full', 'ori']:
                        np.save(os.path.join(single_item_dir, "orientation.npy"), result_dict['orientation_field'])
                        ori_img = cv2.normalize(result_dict['orientation_field'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        cv2.imwrite(os.path.join(single_item_dir, "orientation_grayscale.png"), ori_img)
                    if save_mode == 'full':
                        cv2.imwrite(os.path.join(single_item_dir, "mask.png"), result_dict['mask'])
            
            progress_bar.update(len(batch_paths))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n✨ Inferência em lote concluída em {total_time:.2f} segundos.")
    
    _generate_log(all_results, total_time, batch_size, log_dir=output_dir)

    return all_results