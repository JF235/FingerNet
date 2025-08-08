# test.py

import argparse
import os
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from tqdm import tqdm

# Importa os novos módulos do Lightning
from fingernet.lightning_datamodule import FingerprintDataModule
from fingernet.lightning_model import FingerNetLightning

def save_results(result_item: dict, output_path: str):
    """Salva os resultados de uma única imagem."""
    base_name = os.path.splitext(os.path.basename(result_item['input_path']))[0]
    result_dir = os.path.join(output_path, base_name)
    os.makedirs(result_dir, exist_ok=True)

    # Salvar minúcias
    minutiae_path = os.path.join(result_dir, 'minutiae.txt')
    np.savetxt(minutiae_path, result_item['minutiae'], fmt=['%.0f', '%.0f','%.6f','%.6f'], header='x, y, angle, score', delimiter=',')

    # Salvar imagem melhorada
    Image.fromarray(result_item['enhanced_image']).save(os.path.join(result_dir, 'enhanced.png'))
    
    # Salvar máscara
    Image.fromarray(result_item['segmentation_mask']).save(os.path.join(result_dir, 'mask.png'))

    # Salvar campo de orientação (visual e bruto)
    ori_cpu = result_item['orientation_field']
    ori_norm_visual = ((ori_cpu - ori_cpu.min()) / (ori_cpu.max() - ori_cpu.min() + 1e-8) * 255).astype(np.uint8)
    Image.fromarray(ori_norm_visual).save(os.path.join(result_dir, 'orientation_visual.png'))
    np.save(os.path.join(result_dir, 'orientation_field.npy'), ori_cpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de inferência distribuída para a FingerNet com PyTorch Lightning.")
    parser.add_argument('--input-path', type=str, required=True, help="Caminho para uma imagem ou diretório.")
    parser.add_argument('--output-path', type=str, required=True, help="Diretório onde os resultados serão salvos.")
    parser.add_argument('--weights-path', type=str, required=True, help="Caminho para os pesos .pth do modelo.")
    parser.add_argument('-b', '--batch-size', type=int, default=1, help="Tamanho do lote para inferência.")
    parser.add_argument('--recursive', action='store_true', help="Busca por imagens recursivamente.")
    parser.add_argument('--num-cores', type=int, default=4, help="Número de núcleos de CPU para carregar dados (por GPU).")

    args = parser.parse_args()

    # 1. Instanciar o DataModule
    data_module = FingerprintDataModule(
        input_path=args.input_path,
        batch_size=args.batch_size,
        recursive=args.recursive,
        num_workers=args.num_cores
    )

    # 2. Instanciar o Modelo Lightning
    model_module = FingerNetLightning(weights_path=args.weights_path)

    # 3. Configurar e instanciar o Trainer
    # Ele detectará GPUs automaticamente e usará a estratégia DDP.
    trainer = pl.Trainer(
        accelerator="auto",      # "auto" seleciona gpu, tpu, etc.
        devices=-1,              # -1 usa todas as GPUs disponíveis
        strategy="ddp",          # Usa DistributedDataParallel
        logger=False,            # Desativa logs para inferência limpa
        enable_checkpointing=False
    )

    # 4. Executar a inferência
    # O Trainer gerencia o laço, a distribuição e a coleta dos resultados.
    print("\n--- Iniciando a inferência com PyTorch Lightning ---")
    predictions_per_batch = trainer.predict(model=model_module, datamodule=data_module)
    print("\n--- Inferência concluída. Salvando resultados... ---")

    # 5. Salvar os resultados (após a coleta de todas as GPUs)
    # O trainer.predict coleta os resultados de todos os processos no processo principal.
    # A estrutura é uma lista de listas, então precisamos achatá-la.
    all_results = [item for batch in predictions_per_batch for item in batch]
    
    for result_item in tqdm(all_results, desc="Salvando arquivos", unit="imagem"):
        save_results(result_item, args.output_path)

    print("\n--- Processo concluído com sucesso. ---")