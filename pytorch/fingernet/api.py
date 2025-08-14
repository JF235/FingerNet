import os
import torch
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob

from .model import get_fingernet
from .lightning import FingerNetLightning, FingerprintDataModule

# Defina a precisão do matmul para otimização
torch.set_float32_matmul_precision('medium')

def save_results(result_item: dict, output_path: str):
    """Salva os resultados"""
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

class ResultsSaveCallback(pl.Callback):
    def __init__(self, output_path: str):
        super().__init__()
        self.output_path = output_path

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list, # 'outputs' é o que você retornou do predict_step
        batch: any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        # Este método é chamado em cada processo da GPU ao final de um lote.
        # 'outputs' aqui é a lista de dicionários para o lote atual.
        if outputs:
            for result_item in outputs:
                save_results(result_item, self.output_path)

def run_lightning_inference(
    input_path: str,
    output_path: str,
    weights_path: str,
    batch_size: int = 1,
    recursive: bool = False,
    num_cores: int = 4,
    devices: int | list[int] | str = "auto",
):
    """
    Executa a inferência distribuída com PyTorch Lightning de forma programática.

    Args:
        input_path (str): Caminho para uma imagem, diretório ou lista de arquivos.
        output_path (str): Diretório onde os resultados serão salvos.
        weights_path (str): Caminho para os pesos .pth do modelo.
        batch_size (int): Tamanho do lote para inferência.
        recursive (bool): Busca por imagens recursivamente.
        num_cores (int): Número de núcleos de CPU para carregar dados (por GPU).
        use_all_gpus (bool): Se True, usa todas as GPUs disponíveis. Se False, usa uma única GPU ou CPU.
    """
    # 1. Instanciar o DataModule
    data_module = FingerprintDataModule(
        input=input_path,
        batch_size=batch_size,
        recursive=recursive,
        num_workers=num_cores
    )

    # 2. Instanciar o Modelo Lightning
    model_module = FingerNetLightning(weights_path=weights_path)

    results_saver = ResultsSaveCallback(output_path=output_path)

    # 3. Configurar e instanciar o Trainer
    # O Trainer gerencia a lógica de distribuição (DDP) automaticamente.
    # Não é necessário usar `torchrun`.
    strategy = "auto"
    if torch.cuda.device_count() > 1:
        try:
            # Esta é uma forma comum de detectar um ambiente de notebook
            get_ipython().__class__.__name__
            strategy = "ddp_notebook"
        except NameError:
            # Se não estiver em um notebook, use a estratégia para scripts
            strategy = "ddp_find_unused_parameters_false"

    trainer = pl.Trainer(
        accelerator="auto",
        devices=devices,
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        callbacks=[results_saver] 
    )

    # 4. Executar a inferência
    trainer.predict(model=model_module, datamodule=data_module)    

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