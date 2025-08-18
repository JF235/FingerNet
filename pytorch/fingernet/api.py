import os
import torch
import pytorch_lightning as pl
import numpy as np
from PIL import Image

from .model import DEFAULT_WEIGHTS_PATH
from .lightning import FingerNetLightning, FingerprintDataModule

# Defina a precisão do matmul para otimização
torch.set_float32_matmul_precision('medium')

def save_results(result_item: dict, output_path: str):
    """
    Salva os resultados da inferência na nova estrutura de diretórios.
    Ex: output/mask/101_1.png, output/enhanced/101_1.png, etc.
    """
    # Pega o nome do arquivo original, incluindo a extensão (ex: '101_1.png')
    original_filename = os.path.basename(result_item['input_path'])
    # Pega o nome do arquivo sem a extensão para os arquivos .txt e .npy (ex: '101_1')
    base_name = os.path.splitext(original_filename)[0]

    # --- Salva cada componente em seu respectivo subdiretório ---

    # Salvar minúcias (.txt)
    minutiae_path = os.path.join(output_path, 'minutiae', f"{base_name}.txt")
    np.savetxt(minutiae_path, result_item['minutiae'], fmt=['%.0f', '%.0f', '%.6f', '%.6f'], header='x, y, angle, score', delimiter=',')

    # Salvar imagem melhorada (.png)
    enhanced_path = os.path.join(output_path, 'enhanced', original_filename)
    Image.fromarray(result_item['enhanced_image']).save(enhanced_path)

    # Salvar máscara (.png)
    mask_path = os.path.join(output_path, 'mask', original_filename)
    Image.fromarray(result_item['segmentation_mask']).save(mask_path)

    # Salvar campo de orientação (codificado em PNG)
    ori_cpu = result_item['orientation_field']

    # Converte de radianos para graus, desloca em +90 e salva como uint8
    orientation_path = os.path.join(output_path, 'ori', original_filename)
    angles_deg_shifted = np.round(np.rad2deg(ori_cpu) + 90).astype(np.uint8)
    Image.fromarray(angles_deg_shifted).save(orientation_path)


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
    weights_path: str = DEFAULT_WEIGHTS_PATH,
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