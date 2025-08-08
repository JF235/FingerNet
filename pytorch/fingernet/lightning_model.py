# fingernet/lightning_model.py

import pytorch_lightning as pl
import torch
import numpy as np
from .interface import get_fingernet

class FingerNetLightning(pl.LightningModule):
    """
    Um LightningModule que encapsula o FingerNetWrapper para inferência distribuída.
    """
    def __init__(self, weights_path: str):
        super().__init__()
        self.weights_path = weights_path
        # O modelo será inicializado no método setup para garantir que ele seja
        # movido para o dispositivo correto em cada processo DDP.
        self.model = None
        # Salva os hiperparâmetros (opcional, mas boa prática)
        self.save_hyperparameters()

    def setup(self, stage: str | None = None):
        """Inicializa o modelo. Chamado em cada processo (GPU)."""
        if self.model is None:
            # self.device é fornecido automaticamente pelo Lightning
            self.model = get_fingernet(weights_path=self.weights_path, device=self.device, log=False)

    def predict_step(self, batch: tuple, batch_idx: int) -> list[dict]:
        """
        Executa um passo de inferência em um lote de dados.
        """
        tensors, paths = batch
        
        # Executa a inferência no lote
        results = self.model(tensors)
        
        # Desempacota os resultados para cada imagem no lote e os move para a CPU
        outputs = []
        num_in_batch = tensors.shape[0]
        for i in range(num_in_batch):
            output_item = {
                'input_path': paths[i],
                'minutiae': results['minutiae'][i].cpu().numpy(),
                'enhanced_image': results['enhanced_image'][i].cpu().numpy(),
                'segmentation_mask': results['segmentation_mask'][i].cpu().numpy(),
                'orientation_field': results['orientation_field'][i].cpu().numpy(),
            }
            outputs.append(output_item)
            
        return outputs