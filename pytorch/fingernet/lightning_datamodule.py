# fingernet/lightning_datamodule.py

import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class FingerprintDataset(Dataset):
    """Um Dataset que carrega imagens de impressão digital a partir de uma lista de caminhos."""
    def __init__(self, image_paths: list[str]):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Carrega e pré-processa a imagem
        img_pil = Image.open(img_path).convert('L')
        img_np = np.array(img_pil, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0) # Adiciona dimensão de canal
        
        # Retorna o tensor e o caminho original para salvar os resultados depois
        return img_tensor, img_path

class FingerprintDataModule(pl.LightningDataModule):
    """
    Um LightningDataModule para encontrar e carregar dados de impressão digital para inferência.
    """
    def __init__(self, input_path: str, batch_size: int, recursive: bool, num_workers: int = 4):
        super().__init__()
        self.input_path = input_path
        self.batch_size = batch_size
        self.recursive = recursive
        self.num_workers = num_workers if num_workers is not None else 0
        self.image_paths = []

    def setup(self, stage: str | None = None):
        """Encontra todos os caminhos de imagem. Chamado em cada processo (GPU)."""
        if not self.image_paths:
            print(f"--- Buscando imagens em: {self.input_path} ---")
            if os.path.isfile(self.input_path):
                self.image_paths.append(self.input_path)
            elif os.path.isdir(self.input_path):
                extensoes = ['png', 'jpg', 'jpeg', 'bmp', 'tif']
                for ext in extensoes:
                    pattern = f"{self.input_path}/**/*.{ext.lower()}" if self.recursive else f"{self.input_path}/*.{ext.lower()}"
                    self.image_paths.extend(glob.glob(pattern, recursive=self.recursive))
            
            if not self.image_paths:
                print("Nenhuma imagem encontrada.")
            else:
                print(f"Encontradas {len(self.image_paths)} imagens.")

        self.dataset = FingerprintDataset(self.image_paths)

    def predict_dataloader(self):
        """Cria o DataLoader para inferência."""
        # Lightning usará automaticamente um DistributedSampler aqui quando a estratégia DDP for usada
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )