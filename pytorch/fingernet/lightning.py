import torch
import pytorch_lightning as pl
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import glob
import warnings

from .model import get_fingernet

class FingerNetLightning(pl.LightningModule):
    """
    Um LightningModule que encapsula o FingerNetWrapper para inferência distribuída.
    """
    def __init__(self, weights_path: str):
        super().__init__()

        warnings.filterwarnings(
            "ignore", 
            message="No device id is",
            category=UserWarning
        )

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

class FingerprintDataset(Dataset):
    """Um Dataset que carrega imagens de impressão digital a partir de uma lista de caminhos."""
    def __init__(self, image_paths: list[str], target_size: tuple[int, int]):
        self.image_paths = image_paths
        self.target_size = target_size  # (H, W)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_pil = Image.open(img_path).convert('L')
        img_np = np.array(img_pil, dtype=np.float32) / 255.0

        # Padding para target_size (fundo branco = 1.0)
        h, w = img_np.shape
        th, tw = self.target_size
        pad_h = th - h
        pad_w = tw - w
        if pad_h < 0 or pad_w < 0:
            # Redimensiona para caber no target_size, mantendo aspecto
            scale = min(th / h, tw / w)
            nh, nw = int(h * scale), int(w * scale)
            img_pil = img_pil.resize((nw, nh), Image.BILINEAR)
            img_np = np.array(img_pil, dtype=np.float32) / 255.0
            h, w = img_np.shape
            pad_h = th - h
            pad_w = tw - w
        # Aplica padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        img_np = np.pad(img_np, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=1.0)
        img_np = np.ascontiguousarray(img_np)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        return img_tensor, img_path

class FingerprintDataModule(pl.LightningDataModule):
    """
    Um LightningDataModule para encontrar e carregar dados de impressão digital para inferência.
    """
    def __init__(self, input: str, batch_size: int, recursive: bool, num_workers: int = 4):
        super().__init__()
        self.input = input
        self.batch_size = batch_size
        self.recursive = recursive
        self.num_workers = num_workers if num_workers is not None else 0
        self.image_paths = []

    def setup(self, stage: str | None = None):
        """Encontra todos os caminhos de imagem. Chamado em cada processo (GPU). Também calcula as dimensões mínima e máxima."""
        if not self.image_paths:
            # print(f"--- Buscando imagens em: {self.input} ---")
            if os.path.isfile(self.input):
                # Verifica se é um arquivo de texto (lista de caminhos)
                _, ext = os.path.splitext(self.input)
                if ext.lower() in ['.txt', '.list']:
                    with open(self.input, 'r') as f:
                        for line in f:
                            path = line.strip()
                            if path:
                                self.image_paths.append(path)
                else:
                    self.image_paths.append(self.input)
            elif os.path.isdir(self.input):
                extensoes = ['png']
                for ext in extensoes:
                    pattern = f"{self.input}/**/*.{ext.lower()}" if self.recursive else f"{self.input}/*.{ext.lower()}"
                    self.image_paths.extend(glob.glob(pattern, recursive=self.recursive))
            if not self.image_paths:
                print("Warning: Nenhuma imagem encontrada.")
            # else:
            #     print(f"Encontradas {len(self.image_paths)} imagens.")

        # Calcula as dimensões mínima e máxima
        min_h, min_w = float('inf'), float('inf')
        max_h, max_w = 0, 0
        for img_path in self.image_paths:
            try:
                with Image.open(img_path) as img:
                    h, w = img.size[1], img.size[0]
            except Exception as e:
                print(f"Erro ao abrir {img_path}: {e}")
                continue
            min_h = min(min_h, h)
            min_w = min(min_w, w)
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        self.min_shape = (min_h, min_w)
        self.max_shape = (max_h, max_w)
        # print(f"Menor dimensão encontrada: {self.min_shape}, Maior dimensão encontrada: {self.max_shape}")

        # Alimenta o dataset com a maior dimensão
        print("len(self.image_paths) =", len(self.image_paths))
        self.dataset = FingerprintDataset(self.image_paths, target_size=self.max_shape)

    def predict_dataloader(self):
        """Cria o DataLoader para inferência."""
        # Lightning usará automaticamente um DistributedSampler aqui quando a estratégia DDP for usada
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )