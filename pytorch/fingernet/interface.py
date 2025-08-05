import torch
import os
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist
import cv2

from .model import FingerNet 

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(CURRENT_PATH, "../../models/released_version/Model.pth")

def _preprocess(image_path: str):
    """
    Carrega uma imagem, a converte para tons de cinza, normaliza e aplica padding
    para garantir que suas dimensões sejam múltiplas de 8.
    """
    img = Image.open(image_path).convert('L')
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    original_height, original_width = img_np.shape
    
    pad_height = (8 - original_height % 8) % 8
    pad_width = (8 - original_width % 8) % 8
    
    img_padded = np.pad(img_np, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    
    img_tensor = torch.from_numpy(img_padded).unsqueeze(0).unsqueeze(0)
    
    return img_tensor, (original_height, original_width)

def _angle_delta(A, B, max_D=np.pi * 2):
    """Calcula a menor diferença entre dois ângulos."""
    delta = np.abs(A - B)
    return np.minimum(delta, max_D - delta)

def _nms(minutiae: np.ndarray, dist_thresh: float = 16.0, angle_thresh: float = np.pi/6):
    """Aplica Non-Maximum Suppression (NMS) na lista de minúcias."""
    if minutiae.shape[0] == 0:
        return minutiae

    minutiae = minutiae[minutiae[:, 3].argsort()[::-1]]
    dist_matrix = cdist(minutiae[:, :2], minutiae[:, :2], 'euclidean')
    angle_matrix = cdist(minutiae[:, 2].reshape(-1, 1), minutiae[:, 2].reshape(-1, 1), _angle_delta)
    
    suppress_mask = (dist_matrix < dist_thresh) & (angle_matrix < angle_thresh)
    
    keep = np.ones(len(minutiae), dtype=bool)
    for i in range(len(minutiae)):
        if keep[i]:
            suppress_indices = np.where(suppress_mask[i, i+1:])[0]
            keep[i + 1 + suppress_indices] = False
            
    return minutiae[keep]

def _binarize_mask(seg_map_np: np.ndarray, threshold: float = 0.5, scale_factor: int = 1):
    """Binariza a máscara de segmentação e aplica uma operação morfológica para limpar ruídos."""
    binarized = np.round(seg_map_np).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5 * scale_factor, 5 * scale_factor))
    cleaned_mask = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
    return cleaned_mask

def _detect_minutiae(outputs: dict, cleaned_mask: np.ndarray, threshold: float = 0.5):
    """Detecta e filtra minúcias a partir das saídas do modelo."""
    mnt_score = outputs['minutiae_score'].squeeze().cpu().numpy()
    mnt_orient = outputs['minutiae_orientation'].squeeze().cpu().numpy()
    mnt_x_offset = outputs['minutiae_x_offset'].squeeze().cpu().numpy()
    mnt_y_offset = outputs['minutiae_y_offset'].squeeze().cpu().numpy()

    mnt_score *= cleaned_mask
    rows, cols = np.where(mnt_score > threshold)
    
    if len(rows) == 0:
        return np.empty((0, 4), dtype=np.float32)
        
    scores = mnt_score[rows, cols]
    angles_idx = np.argmax(mnt_orient[:, rows, cols], axis=0)
    x_offsets = np.argmax(mnt_x_offset[:, rows, cols], axis=0)
    y_offsets = np.argmax(mnt_y_offset[:, rows, cols], axis=0)
    
    angles = (angles_idx * 2.0 - 89.0) * np.pi / 180.0
    x_coords = cols * 8 + x_offsets
    y_coords = rows * 8 + y_offsets
    
    minutiae_raw = np.stack([x_coords, y_coords, angles, scores], axis=-1)

    return _nms(minutiae_raw)

def _postprocess(outputs: dict, original_shape: tuple):
    """Orquestra todas as etapas de pós-processamento."""
    seg_map = outputs['segmentation'].squeeze().cpu().numpy()
    seg_up = outputs['segmentation upsample'].squeeze().cpu().numpy()
    ori_up = outputs['orientation upsample'].squeeze().cpu().numpy()
    enh_real = outputs['enhanced_real'].squeeze().cpu().numpy()

    # 1. Binarização da máscara
    cleaned_mask = _binarize_mask(seg_map)
    cleaned_up = _binarize_mask(seg_up, scale_factor=8)

    # 2. Detecção de minúcias
    final_minutiae = _detect_minutiae(outputs, cleaned_mask)

    # 3. Processamento da imagem melhorada e campo de orientação
    orientation_field = (np.argmax(ori_up, axis=0) * 2 - 90) * np.pi / 180.0
    orientation_field[cleaned_up == 0] = 0
    enh_real[cleaned_up == 0] = 0
    enh_visual = (cv2.normalize(enh_real, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
    
    # Retorna na ordem solicitada: mnts, enh, mask, ori
    return final_minutiae, enh_visual, cleaned_up * 255, orientation_field

class FingerNetUtils:
    """
    Classe de alto nível para encapsular o modelo FingerNet e simplificar
    o processo de inferência, orquestrando as etapas de pré e pós-processamento.
    """
    def __init__(self, model: FingerNet, device: torch.device):
        self.model = model
        self.device = device

    def run(self, image_path: str):
        """
        Executa o pipeline completo de inferência para uma determinada imagem.

        Args:
            image_path (str): O caminho para o arquivo de imagem da impressão digital.

        Returns:
            tuple: Uma tupla contendo (minutiae, enhanced_image, mask, orientation_field).
        """
        # Etapa 1: Pré-processamento
        image_tensor, original_shape = _preprocess(image_path)
        image_tensor = image_tensor.to(self.device)

        # Etapa 2: Inferência (sem cálculo de gradientes para otimização)
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Etapa 3: Pós-processamento
        minutiae, enhanced, mask, orientation = _postprocess(outputs, original_shape)

        return minutiae, enhanced, mask, orientation

    def __call__(self, image_path: str):
        """
        Permite que a instância da classe seja chamada diretamente como uma função.
        Delega a execução para o método run().
        """
        return self.run(image_path)


def get_fingernet(weights_path: str = WEIGHTS_PATH, *, log: bool = False):
    """
    Factory para criar e carregar uma instância do FingerNet pronta para uso.
    Parâmetros:
        weights_path (str): Caminho para os pesos do modelo.
        log (bool): Se True, exibe detalhes do carregamento.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Arquivo de pesos não encontrado em: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if log:
        print(f"[FingerNet] Dispositivo selecionado: {device}")
        print(f"[FingerNet] Carregando arquitetura...")
    fingernet_model = FingerNet().to(device)
    if log:
        print(f"[FingerNet] Carregando pesos de: {weights_path}")
    fingernet_model.load_state_dict(torch.load(weights_path, map_location=device))
    fingernet_model.eval()
    if log:
        print("[FingerNet] Modelo pronto para inferência.")
    fnet = FingerNetUtils(model=fingernet_model, device=device)
    if log:
        print("[FingerNet] Instância utilitária criada.")
    return fnet
