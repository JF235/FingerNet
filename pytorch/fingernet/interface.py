import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import os

# Importação da classe FingerNet para anotação de tipo.
from .model import FingerNet

class FingerNetWrapper(nn.Module):
    """
    Um wrapper nn.Module completo e de ponta a ponta para o modelo FingerNet.

    Esta classe encapsula o pré-processamento, a inferência do modelo e
    o pós-processamento completo (incluindo detecção de minúcias e NMS),
    com todas as operações otimizadas para execução em GPU.
    """
    def __init__(self, model: FingerNet):
        """
        Inicializa o wrapper.

        Args:
            fingernet_model (FingerNet): Uma instância do modelo FingerNet
                                         já em modo de avaliação (`.eval()`).
        """
        super().__init__()
        self.fingernet = model

    # ==================================================================
    # MÉTODOS DO PIPELINE PRINCIPAL (PÚBLICOS)
    # ==================================================================

    def forward(self, x: torch.Tensor, minutiae_threshold: float = 0.5) -> dict[str, torch.Tensor]:
        """
        Executa o pipeline completo: pré-processamento, inferência e pós-processamento.

        Args:
            x (torch.Tensor): O lote de tensores de entrada com shape [B, 1, H, W].
            minutiae_threshold (float): O limiar de score para considerar um
                                        ponto como uma minúcia.

        Returns:
            dict[str, torch.Tensor]: Um dicionário contendo os resultados finais,
                                     incluindo uma lista de tensores de minúcias,
                                     mapas de segmentação e imagens melhoradas.
                                     Todos os tensores de saída permanecem no mesmo
                                     dispositivo do tensor de entrada.
        """
        # Etapa 1: Pré-processamento com padding.
        padded_x = self.preprocess(x)

        # Etapa 2: Inferência otimizada na FingerNet.
        with torch.no_grad():
            raw_outputs = self.fingernet(padded_x)

        # Etapa 3: Pós-processamento completo na GPU.
        final_outputs = self.postprocess(raw_outputs, threshold=minutiae_threshold)

        return final_outputs

    @staticmethod
    def preprocess(x: torch.Tensor) -> torch.Tensor:
        """Adiciona padding ao lote para tornar as dimensões múltiplas de 8."""
        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        return F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

    def postprocess(self, outputs: dict, threshold: float) -> dict[str, torch.Tensor]:
        """
        Orquestra todas as etapas de pós-processamento na GPU.
        """
        # 1. Binarização e limpeza da máscara de segmentação
        cleaned_mask = self._post_binarize_mask(outputs['segmentation'])
        cleaned_mask_up = self._post_binarize_mask(outputs['segmentation upsample'], upsample_factor=8)

        # 2. Detecção de minúcias (incluindo NMS)
        # O resultado é uma lista de tensores, um para cada imagem no lote.
        final_minutiae_list = self._post_detect_minutiae(outputs, cleaned_mask, threshold)

        # 3. Processamento do campo de orientação
        ori_up = outputs['orientation upsample']
        orientation_field = (torch.argmax(ori_up, dim=1).float() * 2.0 - 90) * torch.pi / 180.0
        orientation_field = orientation_field * cleaned_mask_up

        # 4. Processamento da imagem melhorada
        enh_real = outputs['enhanced_real'].squeeze(1)
        enh_real = enh_real * cleaned_mask_up
        
        # Normalização Min-Max para visualização
        b, h, w = enh_real.shape
        enh_flat = enh_real.view(b, -1)
        enh_min = enh_flat.min(dim=1, keepdim=True)[0]
        enh_max = enh_flat.max(dim=1, keepdim=True)[0]
        enh_norm = (enh_flat - enh_min) / (enh_max - enh_min + 1e-8)
        enh_visual = (enh_norm.view(b, h, w) * 255).byte()

        return {
            'minutiae': final_minutiae_list,
            'enhanced_image': enh_visual,
            'segmentation_mask': (cleaned_mask_up * 255).byte(),
            'orientation_field': orientation_field
        }

    # ==================================================================
    # MÉTODOS AUXILIARES DE PÓS-PROCESSAMENTO (PRIVADOS)
    # ==================================================================

    def _post_binarize_mask(self, seg_map: torch.Tensor, upsample_factor: int = 1) -> torch.Tensor:
        """Binariza e limpa a máscara de segmentação usando Kornia."""
        seg_map_squeezed = seg_map.squeeze(1)
        binarized = torch.round(seg_map_squeezed)
        kernel = torch.ones(5 * upsample_factor, 5 * upsample_factor, device=seg_map.device)
        # Kornia espera um shape [B, C, H, W], por isso o unsqueeze/squeeze
        cleaned = kornia.morphology.opening(binarized.unsqueeze(1), kernel).squeeze(1)
        return cleaned

    def _post_detect_minutiae(self, outputs: dict, cleaned_mask: torch.Tensor, threshold: float) -> list:
        """Detecta, filtra e aplica NMS nas minúcias para um lote inteiro."""
        mnt_score_batch = outputs['minutiae_score'].squeeze(1) * cleaned_mask
        mnt_orient_batch = outputs['minutiae_orientation']
        mnt_x_offset_batch = outputs['minutiae_x_offset']
        mnt_y_offset_batch = outputs['minutiae_y_offset']
        
        batch_size = mnt_score_batch.shape[0]
        final_minutiae_list = []

        for i in range(batch_size):
            # Encontra coordenadas das minúcias acima do limiar
            rows, cols = torch.where(mnt_score_batch[i] > threshold)
            if rows.shape[0] == 0:
                final_minutiae_list.append(torch.empty((0, 4), device=mnt_score_batch.device))
                continue

            # Extrai scores, ângulos e offsets
            scores = mnt_score_batch[i][rows, cols]
            angles_idx = torch.argmax(mnt_orient_batch[i, :, rows, cols], dim=0)
            x_offsets = torch.argmax(mnt_x_offset_batch[i, :, rows, cols], dim=0)
            y_offsets = torch.argmax(mnt_y_offset_batch[i, :, rows, cols], dim=0)
            
            # Calcula valores finais
            angles = (angles_idx * 2.0 - 89.0) * (torch.pi / 180.0)
            x_coords = cols * 8.0 + x_offsets
            y_coords = rows * 8.0 + y_offsets
            
            minutiae_raw = torch.stack([x_coords, y_coords, angles, scores], dim=-1)
            
            # Aplica NMS
            final_minutiae = self._post_nms(minutiae_raw)
            final_minutiae_list.append(final_minutiae)
            
        return final_minutiae_list

    def _post_nms(self, minutiae: torch.Tensor, dist_thresh: float = 16.0, angle_thresh: float = torch.pi/6) -> torch.Tensor:
        """Aplica Non-Maximum Suppression (NMS) em um tensor de minúcias."""
        if minutiae.shape[0] == 0:
            return minutiae

        # Ordena por score (decrescente)
        order = torch.argsort(minutiae[:, 3], descending=True)
        minutiae = minutiae[order]

        # Calcula matriz de distância Euclidiana e angular
        dist_matrix = torch.cdist(minutiae[:, :2], minutiae[:, :2])
        
        # Cálculo da distância angular via broadcasting
        angles1 = minutiae[:, 2].unsqueeze(1) # [N, 1]
        angles2 = minutiae[:, 2].unsqueeze(0) # [1, N]
        angle_delta = torch.abs(angles1 - angles2)
        angle_matrix = torch.minimum(angle_delta, 2 * torch.pi - angle_delta)

        # Máscara para supressão: True onde a distância E o ângulo são menores que o limiar
        suppress_mask = (dist_matrix < dist_thresh) & (angle_matrix < angle_thresh)
        
        keep = torch.ones(minutiae.shape[0], dtype=torch.bool, device=minutiae.device)
        for i in range(minutiae.shape[0]):
            if keep[i]:
                # Suprime todos os outros pontos que estão muito próximos deste
                # torch.where retorna uma tupla, pegamos o primeiro elemento
                suppress_indices = torch.where(suppress_mask[i, i+1:])[0]
                keep[i + 1 + suppress_indices] = False
                
        return minutiae[keep]

def get_fingernet(weights_path: str, log: bool = True, num_gpus: int = 1) -> FingerNetWrapper:
    """
    Factory para criar e carregar uma instância do FingerNetWrapper pronta para uso.

    Esta função lida com a detecção de dispositivo, carregamento do modelo,
    carregamento dos pesos, e encapsulamento no FingerNetWrapper.

    Args:
        weights_path (str): Caminho para os pesos do modelo (.pth).
        log (bool): Se True, exibe detalhes do processo de carregamento.

    Returns:
        FingerNetWrapper: Uma instância do wrapper pronta para inferência.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Arquivo de pesos não encontrado em: {weights_path}")

    # 1. Detectar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if log: print(f"[FingerNet] Dispositivo selecionado: {device}")

    # 2. Instanciar o modelo base
    if log: print("[FingerNet] Carregando arquitetura FingerNet...")
    fingernet_model = FingerNet()

    # 3. Carregar os pesos (state_dict)
    # map_location=device garante que o modelo carregue corretamente
    # independentemente de onde foi salvo (CPU ou GPU).
    if log: print(f"[FingerNet] Carregando pesos de: {weights_path}")
    fingernet_model.load_state_dict(torch.load(weights_path, map_location=device))

    # 4. Colocar o modelo em modo de avaliação (IMPORTANTE!)
    # Isso desativa camadas como Dropout e usa as médias/variâncias
    # de BatchNorm em vez das estatísticas do lote atual.
    fingernet_model.eval()
    if log: print("[FingerNet] Modelo em modo de avaliação (.eval()).")

    # Verifica se a CUDA está disponível e se há mais de uma GPU
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count() if num_gpus is None else num_gpus
        if log: print(f"[FingerNet] Utilizando {num_gpus} GPUs via nn.DataParallel.")
        # Envolve o modelo com DataParallel
        fingernet_model = nn.DataParallel(fingernet_model, device_ids=list(range(num_gpus)))

    # 5. Criar e mover o wrapper para o dispositivo
    # O wrapper, sendo um nn.Module, moverá o fingernet_model interno
    # para o mesmo dispositivo automaticamente.
    if log: print("[FingerNet] Criando e movendo o wrapper para o dispositivo...")
    fnet_wrapper = FingerNetWrapper(model=fingernet_model).to(device)

    if log: print("\n[FingerNet] Modelo pronto para inferência.")
    
    return fnet_wrapper