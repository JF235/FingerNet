# fingernet.py
"""
Biblioteca principal para o modelo FingerNet.

Este arquivo contém uma única classe `FingerNet` que encapsula:
1. A arquitetura completa da rede em PyTorch.
2. Métodos internos para pré-processamento, pós-processamento e extração de minucias.
3. Métodos públicos para executar a inferência em diferentes estágios (máscara, realce, completo).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import signal, spatial, sparse
import PIL.Image

class FingerNet(nn.Module):
    """
    Implementação da FingerNet em PyTorch, com métodos de inferência integrados.
    """
    def __init__(self):
        super(FingerNet, self).__init__()
        self._build_network()

    # --- Métodos de Construção da Rede (Internos) ---

    def _build_network(self):
        """Define todas as camadas da arquitetura da rede."""
        # Bloco de Normalização
        self.img_norm = self._img_normalization_layer()

        # Blocos VGG para extração de características
        self.conv1_1 = self._conv_bn_prelu(1, 64, 3)
        self.conv1_2 = self._conv_bn_prelu(64, 64, 3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = self._conv_bn_prelu(64, 128, 3)
        self.conv2_2 = self._conv_bn_prelu(128, 128, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = self._conv_bn_prelu(128, 256, 3)
        self.conv3_2 = self._conv_bn_prelu(256, 256, 3)
        self.conv3_3 = self._conv_bn_prelu(256, 256, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloco ASPP Multi-escala
        self.atrousconv4_1 = self._conv_bn_prelu(256, 256, 3, dilation=1)
        self.convori_1_1 = self._conv_bn_prelu(256, 128, 1)
        self.ori_1_2 = nn.Conv2d(128, 90, 1, padding='same')
        self.convseg_1_1 = self._conv_bn_prelu(256, 128, 1)
        self.seg_1_2 = nn.Conv2d(128, 1, 1, padding='same')
        self.atrousconv4_2 = self._conv_bn_prelu(256, 256, 3, dilation=4)
        self.convori_2_1 = self._conv_bn_prelu(256, 128, 1)
        self.ori_2_2 = nn.Conv2d(128, 90, 1, padding='same')
        self.convseg_2_1 = self._conv_bn_prelu(256, 128, 1)
        self.seg_2_2 = nn.Conv2d(128, 1, 1, padding='same')
        self.atrousconv4_3 = self._conv_bn_prelu(256, 256, 3, dilation=8)
        self.convori_3_1 = self._conv_bn_prelu(256, 128, 1)
        self.ori_3_2 = nn.Conv2d(128, 90, 1, padding='same')
        self.convseg_3_1 = self._conv_bn_prelu(256, 128, 1)
        self.seg_3_2 = nn.Conv2d(128, 1, 1, padding='same')
        
        # Bloco de Realce (Enhancement)
        self.enh_img_real_1 = nn.Conv2d(1, 90, 25, padding='same')
        self.enh_img_imag_1 = nn.Conv2d(1, 90, 25, padding='same')
        
        # Bloco de Extração de Minucias
        self.convmnt_1_1 = self._conv_bn_prelu(2, 64, 9)
        self.convmnt_2_1 = self._conv_bn_prelu(64, 128, 5)
        self.convmnt_3_1 = self._conv_bn_prelu(128, 256, 3)
        self.convmnt_o_1_1 = self._conv_bn_prelu(256 + 90, 256, 1)
        self.mnt_o_1_2 = nn.Conv2d(256, 180, 1, padding='same')
        self.convmnt_w_1_1 = self._conv_bn_prelu(256, 256, 1)
        self.mnt_w_1_2 = nn.Conv2d(256, 8, 1, padding='same')
        self.convmnt_h_1_1 = self._conv_bn_prelu(256, 256, 1)
        self.mnt_h_1_2 = nn.Conv2d(256, 8, 1, padding='same')
        self.convmnt_s_1_1 = self._conv_bn_prelu(256, 256, 1)
        self.mnt_s_1_2 = nn.Conv2d(256, 1, 1, padding='same')

    @staticmethod
    def _conv_bn_prelu(in_channels, out_channels, kernel_size, stride=1, dilation=1):
        """Cria um bloco sequencial Conv-BN-PReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding='same', dilation=dilation, bias=True),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.99),
            nn.PReLU(num_parameters=out_channels, init=0.0)
        )

    @staticmethod
    def _img_normalization_layer():
        """Cria uma camada de normalização como um módulo sequencial."""
        class ImgNormalization(nn.Module):
            def __init__(self, m0=0.0, var0=1.0):
                super().__init__()
                self.m0 = m0
                self.var0 = var0
            def forward(self, x):
                m = torch.mean(x, dim=(1, 2, 3), keepdim=True)
                var = torch.var(x, dim=(1, 2, 3), keepdim=True)
                after = torch.sqrt(self.var0 * torch.square(x - m) / (var + 1e-8))
                return torch.where(x > m, self.m0 + after, self.m0 - after)
        return ImgNormalization()

    # --- Métodos de Execução da Rede (Internos) ---

    def _forward_features(self, x):
        """Executa a primeira parte da rede (VGG + ASPP)."""
        bn_image = self.img_norm(x)
        conv = self.conv1_1(bn_image); conv = self.conv1_2(conv); conv = self.pool1(conv)
        conv = self.conv2_1(conv); conv = self.conv2_2(conv); conv = self.pool2(conv)
        conv = self.conv3_1(conv); conv = self.conv3_2(conv); conv = self.conv3_3(conv); conv = self.pool3(conv)
        
        scale1 = self.atrousconv4_1(conv); ori_1 = self.ori_1_2(self.convori_1_1(scale1)); seg_1 = self.seg_1_2(self.convseg_1_1(scale1))
        scale2 = self.atrousconv4_2(conv); ori_2 = self.ori_2_2(self.convori_2_1(scale2)); seg_2 = self.seg_2_2(self.convseg_2_1(scale2))
        scale3 = self.atrousconv4_3(conv); ori_3 = self.ori_3_2(self.convori_3_1(scale3)); seg_3 = self.seg_3_2(self.convseg_3_1(scale3))
        
        ori_out = torch.sigmoid(ori_1 + ori_2 + ori_3)
        seg_out = torch.sigmoid(seg_1 + seg_2 + seg_3)
        return ori_out, seg_out

    def _forward_enhancement(self, x, ori_out, seg_out):
        """Executa a parte de realce da imagem."""
        filter_img_real = self.enh_img_real_1(x)
        filter_img_imag = self.enh_img_imag_1(x)
        
        ori_peak = self._ori_highest_peak(ori_out)
        ori_peak = self._select_max(ori_peak)
        
        upsample_ori = F.interpolate(ori_peak, scale_factor=8, mode='nearest')
        
        enh_img_real = torch.sum(filter_img_real * upsample_ori, dim=1, keepdim=True)
        enh_img_imag = torch.sum(filter_img_imag * upsample_ori, dim=1, keepdim=True)
        
        phase_img = self._atan2(enh_img_imag, enh_img_real)
        upsample_seg = F.interpolate(torch.nn.Softsign()(seg_out), scale_factor=8, mode='nearest')
        
        return enh_img_real, torch.cat((phase_img, upsample_seg), 1)

    def _forward_minutiae(self, enh_seg_img, ori_out):
        """Executa a parte final da rede para extração de minucias."""
        mnt_conv = self.convmnt_1_1(enh_seg_img); mnt_conv = self.pool1(mnt_conv)
        mnt_conv = self.convmnt_2_1(mnt_conv); mnt_conv = self.pool2(mnt_conv)
        mnt_conv = self.convmnt_3_1(mnt_conv); mnt_conv = self.pool3(mnt_conv)
        
        mnt_o_1 = torch.cat((mnt_conv, ori_out), 1)
        mnt_o_out = torch.sigmoid(self.mnt_o_1_2(self.convmnt_o_1_1(mnt_o_1)))
        mnt_w_out = torch.sigmoid(self.mnt_w_1_2(self.convmnt_w_1_1(mnt_conv)))
        mnt_h_out = torch.sigmoid(self.mnt_h_1_2(self.convmnt_h_1_1(mnt_conv)))
        mnt_s_out = torch.sigmoid(self.mnt_s_1_2(self.convmnt_s_1_1(mnt_conv)))
        
        return mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out

    # --- Métodos de Inferência (Públicos) ---

    @staticmethod
    def preprocess_image(image_path):
        """Carrega e pré-processa uma imagem, incluindo padding para ser divisível por 8."""
        img_pil = PIL.Image.open(image_path).convert('L')
        img_np = np.array(img_pil, dtype=np.float32) / 255.0
        
        h, w = img_np.shape
        h_pad = (8 - h % 8) if h % 8 != 0 else 0
        w_pad = (8 - w % 8) if w % 8 != 0 else 0
        
        img_padded = np.pad(img_np, ((0, h_pad), (0, w_pad)), mode='constant', constant_values=0.0)
        
        img_tensor = torch.from_numpy(img_padded).unsqueeze(0).unsqueeze(0)
        return img_tensor, (h, w)

    def predict_orientation_and_mask(self, image_tensor):
        """Executa a rede para obter apenas o campo de orientação e a máscara de segmentação."""
        self.eval()
        with torch.no_grad():
            ori_out, seg_out = self._forward_features(image_tensor)
        
        # Upsample para o tamanho original da imagem
        upsampled_ori = F.interpolate(ori_out, size=image_tensor.shape[2:], mode='nearest')
        upsampled_seg = F.interpolate(seg_out, size=image_tensor.shape[2:], mode='nearest')
        
        return upsampled_ori, upsampled_seg

    def predict_enhancement(self, image_tensor):
        """Executa a rede para obter a máscara, orientação e a imagem realçada."""
        self.eval()
        with torch.no_grad():
            ori_out, seg_out = self._forward_features(image_tensor)
            enh_img, enh_seg_img = self._forward_enhancement(image_tensor, ori_out, seg_out)
        
        upsampled_ori = F.interpolate(ori_out, size=image_tensor.shape[2:], mode='nearest')
        upsampled_seg = F.interpolate(seg_out, size=image_tensor.shape[2:], mode='nearest')

        return upsampled_ori, upsampled_seg, enh_img

    def predict_full(self, image_path):
        """
        Executa o pipeline completo: pré-processa, executa a rede e extrai as minucias.
        """
        self.eval()
        image_tensor, original_dims = self.preprocess_image(image_path)
        
        with torch.no_grad():
            ori_out, seg_out = self._forward_features(image_tensor.to(self.conv1_1[0].weight.device))
            enh_img, enh_seg_img = self._forward_enhancement(image_tensor.to(self.conv1_1[0].weight.device), ori_out, seg_out)
            mnt_outputs = self._forward_minutiae(enh_seg_img, ori_out)
        
        # Converte para numpy e extrai minucias
        outputs_np = [o.cpu().numpy() for o in (enh_img, ori_out, seg_out) + mnt_outputs]
        minutiae = self._extract_minutiae(outputs_np)

        # Recorta os mapas de saída para o tamanho original da imagem
        h, w = original_dims
        upsampled_ori = F.interpolate(ori_out, size=image_tensor.shape[2:], mode='nearest')[0, :, :h, :w]
        upsampled_seg = F.interpolate(seg_out, size=image_tensor.shape[2:], mode='nearest')[0, :, :h, :w]
        enh_img = enh_img[0, :, :h, :w]

        return upsampled_ori, upsampled_seg, enh_img, minutiae

    # --- Métodos Auxiliares de Pós-Processamento (Internos) ---

    @staticmethod
    def _extract_minutiae(outputs, score_thresh=0.5):
        """Extrai e filtra minucias a partir das saídas do modelo."""
        enh_img, ori_out, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = [o.transpose(0, 2, 3, 1) for o in outputs]
        mnt_s_map, seg_map, mnt_w_map, mnt_h_map, mnt_o_map = map(np.squeeze, [mnt_s_out, seg_out, mnt_w_out, mnt_h_out, mnt_o_out])

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seg_map_cleaned = cv2.morphologyEx(np.round(seg_map).astype(np.uint8), cv2.MORPH_OPEN, kernel)

        mnt_s_map = mnt_s_map * seg_map_cleaned
        mnt_sparse = sparse.coo_matrix(mnt_s_map > score_thresh)
        
        if mnt_sparse.nnz == 0: return np.zeros((0, 4))

        mnt_list = np.vstack((mnt_sparse.row, mnt_sparse.col)).T
        
        offsets_w = np.argmax(mnt_w_map, axis=-1)[mnt_list[:, 0], mnt_list[:, 1]]
        offsets_h = np.argmax(mnt_h_map, axis=-1)[mnt_list[:, 0], mnt_list[:, 1]]
        orientations = np.argmax(mnt_o_map, axis=-1)[mnt_list[:, 0], mnt_list[:, 1]]
        scores = mnt_s_map[mnt_list[:, 0], mnt_list[:, 1]]

        final_x = mnt_list[:, 1] * 8 + offsets_w
        final_y = mnt_list[:, 0] * 8 + offsets_h
        final_o = np.deg2rad(orientations * 2 - 89.) % (2 * np.pi)

        minutiae = np.vstack([final_x, final_y, final_o, scores]).T
        return FingerNet._nms(minutiae)

    @staticmethod
    def _nms(mnt):
        """Aplica Non-Maximum Suppression (NMS) na lista de minucias."""
        if mnt.shape[0] == 0: return mnt
        mnt_sort = mnt[mnt[:, 3].argsort()[::-1]]
        dist = spatial.distance.cdist(mnt_sort[:, :2], mnt_sort[:, :2])
        angle_diff = spatial.distance.cdist(mnt_sort[:, 2:3], mnt_sort[:, 2:3], lambda u, v: min(abs(u-v), 2*np.pi - abs(u-v)))
        max_dist, max_angle = 16, np.pi / 6
        keep = np.ones(mnt_sort.shape[0], dtype=bool)
        for i in range(mnt_sort.shape[0]):
            if keep[i]:
                mask = (dist[i, i+1:] < max_dist) & (angle_diff[i, i+1:] < max_angle)
                keep[i+1:][mask] = False
        return mnt_sort[keep]

    @staticmethod
    def _atan2(y, x):
        angle = torch.atan(y / (x + 1e-8))
        angle = torch.where(x > 0, angle, torch.zeros_like(x))
        angle = torch.where((x < 0) & (y >= 0), angle + np.pi, angle)
        angle = torch.where((x < 0) & (y < 0), angle - np.pi, angle)
        return angle

    @staticmethod
    def _ori_highest_peak(y_pred, length=180, stride=2):
        gaussian_pdf = signal.windows.gaussian(length + 1, std=3)
        y = np.reshape(np.arange(stride / 2, length, stride), [1, 1, -1, 1])
        label = np.reshape(np.arange(stride / 2, length, stride), [1, 1, 1, -1])
        delta = np.array(np.abs(label - y), dtype=int)
        delta = np.minimum(delta, length - delta) + length // 2
        glabel = gaussian_pdf[delta].astype(np.float32)
        glabel_tensor = torch.from_numpy(glabel).permute(2, 3, 0, 1).to(y_pred.device)
        return F.conv2d(y_pred, glabel_tensor, padding='same')

    @staticmethod
    def _select_max(x):
        max_vals, _ = torch.max(x, dim=1, keepdim=True)
        x = x / (max_vals + 1e-8)
        x = torch.where(x > 0.999, x, torch.zeros_like(x))
        sum_vals = torch.sum(x, dim=1, keepdim=True)
        x = x / (sum_vals + 1e-8)
        return x
