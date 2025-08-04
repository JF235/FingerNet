# fingernet_cli.py
"""
Utilitário de Linha de Comando (CLI) para o modelo FingerNet.

Permite duas ações principais:
1. `load`: Converte os pesos do formato HDF5 (Keras) para o formato .pth (PyTorch).
2. `infer`: Executa a inferência em uma imagem ou pasta, com diferentes níveis de saída.

Exemplos de Uso:
  # Converter pesos
  python fingernet_cli.py load path/to/fingernet.hdf5 -o fingernet.pth

  # Inferência completa (padrão)
  python fingernet_cli.py infer path/to/images -o results --full

  # Inferência apenas para máscara e orientação
  python fingernet_cli.py infer path/to/image.png -o results --mask
"""

import torch
import numpy as np
import cv2
import os
import glob
import argparse
import h5py
from fingernet import FingerNet

# --- Função de Conversão de Pesos ---

def convert_keras_weights(keras_path, output_path):
    """Carrega os pesos de um HDF5 e salva como um state_dict do PyTorch."""
    print(f"Iniciando a conversão de '{keras_path}'...")
    model = FingerNet()
    state_dict = model.state_dict()
    
    with h5py.File(keras_path, 'r') as f:
        # Mapeamento de nomes de camadas Keras para PyTorch
        layer_map = {
            'conv1_1': 'conv1_1', 'conv1_2': 'conv1_2', 'conv2_1': 'conv2_1', 'conv2_2': 'conv2_2',
            'conv3_1': 'conv3_1', 'conv3_2': 'conv3_2', 'conv3_3': 'conv3_3',
            'conv4_1': 'atrousconv4_1', 'atrousconv4_2': 'atrousconv4_2', 'atrousconv4_3': 'atrousconv4_3',
            'convori_1_1': 'convori_1_1', 'ori_1_2': 'ori_1_2', 'convori_2_1': 'convori_2_1', 'ori_2_2': 'ori_2_2',
            'convori_3_1': 'convori_3_1', 'ori_3_2': 'ori_3_2',
            'convseg_1_1': 'convseg_1_1', 'seg_1_2': 'seg_1_2', 'convseg_2_1': 'convseg_2_1', 'seg_2_2': 'seg_2_2',
            'convseg_3_1': 'convseg_3_1', 'seg_3_2': 'seg_3_2',
            'enh_img_real_1': 'enh_img_real_1', 'enh_img_imag_1': 'enh_img_imag_1',
            'convmnt_1_1': 'convmnt_1_1', 'convmnt_2_1': 'convmnt_2_1', 'convmnt_3_1': 'convmnt_3_1',
            'convmnt_o_1_1': 'convmnt_o_1_1', 'mnt_o_1_2': 'mnt_o_1_2',
            'convmnt_w_1_1': 'convmnt_w_1_1', 'mnt_w_1_2': 'mnt_w_1_2',
            'convmnt_h_1_1': 'convmnt_h_1_1', 'mnt_h_1_2': 'mnt_h_1_2',
            'convmnt_s_1_1': 'convmnt_s_1_1', 'mnt_s_1_2': 'mnt_s_1_2',
        }

        for keras_name, pytorch_name in layer_map.items():
            is_conv_bn_prelu = isinstance(model.get_submodule(pytorch_name), torch.nn.Sequential)

            if is_conv_bn_prelu:
                conv_layer, bn_layer, prelu_layer = model.get_submodule(pytorch_name)
                keras_conv_group = f[keras_name][keras_name]
                
                # Pesos da Convolução
                conv_weights = torch.from_numpy(keras_conv_group['kernel:0'][()]).permute(3, 2, 0, 1)
                conv_bias = torch.from_numpy(keras_conv_group['bias:0'][()])
                state_dict[f'{pytorch_name}.0.weight'] = conv_weights
                state_dict[f'{pytorch_name}.0.bias'] = conv_bias
                
                # Pesos de BN e PReLU
                base_name = keras_name[len('atrousconv'):] if keras_name.startswith('atrousconv') else keras_name[len('conv'):]
                bn_group_name = 'bn-' + base_name
                keras_bn_group = f[bn_group_name][bn_group_name]
                state_dict[f'{pytorch_name}.1.weight'] = torch.from_numpy(keras_bn_group['gamma:0'][()])
                state_dict[f'{pytorch_name}.1.bias'] = torch.from_numpy(keras_bn_group['beta:0'][()])
                state_dict[f'{pytorch_name}.1.running_mean'] = torch.from_numpy(keras_bn_group['moving_mean:0'][()])
                state_dict[f'{pytorch_name}.1.running_var'] = torch.from_numpy(keras_bn_group['moving_variance:0'][()])
                
                prelu_group_name = 'prelu-' + base_name
                state_dict[f'{pytorch_name}.2.weight'] = torch.from_numpy(f[prelu_group_name][prelu_group_name]['alpha:0'][()]).squeeze()

            else: # Camada Conv2D simples
                keras_conv_group = f[keras_name][keras_name]
                conv_weights = torch.from_numpy(keras_conv_group['kernel:0'][()]).permute(3, 2, 0, 1)
                conv_bias = torch.from_numpy(keras_conv_group['bias:0'][()])
                state_dict[f'{pytorch_name}.weight'] = conv_weights
                state_dict[f'{pytorch_name}.bias'] = conv_bias

    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), output_path)
    print(f"✅ Pesos convertidos e salvos com sucesso em '{output_path}'")

# --- Função de Inferência ---

def run_inference(args):
    """Carrega o modelo e executa a inferência com base nos argumentos."""
    if not os.path.exists(args.weights):
        print(f"❌ Erro: Arquivo de pesos '{args.weights}' não encontrado.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Usando dispositivo: {device}")

    model = FingerNet()
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    
    # Prepara a lista de imagens
    if os.path.isfile(args.input_path):
        image_paths = [args.input_path]
    else:
        ext = args.ext if args.ext.startswith('.') else '.' + args.ext
        image_paths = glob.glob(os.path.join(args.input_path, f"*{ext}"))
    
    if not image_paths:
        print(f"⚠️ Nenhuma imagem com extensão '{args.ext}' encontrada em '{args.input_path}'.")
        return

    # Cria as pastas de saída
    output_folders = {'mask': os.path.join(args.o, 'mask'), 'ori': os.path.join(args.o, 'ori')}
    if args.enhance or args.full: output_folders['enh'] = os.path.join(args.o, 'enh')
    if args.full: output_folders['mnt'] = os.path.join(args.o, 'mnt')
    for folder in output_folders.values(): os.makedirs(folder, exist_ok=True)
    
    print(f"📂 Processando {len(image_paths)} imagem(ns). Resultados em '{args.o}'...")

    for img_path in image_paths:
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\nProcessing: {base_filename}...")

        if args.mask:
            tensor, (h, w) = model.preprocess_image(img_path)
            ori, mask = model.predict_orientation_and_mask(tensor.to(device))
            ori, mask = ori[0, :, :h, :w], mask[0, :, :h, :w]
        elif args.enhance:
            tensor, (h, w) = model.preprocess_image(img_path)
            ori, mask, enh = model.predict_enhancement(tensor.to(device))
            ori, mask, enh = ori[0, :, :h, :w], mask[0, :, :h, :w], enh[0, :, :h, :w]
        else: # --full (padrão)
            ori, mask, enh, mnt = model.predict_full(img_path)
        
        # Salva os resultados
        # Converte mapas para imagens 0-255
        mask_img = (np.squeeze(mask.cpu().numpy()) * 255).astype(np.uint8)
        # Para o mapa de orientação, salvamos uma visualização do ângulo
        angle_deg = np.argmax(ori.cpu().numpy(), axis=0) * 2 - 90
        ori_img = ((angle_deg + 90) / 180 * 255).astype(np.uint8) # Normaliza para 0-255

        cv2.imwrite(os.path.join(output_folders['mask'], f"{base_filename}.png"), mask_img)
        cv2.imwrite(os.path.join(output_folders['ori'], f"{base_filename}.png"), ori_img)
        print(f"  -> Máscara e Orientação salvas.")

        if args.enhance or args.full:
            enh_img_np = np.squeeze(enh.cpu().numpy())
            enh_img_norm = cv2.normalize(enh_img_np, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(os.path.join(output_folders['enh'], f"{base_filename}.png"), enh_img_norm.astype(np.uint8))
            print(f"  -> Imagem Realçada salva.")

        if args.full:
            mnt_path = os.path.join(output_folders['mnt'], f"{base_filename}.txt")
            np.savetxt(mnt_path, mnt, fmt=['%d', '%d', '%.4f', '%.4f'], header='x, y, angle_rad, score')
            print(f"  -> {len(mnt)} Minucias salvas.")

# --- Configuração do Parser Principal ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Utilitário de Linha de Comando para FingerNet.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-comando a ser executado')

    # Sub-comando 'load'
    parser_load = subparsers.add_parser('load', help='Converte pesos do Keras (.hdf5) para PyTorch (.pth).')
    parser_load.add_argument('keras_path', type=str, help='Caminho para o arquivo de pesos .hdf5 do Keras.')
    parser_load.add_argument('-o', type=str, required=True, help='Caminho de saída para o arquivo .pth.')
    
    # Sub-comando 'infer'
    parser_infer = subparsers.add_parser('infer', help='Executa a inferência em imagens.')
    parser_infer.add_argument('input_path', type=str, help='Caminho para uma imagem ou uma pasta de imagens.')
    parser_infer.add_argument('--weights', type=str, default='fingernet.pth', help='Caminho para os pesos .pth do modelo.')
    parser_infer.add_argument('--ext', type=str, default='png', help='Extensão dos arquivos de imagem a serem processados (sem o ponto).')
    parser_infer.add_argument('-o', type=str, required=True, help='Pasta de saída para os resultados.')
    
    mode_group = parser_infer.add_mutually_exclusive_group()
    mode_group.add_argument('--mask', action='store_true', help='Executa e salva apenas a máscara e a orientação.')
    mode_group.add_argument('--enhance', action='store_true', help='Executa e salva máscara, orientação e imagem realçada.')
    mode_group.add_argument('--full', action='store_true', default=True, help='Executa o pipeline completo (padrão).')

    args = parser.parse_args()

    if args.command == 'load':
        convert_keras_weights(args.keras_path, args.o)
    elif args.command == 'infer':
        run_inference(args)
