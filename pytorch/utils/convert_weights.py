import torch
import h5py

from fingernet import FingerNet

# =====================
# Constantes e Mapeamentos
# =====================
CONV_BN_PRELU_MAP = {
    'conv1_1': ('feature_extractor.conv1_1', 'bn-1_1', 'prelu-1_1'), 'conv1_2': ('feature_extractor.conv1_2', 'bn-1_2', 'prelu-1_2'),
    'conv2_1': ('feature_extractor.conv2_1', 'bn-2_1', 'prelu-2_1'), 'conv2_2': ('feature_extractor.conv2_2', 'bn-2_2', 'prelu-2_2'),
    'conv3_1': ('feature_extractor.conv3_1', 'bn-3_1', 'prelu-3_1'), 'conv3_2': ('feature_extractor.conv3_2', 'bn-3_2', 'prelu-3_2'), 'conv3_3': ('feature_extractor.conv3_3', 'bn-3_3', 'prelu-3_3'),
    'conv4_1': ('ori_seg_head.atrous_1', 'bn-4_1', 'prelu-4_1'), 'atrousconv4_2': ('ori_seg_head.atrous_2', 'bn-4_2', 'prelu-4_2'), 'atrousconv4_3': ('ori_seg_head.atrous_3', 'bn-4_3', 'prelu-4_3'),
    'convori_1_1': ('ori_seg_head.ori_branch_1.0', 'bn-ori_1_1', 'prelu-ori_1_1'), 'convori_2_1': ('ori_seg_head.ori_branch_2.0', 'bn-ori_2_1', 'prelu-ori_2_1'), 'convori_3_1': ('ori_seg_head.ori_branch_3.0', 'bn-ori_3_1', 'prelu-ori_3_1'),
    'convseg_1_1': ('ori_seg_head.seg_branch_1.0', 'bn-seg_1_1', 'prelu-seg_1_1'), 'convseg_2_1': ('ori_seg_head.seg_branch_2.0', 'bn-seg_2_1', 'prelu-seg_2_1'), 'convseg_3_1': ('ori_seg_head.seg_branch_3.0', 'bn-seg_3_1', 'prelu-seg_3_1'),
    'convmnt_1_1': ('minutiae_head.conv1', 'bn-mnt_1_1', 'prelu-mnt_1_1'), 'convmnt_2_1': ('minutiae_head.conv2', 'bn-mnt_2_1', 'prelu-mnt_2_1'), 'convmnt_3_1': ('minutiae_head.conv3', 'bn-mnt_3_1', 'prelu-mnt_3_1'),
    'convmnt_o_1_1': ('minutiae_head.o_branch.0', 'bn-mnt_o_1_1', 'prelu-mnt_o_1_1'), 'convmnt_w_1_1': ('minutiae_head.w_branch.0', 'bn-mnt_w_1_1', 'prelu-mnt_w_1_1'),
    'convmnt_h_1_1': ('minutiae_head.h_branch.0', 'bn-mnt_h_1_1', 'prelu-mnt_h_1_1'), 'convmnt_s_1_1': ('minutiae_head.s_branch.0', 'bn-mnt_s_1_1', 'prelu-mnt_s_1_1'),
}

FINAL_CONV_MAP = {
    'ori_1_2': 'ori_seg_head.ori_branch_1.1', 'ori_2_2': 'ori_seg_head.ori_branch_2.1', 'ori_3_2': 'ori_seg_head.ori_branch_3.1',
    'seg_1_2': 'ori_seg_head.seg_branch_1.1', 'seg_2_2': 'ori_seg_head.seg_branch_2.1', 'seg_3_2': 'ori_seg_head.seg_branch_3.1',
    'mnt_o_1_2': 'minutiae_head.o_branch.1', 'mnt_w_1_2': 'minutiae_head.w_branch.1', 'mnt_h_1_2': 'minutiae_head.h_branch.1', 'mnt_s_1_2': 'minutiae_head.s_branch.1',
    'enh_img_real_1': 'enhancement_module.gabor_real', 'enh_img_imag_1': 'enhancement_module.gabor_imag',
}

# =====================
# Funções
# =====================

def convert_weights(keras_path: str, output_path: str):
    """
    Carrega os pesos de um HDF5 do Keras e os converte para um state_dict
    do PyTorch, compatível com a nova arquitetura modular do FingerNet.
    """
    print(f"Iniciando a conversão de '{keras_path}'...")
    
    # 1. Instancia o novo modelo modular
    model = FingerNet()
    state_dict = model.state_dict()
    
    # 2. Abre o arquivo de pesos do Keras
    with h5py.File(keras_path, 'r') as f:
        # 3. Itera sobre os blocos Conv-BN-PReLU
        for keras_conv_name, (pt_prefix, keras_bn_name, keras_prelu_name) in CONV_BN_PRELU_MAP.items():
            # Convolução
            state_dict[f'{pt_prefix}.conv.weight'] = torch.from_numpy(f[keras_conv_name][keras_conv_name]['kernel:0'][()]).permute(3, 2, 0, 1)
            state_dict[f'{pt_prefix}.conv.bias'] = torch.from_numpy(f[keras_conv_name][keras_conv_name]['bias:0'][()])
            
            # BatchNorm
            state_dict[f'{pt_prefix}.bn.weight'] = torch.from_numpy(f[keras_bn_name][keras_bn_name]['gamma:0'][()])
            state_dict[f'{pt_prefix}.bn.bias'] = torch.from_numpy(f[keras_bn_name][keras_bn_name]['beta:0'][()])
            state_dict[f'{pt_prefix}.bn.running_mean'] = torch.from_numpy(f[keras_bn_name][keras_bn_name]['moving_mean:0'][()])
            state_dict[f'{pt_prefix}.bn.running_var'] = torch.from_numpy(f[keras_bn_name][keras_bn_name]['moving_variance:0'][()])
            
            # PReLU
            state_dict[f'{pt_prefix}.prelu.weight'] = torch.from_numpy(f[keras_prelu_name][keras_prelu_name]['alpha:0'][()]).squeeze()

        # 4. Itera sobre as camadas de convolução finais (incluindo as de Enhancement)
        for keras_name, pt_name in FINAL_CONV_MAP.items():
            state_dict[f'{pt_name}.weight'] = torch.from_numpy(f[keras_name][keras_name]['kernel:0'][()]).permute(3, 2, 0, 1)
            # AQUI ESTÁ A CORREÇÃO CRÍTICA: Copia o bias para as camadas de enhancement e outras
            if f'{pt_name}.bias' in state_dict:
                 state_dict[f'{pt_name}.bias'] = torch.from_numpy(f[keras_name][keras_name]['bias:0'][()])

    # 5. Carrega o state_dict preenchido e salva o novo arquivo .pth
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), output_path)
    print(f"✅ Pesos convertidos e salvos com sucesso em '{output_path}'")

def verify_weights(keras_path: str, pytorch_path: str):
    """
    Verifica se os pesos no arquivo .pth correspondem aos do arquivo .hdf5 original.
    """
    print("\n--- Iniciando verificação dos pesos ---")
    
    # 1. Carrega o modelo PyTorch com os pesos recém-salvos
    model = FingerNet()
    model.load_state_dict(torch.load(pytorch_path))
    model.eval() # Coloca em modo de avaliação
    pytorch_state_dict = model.state_dict()
    
    errors = 0
    
    # 2. Abre o arquivo Keras para comparação
    with h5py.File(keras_path, 'r') as f:
        # Verifica os blocos ConvBNPReLU
        for keras_conv_name, (pt_prefix, keras_bn_name, keras_prelu_name) in CONV_BN_PRELU_MAP.items():
            print(f"Verificando {keras_conv_name} -> {pt_prefix}...", end=' ')
            
            # Compara Convolução
            keras_kernel = torch.from_numpy(f[keras_conv_name][keras_conv_name]['kernel:0'][()]).permute(3, 2, 0, 1)
            if not torch.allclose(pytorch_state_dict[f'{pt_prefix}.conv.weight'], keras_kernel):
                errors += 1; print("FALHA (kernel)"); continue
            
            # Compara BatchNorm
            keras_gamma = torch.from_numpy(f[keras_bn_name][keras_bn_name]['gamma:0'][()])
            if not torch.allclose(pytorch_state_dict[f'{pt_prefix}.bn.weight'], keras_gamma):
                errors += 1; print("FALHA (bn.weight)"); continue

            # Compara PReLU
            keras_alpha = torch.from_numpy(f[keras_prelu_name][keras_prelu_name]['alpha:0'][()]).squeeze()
            if not torch.allclose(pytorch_state_dict[f'{pt_prefix}.prelu.weight'], keras_alpha):
                errors += 1; print("FALHA (prelu.weight)"); continue
            
            print("OK")

        # Verifica as camadas Conv2D standalone
        for keras_name, pt_name in FINAL_CONV_MAP.items():
            print(f"Verificando {keras_name} -> {pt_name}...", end=' ')
            
            keras_kernel = torch.from_numpy(f[keras_name][keras_name]['kernel:0'][()]).permute(3, 2, 0, 1)
            if not torch.allclose(pytorch_state_dict[f'{pt_name}.weight'], keras_kernel):
                errors += 1; print("FALHA (kernel)"); continue
            
            print("OK")

    print("--- Verificação concluída ---")
    if errors == 0:
        print("✅ Sucesso! Todos os pesos verificados correspondem.")
    else:
        print(f"❌ Falha! {errors} tensor(es) não corresponderam.")
    
    return errors == 0

def main():
    import argparse
    # Define args for input and output
    parser = argparse.ArgumentParser(description="Converter Keras para PyTorch")
    parser.add_argument("--keras", type=str, required=True, help="Caminho para o arquivo Keras (.h5)")
    parser.add_argument("--pytorch", type=str, required=True, help="Caminho para o arquivo PyTorch (.pth)")

    # Make a final check
    parser.add_argument("--verify", action='store_true', help="Verifica se os pesos convertidos correspondem aos originais")
    args = parser.parse_args()

    convert_weights(args.keras, args.pytorch)

    if args.verify:
        verify_weights(args.keras, args.pytorch)


if __name__ == '__main__':
    main()