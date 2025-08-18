from datetime import datetime
import matplotlib.pyplot as plt
import fingernet as fnet
import os
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm

def custom_save(i: int, out: dict, output_dir: str):
    """
    Salva os resultados de forma compativel com deploy original
    - x_enh.png: imagem melhorada
    - x_mnt.png: imagem original com minucias marcadas
    - x_ori.png: campo de orientação na imagem original
    - x_seg.png: máscara de segmentação
    - x.mnt: lista de minucias
    """
    
    # 1. Cria pasta com nome i
    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

    filename = os.path.splitext(os.path.basename(out['input_path']))[0]

    # 2. Salva imagem melhorada
    Image.fromarray(out['enhanced_image']).save(os.path.join(output_dir, str(i), f"{filename}_enh.png"))
    Image.fromarray(out['segmentation_mask']).save(os.path.join(output_dir, str(i), f"{filename}_seg.png"))

    # 3. Cria imagem matplotlib (sem espaço em branco)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(out['original_img'], cmap='gray')
    ax.axis('off')
    fnet.plot_mnt(ax, out['minutiae'], r=15)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join(output_dir, str(i), f"{filename}_mnt.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # 4. Salva campo de orientação (sem espaço em branco)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(out['original_img'], cmap='gray')
    ax.axis('off')
    fnet.plot_ori_field(ax, out['orientation_field'], stride=10)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join(output_dir, str(i), f"{filename}_ori.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # 5. Save minutia list
    with open(os.path.join(output_dir, str(i), f"{filename}.mnt"), 'w') as f:
        f.write(f"{filename}\n")
        f.write(f"{len(out['minutiae'])} {out['original_img'].shape[0]} {out['original_img'].shape[1]}\n")
        for m in out['minutiae']:
            angle = m[2]
            # Wrap to positive
            if angle < 0:
                angle += 2*torch.pi
            f.write(f"{int(m[0])} {int(m[1])} {angle:.6f}\n")

if __name__ == '__main__':

    dataset_dir = '../../datasets/'
    input_dirs = ['NISTSD27/images/','CISL24218/',
                'FVC2002DB2A/','NIST4/','NIST14/']
    input_dirs = [os.path.join(dataset_dir, d) for d in input_dirs]
    output_dir = f'pytorch-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    output_dir = os.path.join(dataset_dir, 'output', output_dir)
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get files
    files = []
    for input_dir in input_dirs:
        current_files = os.listdir(input_dir)
        # Only .bmp
        current_files = [os.path.join(input_dir, f) for f in current_files if f.endswith('.bmp')]
        files.extend(current_files)
    
    # Load grayscale images and convert to tensor
    images = []
    for file in files:
        img = Image.open(file).convert("L")
        img_tensor = transforms.ToTensor()(img)
        # Adiciona dimensoes (B, C, H, W)
        img_tensor = img_tensor.unsqueeze(0)
        images.append(img_tensor)

    # Get fingernet
    model = fnet.get_fingernet(device = "cpu")

    model.eval()

    outputs = []
    for i in tqdm(range(len(images)), desc="Processing images"):
        results = model(images[i])
        output_item = {
            'input_path': files[i],
            'minutiae': results['minutiae'][0].cpu().numpy(),
            'enhanced_image': results['enhanced_image'][0].cpu().numpy(),
            'segmentation_mask': results['segmentation_mask'][0].cpu().numpy(),
            'orientation_field': results['orientation_field'][0].cpu().numpy(),
            'original_img': images[i].cpu().numpy().squeeze()
        }
        outputs.append(output_item)
        tqdm.write(f"Processing: {os.path.basename(files[i])}")
    
    # Save outputs
    for i, out in enumerate(outputs):
        custom_save(i, out, output_dir)