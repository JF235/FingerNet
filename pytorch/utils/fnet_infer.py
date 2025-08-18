from datetime import datetime
from fingernet import get_fingernet, save_results
import os
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm

if __name__ == '__main__':

    dataset_dir = '../../datasets/'
    input_dirs = ['NISTSD27/images/','CISL24218/',
                'FVC2002DB2A/','NIST4/','NIST14/']
    input_dirs = [os.path.join(dataset_dir, d) for d in input_dirs]
    output_dir = f'pytorch-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    output_dir = os.path.join(dataset_dir, 'output', output_dir)

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
        images.append(img_tensor.to(memory_format=torch.channels_last))

    # Get fingernet
    fnet = get_fingernet(device = "cpu")

    fnet.to(memory_format=torch.channels_last)
    fnet.eval()

    outputs = []
    for i in tqdm(range(len(images)), desc="Processing images"):
        results = fnet(images[i])
        output_item = {
            'input_path': files[i],
            'minutiae': results['minutiae'][0].cpu().numpy(),
            'enhanced_image': results['enhanced_image'][0].cpu().numpy(),
            'segmentation_mask': results['segmentation_mask'][0].cpu().numpy(),
            'orientation_field': results['orientation_field'][0].cpu().numpy(),
        }
        outputs.append(output_item)
        tqdm.write(f"Processing: {os.path.basename(files[i])}")
    
    # Save outputs
    for out in outputs:
        save_results(out, output_dir)
