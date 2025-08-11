import shutil
import re
from tqdm import tqdm
import os


def organize_layers(root_dir, out_dir=None):
    """
    Reorganiza a estrutura de diretórios dos assets do FingerNet.
    Para cada asset (ex: mask, enhanced, etc), cria uma estrutura:
        out_dir/asset/iid/sid/ts1k_<iid>_<fid>-<sid>.<ext>
    Args:
        root_dir (str): Caminho para a pasta principal (ex: fingernet_output)
        out_dir (str|None): Caminho para a pasta de saída. Se None, usa root_dir.
    """
    # Regex para extrair iid, fid, sid do nome da pasta
    pattern = re.compile(r"ts1k_(\d{4})_(\d+)-(\d{2})")

    # Descobre todas as pastas de amostras (ex: ts1k_0000_0-00)
    sample_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("ts1k_")]

    # Descobre todos os tipos de asset presentes em uma pasta de amostra
    if not sample_dirs:
        print("Nenhuma pasta de amostra encontrada.")
        return
    example_dir = os.path.join(root_dir, sample_dirs[0])
    assets = [f for f in os.listdir(example_dir) if os.path.isfile(os.path.join(example_dir, f))]
    assets = [os.path.splitext(f)[0] for f in assets]  # sem extensão
    assets = list(set(assets))
    if out_dir is None:
        out_dir = root_dir
    for asset in assets:
        asset_dir = os.path.join(out_dir, asset)
        os.makedirs(asset_dir, exist_ok=True)

    total_files = sum(len([f for f in os.listdir(os.path.join(root_dir, s)) if os.path.isfile(os.path.join(root_dir, s, f))]) for s in sample_dirs)
    with tqdm(total=total_files, desc="Organizando arquivos") as pbar:
        for sample in sample_dirs:
            m = pattern.match(sample)
            if not m:
                print(f"Ignorando pasta com nome inesperado: {sample}")
                continue
            iid, fid, sid = m.groups()
            sample_path = os.path.join(root_dir, sample)
            for file in os.listdir(sample_path):
                file_path = os.path.join(sample_path, file)
                if not os.path.isfile(file_path):
                    continue
                asset_name, ext = os.path.splitext(file)
                asset = asset_name  # ex: mask, enhanced, etc
                # Cria pasta destino: root_dir/asset/iid/sid/
                dest_dir = os.path.join(out_dir, asset, iid, sid)
                os.makedirs(dest_dir, exist_ok=True)
                # Novo nome: ts1k_<iid>_<fid>-<sid><ext>
                new_name = f"ts1k_{iid}_{fid}-{sid}{ext}"
                dest_path = os.path.join(dest_dir, new_name)
                shutil.copy2(file_path, dest_path)
                pbar.update(1)
    print("Organização concluída.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Organiza os assets do FingerNet em camadas por asset/iid/sid.")
    parser.add_argument('root_dir', type=str, help='Pasta principal do FingerNet (ex: fingernet_output)')
    parser.add_argument('-o', '--out_dir', type=str, default=None, help='Pasta de saída (opcional). Se não especificado, sobrescreve na pasta de entrada.')
    args = parser.parse_args()
    organize_layers(args.root_dir, args.out_dir)
