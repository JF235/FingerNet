# inspect_h5.py
import h5py
import argparse

def print_h5_structure(name, obj):
    """Função para ser chamada recursivamente para cada item no arquivo HDF5."""
    # Adiciona indentação baseada na profundidade do nome do objeto
    indent = '  ' * name.count('/')
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}- Dataset: {obj.name} (Shape: {obj.shape})")
    elif isinstance(obj, h5py.Group):
        print(f"{indent}Grupo: {obj.name}")

def main():
    parser = argparse.ArgumentParser(description="Inspeciona e imprime a estrutura de um arquivo HDF5.")
    parser.add_argument("h5_file", type=str, help="Caminho para o arquivo .hdf5 a ser inspecionado.")
    args = parser.parse_args()
    
    try:
        with h5py.File(args.h5_file, 'r') as f:
            print(f"Estrutura do arquivo: {args.h5_file}\n" + "="*40)
            f.visititems(print_h5_structure)
            print("="*40)
    except Exception as e:
        print(f"Não foi possível ler o arquivo: {e}")

if __name__ == "__main__":
    main()