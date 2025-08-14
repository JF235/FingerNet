import argparse
import sys
from pathlib import Path

# Importa as funções principais da sua biblioteca
from fingernet.api import run_lightning_inference
from fingernet.plot import plot_from_result_folder


def infer_command(args):
    """
    Executa a inferência chamando a função da API Python.
    """
    weights_path = args.weights_path
    
    # Lógica para encontrar o caminho dos pesos padrão, se não for fornecido
    if weights_path is None:
        # Assume que o CLI está em fingernet/cli.py, sobe dois níveis para a raiz do projeto
        project_root = Path(__file__).resolve().parent.parent
        default_weights = project_root / "models" / "released_version" / "Model.pth"
        
        if default_weights.exists():
            weights_path = str(default_weights)
            print(f"INFO: --weights-path não fornecido. Usando pesos padrão em: {weights_path}")
        else:
            print(f"ERRO: --weights-path não foi fornecido e o modelo padrão não foi encontrado em '{default_weights}'")
            sys.exit(1)

    # --- PONTO PRINCIPAL DA MUDANÇA ---
    # Chama a função Python diretamente, em vez de usar subprocess.
    print(f"\n--- Iniciando FingerNet via API ---")
    run_lightning_inference(
        input_path=args.input,
        output_path=args.output_path,
        weights_path=weights_path,
        batch_size=args.batch_size,
        recursive=args.recursive,
        num_cores=args.num_cores,
        # Para o CLI, que roda como script, podemos assumir o uso de todas as GPUs
        use_all_gpus=True 
    )


def plot_command(args):
    """
    Plota os resultados de uma pasta de saída, salvando a imagem.
    """
    # Define um caminho de saída padrão se nenhum for fornecido
    output_file = args.output_file
    print(output_file)


    if output_file is not None:
        output_file = Path(args.result_folder) / output_file

    print(args.result_folder)
    print(output_file)
    # Chama a função de plotagem, garantindo que um caminho para salvar seja sempre passado
    plot_from_result_folder(
        result_folder=args.result_folder,
        save_path=output_file  # <-- Passa o caminho para salvar
    )


def main():
    parser = argparse.ArgumentParser(
        prog='fingernet', 
        description='CLI oficial para a biblioteca FingerNet.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Comandos disponíveis')

    # --- Subcomando 'infer' ---
    infer_parser = subparsers.add_parser('infer', help='Executa a inferência em uma imagem ou pasta.')
    infer_parser.add_argument('input', type=str, help='Caminho para a imagem, pasta ou lista de arquivos de entrada.')
    infer_parser.add_argument('--output-path', type=str, default='output', help='Pasta onde os resultados serão salvos. (Padrão: output)')
    infer_parser.add_argument('--weights-path', type=str, default=None, help='Caminho para os pesos .pth do modelo. (Opcional, busca por um padrão se não fornecido)')
    infer_parser.add_argument('-b', '--batch-size', type=int, default=4, help='Tamanho do lote por GPU. (Padrão: 4)')
    infer_parser.add_argument('--recursive', action='store_true', help='Busca por imagens de forma recursiva dentro do diretório de entrada.')
    infer_parser.add_argument('--num-cores', type=int, default=4, help='Número de núcleos de CPU para carregar dados por GPU. (Padrão: 4)')
    infer_parser.set_defaults(func=infer_command)

    # --- Subcomando 'plot' ---
    plot_parser = subparsers.add_parser('plot', help='Gera uma visualização a partir de uma pasta de resultados já processada.')
    plot_parser.add_argument('result_folder', type=str, help='Caminho para a pasta de resultados de uma única imagem que contém os arquivos .txt, .npy, .png.')
    plot_parser.add_argument('--output-file', type=str, default=None, help='Caminho para salvar a imagem de visualização. (Opcional, salva na pasta de resultados por padrão)')
    plot_parser.set_defaults(func=plot_command)

    args = parser.parse_args()
    
    # Executa a função associada ao subcomando ('infer_command' ou 'plot_command')
    args.func(args)


if __name__ == '__main__':
    main()