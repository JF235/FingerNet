import argparse
import sys
import os
from fingernet.plot import plot_from_result_folder

# Caminho para test.py (inferência rápida)
from pathlib import Path
import subprocess



def infer_command(args):
    """
    Executa a inferência rápida usando test.py, repassando argumentos.
    Se --weights-path não for fornecido, usa o modelo padrão em models/released_version/Model.pth relativo ao cli.py.
    """
    test_py = Path(__file__).parent.parent / "test.py"
    if not test_py.exists():
        print(f"test.py não encontrado em {test_py}")
        sys.exit(1)

    # Determina o caminho dos pesos automaticamente se não fornecido
    weights_path = args.weights_path
    if weights_path is None:
        default_weights = Path(__file__).parent.parent.parent / "models" / "released_version" / "Model.pth"
        print(default_weights)
        if default_weights.exists():
            weights_path = str(default_weights)
            print(f"Usando pesos padrão: {weights_path}")
        else:
            print("Erro: --weights-path não fornecido e modelo padrão não encontrado em models/released_version/Model.pth")
            sys.exit(1)

    # Monta comando conforme novo test.py
    cmd = [sys.executable, str(test_py),
           '--input', args.input,
           '--output-path', args.output_path,
           '--weights-path', weights_path]
    if args.batch_size:
        cmd += ['-b', str(args.batch_size)]
    if args.recursive:
        cmd += ['--recursive']
    if args.num_cores:
        cmd += ['--num-cores', str(args.num_cores)]
    print(f"Executando: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def plot_command(args):
    """
    Plota os resultados de uma pasta de saída.
    """
    plot_from_result_folder(args.result_folder)



def main():
    parser = argparse.ArgumentParser(prog='fingernet', description='FingerNet CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subcomando infer
    infer_parser = subparsers.add_parser('infer', help='Executa inferência rápida em uma imagem ou pasta')
    infer_parser.add_argument('input', type=str, help='Imagem, pasta ou lista de arquivos de entrada')
    infer_parser.add_argument('--output-path', type=str, default='output', help='Pasta de saída dos resultados')
    infer_parser.add_argument('--weights-path', type=str, default=None, help='Caminho para os pesos .pth do modelo (opcional)')
    infer_parser.add_argument('-b', '--batch-size', type=int, default=1, help='Tamanho do batch')
    infer_parser.add_argument('--recursive', action='store_true', help='Busca recursiva por imagens')
    infer_parser.add_argument('--num-cores', type=int, default=4, help='Núcleos de CPU para carregar dados')
    infer_parser.set_defaults(func=infer_command)

    # Subcomando plot
    plot_parser = subparsers.add_parser('plot', help='Plota resultados de uma pasta de saída')
    plot_parser.add_argument('result_folder', type=str, help='Pasta de resultados de uma imagem')
    plot_parser.set_defaults(func=plot_command)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
