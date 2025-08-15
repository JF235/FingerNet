import argparse
import sys
import json
from pathlib import Path

# Importa as funções principais da sua biblioteca
from fingernet.api import run_lightning_inference
from fingernet.plot import plot_from_output_folder


def infer_command(args):
    """
    Executa a inferência chamando a função da API Python.
    """
    # Se não foi fornecido arquivo de saída salva na mesma pasta que a entrada
    if args.output_path is None:
        output_path = Path(args.input).parent

    # Converte o argumento de texto para o formato correto (int ou list)
    try:
        # Tenta converter para um número inteiro (ex: "1", "-1")
        devices = int(args.devices)
    except ValueError:
        try:
            # Tenta converter para uma lista (ex: "[2,3]")
            devices = json.loads(args.devices)
            if not isinstance(devices, list):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            # Se falhar, usa como string (ex: "auto")
            devices = args.devices

    print(f"\n--- Iniciando FingerNet via API ---")
    run_lightning_inference(
        input_path=args.input,
        output_path=output_path,
        batch_size=args.batch_size,
        recursive=args.recursive,
        num_cores=args.num_cores,
        devices=devices
    )


def plot_command(args):
    """
    Plota os resultados de uma imagem específica, lendo da nova estrutura de diretórios.
    """
    output_file = args.output_file
    output_path = args.output_path

    # Se output_path não foi fornecido ou está vazio, usa o diretório atual
    if not output_path:
        output_path = "."

    if output_file is None:
        # Salva a imagem de resumo com um nome descritivo na pasta de saída principal
        base_name = Path(args.image_filename).stem
        output_file = Path(output_path) / f"{base_name}_visual_summary.png"
    
    
    # Chama a função de plotagem com os novos argumentos
    plot_from_output_folder(
        output_path=output_path,
        image_filename=args.image_filename,
        save_path=str(output_file)
    )


def main():
    """
    Função principal que configura o argparse e os subcomandos.
    """
    parser = argparse.ArgumentParser(prog='fingernet', description='CLI para a biblioteca FingerNet.')
    subparsers = parser.add_subparsers(dest='command', required=True, help='Comandos disponíveis')

    # --- Subcomando 'infer' (inalterado) ---
    infer_parser = subparsers.add_parser('infer', help='Executa a inferência em uma imagem ou pasta.')
    infer_parser.add_argument('input', type=str, help='Caminho para a imagem ou pasta de entrada.')
    infer_parser.add_argument('--output-path', type=str, default=None, help='Pasta de saída dos resultados.')
    infer_parser.add_argument('--weights-path', type=str, default=None, help='Caminho para os pesos .pth do modelo.')
    infer_parser.add_argument('-b', '--batch-size', type=int, default=4, help='Tamanho do lote por GPU.')
    infer_parser.add_argument('--num-cores', type=int, default=4, help='Núcleos de CPU para carregar dados.')
    infer_parser.add_argument('--devices', type=str, default='auto', help='GPUs a serem usadas. Ex: "auto", "-1", "[2,3]".')
    infer_parser.add_argument('--recursive', action='store_true', help='Busca por imagens de forma recursiva.')
    infer_parser.set_defaults(func=infer_command)

    # --- Subcomando 'plot' (MODIFICADO) ---
    plot_parser = subparsers.add_parser('plot', help='Gera uma visualização para uma imagem específica a partir da pasta de resultados.')
    plot_parser.add_argument('output_path', type=str, help='Caminho para a pasta principal de resultados (ex: output/).')
    plot_parser.add_argument('image_filename', type=str, help='Nome do arquivo da imagem original (ex: 101_1.png).')
    plot_parser.add_argument('--output-file', type=str, default=None, help='Caminho para salvar a imagem de visualização. (Opcional)')
    plot_parser.set_defaults(func=plot_command)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
