#!/usr/bin/env python
import argparse
import os
import glob

# Importa apenas as funções de alto nível necessárias
from fingernet import get_fingernet, run_inference

def _get_image_files(input_path: str) -> list[str]:
    """
    Obtém uma lista de caminhos de imagem a partir de um arquivo ou diretório.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Caminho de entrada não encontrado: {input_path}")
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        supported_exts = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif')
        image_files = []
        for ext in supported_exts:
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
        if not image_files:
            raise FileNotFoundError(f"Nenhum arquivo de imagem suportado encontrado em: {input_path}")
        return image_files
    raise ValueError(f"Caminho de entrada não é um arquivo ou diretório válido: {input_path}")


def main():
    """Função principal para a interface de linha de comando."""
    parser = argparse.ArgumentParser(
        description="FingerNet CLI: Ferramenta para extrair características de impressões digitais."
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help="Comandos disponíveis")

    commands = {
        'full': 'Executa o pipeline completo e salva todas as saídas.',
        'enhance': 'Salva apenas a imagem de impressão digital melhorada.',
        'mnt': 'Salva apenas os dados das minúcias.',
        'ori': 'Salva apenas o campo de orientação.'
    }
    for cmd, help_text in commands.items():
        cmd_parser = subparsers.add_parser(cmd, help=help_text)
        cmd_parser.add_argument('input', help='Caminho para um arquivo de imagem ou um diretório.')
        cmd_parser.add_argument(
            '-o', '--output', 
            default='fingernet_output', 
            help='Diretório de saída principal. (Padrão: fingernet_output)'
        )
    
    args = parser.parse_args()

    try:
        # 1. Obter a lista de arquivos de imagem
        image_files = _get_image_files(args.input)
        
        # 2. Carregar o modelo FingerNet
        print("Carregando modelo FingerNet...")
        fnet = get_fingernet(log=False)
        
        # 3. Executar a inferência, delegando o salvamento para a função
        # O comando da CLI ('full', 'mnt', etc.) é passado diretamente para o parâmetro 'save_mode'
        run_inference(
            fnet=fnet,
            image_paths=image_files,
            output_dir=args.output,
            save_mode=args.command,
            batch_size=8
        )

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Erro: {e}")
    except Exception as e:
        print(f"\n❌ Ocorreu um erro inesperado: {e}")


if __name__ == '__main__':
    main()