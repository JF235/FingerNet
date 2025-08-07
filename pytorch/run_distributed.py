# run_distributed.py

import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Importa as funções originais e inalteradas dos seus módulos
from fingernet import get_fingernet
from fingernet.cli import _get_image_files # Reutiliza o helper para encontrar imagens
from fingernet.distributed import run_distributed_inference # Importa a nova lógica

def setup_distributed_environment():
    """
    Inicializa o ambiente de comunicação distribuída do PyTorch.
    `torchrun` define as variáveis de ambiente necessárias.
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"[Rank {local_rank}] Ambiente distribuído inicializado na GPU {local_rank}.")
    return local_rank

def create_parser():
    """
    Cria um parser de argumentos de linha de comando.
    É quase idêntico ao do `fingernet/cli.py` original.
    """
    parser = argparse.ArgumentParser(
        description="FingerNet Distributed CLI: Ferramenta para extrair características de impressões digitais em Múltiplas GPUs."
    )
    # Comandos para manter a compatibilidade da interface
    subparsers = parser.add_subparsers(dest='command', required=True, help="Comandos disponíveis (save_mode)")

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
            default='fingernet_output_dist',
            help='Diretório de saída principal. (Padrão: fingernet_output_dist)'
        )
        cmd_parser.add_argument(
            '-r', '--recursive',
            action='store_true',
            help='Busca recursiva por imagens em subdiretórios.'
        )
        cmd_parser.add_argument(
            '--batch-size',
            type=int,
            default=8,
            help='Tamanho do lote (batch size) POR GPU. (Padrão: 8)'
        )
    return parser

def main():
    """
    Ponto de entrada principal para o lançador distribuído.
    """
    parser = create_parser()
    args = parser.parse_args()

    # 1. Configurar o ambiente distribuído
    local_rank = setup_distributed_environment()
    is_main_process = local_rank == 0
    world_size = dist.get_world_size()

    try:
        # 2. Obter a lista de arquivos (todos os processos fazem isso para ter a contagem total)
        image_files = _get_image_files(args.input, recursive=args.recursive)

        if is_main_process:
            print(f"Encontradas {len(image_files)} imagens. Iniciando inferência em {world_size} GPUs.")

        # 3. Carregar o modelo e envolvê-lo com DDP
        device = torch.device(f"cuda:{local_rank}")
        fnet_util = get_fingernet(log=is_main_process)
        
        # Move a instância do modelo para a GPU designada para este processo
        fnet_util.model.to(device)
        fnet_util.device = device
        
        # Envolve o modelo com DistributedDataParallel
        fnet_util.model = DDP(fnet_util.model, device_ids=[local_rank])

        # 4. Executar a inferência usando a nova função de orquestração
        run_distributed_inference(
            fnet=fnet_util,
            image_paths=image_files,
            output_dir=args.output,
            save_mode=args.command,
            batch_size=args.batch_size,
            is_main_process=is_main_process,
        )

    except (FileNotFoundError, ValueError) as e:
        if is_main_process:
            print(f"\n❌ Erro: {e}")
    except Exception as e:
        if is_main_process:
            print(f"\n❌ Ocorreu um erro inesperado: {e}")
    finally:
        # 5. Limpar o ambiente distribuído
        dist.destroy_process_group()
        if is_main_process:
            print("Ambiente distribuído finalizado.")

if __name__ == '__main__':
    main()