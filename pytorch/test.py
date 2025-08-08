import argparse
from fingernet import run_inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de inferência em lote para a FingerNet.")
    parser.add_argument('--input-path', type=str, required=True, help="Caminho para uma imagem ou diretório.")
    parser.add_argument('--output-path', type=str, required=True, help="Diretório onde os resultados serão salvos.")
    parser.add_argument('--weights-path', type=str, required=True, help="Caminho para os pesos .pth do modelo.")
    parser.add_argument('-b', '--batch-size', type=int, default=1, help="Tamanho do lote para inferência.")
    parser.add_argument('--recursive', action='store_true', help="Busca por imagens recursivamente.")

    args = parser.parse_args()

    run_inference(
        input_path=args.input_path,
        output_path=args.output_path,
        weights_path=args.weights_path,
        recursive=args.recursive,
        batch_size=args.batch_size,
        num_gpus=None, # Try to use all
    )