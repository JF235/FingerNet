# Load

```
$ python fingernet_cli.py load ../models/released_version/Model.model -o ../models/released_version/Model.pth
Iniciando a conversão de '../models/released_version/Model.model'...
✅ Pesos convertidos e salvos com sucesso em '../models/released_version/Model.pth'
```

# Infer

```
$ python fingernet_cli.py infer ../datasets/FVC2002DB2A/1_1.bmp  --weight ../models/released_version/Model.pth -o ../output/torch_inference
🧠 Usando dispositivo: cuda
📂 Processando 1 imagem(ns). Resultados em '../output/torch_inference'...

Processing: 1_1...
  -> Máscara e Orientação salvas.
  -> Imagem Realçada salva.
  -> 33 Minucias salvas.
```