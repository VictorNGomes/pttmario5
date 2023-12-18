
# Projeto de Fine Tuning para Sumarização em Português usando MLOps

## Descrição do Projeto
Este repositório apresenta uma implementação de fine-tuning utilizando o modelo T5 (Text-to-Text Transfer Transformer) um Large Language Model (LLM) para tarefa de sumarização em português com o objetivo de integrar com as ferramentas MLflow, Gradio e Beam Cloud. O projeto utiliza dois conjuntos de dados específicos para o idioma, além das bibliotecas MLflow e Apache Beam Cloud para gerenciamento de experimentos e processamento distribuído.

## Objetivo

O objetivo principal deste projeto é treinar um modelo de sumarização em português utilizando a arquitetura de uma LLM, aprimorando o desempenho do modelo por meio de Fine Tuning. O uso de dois conjuntos de dados específicos visa melhorar a generalização do modelo para diferentes contextos e estilos de texto em português.

## Datasets

O projeto utiliza dois conjuntos de dados obtidos de fontes confiáveis em português para o treinamento do modelo de sumarização. Os datasets estão disponíveis em [Hugging Face Datasets](https://huggingface.co/datasets) e são fundamentais para garantir a qualidade e diversidade dos dados de treinamento.

### Corpus TéMario
Consiste em 100 textos jornalísticos. 60 textos constam do jornal online Folha de São Paulo  e estão distribuídos igualmente nas seções Especial, Mundo e Opinião; os 40 textos restantes foram publicados no Jornal do Brasil.  
- http://www.nilc.icmc.usp.br/nilc/tools/TeMario.zip

- HuggingFace: [VictorNGomes/pttmario5](https://huggingface.co/VictorNGomes/pttmario5)

### XLsum
XLSum é um conjunto de dados amplo e diversificado que inclui 1,35 milhão de pares de artigos e resumos anotados profissionalmente da BBC, obtidos por meio de um conjunto de heurísticas cuidadosamente elaboradas. O conjunto de dados abrange 45 idiomas, variando de baixo a alto recurso, muitos dos quais não possuem atualmente um conjunto de dados público disponível. O XL-Sum é altamente abstrato, conciso e de alta qualidade, conforme indicado por avaliações humanas e avaliações intrínsecas.

- [csebuetnlp/xlsum](https://huggingface.co/datasets/csebuetnlp/xlsum/viewer/portuguese)


## Modelo
A principio o melo atual é um Fine Tuning do [unicamp-dl/ptt5-base-portuguese-vocab](https://huggingface.co/unicamp-dl/ptt5-base-portuguese-vocab) um modelo T5 pré-treinado no corpus BrWac, uma extensa compilação de páginas web em português, aprimorando significativamente o desempenho do T5 em tarefas relacionadas à similaridade e implicação de sentenças em português. Este modelo está disponível em três escalas (pequeno, básico e grande) e possui dois vocabulários distintos: o vocabulário original T5 do Google e o nosso, que foi treinado especificamente na Wikipédia em português.

O modelo ajustado encontra-se em: [VictorNGomes/pttmario5](https://huggingface.co/VictorNGomes/pttmario5)


### Fine-Tuning
O processo de fine-tuning envolve treinar o modelo pré-treinado em um conjunto de dados menor, rotulado e específico para uma tarefa particular. 
Fine-tuning T5 refere-se ao processo de ajustar um modelo pré-treinado chamado T5 (Text-to-Text Transfer Transformer) para uma tarefa de aprendizado específica.



## MLflow

O MLflow é utilizado como a plataforma central para o gerenciamento de experimentos. Ele permite o rastreamento dos parâmetros, métricas e artefatos do modelo durante o treinamento. O uso do MLflow facilita a reprodução de experimentos, a comparação de modelos e a visualização do desempenho ao longo do tempo.

### Instalação do MLflow

```bash
pip install mlflow
```

Para vizalização gráfica do MLflow é exutado o script mlflow_up

## Beam Cloud

O Apache Beam Cloud é empregado para realizar o processamento distribuído durante o treinamento. Ele proporciona escalabilidade e eficiência na manipulação de grandes volumes de dados, otimizando o tempo de treinamento do modelo. Neste caso foi criada uam API para realizar inferencia.

### Inferencia pelo Beam Cloud

```bash
import requests

url = 'https://5xdli-657f654e0c95350007f20190.apps.beam.cloud'

headers = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Authorization': 'TOKEN_BEAM_CLOUD',
    'Content-Type': 'application/json',
}

data = {}

response = requests.post(url, headers=headers, json=data)

print(response.status_code)
print(response.text)
```
## Gradio
Gradio é uma biblioteca de Python que facilita a criação de interfaces de usuário para modelos de aprendizado de máquina.
Com o Gradio, é possível criar interfaces de usuário para modelos de visão computacional, processamento de linguagem natural, regressão, classificação e muitas outras tarefas de aprendizado de máquina. Além disso, ele suporta uma variedade de frameworks populares, como TensorFlow, PyTorch, Scikit-learn, entre outros.

[Interface](https://huggingface.co/spaces/VictorNGomes/Gradio-Summarization-pt-br)



## Estrutura do Repositório

- `train.py`: Script principal para treinamento do modelo. Aceita um conjunto de dados como entrada e executa o Fine Tuning da LLM para sumarização.
- `mlflow_up.py` Script para inciar a interface gráfica do MLflow

## Como Usar

1. Acompanhe o progresso do treinamento utilizando o MLflow:

```bash
export NGROK_AUTH_TOKEN = 'token'
python mlflow_up.py
```

2. Execute o script de treinamento:


```bash
python train.py --model_checkpoint VictorNGomes/pttmario5\
                --tokenizer VictorNGomes/pttmario5 \
                --dataset VictorNGomes/CorpusTeMario\
                --text_column texto\
                --summary_column sumario\
                --experiment_name seu_experimento \
                --push_to_huggingface

```



Abra o navegador e acesse `http://localhost:5000` ou link do NGROK para visualizar os experimentos.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para melhorias, correções ou novas funcionalidades.

## Refrencias
 - [PTT5](https://github.com/unicamp-dl/PTT5)
 - [TéMario](http://www.nilc.icmc.usp.br/nilc/index.php/tools-and-resources}
 - [Notebook based](https://github.com/rohan-paul/LLM-FineTuning-Large-Language-Models/blob/main/Other-Language_Models_BERT_related/Fine_Tuning_Pegasus_for_Text_Summarization.ipynb)
## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
