
# Projeto de Fine Tuning para Sumarização em Português usando MLOps

## Descrição do Projeto
Este repositório apresenta uma implementação de fine-tuning utilizando o modelo T5 (Text-to-Text Transfer Transformer) um Large Language Model (LLM) para tarefa de sumarização em português com o objetivo de integrar com as ferramentas MLflow, Gradio e Beam Cloud. O projeto utiliza dois conjuntos de dados específicos para o idioma, além das bibliotecas MLflow e Apache Beam Cloud para gerenciamento de experimentos e processamento distribuído.

## Objetivo

O objetivo principal deste projeto é treinar um modelo de sumarização em português utilizando a arquitetura de uma LLM, aprimorando o desempenho do modelo por meio de Fine Tuning. O uso de dois conjuntos de dados específicos visa melhorar a generalização do modelo para diferentes contextos e estilos de texto em português.

## Datasets

O projeto utiliza dois conjuntos de dados obtidos de fontes confiáveis em português para o treinamento do modelo de sumarização. Os datasets estão disponíveis em [Hugging Face Datasets](https://huggingface.co/datasets) e são fundamentais para garantir a qualidade e diversidade dos dados de treinamento.

## T5 Fine-Tuning
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

O Apache Beam Cloud é empregado para realizar o processamento distribuído durante o treinamento. Ele proporciona escalabilidade e eficiência na manipulação de grandes volumes de dados, otimizando o tempo de treinamento do modelo.

### Instalação do Beam Cloud

```bash
pip install apache-beam
```
## Gradio
Gradio é uma biblioteca de Python que facilita a criação de interfaces de usuário para modelos de aprendizado de máquina.
Com o Gradio, é possível criar interfaces de usuário para modelos de visão computacional, processamento de linguagem natural, regressão, classificação e muitas outras tarefas de aprendizado de máquina. Além disso, ele suporta uma variedade de frameworks populares, como TensorFlow, PyTorch, Scikit-learn, entre outros.

### Instalação do Gradio
```bash
pip install gradio
```

## Estrutura do Repositório

- `train.py`: Script principal para treinamento do modelo. Aceita um conjunto de dados como entrada e executa o Fine Tuning da LLM para sumarização.
- `datasets/`: Diretório que contém os conjuntos de dados utilizados para o treinamento.
- `mlflow/`: Diretório para armazenar os logs e artefatos do MLflow.
- `beam_cloud/`: Diretório que contém scripts e configurações para o processamento distribuído com Apache Beam Cloud.

## Como Usar

1. Clone este repositório:

```bash
git clone https://github.com/seu-usuario/mlops-sumarizacao.git
cd mlops-sumarizacao
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute o script de treinamento:

```bash
python train.py --dataset caminho/para/dataset
```

4. Acompanhe o progresso do treinamento utilizando o MLflow:

```bash
mlflow ui
```

Abra o navegador e acesse `http://localhost:5000` para visualizar os experimentos.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para melhorias, correções ou novas funcionalidades.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
