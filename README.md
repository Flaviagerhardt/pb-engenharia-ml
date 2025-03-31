# Projeto: Preditor de Arremessos do Kobe Bryant

## Visão Geral

Este projeto tem como objetivo construir um sistema de previsão para identificar se um arremesso realizado pelo astro da NBA **Kobe Bryant** resultou em cesta ou não. O projeto aborda a **Engenharia de Machine Learning** de forma completa, desde a ingestão dos dados até a operação e monitoramento de modelos.

## Estrutura do Projeto

A estrutura a seguir representa a organização dos diretórios e arquivos do projeto, seguindo o framework TDSP (Team Data Science Process) da Microsoft.

PB_ENGENHARIA_ML/
│
├── data/                           # Diretório de dados
│   ├── processed/                  # Dados processados
│   │   ├── base_test.parquet       # Conjunto de teste
│   │   ├── base_train.parquet      # Conjunto de treino
│   │   ├── data_filtered.parquet   # Dados filtrados
│   │   └── predicoes_producao.parquet # Predições em produção
│   │
│   └── raw/                        # Dados brutos (originais)
│       ├── dataset_kobe_dev.parquet  # Dataset de desenvolvimento
│       └── dataset_kobe_prod.parquet # Dataset de produção
│
├── mlruns/                         # Diretório do MLflow para tracking de experimentos
│
├── src/                            # Código-fonte do projeto
│   ├── aplicacao.py                # Script para aplicação do modelo em produção
│   ├── dashboard.py                # Dashboard Streamlit para monitoramento
│   ├── preparacao_dados            # Script para preparação e pré-processamento dos dados
│   └── treinamento_dados           # Script para treinamento e avaliação de modelos
│
├── .gitignore                      # Arquivo de configuração do Git
├── logs.log                        # Arquivo de logs do sistema
├── modelo_kobe_final.pkl           # Modelo serializado final
├── README.md                       # Documentação do projeto
└── requirements.txt                # Dependências do projeto

## Descrição dos Componentes

### Diretórios de Dados
- **data/raw**: Armazena os dados brutos, sem nenhuma modificação.
  - `dataset_kobe_dev.parquet`: Dados de desenvolvimento para treinar o modelo.
  - `dataset_kobe_prod.parquet`: Dados de produção para aplicar o modelo.

- **data/processed**: Contém dados após processamento e transformações.
  - `data_filtered.parquet`: Dados após filtragem e limpeza.
  - `base_train.parquet`: Conjunto de dados de treino.
  - `base_test.parquet`: Conjunto de dados de teste.
  - `predicoes_producao.parquet`: Resultados das predições em produção.

### Código-fonte
- **src/**: Contém todos os scripts do projeto.
  - `preparacao_dados`: Script para processar os dados brutos.
  - `treinamento_dados`: Script para treinar e avaliar os modelos.
  - `aplicacao.py`: Script para aplicar o modelo em novos dados.
  - `dashboard.py`: Dashboard interativo para monitoramento.

### Artefatos e Configurações
- `modelo_kobe_final.pkl`: Modelo treinado e serializado.
- `requirements.txt`: Lista de dependências do projeto.
- `README.md`: Documentação do projeto.
- `mlruns/`: Diretório onde o MLflow armazena métricas, parâmetros e artefatos.

Esta estrutura segue boas práticas de organização de projetos de ciência de dados, separando claramente dados de código, e mantendo a rastreabilidade do processo através do MLflow.

## Pipeline de Machine Learning

Segue abaixo um diagrama de fluxo que mostra como os dados e processos se conectam, desde a preparação dos dados até a aplicação em produção e monitoramento, dividido em fases de desenvolvimento e produção.

![alt text](image.png)

### 1. Preparação de Dados
- **Objetivo**: Preparar os dados para treinamento do modelo.
- **Implementação**: Script `src/preparacao_dados`.
- **Processo**:
  - Carregamento dos dados brutos (dataset_kobe_dev.parquet)
  - Seleção de colunas relevantes: `lat`, `lon`, `minutes_remaining`, `period`, `playoffs`, `shot_distance`, `shot_made_flag`
  - Remoção de valores faltantes
  - Divisão em conjuntos de treino (80%) e teste (20%), estratificados pela variável alvo
  - Salva os dados processados em `data/processed/`

**Métricas registradas no MLflow**:
- Número de linhas filtradas: 20.285
- Número de colunas: 7
- Tamanho do conjunto de treino: 16.228
- Tamanho do conjunto de teste: 4.057

### 2. Treinamento do Modelo
- **Objetivo**: Treinar e avaliar diferentes modelos para prever os arremessos.
- **Implementação**: Script `src/treinamento_dados`.
- **Modelos treinados**:
  1. **Regressão Logística**
     - Log Loss: 0,7157
     - F1 Score: 0,5139
  2. **Árvore de Decisão**
     - Log Loss: 0,7128
     - F1 Score: 0,4373
- **Modelo escolhido**: Regressão Logística (melhor F1 Score)
- **Armazenamento**: O modelo final é salvo utilizando PyCaret e registrado no MLflow para rastreamento.

### 3. Aplicação em Produção
- **Objetivo**: Aplicar o modelo treinado em dados de produção.
- **Implementação**: Script `src/aplicacao.py`.
- **Processo**:
  - Carregamento do modelo treinado
  - Carregamento e pré-processamento dos dados de produção
  - Geração de previsões
  - Cálculo de métricas de desempenho quando a variável alvo está disponível
  - Salva as previsões em `data/processed/predicoes_producao.parquet`

**Métricas registradas em produção**:
- Log Loss em produção: 0,9058
- F1 Score em produção: 0

### 4. Monitoramento
- **Objetivo**: Monitorar o desempenho do modelo em produção.
- **Implementação**: Dashboard interativo em `src/dashboard.py` usando Streamlit.
- **Funcionalidades**:
  - Visualização dos dados e previsões
  - Métricas de desempenho (Log Loss, F1 Score)
  - Gráficos de distribuição de acertos/erros por distância
  - Mapa de arremessos (visualização espacial)
  - Suporte para upload de novos arquivos de previsão

