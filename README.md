# Análise de Sentimentos de Reviews de Produtos - Desafio Técnico

Este projeto visa transformar grandes volumes de avaliações de produtos em insights acionáveis, utilizando técnicas de Inteligência Artificial Generativa (GenAI) e Machine Learning.

Desenvolvido no contexto de um desafio técnico: um grande e-commerce brasileiro precisa de uma forma mais inteligente de analisar os milhares de reviews de seus produtos. A gerente de marketing precisa de **insights rápidos e acionáveis**, sem ter que ler tudo manualmente.

> O desafio é construir um serviço de análise de marketing que transforma reviews brutos em inteligência de negócio, tudo acessível via API.

### 🎯 Etapas do Desafio

1. RAG: Processar os Reviews de produtos e armazena-los em um banco de dados vetorial

2. Análise de Sentimento: Criar um serviço que analisa os reviews e gera insights em relação ao sentimento dos clientes em relação aos produtos.

3. API de Inferência: Criar uma API que permita consultar os insights de sentimento dos produtos.

## 💡 Solução Proposta

A solução foi desenvolvida contendo diferentes componentes que se integram para formar o sistema completo:

1.  **Ingestão**: Script `scripts/ingest.py` que extrai do banco de dados SQL as avaliações de produtos, realiza os embeddings dos textos, previsão do sentimento e armazena no banco vetorial (utilizado `chromadb`).

2.  **Análise de Sentimento**: Com foco em otimizar o processo e custo do sistema um modelo de classificação de sentimentos foi treinado para identificar se um review é positivo, negativo ou neutro, utilizando uma abordagem híbrida de rotulagem:

    -   **Weak Labels**: Rótulos iniciais gerados a partir da nota do review (score), permitindo uma cobertura rápida e de baixo custo de todo o dataset.
    -   **Gold Labels**: Rótulos de alta precisão gerados por um Large Language Model (LLM, `gpt-4o-mini`) para um subconjunto de dados onde o modelo inicial tinha baixa confiança, refinando a qualidade do treinamento.

3.  **API de Inferência**: Uma API construída com **FastAPI** que expõe as informações e gera uma análise completa, incluindo a distribuição de sentimentos, pontos positivos/negativos e os reviews mais relevantes.

4.  **MCP Server**: Um servidor MCP foi criado permitindo que Agentes de IA consultem as informações de sentimento dos produtos de forma eficiente.

5.  **Dashboard**: Uma aplicação interativa (usando Streamlit) exemplificando uma possível forma de consumo da API. Além disso há uma interface de Chat que permite a consulta utilizando um LLM.

## 🛠️ Detalhamento da Solução

### 1. Ingestão de Dados (`scripts/ingest.py`)

O projeto utiliza o [Olist E-commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), um conjunto de dados público com informações de pedidos, produtos e reviews de clientes. O foco principal está na tabela `order_reviews`.

-   **Limpeza de Dados**: Foram identificados e removidos reviews duplicados no banco de dados.

-   **Filtragem de Ambiguidade**: Para garantir que um review se refira a um único produto, foram selecionados apenas os reviews de pedidos que continham um único tipo de produto. Esta decisão, embora reduza o número de amostras, aumenta a confiabilidade da análise, afetando apenas ~5% do dataset original.

-   **Criação de IDs Únicos**: Foi criado um `doc_id` único para cada review de produto em um pedido, facilitando o rastreamento e a junção de dados.

-   **Embeddings de Texto**: Os textos dos reviews foram transformados em vetores numéricos (embeddings) utilizando o modelo `paraphrase-multilingual-MiniLM-L12-v2` da biblioteca `SentenceTransformer`. Este modelo foi escolhido por ser leve e ter bom desempenho em português.

-   **Batch**: Todo o processo de ingestão é realizado em batchs, onde os reviews são extraídos do banco de dados SQL, processados (o que inclui criação dos embeddings e predição do sentimento) e armazenados em um banco de dados vetorial

### 2. Análise de Sentimento (`scripts/refine.py` e `nb2-classification.ipynb`)

Para classificar os sentimentos dos reviews, foi adotada uma abordagem que otimizasse o uso de LLMs apenas onde necessário, garantindo eficiência e custo-benefício. Por isso um modelo de classificação foi treinado.

-   **Modelo de Classificação**: Foi utilizado um classificador **XGBoost**, conhecido por sua performance e eficiência em dados tabulares e estruturados. Utilizamos como feature os próprios embeddings gerados.

    -   **Treinamento com Weak Labels**: Um modelo inicial foi treinado usando os "weak labels" derivados do `review_score` (1-2: negativo, 3: neutro, 4-5: positivo). A avaliação deste modelo (via validação cruzada) mostrou bom desempenho para as classes "positivo" e "negativo", mas uma performance fraca para a classe "neutro", que é inerentemente mais ambígua.

    -   **Refinamento com Gold Labels**: O modelo treinado com weak labels foi usado para prever em todo o conjunto de treino. As amostras onde o modelo apresentou baixa confiança (probabilidade < 80%) foram separadas. Essas amostras "incertas" foram então rotuladas pelo `gpt-4o-mini`, gerando os "gold labels". O script `scripts/refine.py` foi criado para automatizar esse processo.

    -   **Treinamento Final com Pesos**: Um novo dataset de treino foi montado, combinando as amostras "weak" de alta confiança com as amostras "gold". O modelo final foi treinado no dataset combinado. Para dar mais importância aos rótulos de alta qualidade, foi aplicado um sistema de pesos durante o treinamento (`sample_weight`), com um peso maior para os "gold labels". Os melhores pesos foram encontrados através de um Grid Search.

#### Resultados do Modelo

O modelo final, treinado com a abordagem híbrida, foi avaliado no conjunto de _holdout_ (nunca visto em treino) e demonstrou uma **melhora significativa no F1-Score**, especialmente para a classe **neutro**, validando a eficácia da estratégia de refinamento com LLM.

| Modelo                 | F1-Score (Neutro) | F1-Score (Negativo) | F1-Score (Positivo) |
| :--------------------- | :---------------: | :-----------------: | :-----------------: |
| Apenas Weak Labels     |      0.1250       |       0.8077        |       0.8939        |
| **Weak + Gold Labels** |    **0.1875**     |     **0.8381**      |     **0.9049**      |

### 3. API de Inferência (`app/api`)

A API foi construída utilizando **FastAPI**, permitindo consultas eficientes e escaláveis. O endpoint principal é `product_sentiment`, que recebe um `product_id` e retorna:

```json
{
    "sentiment": "positivo",

    "summary": "Os reviews do produto destacam uma mistura de satisfação e insatisfação. Muitos usuários elogiam a qualidade do produto, mencionando que tudo estava dentro do esperado e que não tiveram problemas na execução do pedido. No entanto, há várias reclamações sobre a entrega, incluindo a não recepção de itens completos, com alguns clientes recebendo apenas parte do pedido ou produtos diferentes do que esperavam. Além disso, há críticas sobre a falta de instruções claras para a utilização do produto. A entrega foi mencionada como pontual em alguns casos, mas também houve relatos de danos na embalagem.",

    "sentiment_distrib": {
        "Positivo": 0.68,
        "Negativo": 0.21,
        "Neutro": 0.11
    },

    "positive_points": [
        "Produto atende às expectativas",
        "Satisfação com a execução do pedido",
        "Produto é considerado ótimo"
    ],

    "negative_points": [
        "Problemas com a entrega de produtos",
        "Falta de um relógio na entrega",
        "Produto ainda não foi instalado"
    ],

    "top_reviews": [
        "entregaram antes da data, o produto é bom.\nobrigada",
        "Foi entregue no prazo, veio tudo correto, ganhei um mimos, amei meus produtos. Obrigada. Super recomendo.obg",
        "entregou antes do prazo.super recomendo"
    ]
}
```

### 4. MCP Server (`app/mcp_server`)

O servidor MCP foi implementado para permitir que Agentes de IA consultem as informações de sentimento dos produtos de forma eficiente. Ele expõe uma tool que pode ser utilizado por agentes para obter análises detalhadas de sentimentos, facilitando a integração com sistemas de IA.

### 5. Dashboard (`dashboard`)

Um dashboard interativo foi desenvolvido utilizando **Streamlit**, permitindo que usuários explorem os dados de forma visual e intuitiva.

O dashboard inclui:

-   **Visualização do Retorno da API**: Exibe os resultados da análise de sentimentos de um produto específico, incluindo a distribuição de sentimentos, pontos positivos/negativos e os reviews mais relevantes.

-   **Interface de Chat**: Permite consultas utilizando um LLM, onde o usuário pode fazer perguntas sobre os produtos e receber respostas baseadas nos dados analisados.

## 📁 Estrutura do Projeto

O projeto é organizado da seguinte forma:

```
.
├── app/                # Aplicação FastAPI e servidor MCP
├── dashboard/          # Aplicação do Dashboard
├── data/               # Dados brutos, processados
├── docker/             # Arquivos Docker para conteinerização
├── model/              # Modelo treinado e serializado
├── notebooks/          # Notebooks de exploração e modelagem
└── scripts/            # Scripts de ingestão e refinamento
```

## 🔮 Próximos Passos e Melhorias

-   **Modelo de Embedding**: Explorar o uso de outros modelos de embedding para melhorar avaliar se existe melhora na performance do modelo de classificação.

-   **Aprimoramento do Modelo de Classificação**: Continuar refinando o modelo de classificação, usando outra arquitetura ou técnicas de ensemble, treinar com dados anotados manualmente para avaliar se há melhora na performance, etc.

-   **Monitoramento de Modelos**: Monitorar o desempenho do modelo de classificação em produção, detectando _data drift_ ou _concept drift_.

-   **Melhoria da API**: Adicionar endpoints adicionais para consultas mais específicas, como análises por categoria de produto ou período de tempo.
