# Tech Challenge — Fase 3 | Predição e inteligência analítica para alfabetização no Brasil

Projeto acadêmico da pós-graduação da FIAP, desenvolvido a partir da camada **Gold** do [projeto de Engenharia de Dados da Fase 2](https://github.com/vitoriags/pipeline-alfabetizacao-brasil). O objetivo é investigar fatores associados aos resultados de alfabetização e avaliar a viabilidade de uma classificação supervisionada, com interpretação dos resultados e análise territorial.

> **Escopo e cautela:** o modelo é experimental. Suas previsões não devem ser usadas isoladamente para diagnósticos individuais, seleção de estudantes ou decisões automatizadas de política pública.

## Problema e objetivo analítico

O desafio investiga se indicadores educacionais e territoriais podem ajudar a estimar a classificação binária de alfabetização de estudantes avaliados no Brasil. A variável-alvo é `alfabetizado` (`0`: não alfabetizado; `1`: alfabetizado). O objetivo estratégico é complementar análises de desigualdades territoriais e cumprimento de metas de alfabetização, reconhecendo os limites das variáveis disponíveis.

## Dados e origem

- **Fonte de referência:** dados educacionais tratados na Fase 2, enriquecidos com indicadores municipais do IBGE e do Censo Escolar, conforme documentado na base Gold original.
- **Base principal:** `data/gold/base_analitica_alfabetizacao.parquet` (3.354.661 registros válidos, 24 colunas, anos de 2023 e 2024).
- **Treinamento:** 1.502.809 registros de 2023, redes públicas estadual e municipal.
- **Teste temporal:** 1.851.828 registros de 2024, mesmas redes, reservados para a avaliação final.
- **Features:** 11 variáveis contextuais: população, área, densidade, infraestrutura e rede de ensino. A seleção está em `src/preprocessing/features.py`.

**Os arquivos Parquet e os microdados individuais não são versionados neste repositório.** São um pré-requisito para reproduzir a execução. Consulte [Como reproduzir](docs/reproducao.md) para disponibilizar a camada Gold produzida na Fase 2.

## Metodologia

1. Auditoria e análise exploratória dos dados, incluindo granularidade, valores faltantes, distribuição e padrões municipais.
2. Divisão temporal: experimentação e escolha dos parâmetros em 2023; teste final em 2024.
3. Comparação experimental com `DummyClassifier`, Regressão Logística e HistGradientBoosting; validação cruzada por município (`GroupKFold`, três folds) documentada no notebook.
4. Modelo final: `HistGradientBoostingClassifier`, integrado a `Pipeline`/`ColumnTransformer` do Scikit-learn, com imputação de valores faltantes, padronização de numéricas e one-hot encoding de rede de ensino.
5. Avaliação por métricas de discriminação, classificação e análise temporal; interpretação exploratória por Permutation Importance e SHAP no notebook.
6. Análise territorial e confronto retrospectivo com metas municipais.

A auditoria temporal indica uso de referências anteriores do Censo Escolar. Entretanto, a data exata de disponibilidade dos dados e a natureza majoritariamente municipal das features limitam a interpretação como previsão individual operacional e a generalização causal.

## Executar a pipeline

Recomendado: Python 3.10+ em ambiente virtual, com memória suficiente para o volume de dados.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Disponibilize **antes** `data/gold/base_analitica_alfabetizacao.parquet` e execute **na raiz do projeto**:

```bash
python run_pipeline.py
```

O script coordena treinamento do modelo final e avaliação temporal, em processos separados para reduzir picos de memória, gerando a cada execução arquivos numerados:

```text
models/modelo_001.joblib
reports/metricas_modelo_001.json
reports/figures/avaliacao_modelo_001.png
```

Em uma nova execução, a numeração avança para `002`, `003` etc. **Isso representa diferentes execuções da mesma configuração, não diferentes algoritmos.** Como o modelo usa `random_state=42`, novas execuções com dados e versões iguais podem apresentar as mesmas métricas.

### Inferência com uma execução específica

```python
import joblib
import pandas as pd
from src.modeling.inferencia import prever_risco
from src.preprocessing.features import FEATURES

modelo = joblib.load("models/modelo_001.joblib")  # somente arquivo confiável
# Exemplo ilustrativo: substitua por um DataFrame com as 11 features.
dados = pd.read_parquet(
    "data/gold/base_analitica_alfabetizacao.parquet",
    columns=FEATURES,
).head(5)
resultado = prever_risco(dados, modelo=modelo)
print(resultado[["risco_nao_alfabetizacao"]])
```

> A função `carregar_modelo()` de `inferencia.py` ainda pode apontar para o nome antigo `histgradientboosting_alfabetizacao_2023.joblib`; para modelos **numerados**, carregue explicitamente o arquivo da execução desejada, como no exemplo acima.

## Resultados do modelo final — Teste temporal 2024

| Indicador | Resultado |
|---|---:|
| Accuracy | 60,67% |
| Balanced Accuracy | 53,15% |
| ROC-AUC | 0,5906 |
| Average Precision (classe 0) | 0,4833 |
| Precisão da classe não alfabetizado | 54,07% |
| Recall da classe não alfabetizado | 14,72% |
| F1 da classe não alfabetizado | 0,2314 |

Dos 744.725 alunos não alfabetizados no conjunto de teste, **109.651 foram identificados** como classe 0 e **635.074 não foram identificados**, considerando a regra padrão de classificação. O modelo apresenta capacidade discriminativa limitada e não está validado para apoiar decisões individuais.

Na validação cruzada por município em 2023, o ROC-AUC médio foi **0,5813** (desvio-padrão **0,0017**). A Permutation Importance e o SHAP destacaram bibliotecas, internet para aprendizagem e densidade demográfica como variáveis influentes nas *previsões*, não como causas demonstradas da alfabetização.

## Insights territoriais e aplicação estratégica

Na análise retrospectiva de **2.696 municípios** com metas comparáveis em 2024, **1.301 (48,3%)** ficaram abaixo das respectivas metas. Por região, os percentuais de municípios abaixo da meta nessa amostra foram: Sul **69,77%**, Norte **61,22%**, Nordeste **50,64%**, Sudeste **36,75%** e Centro-Oeste **28,12%**.

A associação entre riscos preditos e déficits municipais foi fraca. Assim, a recomendação é priorizar **indicadores observados** (resultado de alfabetização, distância até a meta e participação na avaliação), usando as previsões apenas como informação experimental complementar. Os recortes regionais referem-se à amostra filtrada, **não são estatísticas oficiais regionais ponderadas**.

## Limitações

- Predominância de variáveis municipais para um alvo individual; alunos do mesmo contexto podem receber previsões iguais.
- Desempenho baixo para detecção da classe de interesse com o limiar avaliado.
- Dependência externa da camada Gold produzida na Fase 2; a clonagem isolada do repositório não inclui os dados.
- A disponibilidade histórica exata de cada variável precisa ser considerada para uso prospectivo.
- Importância de variáveis não implica efeito causal; análise de metas é retrospectiva, não previsão de descumprimento futuro.
- Os resultados territoriais decorrem de filtros amostrais, não abrangem todos os municípios indistintamente.

## Estrutura principal

```text
notebooks/                    # auditoria, EDA e experimentação
src/preprocessing/features.py  # definição das 11 features
src/modeling/treinamento.py    # pipeline e treinamento
src/modeling/inferencia.py     # predição de risco
src/evaluation/avaliacao.py              # cálculo das métricas de avaliação
src/visualization/relatorio_visual.py    # geração dos gráficos e relatório visual
run_pipeline.py                # ponto de entrada de ponta a ponta
models/                        # artefatos locais gerados (.joblib)
reports/                       # métricas JSON e gráficos PNG
images/                        # imagens de apoio à documentação
docs/                          # documentação e reprodução
```

## Evoluções possíveis

Dados escolares e individuais adicionais, validação temporal em mais anos, calibração, métricas de custo e, se o objetivo institucional for metas, desenvolvimento de um **modelo municipal específico**. Nenhuma dessas evoluções é apresentada como resultado já implementado.

## Referências internas

- `notebooks/00_data_audit_and_integration.ipynb`
- `notebooks/02_eda_base_analitica.ipynb`
- `notebooks/03_preparacao_modelagem.ipynb`
- `docs/reproducao.md`
- [Projeto de Engenharia de Dados — Fase 2](https://github.com/vitoriags/pipeline-alfabetizacao-brasil)
