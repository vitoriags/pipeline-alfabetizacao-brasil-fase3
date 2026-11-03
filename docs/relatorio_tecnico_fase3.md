# Relatório técnico — Tech Challenge Fase 3

## Objetivo e desenho do estudo

Estimar a classificação de alfabetização dos estudantes com base em variáveis educacionais e territoriais, interpretando limites da predição individual e relacionando indicadores territoriais a metas municipais. Base: Gold da Fase 2, acrescida de dados individuais; alvo binário `alfabetizado` (`0`/`1`).

## Dados e qualidade

A base analítica contém 3.354.661 registros válidos. O treinamento foi realizado sobre 1.502.809 registros de 2023 de redes públicas; o teste temporal sobre 1.851.828 registros de 2024. A análise exploratória documenta auditoria de granularidade e maior parte das covariáveis em nível municipal. Não foram usadas proficiência individual nem taxas agregadas contemporâneas de alfabetização como features.

## Modelagem

Experimentos compararam baseline, Regressão Logística e HistGradientBoosting. O modelo selecionado foi HistGradientBoosting com `max_iter=150`, `max_leaf_nodes=7`, `learning_rate=0.05`, `min_samples_leaf=100`, `l2_regularization=5`, `early_stopping=False` e `random_state=42`. Imputação, padronização e one-hot encoding fazem parte de pipeline Scikit-learn.

Validação por município com GroupKFold (3 folds): ROC-AUC médio 0,5813 ± 0,0017. O teste de 2024 foi mantido fora da escolha de hiperparâmetros.

## Resultados do teste temporal de 2024

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,6067 |
| Balanced accuracy | 0,5315 |
| ROC-AUC | 0,5906 |
| Average precision (classe 0) | 0,4833 |
| Precision (classe 0) | 0,5407 |
| Recall (classe 0) | 0,1472 |
| F1-score (classe 0) | 0,2314 |

A matriz de confusão, considerando **classe de interesse 0 como positiva**, fornece TP=109.651, FN=635.074, FP=93.160, TN=1.013.943. Com o limiar padrão, 85,28% dos alunos não alfabetizados não foram identificados. Não recomendamos utilizar o modelo para diagnóstico individual ou decisão automatizada.

## Interpretabilidade

Permutation Importance e SHAP do notebook destacaram biblioteca/sala de leitura, internet para aprendizagem e densidade demográfica entre as variáveis mais influentes nas previsões. Resultados são associações preditivas, **não efeitos causais**.

## Análise territorial e metas

Em 2.696 municípios com metas comparáveis, 1.301 (48,3%) ficaram abaixo da meta em 2024. Percentuais de municípios abaixo da meta: Sul 69,77%; Norte 61,22%; Nordeste 50,64%; Sudeste 36,75%; Centro-Oeste 28,12%.

A correlação de Spearman entre risco médio municipal previsto e proporção observada de não alfabetizados foi 0,4214; MAE municipal 13,34 pontos percentuais. Apenas 9 municípios foram comuns entre os top 10% de maior déficit e de maior risco previsto (6,87% de sobreposição). O risco individual agregado **não** equivale à probabilidade de descumprimento de meta futura.

## Aplicação estratégica

Priorizar análise de indicadores observados de alfabetização, déficit frente à meta e participação na avaliação. Usar a predição como complemento experimental, não como critério decisório isolado. Para novas pesquisas: ampliar variáveis no nível escolar/individual; avaliar mais períodos; propor modelo municipal se o desfecho for cumprimento de meta.

## Reprodutibilidade e limitações

O comando `python run_pipeline.py` gera novos artefatos numerados de modelo, avaliação JSON e PNG, desde que a base Gold exista no caminho esperado. Os microdados e o modelo não são versionados; a obtenção da Gold exige a Fase 2 (documentada em `docs/reproducao.md`). As condições de disponibilidade temporal das features e a limitação de explicação causal permanecem ressalvas importantes.

**Fonte dos números:** notebooks `00_data_audit_and_integration.ipynb`, `02_eda_base_analitica.ipynb` e `03_preparacao_modelagem.ipynb` enviados para revisão, e saída validada do script de avaliação.
