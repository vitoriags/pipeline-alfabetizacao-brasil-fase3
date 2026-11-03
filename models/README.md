# Modelos treinados

Esta pasta armazena os modelos gerados automaticamente
pelo pipeline de Machine Learning.

O algoritmo selecionado é o HistGradientBoostingClassifier,
com os hiperparâmetros definidos na etapa de validação.

Para treinar e avaliar o modelo, execute na raiz do projeto:

    python run_pipeline.py

Cada execução gera um modelo com numeração sequencial,
como modelo_001.joblib e modelo_002.joblib.

Os arquivos .joblib não são versionados no Git.
