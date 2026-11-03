# Reprodução do Tech Challenge — Fase 3

## 1. Requisitos

- Python 3.10+ e `pip`.
- Recursos de CPU/RAM suficientes para treinar um classificador sobre aproximadamente 1,5 milhão de registros e avaliar aproximadamente 1,85 milhão.
- Acesso à camada Gold gerada pela Fase 2; **os dados não acompanham o clone** por volume e cuidados de divulgação de registros individuais.

## 2. Obter a base Gold

A origem é o projeto [pipeline-alfabetizacao-brasil — Fase 2](https://github.com/vitoriags/pipeline-alfabetizacao-brasil). Consulte suas instruções de obtenção/transformação e gere a base `base_analitica_alfabetizacao.parquet`. A reconstrução integral pode exigir acesso e credenciais das fontes originais, incluindo a tabela individual de alunos; portanto, este repositório **não oferece descarga automática autônoma dos dados**.

Após obter o arquivo, copie-o para a nova instalação da Fase 3:

```bash
mkdir -p data/gold
cp /CAMINHO/DA/FASE2/data/gold/base_analitica_alfabetizacao.parquet data/gold/
```

No comando acima, substitua `/CAMINHO/DA/FASE2` pelo caminho real do projeto de origem. Evite versionar ou compartilhar microdados individuais. Nos notebooks, outras tabelas Gold e `data/silver/alunos.parquet` podem ser necessárias; copie-as também conforme os caminhos referenciados.

## 3. Preparar o ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux, WSL, macOS
python -m pip install -r requirements.txt
python -m pip check
```

No Windows PowerShell sem WSL, a ativação é `.venv\\Scripts\\Activate.ps1`.

## 4. Verificar a base

```bash
python - <<'PY'
import pandas as pd
p = 'data/gold/base_analitica_alfabetizacao.parquet'
df = pd.read_parquet(p, columns=['ano', 'alfabetizado', 'rede_codigo'])
print(df.shape)
print(df.groupby('ano').size())
PY
```

## 5. Executar

```bash
python run_pipeline.py
```

Os modelos gerados são sempre `HistGradientBoostingClassifier`, com parâmetros do notebook final. Para evitar sobrescritas, cada execução recebe um número sequencial e gera o `.joblib`, o JSON de métricas e a imagem de avaliação correspondentes.

## 6. Interpretar corretamente

Os testes documentados reproduziram aproximadamente **ROC-AUC 0,5906** e **recall da classe 0 14,72%** no teste temporal de 2024. O desempenho não é adequado para recomendações individuais. Repetições em hardware e versões diferentes podem exigir verificação de equivalência numérica.

**Segurança:** não execute arquivos Joblib de origem desconhecida. Não publique chaves de acesso, microdados ou dados identificáveis.

## 7. Limites de reprodutibilidade

A pipeline de ML funciona em ambiente com dados disponíveis. **Não é um projeto 100% autocontido**: a regeneração da camada Gold pertence à Fase 2 e pode depender de credenciais externas. A etapa de geração de gráficos no `run_pipeline.py` também depende da versão atual do código de avaliação, que deve expor `avaliar_modelo(..., caminho_relatorio_visual=...)`.
