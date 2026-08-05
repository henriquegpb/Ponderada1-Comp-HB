# Baseline clássico de análise de sentimentos — B2W-Reviews01

Entrega da P-Comp 1: comparação entre **Bag of Words** e **TF-IDF** com cinco classificadores clássicos (Naive Bayes, Regressão Logística, SVM linear, Random Forest e Gradient Boosting).

## Arquivos da entrega

- `baseline_sentimentos_b2w.ipynb`: notebook completo, comentado e já executado.
- `RELATORIO.md`: relatório curto com metodologia, resultados, matriz de confusão e análise de erros.
- `RELATORIO.pdf`: versão pronta para anexar do relatório técnico.
- `requirements.txt`: dependências para reprodução.
- `artifacts/`: tabelas e figuras geradas pela execução do notebook.

## Como reproduzir

Recomenda-se Python 3.11 ou 3.12. A partir da raiz do projeto:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook baseline_sentimentos_b2w.ipynb
```

No Jupyter, execute **Run All**. O notebook baixa e armazena o CSV oficial em `data/` na primeira execução. A semente é fixa (`RANDOM_STATE = 42`). Os vetorizadores são ajustados apenas no treino, impedindo vazamento do vocabulário e dos pesos IDF do teste.

Para atualizar o PDF após editar `RELATORIO.md`, execute `python scripts/render_report.py`. O script procura Google Chrome/Chromium no macOS, Linux ou Windows.


## Decisões principais

- Polaridade binária: notas 1–2 são negativas; 4–5 são positivas; nota 3 é neutra e não participa do problema binário.
- Textos vazios e duplicatas exatas são removidos antes da divisão.
- A análise descritiva usa o corpus completo; os dez experimentos usam uma amostra estratificada e reprodutível de 30 mil textos por viabilidade computacional dos ensembles de árvores.
- Precisão, revocação e F1 são macro-médias, para atribuir o mesmo peso às duas classes; a acurácia também é informada.
- Não é usado modelo pré-treinado de classificação, deep learning ou LLM. O modelo pequeno do spaCy é empregado somente para tokenização e lematização linguística.

## Fonte e licença do dataset

B2W-Reviews01, Americanas/B2W Digital: <https://github.com/americanas-tech/b2w-reviews01>. O corpus é distribuído sob CC BY-NC-SA 4.0; o CSV não é versionado neste repositório e é baixado diretamente da fonte oficial.
