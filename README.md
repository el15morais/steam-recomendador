# Recomendador de Jogos da Steam — perfil recente do jogador

Projeto da disciplina de **Projeto de Mineração de Dados** (Centro Universitário FAG).
Metodologia **CRISP-DM**.

**Equipe:** Antonio Landim · Gabriel Morais · Kaue Correa · Pedro Paulo

## Pergunta de pesquisa
> Usar vários jogos recentes (5) gera recomendações mais coerentes do que usar apenas um jogo?

Na parte 1 (seminário) a proposta usava só metadados dos jogos, e a avaliação era manual.
Seguindo o feedback do professor, **incorporamos uma segunda base com interações reais de usuários**.
Com ela dá para montar perfis reais, medir a qualidade das recomendações com métricas objetivas
(Precision, Recall, HitRate e NDCG @10) e comparar modelos de conteúdo com modelos colaborativos.

## Resultados (conjunto de teste, 3.500 usuários)

| Modelo | NDCG@10 (1 jogo) | NDCG@10 (5 jogos) | HitRate@10 (5 jogos) | Cobertura (5 jogos) | 1 × 5 significativo? |
|---|---|---|---|---|---|
| Popularidade | 0,017 | 0,017 | 7,7% | 0,1% | — |
| Conteúdo TF-IDF (média) | 0,019 | 0,022 | 8,0% | 33,6% | não |
| Conteúdo TF-IDF (máximo) | 0,019 | 0,025 | 8,3% | 36,3% | sim |
| Colaborativo item-item | 0,040 | 0,052 | 18,7% | 29,2% | sim |
| Fatoração SVD (16 dim.) | 0,034 | 0,038 | 15,4% | 1,1% | sim |
| Híbrido (α=0,4) | 0,039 | 0,055 | 19,4% | 29,7% | sim |

- **Pergunta de pesquisa:** sim — o perfil de 5 jogos foi melhor que o de 1 jogo em 4 dos 5 modelos personalizados (Wilcoxon pareado, 5%).
- **Melhor modelo:** Híbrido (α=0,4) — NDCG@10 de 0,039 → 0,055 ao passar de 1 para 5 jogos; acerta pelo menos um jogo em 19,4% dos usuários (aleatório: ~0,2%).
- **Custo:** perfis maiores concentram as recomendações em jogos populares (cobertura do catálogo cai).
- Figuras em `reports/figures/`, tabelas em `reports/results/`, notebooks executados em `notebooks/`.

## Bases de dados

| Base | Conteúdo | Link |
|---|---|---|
| A — Steam Games Dataset | metadados de ~145 mil jogos: gêneros, categorias, tags, avaliações | [Kaggle](https://www.kaggle.com/datasets/hubertsidorowicz/steam-games-dataset-daily-updates) |
| B — Game Recommendations on Steam | ~41 milhões de recomendações usuário × jogo, com data e horas jogadas | [Kaggle](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) |

As bases são unidas pelo `app_id`. Os arquivos brutos **não vão para o GitHub** (são grandes demais).

## Estrutura

```
├── data/
│   ├── raw/                 # bases brutas (baixadas com scripts/baixar_dados.py)
│   └── processed/           # saídas da preparação (parquet)
├── notebooks/
│   ├── 00_executar_no_colab.ipynb        # roda tudo no Colab (01 → 02 → 03)
│   ├── 01_entendimento_dos_dados.ipynb   # CRISP-DM 1 e 2 — EDA
│   ├── 02_preparacao_dos_dados.ipynb     # CRISP-DM 3 — limpeza, junção, divisão temporal
│   └── 03_modelagem_e_avaliacao.ipynb    # CRISP-DM 4 e 5 — modelos, métricas, testes
├── src/
│   ├── config.py            # caminhos e parâmetros (fonte única)
│   ├── carregamento.py      # leitura das bases (tipos enxutos, leitura em blocos)
│   ├── preparacao.py        # limpeza, filtro de qualidade, TF-IDF text, divisão temporal
│   ├── modelos.py           # Popularidade, TF-IDF, item-item, SVD, Híbrido
│   ├── avaliacao.py         # métricas @K, bootstrap, Wilcoxon
│   └── graficos.py          # padrão visual das figuras do artigo
├── api/main.py              # FastAPI (CRISP-DM 6 — Implantação)
├── reports/
│   ├── figures/             # figuras numeradas na ordem do artigo
│   └── results/             # tabelas (CSV/JSON) usadas no artigo
└── scripts/baixar_dados.py
```

## Como executar

```bash
git clone <url-do-repositorio>
cd steam-recomendador
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/baixar_dados.py                       # baixa as duas bases para data/raw/
jupyter notebook                                     # execute 01 -> 02 -> 03, nessa ordem
```

**Google Colab (mais simples):** abra `notebooks/00_executar_no_colab.ipynb` no Colab (Arquivo → Abrir notebook → GitHub) e use **Ambiente de execução → Executar tudo**. Ele baixa as bases, executa os três notebooks e baixa um `resultados.zip`.

**Colab manual:** clone o repositório numa célula (`!git clone ...`), entre na pasta com
`%cd steam-recomendador`, rode `!pip install -r requirements.txt` e `!python scripts/baixar_dados.py`
e abra os notebooks da pasta `notebooks/`.

**Memória:** por padrão usamos 15% dos usuários da base B (amostra determinística por `user_id`).
Para mudar sem editar o código, defina a variável de ambiente antes de abrir o Jupyter:
`PERCENTUAL_USUARIOS=30 jupyter notebook`.

### API

Depois de executar o notebook 03 (ele salva `models/recomendador.joblib`):

```bash
uvicorn api.main:app --reload
```

Abra http://127.0.0.1:8000/docs. Exemplo:

```bash
curl -X POST http://127.0.0.1:8000/recomendar \
  -H "Content-Type: application/json" \
  -d '{"jogos": ["ELDEN RING", "Cyberpunk 2077", "Red Dead Redemption 2"], "k": 10}'
```

## CRISP-DM → onde está cada etapa

| Etapa | Onde |
|---|---|
| 1. Entendimento do negócio | notebook 01 (início) |
| 2. Entendimento dos dados | notebook 01 — Figuras 1 a 6, tabelas de colunas e faltantes |
| 3. Preparação dos dados | notebook 02 + `src/preparacao.py` — Figura 7, tabela do funil |
| 4. Modelagem | notebook 03 + `src/modelos.py` — ajuste na validação (Figura 8) |
| 5. Avaliação | notebook 03 + `src/avaliacao.py` — Figuras 9 e 10, Wilcoxon, ablação, avaliação qualitativa |
| 6. Implantação | `api/main.py` (FastAPI) |

## Protocolo de avaliação (resumo)
1. Para cada usuário, as interações são ordenadas por data; os **3 jogos mais recentes** ficam escondidos (teste).
2. O perfil é formado pelos **últimos 1 ou 5 jogos** do histórico restante.
3. Cada modelo gera um top-10 (sem repetir jogos que o usuário já tem).
4. Métricas: Precision@10, Recall@10, HitRate@10, NDCG@10, cobertura do catálogo.
5. Hiperparâmetros (dimensões do SVD e peso α do híbrido) ajustados num grupo de **validação**; resultado final num grupo de **teste** separado.
6. Diferença 1 × 5 jogos testada com **Wilcoxon pareado**; intervalos de confiança por **bootstrap**.
