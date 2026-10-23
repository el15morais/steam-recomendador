"""
Gera os notebooks 01, 02 e 03 a partir deste script (fonte única de verdade).
Uso: python notebooks/_construir_notebooks.py
Depois é só abrir os .ipynb e executar.
"""
from pathlib import Path

import nbformat as nbf

PASTA = Path(__file__).resolve().parent

SETUP = '''import sys
from pathlib import Path

RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config, carregamento as c, preparacao as p, graficos as g

config.criar_pastas()
g.aplicar_estilo()
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
print("Percentual de usuários amostrados:", config.PERCENTUAL_USUARIOS, "%")'''


def md(texto):
    return nbf.v4.new_markdown_cell(texto.strip())


def code(texto):
    return nbf.v4.new_code_cell(texto.strip())


def salvar(nome, celulas):
    nb = nbf.v4.new_notebook()
    nb["cells"] = celulas
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, PASTA / nome)
    print("gerado:", nome)


# =============================================================================
# 01 — Entendimento do negócio e dos dados
# =============================================================================
nb01 = [
md("""
# 01 — Entendimento do Negócio e dos Dados
**Projeto:** Recomendador de jogos da Steam baseado no perfil recente do jogador
**Metodologia:** CRISP-DM — etapas 1 (Entendimento do Negócio) e 2 (Entendimento dos Dados)

## 1. Entendimento do Negócio
- **Problema:** a Steam tem dezenas de milhares de jogos; achar o próximo jogo parecido com o que a pessoa gosta é difícil.
- **Pergunta de pesquisa:** usar vários jogos recentes (5) gera recomendações mais coerentes do que usar apenas um jogo?
- **Mudança em relação ao seminário (feedback do professor):** incorporamos uma segunda base, com **interações reais de usuários**.
  Isso permite (a) montar perfis reais de jogadores, (b) medir a qualidade das recomendações com métricas objetivas e
  (c) comparar modelos de conteúdo com modelos colaborativos.

| Base | Fonte | Papel no projeto |
|---|---|---|
| A — Steam Games Dataset | Kaggle (Hubert Sidorowicz) | metadados: gêneros, categorias, tags, avaliações |
| B — Game Recommendations on Steam | Kaggle (Anton Kozyriev) | interações usuário × jogo (recomendou? horas? data) |
"""),
code(SETUP),
md("## 2. Base A — metadados dos jogos"),
code('''A = c.carregar_base_a()
print(f"Base A: {A.shape[0]:,} linhas x {A.shape[1]} colunas | {c.memoria_mb(A):,.0f} MB em memória")
A.head()'''),
code('''faltantes_a = c.resumo_colunas(A)
faltantes_a.to_csv(config.RESULTADOS / "tabela_colunas_base_a.csv")
faltantes_a'''),
code('''problemas_a = {
    "linhas": len(A),
    "app_id_duplicados": int(A["app_id"].duplicated().sum()),
    "nome_vazio": int(A["name"].isna().sum()),
    "sem_genero": int(A["genres"].map(p.parse_lista).map(len).eq(0).sum()),
    "sem_tags": int(A["tags"].map(p.parse_lista).map(len).eq(0).sum()) if "tags" in A else None,
    "sem_nenhuma_avaliacao": int(((A["positive"].fillna(0) + A["negative"].fillna(0)) == 0).sum()),
}
problemas_a'''),
md("### Gêneros mais frequentes (figura usada no seminário, agora reproduzível em código)"),
code('''generos = (A.drop_duplicates("app_id")["genres"].map(p.parse_lista)
             .explode().dropna().value_counts().head(10).sort_values())
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(generos.index, generos.values, color=g.AZUL, height=0.7)
g.rotular_barras(ax, horizontal=True)
ax.set_title("Figura 1 — Os 10 gêneros mais frequentes (base A)")
ax.set_xlabel("Quantidade de jogos")
g.eixo_br(ax, "x")
ax.grid(axis="y", visible=False)
g.salvar(fig, "fig01_generos_top10")
plt.show()'''),
md("### Volume de avaliações por jogo — justificativa para um volume mínimo no filtro de qualidade"),
code('''total_av = (A["positive"].fillna(0) + A["negative"].fillna(0))
print(total_av.describe(percentiles=[.25, .5, .75, .9, .99]))
print(f"Jogos com menos de {config.MIN_AVALIACOES_QUALIDADE} avaliações: {(total_av < config.MIN_AVALIACOES_QUALIDADE).mean():.1%}")

fig, ax = plt.subplots()
ax.hist(np.log10(total_av + 1), bins=50, color=g.AZUL, edgecolor="white", linewidth=0.5)
ax.axvline(np.log10(config.MIN_AVALIACOES_QUALIDADE + 1), color=g.TEXTO_SECUNDARIO, ls="--", lw=1)
ax.text(np.log10(config.MIN_AVALIACOES_QUALIDADE + 1) + 0.05, ax.get_ylim()[1] * 0.9,
        f"mínimo = {config.MIN_AVALIACOES_QUALIDADE}", color=g.TEXTO_SECUNDARIO)
ax.set_title("Figura 2 — Distribuição do total de avaliações por jogo (escala log)")
ax.set_xlabel("log10(total de avaliações + 1)")
ax.set_ylabel("Quantidade de jogos")
g.salvar(fig, "fig02_distribuicao_avaliacoes")
plt.show()'''),
md("### Faixas de taxa negativa (regra de negócio apresentada no seminário)"),
code('''taxa = np.where(total_av > 0, A["negative"].fillna(0) / total_av.replace(0, np.nan), np.nan)
faixas = pd.Series([p.classificar_taxa_negativa(t, n) for t, n in zip(np.nan_to_num(taxa, nan=1), total_av)])
ordem = ["Extremamente positivas", "Muito positivas", "Positivas",
         "Ligeiramente positivas / mistas", "Negativas (descartadas)", "Poucas avaliações"]
contagem_faixas = faixas.value_counts().reindex(ordem, fill_value=0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(contagem_faixas.index[::-1], contagem_faixas.values[::-1], color=g.AZUL, height=0.7)
g.rotular_barras(ax, horizontal=True)
ax.set_title("Figura 3 — Jogos por faixa de taxa negativa (base A)")
ax.set_xlabel("Quantidade de jogos")
g.eixo_br(ax, "x")
ax.grid(axis="y", visible=False)
g.salvar(fig, "fig03_faixas_taxa_negativa")
plt.show()
contagem_faixas'''),
md("## 3. Base B — interações de usuários"),
code('''gb = c.carregar_b_games()
users = c.carregar_b_users()
meta = c.carregar_b_metadata()
recs, resumo_recs = c.carregar_recomendacoes()

print(f"games.csv:          {gb.shape}")
print(f"users.csv:          {users.shape}")
print(f"games_metadata:     {meta.shape}")
print(f"recommendations:    {resumo_recs['linhas_base_completa']:,} linhas na base completa")
print(f"amostra ({config.PERCENTUAL_USUARIOS}% usuários): {len(recs):,} linhas | {c.memoria_mb(recs):,.0f} MB")
recs.head()'''),
code('''tab_b = pd.concat({"games.csv": c.resumo_colunas(gb), "recommendations.csv": c.resumo_colunas(recs)})
tab_b.to_csv(config.RESULTADOS / "tabela_colunas_base_b.csv")
tab_b'''),
md("""### Desbalanceamento da variável `is_recommended`
A grande maioria das avaliações é positiva. Isso orienta a decisão de modelagem: tratamos o problema como
**feedback implícito positivo** (o usuário recomendou o jogo = gostou), e não como classificação binária,
onde a classe negativa seria minoritária e pouco informativa para recomendar."""),
code('''pos = resumo_recs["positivas_base_completa"]
neg = resumo_recs["linhas_base_completa"] - pos
bal = pd.Series({"Recomendou (positiva)": pos, "Não recomendou (negativa)": neg})

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.barh(bal.index[::-1], bal.values[::-1], color=g.AZUL, height=0.6)
for i, v in enumerate(bal.values[::-1]):
    ax.text(v, i, f"  {g.br(v)} ({v / bal.sum():.1%})", va="center", color=g.TEXTO_SECUNDARIO)
ax.margins(x=0.35)
ax.set_title("Figura 4 — Balanceamento de is_recommended (base B completa)")
ax.set_xlabel("Quantidade de avaliações")
g.eixo_br(ax, "x")
ax.grid(axis="y", visible=False)
g.salvar(fig, "fig04_balanceamento_is_recommended")
plt.show()'''),
md("### Interações por usuário — a maioria avalia poucos jogos (cauda longa)"),
code('''por_usuario = recs[recs["is_recommended"]].groupby("user_id").size()
print(por_usuario.describe(percentiles=[.5, .75, .9, .99]))
print(f"Usuários com >= {config.MIN_INTERACOES_USUARIO} jogos recomendados: "
      f"{(por_usuario >= config.MIN_INTERACOES_USUARIO).mean():.1%}")

fig, ax = plt.subplots()
ax.hist(por_usuario.clip(upper=50), bins=np.arange(0.5, 51.5, 1), color=g.AZUL, edgecolor="white", linewidth=0.5)
ax.axvline(config.MIN_INTERACOES_USUARIO, color=g.TEXTO_SECUNDARIO, ls="--", lw=1)
ax.set_yscale("log")
ax.set_title("Figura 5 — Jogos recomendados por usuário (valores acima de 50 agrupados)")
ax.set_xlabel("Jogos recomendados pelo usuário")
ax.set_ylabel("Usuários (escala log)")
g.salvar(fig, "fig05_interacoes_por_usuario")
plt.show()'''),
md("### Concentração da popularidade — poucos jogos recebem a maior parte das interações"),
code('''por_jogo = recs[recs["is_recommended"]].groupby("app_id").size().sort_values(ascending=False)
x = np.arange(1, len(por_jogo) + 1) / len(por_jogo) * 100
y = por_jogo.cumsum().to_numpy() / por_jogo.sum() * 100
top10 = y[int(len(y) * 0.10) - 1]

fig, ax = plt.subplots()
ax.plot(x, y, color=g.AZUL)
ax.plot([0, 100], [0, 100], color=g.GRADE, lw=1, ls="--")
ax.scatter([10], [top10], color=g.AZUL, s=40, zorder=3, edgecolor="white", linewidth=1.5)
ax.annotate(f"10% dos jogos = {top10:.0f}% das interações", (10, top10), xytext=(18, top10 - 12),
            color=g.TEXTO, arrowprops=dict(arrowstyle="-", color=g.TEXTO_SECUNDARIO, lw=0.8))
ax.set_title("Figura 6 — Curva de concentração das interações por jogo")
ax.set_xlabel("% dos jogos (do mais para o menos popular)")
ax.set_ylabel("% acumulado das interações")
g.salvar(fig, "fig06_concentracao_popularidade")
plt.show()'''),
md("## 4. Junção das bases A e B"),
code('''cobertura = gb["app_id"].isin(A["app_id"]).mean()
tags_a = A["tags"].map(p.parse_lista).map(len) if "tags" in A else pd.Series([0])
tags_b = meta["tags"].map(p.parse_lista).map(len)
print(f"Jogos da base B encontrados na base A: {cobertura:.1%}")
print(f"Média de tags por jogo — base A: {tags_a.mean():.1f} | base B: {tags_b.mean():.1f}")'''),
code('''resumo_entendimento = {
    "base_a_linhas": int(len(A)),
    "base_a_colunas_usadas": int(A.shape[1]),
    **{f"base_a_{k}": v for k, v in problemas_a.items()},
    "base_b_jogos": int(len(gb)),
    "base_b_usuarios": int(len(users)),
    **{f"base_b_{k}": v for k, v in resumo_recs.items()},
    "cobertura_b_em_a": float(cobertura),
    "top10pct_jogos_share_interacoes": float(top10),
    "mediana_jogos_por_usuario_amostra": float(por_usuario.median()),
}
with open(config.RESULTADOS / "resumo_entendimento.json", "w", encoding="utf-8") as f:
    json.dump(resumo_entendimento, f, ensure_ascii=False, indent=2, default=float)
resumo_entendimento'''),
md("""## Principais achados desta etapa (preencher com os números acima para o artigo)
- **Qualidade dos dados (A):** duplicatas de `app_id`, nomes vazios e jogos sem gênero/tags → tratados na preparação.
- **Avaliações:** distribuição muito assimétrica; muitos jogos com poucas avaliações → a taxa negativa precisa de volume mínimo.
- **Desbalanceamento (B):** `is_recommended` majoritariamente positivo → modelagem como feedback implícito positivo.
- **Cauda longa:** poucos jogos concentram a maior parte das interações → baseline de popularidade é obrigatório e a cobertura do catálogo deve ser medida.
- **Junção:** a maior parte dos jogos de B existe em A → o enriquecimento com gêneros/categorias é viável.
"""),
]

# =============================================================================
# 02 — Preparação dos dados
# =============================================================================
nb02 = [
md("""
# 02 — Preparação dos Dados
**CRISP-DM — etapa 3.** Limpeza, tratamento de faltantes, transformação, junção das bases e divisão treino/teste.
Toda etapa registra quantas linhas sobraram (tabela "funil"), que vai para o artigo.
"""),
code(SETUP),
code('''funil = []
A = p.limpar_base_a(c.carregar_base_a(), funil)
b = p.limpar_b_games(c.carregar_b_games(), c.carregar_b_metadata(), funil)
catalogo = p.montar_catalogo(A, b, funil)
print(f"Catálogo unificado: {len(catalogo):,} jogos | com características: {catalogo['tem_features'].mean():.1%}")'''),
md("""### Transformação: texto de características para o TF-IDF
Gêneros + categorias + tags (união das tags de A e B) são normalizados (minúsculas, sem acento) e cada expressão
vira um token único (`Open World` → `open_world`), para o TF-IDF não separar palavras que só fazem sentido juntas."""),
code('''catalogo.sort_values("total_avaliacoes", ascending=False)[
    ["nome", "genres", "tags_unificadas", "texto_features"]].head(5)'''),
md("### Tratamento de faltantes"),
code('''faltantes_cat = pd.DataFrame({
    "sem gênero": catalogo["genres"].map(len).eq(0).mean(),
    "sem categoria": catalogo["categories"].map(len).eq(0).mean(),
    "sem tag": catalogo["tags_unificadas"].map(len).eq(0).mean(),
    "sem nenhuma característica (removido)": (~catalogo["tem_features"]).mean(),
    "sem avaliações": catalogo["total_avaliacoes"].eq(0).mean(),
    "fora da base A (avaliações vieram da base B)": (~catalogo["presente_na_base_a"]).mean(),
}, index=["% do catálogo"]).T
faltantes_cat.to_csv(config.RESULTADOS / "tabela_faltantes_catalogo.csv")
faltantes_cat.style.format("{:.1%}")'''),
md("""**Decisões:**
- jogo **sem nenhuma característica** → removido (não há como compará-lo por conteúdo);
- jogo **sem gênero/categoria, mas com tags** → mantido (as tags carregam a informação);
- jogo **fora da base A** → avaliações reconstruídas de `positive_ratio × user_reviews` da base B;
- **avaliações faltantes** → contadas como zero, e o filtro de qualidade exige volume mínimo."""),
md("### Filtro de qualidade (proposta do seminário + volume mínimo)"),
code('''ordem = ["Extremamente positivas", "Muito positivas", "Positivas",
         "Ligeiramente positivas / mistas", "Negativas (descartadas)", "Poucas avaliações"]
dist_q = catalogo["classificacao_qualidade"].value_counts().reindex(ordem, fill_value=0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(dist_q.index[::-1], dist_q.values[::-1], color=g.AZUL, height=0.7)
g.rotular_barras(ax, horizontal=True)
ax.set_title("Figura 7 — Catálogo por classificação de qualidade")
ax.set_xlabel("Quantidade de jogos")
g.eixo_br(ax, "x")
ax.grid(axis="y", visible=False)
g.salvar(fig, "fig07_catalogo_qualidade")
plt.show()
print(f"Passam no filtro (>= {config.MIN_AVALIACOES_QUALIDADE} avaliações e taxa negativa <= "
      f"{config.TAXA_NEGATIVA_DESCARTE:.0%}): {catalogo['passa_filtro_qualidade'].mean():.1%}")'''),
md("""### Por que o limite inferior de Wilson?
A taxa negativa simples trata 3 avaliações positivas em 3 igual a 30 000 em 30 000. O limite inferior do intervalo
de Wilson (95%) corrige isso: com poucas avaliações a incerteza é grande e o valor cai."""),
code('''exemplo = pd.DataFrame({"positivas": [3, 30, 300, 30000], "total": [3, 33, 330, 33000]})
exemplo["taxa_positiva"] = exemplo["positivas"] / exemplo["total"]
exemplo["wilson_inferior"] = p.limite_inferior_wilson(exemplo["positivas"], exemplo["total"])
exemplo'''),
md("### Interações: limpeza, desbalanceamento e filtros mínimos"),
code('''recs, _ = c.carregar_recomendacoes()
interacoes = p.preparar_interacoes(recs, catalogo, funil)
del recs
print(f"Interações finais: {len(interacoes):,} | usuários: {interacoes['user_id'].nunique():,} | "
      f"jogos: {interacoes['app_id'].nunique():,}")'''),
md("""**Tratamento do desbalanceamento:** como as avaliações negativas são minoria e o objetivo é recomendar o que
a pessoa vai gostar, usamos apenas as interações positivas como sinal (feedback implícito). O viés de popularidade
(cauda longa) é tratado na avaliação: comparamos com um baseline de popularidade e medimos a cobertura do catálogo."""),
md("### Funil da preparação (tabela para o artigo)"),
code('''tabela_funil = p.funil_para_df(funil)
tabela_funil.to_csv(config.ARQ_FUNIL, index=False)
tabela_funil'''),
md("""### Divisão temporal treino/teste
Para cada usuário, os **3 jogos mais recentes** ficam escondidos (teste) e o restante é o histórico (treino).
O perfil usado para recomendar são os últimos 1 ou 5 jogos do histórico. Assim, simulamos a situação real:
"o que essa pessoa vai jogar depois?" — sem usar informação do futuro."""),
code('''treino, teste = p.divisao_temporal(interacoes)
catalogo_modelo = catalogo[catalogo["app_id"].isin(interacoes["app_id"])].reset_index(drop=True)

print(f"Treino: {len(treino):,} interações | Teste: {len(teste):,} interações")
print(f"Jogos no catálogo do modelo: {len(catalogo_modelo):,}")
print(f"Checagem de vazamento — data máxima do treino <= data mínima do teste por usuário: "
      f"{(treino.groupby('user_id')['date'].max() <= teste.groupby('user_id')['date'].min()).mean():.1%}")'''),
code('''catalogo_modelo.to_parquet(config.ARQ_CATALOGO, index=False)
interacoes.to_parquet(config.ARQ_INTERACOES, index=False)
treino.to_parquet(config.ARQ_TREINO, index=False)
teste.to_parquet(config.ARQ_TESTE, index=False)
print("Arquivos salvos em", config.DADOS_PROCESSADOS)'''),
]

# =============================================================================
# 03 — Modelagem e avaliação
# =============================================================================
nb03 = [
md("""
# 03 — Modelagem e Avaliação
**CRISP-DM — etapas 4 e 5.** Treinamos 5 modelos com a mesma interface (perfil → ranking de jogos), ajustamos
hiperparâmetros num conjunto de **validação** e reportamos o resultado num conjunto de **teste** separado.

| Modelo | Tipo | Por que entra |
|---|---|---|
| Popularidade | baseline | qualquer modelo precisa superar "recomendar o mais popular" |
| Conteúdo TF-IDF (média / máximo) | baseado em conteúdo | proposta original do seminário |
| Colaborativo item-item | filtragem colaborativa | usa o comportamento real dos usuários (base B) |
| Fatoração SVD | fatoração de matriz | captura padrões latentes de gosto |
| Híbrido | conteúdo + colaborativo | une as duas fontes de informação |
"""),
code(SETUP + '''

import time
import joblib
from src import modelos as m, avaliacao as av'''),
code('''catalogo = pd.read_parquet(config.ARQ_CATALOGO)
treino = pd.read_parquet(config.ARQ_TREINO)
teste = pd.read_parquet(config.ARQ_TESTE)

indice = m.IndiceCatalogo(catalogo)
matriz_ui = m.matriz_usuario_item(treino, indice)
casos = av.montar_casos(treino, teste, indice, n_usuarios=config.N_USUARIOS_AVALIACAO)
validacao, casos_teste = av.separar_validacao(casos)

densidade = matriz_ui.nnz / (matriz_ui.shape[0] * matriz_ui.shape[1])
print(f"Matriz usuário x jogo: {matriz_ui.shape} | densidade {densidade:.4%}")
print(f"Usuários avaliados: {len(casos):,} (validação {len(validacao):,} | teste {len(casos_teste):,})")'''),
md("## Treinamento"),
code('''tempos = {}
def treinar(nome, fabrica):
    t = time.time()
    modelo = fabrica()
    tempos[nome] = time.time() - t
    print(f"{nome:<32} treinado em {tempos[nome]:.1f}s")
    return modelo

pop = treinar("Popularidade", lambda: m.Popularidade(indice, treino))
tfidf_media = treinar("Conteúdo TF-IDF (média)", lambda: m.ConteudoTFIDF(indice, "media"))
tfidf_max = treinar("Conteúdo TF-IDF (máximo)", lambda: m.ConteudoTFIDF(indice, "maximo"))
item_item = treinar("Colaborativo item-item", lambda: m.ColaborativoItemItem(indice, matriz_ui))
print(f"Vocabulário TF-IDF: {len(tfidf_media.vetorizador.vocabulary_):,} tokens")'''),
md("## Ajuste de hiperparâmetros na validação (perfil de 5 jogos)"),
md("### SVD — número de dimensões latentes"),
code('''grade_svd = []
for d in [16, 32, 64, 128]:
    if d >= min(matriz_ui.shape):
        continue
    modelo = m.FatoracaoSVD(indice, matriz_ui, n_componentes=d)
    _, r = av.avaliar(modelo, validacao, tamanho_perfil=5)
    grade_svd.append({"dimensoes": d, "ndcg@10": r["ndcg@10"]})
grade_svd = pd.DataFrame(grade_svd)
melhor_d = int(grade_svd.loc[grade_svd["ndcg@10"].idxmax(), "dimensoes"])
svd = treinar("Fatoração SVD", lambda: m.FatoracaoSVD(indice, matriz_ui, n_componentes=melhor_d))
grade_svd'''),
md("### Híbrido — peso α do conteúdo (α = 0 → só colaborativo; α = 1 → só conteúdo)"),
code('''grade_alpha = []
for alpha in np.round(np.arange(0, 1.01, 0.1), 1):
    h = m.Hibrido(indice, tfidf_media, item_item, peso_conteudo=alpha)
    _, r = av.avaliar(h, validacao, tamanho_perfil=5)
    grade_alpha.append({"alpha": alpha, "ndcg@10": r["ndcg@10"], "hit_rate@10": r["hit_rate@10"]})
grade_alpha = pd.DataFrame(grade_alpha)
melhor_alpha = float(grade_alpha.loc[grade_alpha["ndcg@10"].idxmax(), "alpha"])
hibrido = m.Hibrido(indice, tfidf_media, item_item, peso_conteudo=melhor_alpha)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(grade_alpha["alpha"], grade_alpha["ndcg@10"], color=g.AZUL, marker="o", markersize=6,
        markeredgecolor="white", markeredgewidth=1.5)
ax.scatter([melhor_alpha], [grade_alpha["ndcg@10"].max()], s=90, facecolor="none", edgecolor=g.TEXTO, zorder=4)
ax.annotate(f"melhor α = {melhor_alpha:.1f}", (melhor_alpha, grade_alpha["ndcg@10"].max()),
            xytext=(10, -18), textcoords="offset points", color=g.TEXTO)
ax.set_title("Figura 8 — Ajuste do peso do conteúdo no híbrido (validação)")
ax.set_xlabel("α (peso do conteúdo)")
ax.set_ylabel("NDCG@10")
g.salvar(fig, "fig08_ajuste_alpha_hibrido")
plt.show()
grade_alpha'''),
md("## Avaliação final no conjunto de teste — 1 jogo × 5 jogos"),
code('''modelos = [pop, tfidf_media, tfidf_max, item_item, svd, hibrido]
resultados, por_usuario = [], {}
for modelo in modelos:
    for tam in (1, 5):
        u, r = av.avaliar(modelo, casos_teste, tamanho_perfil=tam, popularidade=pop.scores)
        r["ndcg_ic95"] = av.ic_bootstrap(u["ndcg"])
        resultados.append(r)
        por_usuario[(modelo.nome, tam)] = u

tabela = pd.DataFrame(resultados)
tabela.to_csv(config.RESULTADOS / "resultados_teste.csv", index=False)
tabela.drop(columns=["filtro_qualidade", "usuarios"]).set_index(["modelo", "perfil"])'''),
code('''t = tabela.copy()
t["ic_inf"] = t["ndcg_ic95"].map(lambda x: x[0])
t["ic_sup"] = t["ndcg_ic95"].map(lambda x: x[1])
nomes = [mdl.nome for mdl in modelos]
y = np.arange(len(nomes))
altura = 0.38

fig, ax = plt.subplots(figsize=(9, 5))
for i, (perfil, cor) in enumerate([("1 jogo", g.AZUL), ("5 jogos", g.LARANJA)]):
    d = t[t["perfil"] == perfil].set_index("modelo").loc[nomes]
    pos = y + (i - 0.5) * altura
    ax.barh(pos, d["ndcg@10"], height=altura - 0.04, color=cor, label=f"Perfil com {perfil}")
    ax.errorbar(d["ndcg@10"], pos, xerr=[d["ndcg@10"] - d["ic_inf"], d["ic_sup"] - d["ndcg@10"]],
                fmt="none", ecolor=g.TEXTO_SECUNDARIO, elinewidth=1, capsize=2)
    for v, sup, yy in zip(d["ndcg@10"], d["ic_sup"], pos):
        ax.text(sup, yy, "  " + g.br(v, "{:.3f}"), va="center", fontsize=8.5, color=g.TEXTO_SECUNDARIO)
ax.set_yticks(y, nomes)
ax.invert_yaxis()
ax.margins(x=0.18)
ax.set_xlabel("NDCG@10 (barras de erro: IC 95% por bootstrap)")
g.eixo_br(ax, "x", "{:.2f}")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.set_title("Figura 9 — Desempenho no teste: perfil de 1 jogo × 5 jogos", pad=28)
ax.grid(axis="y", visible=False)
g.salvar(fig, "fig09_ndcg_1x5_modelos")
plt.show()'''),
md("### Teste estatístico pareado (Wilcoxon) — a diferença entre 1 e 5 jogos é significativa?"),
code('''testes = []
for modelo in modelos:
    if modelo is pop:
        continue  # popularidade ignora o perfil: 1 e 5 jogos são idênticos
    a = por_usuario[(modelo.nome, 1)]["ndcg"]
    b_ = por_usuario[(modelo.nome, 5)]["ndcg"]
    testes.append({"modelo": modelo.nome, "ndcg_1_jogo": a.mean(), "ndcg_5_jogos": b_.mean(),
                   **av.comparar_pareado(a, b_)})
testes = pd.DataFrame(testes)
testes["significativo_5%"] = testes["p_valor"] < 0.05
testes.to_csv(config.RESULTADOS / "teste_wilcoxon_1x5.csv", index=False)
testes'''),
md("### Quantos jogos recentes usar? (curva de 1 a 5 jogos no perfil)"),
code('''curva = []
for modelo in [tfidf_media, item_item, hibrido]:
    for tam in range(1, 6):
        _, r = av.avaliar(modelo, casos_teste, tamanho_perfil=tam)
        curva.append({"modelo": modelo.nome, "jogos_no_perfil": tam, "ndcg@10": r["ndcg@10"]})
curva = pd.DataFrame(curva)
curva.to_csv(config.RESULTADOS / "curva_tamanho_perfil.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.2))
for cor, (nome, d) in zip(g.CORES, curva.groupby("modelo", sort=False)):
    ax.plot(d["jogos_no_perfil"], d["ndcg@10"], color=cor, marker="o", markersize=6,
            markeredgecolor="white", markeredgewidth=1.5, label=nome)
ax.set_xticks(range(1, 6))
ax.set_xlim(0.8, 5.2)
ax.set_xlabel("Jogos recentes usados no perfil")
ax.set_ylabel("NDCG@10")
ax.set_title("Figura 10 — Efeito do tamanho do perfil no desempenho (teste)")
ax.legend(loc="best")
g.salvar(fig, "fig10_curva_tamanho_perfil")
plt.show()'''),
md("### Filtro de qualidade: ajuda ou atrapalha?"),
code('''abl = []
for modelo in [tfidf_media, item_item, hibrido]:
    for filtro in (False, True):
        _, r = av.avaliar(modelo, casos_teste, tamanho_perfil=5, filtro_qualidade=filtro, popularidade=pop.scores)
        abl.append(r)
ablacao = pd.DataFrame(abl)[["modelo", "filtro_qualidade", "precision@10", "recall@10",
                             "hit_rate@10", "ndcg@10", "cobertura_catalogo", "popularidade_media_recs"]]
ablacao.to_csv(config.RESULTADOS / "ablacao_filtro_qualidade.csv", index=False)
ablacao'''),
md("""## Avaliação qualitativa (proposta do seminário: notas 0, 1 e 2)
Perfis montados à mão com jogos conhecidos. O CSV exportado tem uma coluna `nota_relevancia` em branco para o grupo
preencher (2 = relevante, 1 = parcialmente relevante, 0 = irrelevante). Jogos que não existirem no catálogo são ignorados."""),
code('''perfis_teste = {
    "Seminário (FPS + RPG + Souls)": ["Call of Duty", "Cyberpunk 2077", "ELDEN RING", "Counter-Strike", "Red Dead Redemption 2"],
    "Indie / relaxante": ["Stardew Valley", "Terraria", "Hades", "Hollow Knight", "Celeste"],
    "Estratégia": ["Civilization VI", "Total War", "Crusader Kings", "Stellaris", "Age of Empires"],
}

def resolver(nomes):
    ids = []
    for nome in nomes:
        achados = indice.buscar_por_nome(nome)
        if len(achados):
            ids.append(int(achados.iloc[0]["app_id"]))
        else:
            print(f"  aviso: '{nome}' não encontrado no catálogo")
    return ids

linhas = []
for perfil_nome, nomes_jogos in perfis_teste.items():
    ids = resolver(nomes_jogos)
    if not ids:
        continue
    for tam in (1, 5):
        entrada = ids[-tam:]
        recs_q = hibrido.recomendar(entrada, k=10, filtro_qualidade=True)
        for pos_, linha in enumerate(recs_q.itertuples(), 1):
            linhas.append({"perfil": perfil_nome, "jogos_no_perfil": tam,
                           "entrada": " | ".join(indice.catalogo.set_index("app_id").loc[entrada, "nome"]),
                           "posicao": pos_, "recomendado": linha.nome, "score": round(linha.score, 4),
                           "nota_relevancia": ""})
qualitativa = pd.DataFrame(linhas)
qualitativa.to_csv(config.RESULTADOS / "avaliacao_qualitativa_para_preencher.csv", index=False)
qualitativa.head(20)'''),
md("## Salvando o modelo final para a API (CRISP-DM — etapa 6, Implantação)"),
code('''joblib.dump({"indice": indice, "modelo": hibrido, "alpha": melhor_alpha, "svd_dimensoes": melhor_d},
            config.MODELOS / "recomendador.joblib", compress=3)
resumo_modelagem = {"melhor_alpha": melhor_alpha, "svd_dimensoes": melhor_d,
                    "tempos_treino_s": tempos, "densidade_matriz": densidade,
                    "usuarios_validacao": len(validacao), "usuarios_teste": len(casos_teste)}
with open(config.RESULTADOS / "resumo_modelagem.json", "w", encoding="utf-8") as f:
    json.dump(resumo_modelagem, f, ensure_ascii=False, indent=2, default=float)
resumo_modelagem'''),
]

if __name__ == "__main__":
    salvar("01_entendimento_dos_dados.ipynb", nb01)
    salvar("02_preparacao_dos_dados.ipynb", nb02)
    salvar("03_modelagem_e_avaliacao.ipynb", nb03)
