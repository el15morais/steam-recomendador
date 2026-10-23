"""
Preparação dos dados (CRISP-DM — etapa 3).

- limpeza da base A (metadados) e da base B (jogos + interações);
- junção das duas bases pelo app_id;
- filtro de qualidade (taxa negativa, proposta do seminário) com volume mínimo;
- construção do texto de características para o TF-IDF;
- filtragem de usuários/jogos e divisão temporal treino/teste.

Cada função de limpeza devolve também um "funil" (quantas linhas sobraram em
cada passo), que vira tabela no artigo.
"""
from __future__ import annotations

import ast
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Utilitários de texto
# ---------------------------------------------------------------------------
def parse_lista(valor) -> list[str]:
    """
    Converte uma célula em lista de strings, aceitando os formatos que aparecem
    em datasets da Steam:
      "['Action', 'Indie']"   (literal Python)
      '["Action", "Indie"]'   (JSON)
      "{'FPS': 1200, ...}"    (dicionário tag -> votos; ficamos com as chaves)
      "Action,Indie"          (separado por vírgula / ; / |)
      ['Action', 'Indie']     (lista já pronta)
    """
    if isinstance(valor, (list, tuple, np.ndarray)):
        return [str(v).strip() for v in valor if str(v).strip()]
    if isinstance(valor, dict):
        return [str(k).strip() for k in valor if str(k).strip()]
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return []

    texto = str(valor).strip()
    if not texto or texto in {"[]", "{}", "nan", "None"}:
        return []

    if texto[0] in "[{":
        for leitor in (json.loads, ast.literal_eval):
            try:
                return parse_lista(leitor(texto))
            except (ValueError, SyntaxError, TypeError):
                continue

    partes = re.split(r"[,;|]", texto)
    return [p.strip().strip("'\"") for p in partes if p.strip().strip("'\"")]


def normalizar_token(texto: str) -> str:
    """'Open World' -> 'open_world'; 'Ação' -> 'acao'. Mantém expressões juntas."""
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    texto = re.sub(r"[^a-z0-9]+", "_", texto.lower()).strip("_")
    return texto


def tokens(lista: list[str]) -> list[str]:
    vistos, saida = set(), []
    for item in lista:
        tok = normalizar_token(item)
        if tok and tok not in vistos:
            vistos.add(tok)
            saida.append(tok)
    return saida


def _registrar(funil: list, base: str, etapa: str, n: int) -> None:
    funil.append({"base": base, "etapa": etapa, "registros": int(n)})


# ---------------------------------------------------------------------------
# Base A — metadados
# ---------------------------------------------------------------------------
def limpar_base_a(df: pd.DataFrame, funil: list | None = None) -> pd.DataFrame:
    funil = funil if funil is not None else []
    _registrar(funil, "A (metadados)", "bruto", len(df))

    df = df.dropna(subset=["app_id"]).copy()
    df["app_id"] = df["app_id"].astype("int64")
    _registrar(funil, "A (metadados)", "sem app_id nulo", len(df))

    df = df.drop_duplicates(subset="app_id", keep="last")
    _registrar(funil, "A (metadados)", "sem app_id duplicado", len(df))

    df = df[df["name"].notna() & (df["name"].astype(str).str.strip() != "")]
    _registrar(funil, "A (metadados)", "com nome preenchido", len(df))

    for col in ("genres", "categories", "tags", "developers", "publishers"):
        if col in df.columns:
            df[col] = df[col].map(parse_lista)
        else:
            df[col] = [[] for _ in range(len(df))]

    for col in ("positive", "negative", "recommendations", "average_playtime_forever"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Base B — jogos
# ---------------------------------------------------------------------------
def limpar_b_games(games: pd.DataFrame, metadata: pd.DataFrame, funil: list | None = None) -> pd.DataFrame:
    funil = funil if funil is not None else []
    _registrar(funil, "B (jogos)", "bruto", len(games))

    games = games.drop_duplicates(subset="app_id", keep="last").copy()
    _registrar(funil, "B (jogos)", "sem app_id duplicado", len(games))

    meta = metadata.drop_duplicates(subset="app_id", keep="last")[["app_id", "tags"] +
                                                                  (["description"] if "description" in metadata else [])]
    meta = meta.rename(columns={"tags": "tags_b", "description": "descricao_b"})
    meta["tags_b"] = meta["tags_b"].map(parse_lista)

    games = games.merge(meta, on="app_id", how="left")
    games["tags_b"] = games["tags_b"].map(lambda v: v if isinstance(v, list) else [])
    return games.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Qualidade (proposta do seminário + correção por volume)
# ---------------------------------------------------------------------------
def limite_inferior_wilson(positivas, total, z: float = 1.96):
    """
    Limite inferior do intervalo de Wilson para a proporção de avaliações
    positivas. Penaliza jogos com poucas avaliações: 3/3 positivas não vale o
    mesmo que 30 000/30 000.
    """
    positivas = np.asarray(positivas, dtype=float)
    total = np.asarray(total, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 0, positivas / total, 0.0)
        denom = 1 + z**2 / total
        centro = p + z**2 / (2 * total)
        margem = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
        lb = np.where(total > 0, (centro - margem) / denom, 0.0)
    return np.clip(lb, 0, 1)


def classificar_taxa_negativa(taxa: float, total: float) -> str:
    """Faixas apresentadas no seminário (equivalentes à classificação da Steam)."""
    if total < config.MIN_AVALIACOES_QUALIDADE:
        return "Poucas avaliações"
    if taxa <= 0.05:
        return "Extremamente positivas"
    if taxa <= 0.10:
        return "Muito positivas"
    if taxa <= 0.20:
        return "Positivas"
    if taxa <= 0.30:
        return "Ligeiramente positivas / mistas"
    return "Negativas (descartadas)"


# ---------------------------------------------------------------------------
# Catálogo unificado (junção A + B)
# ---------------------------------------------------------------------------
def montar_catalogo(base_a: pd.DataFrame, b_games: pd.DataFrame, funil: list | None = None) -> pd.DataFrame:
    """
    O catálogo parte dos jogos da base B (são os que têm interações) e é
    enriquecido com gêneros, categorias e tags da base A.
    """
    funil = funil if funil is not None else []

    colunas_a = ["app_id", "name", "genres", "categories", "tags", "developers",
                 "positive", "negative", "price", "release_date"]
    colunas_a = [c for c in colunas_a if c in base_a.columns]
    a = base_a[colunas_a].rename(columns={"price": "price_a", "release_date": "release_date_a"})

    cat = b_games.merge(a, on="app_id", how="left", indicator=True)
    cat["presente_na_base_a"] = cat["_merge"].eq("both")
    cat = cat.drop(columns="_merge")
    _registrar(funil, "Catálogo (A+B)", "jogos da base B", len(cat))
    _registrar(funil, "Catálogo (A+B)", "jogos também presentes na base A", int(cat["presente_na_base_a"].sum()))

    for col in ("genres", "categories", "tags", "developers"):
        if col not in cat.columns:
            cat[col] = None
        cat[col] = cat[col].map(lambda v: v if isinstance(v, list) else [])

    cat["nome"] = cat["title"].fillna(cat.get("name"))

    # Tags: união da base A com a base B (a B costuma ter mais tags por jogo)
    cat["tags_unificadas"] = [list(dict.fromkeys(ta + tb)) for ta, tb in zip(cat["tags"], cat["tags_b"])]

    # Texto de características para o TF-IDF: gêneros + categorias + tags.
    # Cada expressão vira um token só ("open_world"), para o TF-IDF não
    # quebrar "Open World" em "open" e "world".
    cat["texto_features"] = [
        " ".join(tokens(g + c + t))
        for g, c, t in zip(cat["genres"], cat["categories"], cat["tags_unificadas"])
    ]
    cat["tem_features"] = cat["texto_features"].str.len() > 0

    # Avaliações: base A quando existir; senão, reconstruídas da base B
    pos_b = (cat["user_reviews"] * cat["positive_ratio"] / 100).round()
    neg_b = cat["user_reviews"] - pos_b
    usa_a = cat["presente_na_base_a"] & ((cat["positive"].fillna(0) + cat["negative"].fillna(0)) > 0)
    cat["avaliacoes_positivas"] = np.where(usa_a, cat["positive"], pos_b).astype(float)
    cat["avaliacoes_negativas"] = np.where(usa_a, cat["negative"], neg_b).astype(float)
    cat["total_avaliacoes"] = cat["avaliacoes_positivas"] + cat["avaliacoes_negativas"]

    with np.errstate(divide="ignore", invalid="ignore"):
        cat["taxa_negativa"] = np.where(cat["total_avaliacoes"] > 0,
                                        cat["avaliacoes_negativas"] / cat["total_avaliacoes"], np.nan)
    cat["wilson_inferior"] = limite_inferior_wilson(cat["avaliacoes_positivas"], cat["total_avaliacoes"])
    cat["classificacao_qualidade"] = [classificar_taxa_negativa(t, n)
                                      for t, n in zip(cat["taxa_negativa"].fillna(1), cat["total_avaliacoes"])]
    cat["passa_filtro_qualidade"] = (
        (cat["total_avaliacoes"] >= config.MIN_AVALIACOES_QUALIDADE)
        & (cat["taxa_negativa"] <= config.TAXA_NEGATIVA_DESCARTE)
    )

    colunas_finais = [
        "app_id", "nome", "date_release", "price_final", "genres", "categories",
        "tags_unificadas", "texto_features", "tem_features", "presente_na_base_a",
        "avaliacoes_positivas", "avaliacoes_negativas", "total_avaliacoes",
        "taxa_negativa", "wilson_inferior", "classificacao_qualidade", "passa_filtro_qualidade",
    ]
    colunas_finais = [c for c in colunas_finais if c in cat.columns]
    return cat[colunas_finais].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Interações
# ---------------------------------------------------------------------------
def preparar_interacoes(recs: pd.DataFrame, catalogo: pd.DataFrame,
                        funil: list | None = None) -> pd.DataFrame:
    """
    Mantém só recomendações positivas (o usuário gostou do jogo), remove
    duplicatas e aplica os filtros mínimos de usuário e de jogo.
    """
    funil = funil if funil is not None else []
    base = "B (interações)"
    _registrar(funil, base, "amostra de usuários", len(recs))

    df = recs.dropna(subset=["date", "user_id", "app_id"])
    _registrar(funil, base, "sem data/usuário/jogo nulo", len(df))

    df = df[df["is_recommended"]]
    _registrar(funil, base, "só recomendações positivas", len(df))

    ids_catalogo = set(catalogo.loc[catalogo["tem_features"], "app_id"])
    df = df[df["app_id"].isin(ids_catalogo)]
    _registrar(funil, base, "jogo no catálogo com características", len(df))

    df = (df.sort_values(["user_id", "date"])
            .drop_duplicates(subset=["user_id", "app_id"], keep="last"))
    _registrar(funil, base, "sem par usuário-jogo duplicado", len(df))

    # Filtros iterativos: remover jogos raros pode deixar usuários abaixo do
    # mínimo e vice-versa; repetimos até estabilizar.
    for _ in range(10):
        antes = len(df)
        cont_jogo = df["app_id"].map(df["app_id"].value_counts())
        df = df[cont_jogo >= config.MIN_INTERACOES_JOGO]
        cont_user = df["user_id"].map(df["user_id"].value_counts())
        df = df[cont_user >= config.MIN_INTERACOES_USUARIO]
        if len(df) == antes:
            break
    _registrar(funil, base,
               f"usuários >= {config.MIN_INTERACOES_USUARIO} e jogos >= {config.MIN_INTERACOES_JOGO} interações",
               len(df))

    return df[["user_id", "app_id", "date", "hours"]].reset_index(drop=True)


def divisao_temporal(interacoes: pd.DataFrame, n_teste: int = config.N_ITENS_TESTE):
    """
    Para cada usuário, os n_teste jogos MAIS RECENTES vão para o teste
    (escondidos) e o restante fica no treino. Isso simula o uso real:
    olhar o que a pessoa jogou até agora e prever o que ela joga depois.
    """
    df = interacoes.sort_values(["user_id", "date", "app_id"], kind="mergesort").copy()
    df["ordem_reversa"] = df.groupby("user_id").cumcount(ascending=False)
    teste = df[df["ordem_reversa"] < n_teste].drop(columns="ordem_reversa")
    treino = df[df["ordem_reversa"] >= n_teste].drop(columns="ordem_reversa")
    return treino.reset_index(drop=True), teste.reset_index(drop=True)


def funil_para_df(funil: list) -> pd.DataFrame:
    df = pd.DataFrame(funil)
    df["removidos"] = df.groupby("base")["registros"].diff().fillna(0).astype(int) * -1
    return df
