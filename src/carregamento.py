"""
Carregamento das bases brutas.

Base A (metadados) — Steam Games Dataset, Hubert Sidorowicz (Kaggle).
Base B (interações) — Game Recommendations on Steam, Anton Kozyriev (Kaggle).

As funções leem só as colunas necessárias e usam tipos enxutos, porque os
arquivos são grandes (steam_games.csv ~1 GB; recommendations.csv ~41 mi linhas).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Colunas da base A que o projeto usa. Lemos o cabeçalho antes e pegamos só as
# que existirem — assim uma mudança no dataset (ele é atualizado diariamente)
# não quebra o código.
COLUNAS_A = [
    "app_id", "name", "release_date", "price",
    "positive", "negative", "recommendations",
    "genres", "categories", "tags",
    "developers", "publishers",
    "estimated_owners", "average_playtime_forever",
]

# Nomes alternativos já vistos em versões de datasets da Steam
SINONIMOS_A = {"appid": "app_id", "AppID": "app_id", "Name": "name"}


def _colunas_disponiveis(caminho: Path) -> list[str]:
    return pd.read_csv(caminho, nrows=0).columns.tolist()


def carregar_base_a(caminho: Path = config.ARQUIVO_BASE_A) -> pd.DataFrame:
    """Lê a base A (metadados dos jogos) apenas com as colunas úteis."""
    cabecalho = _colunas_disponiveis(caminho)
    renomear = {c: SINONIMOS_A[c] for c in cabecalho if c in SINONIMOS_A}
    desejadas = set(COLUNAS_A)
    usar = [c for c in cabecalho if c in desejadas or c in renomear]

    faltando = desejadas - {renomear.get(c, c) for c in usar}
    if faltando:
        print(f"[base A] colunas não encontradas (seguem sem elas): {sorted(faltando)}")

    df = pd.read_csv(caminho, usecols=usar, low_memory=False).rename(columns=renomear)
    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce")
    return df


def carregar_b_games(caminho: Path = config.ARQUIVO_B_GAMES) -> pd.DataFrame:
    """games.csv da base B: título, nota agregada, preço, plataformas."""
    return pd.read_csv(caminho, parse_dates=["date_release"])


def carregar_b_users(caminho: Path = config.ARQUIVO_B_USERS) -> pd.DataFrame:
    return pd.read_csv(caminho, dtype={"user_id": "int32", "products": "int32", "reviews": "int32"})


def carregar_b_metadata(caminho: Path = config.ARQUIVO_B_METADATA) -> pd.DataFrame:
    """games_metadata.json (um JSON por linha): app_id, description, tags."""
    registros = []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                registros.append(json.loads(linha))
    df = pd.DataFrame(registros)
    if "tags" not in df.columns:
        df["tags"] = [[] for _ in range(len(df))]
    return df


def carregar_recomendacoes(
    caminho: Path = config.ARQUIVO_B_RECS,
    percentual_usuarios: int = config.PERCENTUAL_USUARIOS,
    tamanho_bloco: int = config.TAMANHO_BLOCO,
    retornar_resumo: bool = True,
):
    """
    Lê recommendations.csv em blocos, aplicando a amostragem de usuários
    (user_id % 100 < percentual) durante a leitura para economizar memória.

    Retorna (interacoes, resumo), onde `resumo` guarda contagens da base
    INTEIRA (antes da amostra) — usadas na análise de desbalanceamento.
    """
    tipos = {"app_id": "int32", "user_id": "int32", "is_recommended": "bool", "hours": "float32"}
    colunas = ["app_id", "date", "is_recommended", "hours", "user_id"]

    blocos = []
    total_linhas = 0
    total_positivas = 0
    for bloco in pd.read_csv(caminho, usecols=colunas, dtype=tipos, chunksize=tamanho_bloco):
        total_linhas += len(bloco)
        total_positivas += int(bloco["is_recommended"].sum())
        if percentual_usuarios < 100:
            bloco = bloco[(bloco["user_id"] % 100) < percentual_usuarios]
        bloco["date"] = pd.to_datetime(bloco["date"], errors="coerce")
        blocos.append(bloco)

    df = pd.concat(blocos, ignore_index=True)
    resumo = {
        "linhas_base_completa": total_linhas,
        "positivas_base_completa": total_positivas,
        "proporcao_positivas_base_completa": total_positivas / max(total_linhas, 1),
        "percentual_usuarios_amostrados": percentual_usuarios,
        "linhas_amostra": len(df),
    }
    return (df, resumo) if retornar_resumo else df


def memoria_mb(df: pd.DataFrame) -> float:
    """Memória ocupada por um DataFrame, em MB (para documentar no artigo)."""
    return float(df.memory_usage(deep=True).sum() / 1e6)


def resumo_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela de tipos, faltantes e cardinalidade — vai direto para o artigo."""
    def _unicos(serie: pd.Series):
        if serie.dtype == object:
            exemplo = serie.dropna().head(1000)
            if not exemplo.map(np.isscalar).all():   # colunas com listas
                return np.nan
        return serie.nunique(dropna=True)

    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "nao_nulos": df.notna().sum(),
        "faltantes": df.isna().sum(),
        "faltantes_%": (df.isna().mean() * 100).round(2),
        "valores_unicos": [_unicos(df[c]) for c in df.columns],
    })
