"""
Avaliação offline (CRISP-DM — etapa 5).

Protocolo:
  - para cada usuário de teste, o histórico de treino está em ordem de data;
  - o PERFIL é formado pelos últimos 1 ou 5 jogos do treino (perfil recente);
  - os N_ITENS_TESTE jogos seguintes (escondidos) são o gabarito;
  - jogos já vistos no treino não podem ser recomendados.

Métricas @K (K = 10):
  Precision@K — fração da lista que acertou;
  Recall@K    — fração do gabarito encontrada;
  HitRate@K   — acertou pelo menos um?
  NDCG@K      — premia acertos nas primeiras posições.
Métricas de lista:
  Cobertura   — % do catálogo que aparece em alguma recomendação;
  Popularidade média — quão "óbvias" são as recomendações.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .modelos import IndiceCatalogo, Recomendador, mascara_qualidade, top_k


def montar_casos(treino: pd.DataFrame, teste: pd.DataFrame, indice: IndiceCatalogo,
                 n_usuarios: int | None = None, semente: int = config.SEMENTE) -> pd.DataFrame:
    """Um caso por usuário: histórico de treino ordenado e gabarito de teste."""
    tr = treino.sort_values(["user_id", "date", "app_id"], kind="mergesort")
    historico = tr.groupby("user_id")["app_id"].apply(lambda s: indice.para_posicoes(s.tolist()))
    alvo = teste.groupby("user_id")["app_id"].apply(lambda s: indice.para_posicoes(s.tolist()))

    casos = pd.DataFrame({"historico": historico, "alvo": alvo}).dropna()
    casos = casos[casos["historico"].map(len) >= config.TAMANHO_PERFIL]
    casos = casos[casos["alvo"].map(len) > 0]
    if n_usuarios is not None and len(casos) > n_usuarios:
        casos = casos.sample(n=n_usuarios, random_state=semente)
    return casos.reset_index()


def separar_validacao(casos: pd.DataFrame, fracao: float = config.FRACAO_VALIDACAO,
                      semente: int = config.SEMENTE):
    """Usuários de validação (ajuste do híbrido) x usuários de teste (resultado final)."""
    validacao = casos.sample(frac=fracao, random_state=semente)
    teste = casos.drop(validacao.index)
    return validacao.reset_index(drop=True), teste.reset_index(drop=True)


def _metricas_usuario(recs: np.ndarray, alvo: set[int], k: int) -> tuple[float, float, float, float]:
    acertos = np.array([r in alvo for r in recs[:k]], dtype=float)
    n_acertos = acertos.sum()
    precision = n_acertos / k
    recall = n_acertos / len(alvo)
    hit = float(n_acertos > 0)
    descontos = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((acertos * descontos).sum())
    idcg = float(descontos[: min(len(alvo), k)].sum())
    return precision, recall, hit, dcg / idcg


def avaliar(modelo: Recomendador, casos: pd.DataFrame, tamanho_perfil: int,
            k: int = config.K, filtro_qualidade: bool = False, lote: int = 500,
            popularidade: np.ndarray | None = None):
    """
    Avalia um modelo em um cenário (perfil de 1 ou de 5 jogos).
    Retorna (métricas por usuário, resumo do cenário).
    """
    mascara = mascara_qualidade(modelo.indice) if filtro_qualidade else None
    linhas, todas_recs = [], []

    for ini in range(0, len(casos), lote):
        bloco = casos.iloc[ini: ini + lote]
        perfis = [h[-tamanho_perfil:] for h in bloco["historico"]]
        excluir = [set(h) for h in bloco["historico"]]
        scores = modelo.pontuar_lote(perfis)
        recs = top_k(scores, excluir, k, mascara)
        todas_recs.append(recs)
        for uid, alvo, r in zip(bloco["user_id"], bloco["alvo"], recs):
            p, rc, h, n = _metricas_usuario(r, set(alvo), k)
            linhas.append((uid, p, rc, h, n))

    por_usuario = pd.DataFrame(linhas, columns=["user_id", "precision", "recall", "hit_rate", "ndcg"])
    recs = np.vstack(todas_recs)
    resumo = {
        "modelo": modelo.nome,
        "perfil": f"{tamanho_perfil} jogo" + ("s" if tamanho_perfil > 1 else ""),
        "filtro_qualidade": filtro_qualidade,
        "usuarios": len(por_usuario),
        f"precision@{k}": por_usuario["precision"].mean(),
        f"recall@{k}": por_usuario["recall"].mean(),
        f"hit_rate@{k}": por_usuario["hit_rate"].mean(),
        f"ndcg@{k}": por_usuario["ndcg"].mean(),
        "cobertura_catalogo": len(np.unique(recs)) / modelo.indice.n,
    }
    if popularidade is not None:
        resumo["popularidade_media_recs"] = float(popularidade[recs].mean())
    return por_usuario, resumo


def ic_bootstrap(valores, n: int = 1000, confianca: float = 0.95, semente: int = config.SEMENTE):
    """Intervalo de confiança da média por bootstrap."""
    rng = np.random.default_rng(semente)
    valores = np.asarray(valores)
    medias = rng.choice(valores, size=(n, len(valores)), replace=True).mean(axis=1)
    alfa = (1 - confianca) / 2
    return float(np.quantile(medias, alfa)), float(np.quantile(medias, 1 - alfa))


def comparar_pareado(a: pd.Series, b: pd.Series) -> dict:
    """
    Teste de Wilcoxon pareado (mesmos usuários, dois cenários).
    H0: não há diferença entre perfil de 1 jogo e perfil de 5 jogos.
    """
    diff = np.asarray(b) - np.asarray(a)
    if np.allclose(diff, 0):
        return {"diferenca_media": 0.0, "estatistica": np.nan, "p_valor": 1.0}
    est, p = stats.wilcoxon(a, b, zero_method="wilcox")
    return {"diferenca_media": float(diff.mean()), "estatistica": float(est), "p_valor": float(p)}
