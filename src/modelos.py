"""
Modelos de recomendação (CRISP-DM — etapa 4).

Todos recebem um PERFIL (lista de jogos recentes do jogador) e devolvem uma
pontuação para cada jogo do catálogo. A mesma interface vale para 1 ou 5 jogos,
o que permite responder à pergunta de pesquisa com qualquer modelo.

Modelos:
  1. Popularidade        — linha de base: recomenda o que mais gente gostou.
  2. Conteúdo (TF-IDF)   — proposta original do grupo: gêneros + categorias +
                           tags -> TF-IDF -> similaridade de cosseno.
                           Agregação do perfil por "média" (centroide) ou
                           "máximo" (melhor similaridade com qualquer jogo do perfil).
  3. Colaborativo item-item — "quem gostou de X também gostou de Y",
                           cosseno entre colunas da matriz usuário x jogo.
  4. Fatoração (TruncatedSVD) — vetores latentes de jogos aprendidos da matriz
                           usuário x jogo.
  5. Híbrido             — combinação ponderada de conteúdo + item-item.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config


# ---------------------------------------------------------------------------
# Índice do catálogo e matriz de interações
# ---------------------------------------------------------------------------
class IndiceCatalogo:
    """Mapeia app_id <-> posição (0..n-1) no catálogo."""

    def __init__(self, catalogo: pd.DataFrame):
        self.catalogo = catalogo.reset_index(drop=True)
        self.app_ids = self.catalogo["app_id"].to_numpy()
        self.posicao = {int(a): i for i, a in enumerate(self.app_ids)}
        self.n = len(self.app_ids)

    def para_posicoes(self, app_ids) -> list[int]:
        return [self.posicao[int(a)] for a in app_ids if int(a) in self.posicao]

    def para_app_ids(self, posicoes) -> list[int]:
        return [int(self.app_ids[p]) for p in posicoes]

    @staticmethod
    def _normalizar_nome(texto) -> str:
        """'Sid Meier’s Civilization® VI' -> 'sid meiers civilization vi'"""
        texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().lower()
        texto = re.sub(r"[^a-z0-9 ]+", "", texto)
        return re.sub(r"\s+", " ", texto).strip()

    def buscar_por_nome(self, nome: str) -> pd.DataFrame:
        if "_nome_normalizado" not in self.catalogo:
            self.catalogo["_nome_normalizado"] = self.catalogo["nome"].map(self._normalizar_nome)
        alvo = self._normalizar_nome(nome)
        mascara = self.catalogo["_nome_normalizado"].str.contains(alvo, na=False, regex=False)
        return self.catalogo.loc[mascara, ["app_id", "nome", "total_avaliacoes"]].sort_values(
            "total_avaliacoes", ascending=False)


def matriz_usuario_item(treino: pd.DataFrame, indice: IndiceCatalogo) -> sparse.csr_matrix:
    """Matriz binária (usuários x jogos) com 1 onde o usuário recomendou o jogo."""
    usuarios = pd.Index(treino["user_id"].unique())
    linhas = usuarios.get_indexer(treino["user_id"])
    colunas = treino["app_id"].map(indice.posicao).to_numpy()
    validos = ~pd.isna(colunas)
    dados = np.ones(validos.sum(), dtype=np.float32)
    return sparse.csr_matrix(
        (dados, (linhas[validos], colunas[validos].astype(int))),
        shape=(len(usuarios), indice.n),
    )


def _matriz_perfis(perfis: list[list[int]], n: int, media: bool = True) -> sparse.csr_matrix:
    """Uma linha por perfil, com 1 (ou 1/len) nas posições dos jogos do perfil."""
    linhas, colunas, dados = [], [], []
    for i, perfil in enumerate(perfis):
        peso = 1.0 / len(perfil) if (media and perfil) else 1.0
        for p in perfil:
            linhas.append(i)
            colunas.append(p)
            dados.append(peso)
    return sparse.csr_matrix((dados, (linhas, colunas)), shape=(len(perfis), n), dtype=np.float32)


def _minmax_linhas(m: np.ndarray) -> np.ndarray:
    mn = m.min(axis=1, keepdims=True)
    mx = m.max(axis=1, keepdims=True)
    amplitude = np.where(mx - mn > 0, mx - mn, 1.0)
    return (m - mn) / amplitude


# ---------------------------------------------------------------------------
# Classe base
# ---------------------------------------------------------------------------
class Recomendador:
    nome = "base"

    def __init__(self, indice: IndiceCatalogo):
        self.indice = indice

    def pontuar_lote(self, perfis: list[list[int]]) -> np.ndarray:
        """Matriz (len(perfis) x n_jogos) de pontuações. Maior = mais recomendado."""
        raise NotImplementedError

    def recomendar(self, perfil_app_ids, k: int = config.K, excluir_app_ids=(),
                   filtro_qualidade: bool = False) -> pd.DataFrame:
        """Uso interativo / API: recebe app_ids e devolve o top-k com nome e score."""
        perfil = self.indice.para_posicoes(perfil_app_ids)
        if not perfil:
            raise ValueError("Nenhum dos jogos informados está no catálogo.")
        excluir = set(perfil) | set(self.indice.para_posicoes(excluir_app_ids))
        scores = self.pontuar_lote([perfil])[0]
        top = top_k(scores[None, :], [excluir], k, mascara_qualidade(self.indice) if filtro_qualidade else None)[0]
        cat = self.indice.catalogo.iloc[top]
        return pd.DataFrame({
            "app_id": cat["app_id"].to_numpy(),
            "nome": cat["nome"].to_numpy(),
            "score": scores[top],
            "classificacao_qualidade": cat["classificacao_qualidade"].to_numpy(),
        })


# ---------------------------------------------------------------------------
# 1. Popularidade
# ---------------------------------------------------------------------------
class Popularidade(Recomendador):
    nome = "Popularidade"

    def __init__(self, indice: IndiceCatalogo, treino: pd.DataFrame):
        super().__init__(indice)
        contagem = treino["app_id"].map(indice.posicao).value_counts()
        self.scores = np.zeros(indice.n, dtype=np.float32)
        self.scores[contagem.index.astype(int)] = contagem.to_numpy()

    def pontuar_lote(self, perfis):
        return np.tile(self.scores, (len(perfis), 1))


# ---------------------------------------------------------------------------
# 2. Conteúdo — TF-IDF + cosseno
# ---------------------------------------------------------------------------
class ConteudoTFIDF(Recomendador):
    def __init__(self, indice: IndiceCatalogo, agregacao: str = "media", min_df: int = 2):
        super().__init__(indice)
        if agregacao not in {"media", "maximo"}:
            raise ValueError("agregacao deve ser 'media' ou 'maximo'")
        self.agregacao = agregacao
        self.nome = f"Conteúdo TF-IDF ({'média' if agregacao == 'media' else 'máximo'})"
        # token_pattern pega cada token inteiro ("open_world"), sem quebrar.
        self.vetorizador = TfidfVectorizer(token_pattern=r"[^ ]+", min_df=min_df, lowercase=False)
        self.X = self.vetorizador.fit_transform(indice.catalogo["texto_features"].fillna(""))  # já L2

    def pontuar_lote(self, perfis):
        if self.agregacao == "media":
            P = _matriz_perfis(perfis, self.indice.n, media=True)
            centroides = normalize(P @ self.X)
            return (centroides @ self.X.T).toarray()
        saida = np.empty((len(perfis), self.indice.n), dtype=np.float32)
        for i, perfil in enumerate(perfis):
            saida[i] = (self.X[perfil] @ self.X.T).max(axis=0).toarray().ravel()
        return saida


# ---------------------------------------------------------------------------
# 3. Colaborativo item-item
# ---------------------------------------------------------------------------
class ColaborativoItemItem(Recomendador):
    """
    Item-kNN: similaridade de cosseno entre jogos calculada a partir dos
    usuários em comum. Para caber na memória, guardamos só os `vizinhos`
    jogos mais parecidos com cada jogo (matriz esparsa).
    """

    def __init__(self, indice: IndiceCatalogo, matriz_ui: sparse.csr_matrix,
                 vizinhos: int = 200, bloco: int = 1000):
        super().__init__(indice)
        self.vizinhos = vizinhos
        self.nome = "Colaborativo item-item"
        # Normaliza cada coluna (jogo) -> produto interno vira cosseno entre jogos
        Xn = normalize(matriz_ui.tocsc().astype(np.float32), axis=0)
        XnT = Xn.T.tocsr()
        n = indice.n
        k = min(vizinhos, n - 1)

        linhas, colunas, dados = [], [], []
        for ini in range(0, n, bloco):
            fim = min(ini + bloco, n)
            S = (XnT[ini:fim] @ Xn).toarray()          # (bloco x jogos)
            S[np.arange(fim - ini), np.arange(ini, fim)] = 0  # ignora o próprio jogo
            top = np.argpartition(-S, k - 1, axis=1)[:, :k]
            valores = np.take_along_axis(S, top, axis=1)
            for i in range(fim - ini):
                ok = valores[i] > 0
                linhas.append(np.full(ok.sum(), ini + i))
                colunas.append(top[i][ok])
                dados.append(valores[i][ok])
        self.S = sparse.csr_matrix(
            (np.concatenate(dados), (np.concatenate(linhas), np.concatenate(colunas))),
            shape=(n, n), dtype=np.float32,
        )

    def pontuar_lote(self, perfis):
        P = _matriz_perfis(perfis, self.indice.n, media=True)
        return (P @ self.S).toarray()


# ---------------------------------------------------------------------------
# 4. Fatoração de matriz — TruncatedSVD
# ---------------------------------------------------------------------------
class FatoracaoSVD(Recomendador):
    """
    PureSVD (Cremonesi; Koren; Turrin, 2010): o perfil é projetado no espaço
    latente e reconstruído, x̂ = x · V · Vᵀ, onde V são os vetores latentes dos
    jogos aprendidos pelo TruncatedSVD na matriz usuário x jogo.
    """

    def __init__(self, indice: IndiceCatalogo, matriz_ui: sparse.csr_matrix,
                 n_componentes: int = config.COMPONENTES_SVD, semente: int = config.SEMENTE):
        super().__init__(indice)
        n_componentes = min(n_componentes, min(matriz_ui.shape) - 1)
        self.n_componentes = n_componentes
        self.nome = f"Fatoração SVD ({n_componentes} dim.)"
        self.svd = TruncatedSVD(n_components=n_componentes, random_state=semente)
        self.svd.fit(matriz_ui)
        self.V = self.svd.components_.T  # (jogos x dimensões latentes)

    def pontuar_lote(self, perfis):
        P = _matriz_perfis(perfis, self.indice.n, media=False)
        return np.asarray(P @ self.V) @ self.V.T


# ---------------------------------------------------------------------------
# 5. Híbrido
# ---------------------------------------------------------------------------
class Hibrido(Recomendador):
    def __init__(self, indice: IndiceCatalogo, conteudo: Recomendador,
                 colaborativo: Recomendador, peso_conteudo: float = 0.5):
        super().__init__(indice)
        self.conteudo = conteudo
        self.colaborativo = colaborativo
        self.peso_conteudo = peso_conteudo

    @property
    def nome(self):
        return f"Híbrido (α={self.peso_conteudo:.1f})"

    def pontuar_lote(self, perfis):
        a = _minmax_linhas(self.conteudo.pontuar_lote(perfis))
        b = _minmax_linhas(self.colaborativo.pontuar_lote(perfis))
        return self.peso_conteudo * a + (1 - self.peso_conteudo) * b


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def mascara_qualidade(indice: IndiceCatalogo) -> np.ndarray:
    """True para jogos que passam no filtro de qualidade do seminário."""
    return indice.catalogo["passa_filtro_qualidade"].to_numpy(dtype=bool)


def top_k(scores: np.ndarray, excluir: list[set[int]], k: int,
          mascara: np.ndarray | None = None) -> np.ndarray:
    """Top-k por linha, ignorando jogos já vistos e (opcional) reprovados no filtro."""
    s = scores.astype(np.float64, copy=True)
    if mascara is not None:
        s[:, ~mascara] = -np.inf
    for i, ex in enumerate(excluir):
        if ex:
            s[i, list(ex)] = -np.inf
    k = min(k, s.shape[1])
    candidatos = np.argpartition(-s, k - 1, axis=1)[:, :k]
    ordem = np.take_along_axis(s, candidatos, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(candidatos, ordem, axis=1)
