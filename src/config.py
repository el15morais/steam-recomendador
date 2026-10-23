"""
Configurações centrais do projeto.

Todos os caminhos e parâmetros ficam aqui para que notebooks, scripts e API
usem exatamente os mesmos valores (reprodutibilidade).
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]

DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"
FIGURAS = RAIZ / "reports" / "figures"
RESULTADOS = RAIZ / "reports" / "results"
MODELOS = RAIZ / "models"

# Base A — Steam Games Dataset (Hubert Sidorowicz, Kaggle): metadados dos jogos
PASTA_BASE_A = DADOS_BRUTOS / "steam_games"
ARQUIVO_BASE_A = PASTA_BASE_A / "steam_games.csv"

# Base B — Game Recommendations on Steam (Anton Kozyriev, Kaggle): interações
PASTA_BASE_B = DADOS_BRUTOS / "game_recommendations"
ARQUIVO_B_GAMES = PASTA_BASE_B / "games.csv"
ARQUIVO_B_USERS = PASTA_BASE_B / "users.csv"
ARQUIVO_B_RECS = PASTA_BASE_B / "recommendations.csv"
ARQUIVO_B_METADATA = PASTA_BASE_B / "games_metadata.json"

# Saídas da preparação
ARQ_CATALOGO = DADOS_PROCESSADOS / "catalogo.parquet"
ARQ_INTERACOES = DADOS_PROCESSADOS / "interacoes.parquet"
ARQ_TREINO = DADOS_PROCESSADOS / "treino.parquet"
ARQ_TESTE = DADOS_PROCESSADOS / "teste.parquet"
ARQ_FUNIL = RESULTADOS / "funil_preparacao.csv"

# ---------------------------------------------------------------------------
# Parâmetros da preparação
# ---------------------------------------------------------------------------
SEMENTE = 42

# Amostragem determinística de usuários: mantém user_id % 100 < PERCENTUAL.
# A base B tem ~41 milhões de linhas; 100% não cabe confortavelmente no Colab.
# Pode ser alterado sem mexer no código: PERCENTUAL_USUARIOS=30 jupyter ...
PERCENTUAL_USUARIOS = int(os.environ.get("PERCENTUAL_USUARIOS", 15))

# Quantos usuários de teste entram na avaliação (sorteados). Avaliar todos
# seria lento e não muda as conclusões; 5 000 dá intervalos de confiança estreitos.
N_USUARIOS_AVALIACAO = int(os.environ.get("N_USUARIOS_AVALIACAO", 5000))

# Tamanho do bloco de leitura do recommendations.csv
TAMANHO_BLOCO = 5_000_000

# Um usuário só entra no experimento se tiver jogos suficientes para
# montar o perfil (5 jogos) e ainda sobrar jogos "escondidos" para avaliar.
TAMANHO_PERFIL = 5
N_ITENS_TESTE = 3
MIN_INTERACOES_USUARIO = TAMANHO_PERFIL + N_ITENS_TESTE  # 8

# Um jogo precisa de um mínimo de interações positivas para entrar no catálogo
# avaliado (jogos quase sem uso não podem ser aprendidos pelo colaborativo).
MIN_INTERACOES_JOGO = 5

# Filtro de qualidade (seminário): taxa negativa = neg / (pos + neg)
MIN_AVALIACOES_QUALIDADE = 50      # abaixo disso a taxa não é confiável
TAXA_NEGATIVA_DESCARTE = 0.30      # > 30% negativas -> descartado (seminário)

# ---------------------------------------------------------------------------
# Parâmetros da modelagem / avaliação
# ---------------------------------------------------------------------------
K = 10                       # tamanho da lista de recomendação
COMPONENTES_SVD = 64         # dimensões latentes da fatoração (TruncatedSVD)
FRACAO_VALIDACAO = 0.3       # fração dos usuários de teste usada para ajustar o híbrido


def criar_pastas() -> None:
    """Cria as pastas de saída, se ainda não existirem."""
    for pasta in (DADOS_BRUTOS, DADOS_PROCESSADOS, FIGURAS, RESULTADOS, MODELOS):
        pasta.mkdir(parents=True, exist_ok=True)
