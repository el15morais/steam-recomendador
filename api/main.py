"""
API de consulta do recomendador (CRISP-DM — etapa 6, Implantação).

Pré-requisito: executar o notebook 03, que salva models/recomendador.joblib.

Rodar (na raiz do repositório):
    uvicorn api.main:app --reload
Documentação interativa: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

ARQUIVO_MODELO = config.MODELOS / "recomendador.joblib"
if not ARQUIVO_MODELO.exists():
    raise RuntimeError("Modelo não encontrado. Execute antes o notebook 03_modelagem_e_avaliacao.ipynb.")

artefato = joblib.load(ARQUIVO_MODELO)
indice = artefato["indice"]
modelo = artefato["modelo"]

app = FastAPI(
    title="Recomendador de Jogos da Steam",
    description="Recomenda jogos a partir de 1 a 5 jogos jogados recentemente.",
    version="1.0.0",
)


class Pedido(BaseModel):
    jogos: list[str] = Field(..., min_length=1, max_length=5,
                             description="Até 5 jogos recentes: app_id ou parte do nome",
                             examples=[["ELDEN RING", "Cyberpunk 2077", "730"]])
    k: int = Field(config.K, ge=1, le=50)
    filtro_qualidade: bool = True


def _resolver(jogo: str) -> tuple[int, str] | None:
    if jogo.strip().isdigit() and int(jogo) in indice.posicao:
        app_id = int(jogo)
    else:
        achados = indice.buscar_por_nome(jogo)
        if achados.empty:
            return None
        app_id = int(achados.iloc[0]["app_id"])
    nome = indice.catalogo.iloc[indice.posicao[app_id]]["nome"]
    return app_id, nome


@app.get("/")
def status():
    return {"status": "ok", "modelo": modelo.nome, "jogos_no_catalogo": indice.n}


@app.get("/jogos")
def buscar_jogos(busca: str = Query(..., min_length=2), limite: int = Query(10, ge=1, le=50)):
    """Busca jogos do catálogo pelo nome (para descobrir o app_id)."""
    achados = indice.buscar_por_nome(busca).head(limite)
    return achados[["app_id", "nome"]].to_dict(orient="records")


@app.post("/recomendar")
def recomendar(pedido: Pedido):
    resolvidos, nao_encontrados = [], []
    for jogo in pedido.jogos:
        r = _resolver(jogo)
        (resolvidos.append(r) if r else nao_encontrados.append(jogo))
    if not resolvidos:
        raise HTTPException(status_code=404, detail=f"Nenhum jogo encontrado: {nao_encontrados}")

    recs = modelo.recomendar([a for a, _ in resolvidos], k=pedido.k,
                             filtro_qualidade=pedido.filtro_qualidade)
    return {
        "perfil": [{"app_id": a, "nome": n} for a, n in resolvidos],
        "nao_encontrados": nao_encontrados,
        "modelo": modelo.nome,
        "recomendacoes": recs.assign(score=recs["score"].round(4)).to_dict(orient="records"),
    }
