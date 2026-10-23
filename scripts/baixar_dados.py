"""
Baixa as duas bases do Kaggle para data/raw/.

Uso (na raiz do repositório):
    python scripts/baixar_dados.py

Requer `pip install kagglehub`. Bases públicas normalmente baixam sem login;
se o Kaggle pedir autenticação, crie um token em kaggle.com > Settings > API
e siga https://github.com/Kaggle/kagglehub#authenticate.

Se preferir baixar manualmente, extraia os arquivos nas pastas:
    data/raw/steam_games/            -> steam_games.csv
    data/raw/game_recommendations/   -> games.csv, users.csv,
                                        recommendations.csv, games_metadata.json
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

BASES = {
    "hubertsidorowicz/steam-games-dataset-daily-updates": config.PASTA_BASE_A,
    "antonkozyriev/game-recommendations-on-steam": config.PASTA_BASE_B,
}


def main() -> None:
    try:
        import kagglehub
    except ImportError:
        sys.exit("Instale antes: pip install kagglehub")

    for slug, destino in BASES.items():
        print(f"\nBaixando {slug} ...")
        origem = Path(kagglehub.dataset_download(slug))
        destino.mkdir(parents=True, exist_ok=True)
        for arquivo in origem.rglob("*"):
            if arquivo.is_file():
                alvo = destino / arquivo.name
                if not alvo.exists():
                    shutil.copy2(arquivo, alvo)
                print(f"  {alvo.relative_to(config.RAIZ)}  ({arquivo.stat().st_size / 1e6:.0f} MB)")

    print("\nPronto. Confira se os nomes batem com os de src/config.py.")


if __name__ == "__main__":
    main()
