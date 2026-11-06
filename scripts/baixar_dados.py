"""
Baixa as duas bases do Kaggle para data/raw/, em versões FIXAS.

Uso (na raiz do repositório):
    python scripts/baixar_dados.py

A base A (Steam Games Dataset) é atualizada todos os dias no Kaggle. Para que
os resultados do artigo sejam reproduzíveis, usamos sempre a mesma versão:
a v89, de 28/10/2026 (144.981 jogos), que gerou os resultados publicados.
A base B (Game Recommendations on Steam) não muda; fixamos a v28 por garantia.

Requer `pip install kagglehub`. Bases públicas normalmente baixam sem login;
se o Kaggle pedir autenticação, crie um token em kaggle.com > Settings > API
e siga https://github.com/Kaggle/kagglehub#authenticate.

Download manual (mesmas versões): baixe pela aba "Data" de cada base no Kaggle,
escolhendo a versão indicada, e extraia em:
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
    "hubertsidorowicz/steam-games-dataset-daily-updates/versions/89": config.PASTA_BASE_A,
    "antonkozyriev/game-recommendations-on-steam/versions/28": config.PASTA_BASE_B,
}
MARCADOR = "VERSAO_KAGGLE.txt"


def main() -> None:
    try:
        import kagglehub
    except ImportError:
        sys.exit("Instale antes: pip install kagglehub")

    for handle, destino in BASES.items():
        marcador = destino / MARCADOR
        if marcador.exists() and marcador.read_text().strip() == handle:
            print(f"{handle}: já baixada.")
            continue

        print(f"\nBaixando {handle} ...")
        origem = Path(kagglehub.dataset_download(handle))
        if destino.exists():
            shutil.rmtree(destino)          # remove versão anterior, se houver
        destino.mkdir(parents=True, exist_ok=True)
        for arquivo in origem.rglob("*"):
            if arquivo.is_file():
                shutil.copy2(arquivo, destino / arquivo.name)
                print(f"  {(destino / arquivo.name).relative_to(config.RAIZ)}  ({arquivo.stat().st_size / 1e6:.0f} MB)")
        marcador.write_text(handle)

    print("\nPronto.")


if __name__ == "__main__":
    main()
