# Roteiro do artigo (modelo do professor → o que entra em cada seção)

Formato: A4, margens 3 cm, Arial 12, espaçamento 1,5, **14 a 20 páginas**, PDF, nome de todos os integrantes.
Título em fonte 14, caixa alta, negrito, centralizado, no máximo 12 palavras.

Sugestão de título (11 palavras):
**RECOMENDAÇÃO DE JOGOS DA STEAM BASEADA NO PERFIL RECENTE DO JOGADOR**

| Seção | Páginas (aprox.) | Conteúdo | Material do repositório |
|---|---|---|---|
| Resumo | 0,5 | 200–300 palavras: contexto, objetivo, método, resultado, conclusão | números de `resultados_teste.csv` e `teste_wilcoxon_1x5.csv` |
| Introdução | 1,5 | catálogo enorme da Steam, sobrecarga de escolha, papel dos recomendadores, por que perfil recente | — |
| Objetivos | 0,5 | geral + específicos (abaixo) | — |
| Fundamentação teórica | 3 | CRISP-DM; sistemas de recomendação (conteúdo × colaborativo × híbrido); TF-IDF e cosseno; item-kNN; fatoração (SVD); feedback implícito; métricas top-N (Precision, Recall, NDCG); intervalo de Wilson | referências abaixo |
| Metodologia (CRISP-DM) | 6–7 | as 6 etapas, com figuras e tabelas | Figuras 1–8, funil, tabelas de faltantes |
| Resultados e conclusões | 3–4 | tabela final, Figura 9, Wilcoxon, Figura 10, ablação do filtro, avaliação qualitativa; responder: funcionou? faz sentido? gera valor? limitações e trabalhos futuros | Figuras 9–10, CSVs de resultados |
| Referências | 1 | ABNT, só as citadas | lista abaixo |

## Objetivos (rascunho)
**Geral:** desenvolver e avaliar um sistema de recomendação de jogos da Steam que use o perfil recente do jogador,
verificando se perfis formados por cinco jogos produzem recomendações mais coerentes do que perfis de um único jogo.

**Específicos:**
1. Integrar uma base de metadados de jogos e uma base de interações de usuários da Steam;
2. Limpar, transformar e tratar dados faltantes e desbalanceados das duas bases;
3. Implementar modelos de recomendação por conteúdo (TF-IDF), colaborativos (item-item e SVD) e híbrido;
4. Avaliar os modelos com métricas de ranqueamento em um protocolo temporal, comparando perfis de 1 e de 5 jogos;
5. Disponibilizar o modelo final por meio de uma API.

## Metodologia — o que escrever em cada etapa
- **Entendimento do negócio:** problema, pergunta, mudança após o seminário (inclusão da base B e por quê).
- **Entendimento dos dados:** descrição das duas bases (linhas, colunas usadas), Figuras 1–6, problemas encontrados
  (duplicatas, faltantes, poucas avaliações, desbalanceamento de `is_recommended`, cauda longa).
- **Preparação:** limpeza (funil), faltantes (tabela + decisões), transformação (texto de características, tokens),
  junção A+B, filtro de qualidade com Wilson, amostragem de usuários e justificativa (memória), divisão temporal.
- **Modelagem:** cada modelo em 1 parágrafo + justificativa; ajuste de hiperparâmetros na validação (Figura 8).
- **Avaliação:** protocolo, métricas, teste estatístico, avaliação qualitativa 0/1/2 (preencher o CSV).
- **Implantação:** API FastAPI (endpoints `/jogos` e `/recomendar`) e como seria usada (ex.: seção "o que jogar depois").

## Referências (ABNT) — conferir dados antes de entregar
CHAPMAN, P. et al. **CRISP-DM 1.0**: step-by-step data mining guide. [S. l.]: SPSS, 2000.

CREMONESI, P.; KOREN, Y.; TURRIN, R. Performance of recommender algorithms on top-N recommendation tasks. In: ACM CONFERENCE ON RECOMMENDER SYSTEMS, 4., 2010, Barcelona. **Proceedings** [...]. New York: ACM, 2010. p. 39-46.

FAYYAD, U.; PIATETSKY-SHAPIRO, G.; SMYTH, P. From data mining to knowledge discovery in databases. **AI Magazine**, v. 17, n. 3, p. 37-54, 1996.

HARRIS, C. R. et al. Array programming with NumPy. **Nature**, v. 585, p. 357-362, 2020.

HU, Y.; KOREN, Y.; VOLINSKY, C. Collaborative filtering for implicit feedback datasets. In: IEEE INTERNATIONAL CONFERENCE ON DATA MINING, 8., 2008, Pisa. **Proceedings** [...]. Piscataway: IEEE, 2008. p. 263-272.

HUNTER, J. D. Matplotlib: a 2D graphics environment. **Computing in Science & Engineering**, v. 9, n. 3, p. 90-95, 2007.

JÄRVELIN, K.; KEKÄLÄINEN, J. Cumulated gain-based evaluation of IR techniques. **ACM Transactions on Information Systems**, v. 20, n. 4, p. 422-446, 2002.

KOREN, Y.; BELL, R.; VOLINSKY, C. Matrix factorization techniques for recommender systems. **Computer**, v. 42, n. 8, p. 30-37, 2009.

KOZYRIEV, A. **Game Recommendations on Steam**. Kaggle, 2024. Disponível em: https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam. Acesso em: [data].

LOPS, P.; DE GEMMIS, M.; SEMERARO, G. Content-based recommender systems: state of the art and trends. In: RICCI, F. et al. (ed.). **Recommender Systems Handbook**. Boston: Springer, 2011. p. 73-105.

MCKINNEY, W. Data structures for statistical computing in Python. In: PYTHON IN SCIENCE CONFERENCE, 9., 2010, Austin. **Proceedings** [...]. 2010. p. 56-61.

PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. **Journal of Machine Learning Research**, v. 12, p. 2825-2830, 2011.

RICCI, F.; ROKACH, L.; SHAPIRA, B. (ed.). **Recommender Systems Handbook**. 3. ed. New York: Springer, 2022.

SALTON, G.; BUCKLEY, C. Term-weighting approaches in automatic text retrieval. **Information Processing & Management**, v. 24, n. 5, p. 513-523, 1988.

SARWAR, B. et al. Item-based collaborative filtering recommendation algorithms. In: INTERNATIONAL WORLD WIDE WEB CONFERENCE, 10., 2001, Hong Kong. **Proceedings** [...]. New York: ACM, 2001. p. 285-295.

SIDOROWICZ, H. **Steam Games Dataset (daily updates)**. Kaggle, 2026. Disponível em: https://www.kaggle.com/datasets/hubertsidorowicz/steam-games-dataset-daily-updates. Acesso em: [data].

WASKOM, M. L. seaborn: statistical data visualization. **Journal of Open Source Software**, v. 6, n. 60, p. 3021, 2021.

WILSON, E. B. Probable inference, the law of succession, and statistical inference. **Journal of the American Statistical Association**, v. 22, n. 158, p. 209-212, 1927.
