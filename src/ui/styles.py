"""
Estilo visual da interface de demonstração.

Paleta institucional: azul-marinho para identidade, vermelho reservado a alerta,
verde a resultado sem indício. A escolha não é estética apenas — num sistema que
sinaliza pessoas, cor é informação, e usar vermelho para decoração gastaria o
sinal que precisa significar "atenção".

O CSS é mínimo de propósito: o suficiente para não parecer aplicação Streamlit
padrão, sem reescrever o framework. Regras demais quebram a cada atualização da
biblioteca e ninguém lembra por que existiam.
"""

from __future__ import annotations

MARINHO = "#0B2545"
MARINHO_CLARO = "#13315C"
AZUL_SUAVE = "#E8EEF7"
VERMELHO = "#B3261E"
VERMELHO_FUNDO = "#FDECEA"
VERDE = "#1B6B3A"
VERDE_FUNDO = "#E8F5EC"
CINZA_TEXTO = "#4A5568"
BORDA = "#DDE3EC"

CSS = f"""
<style>
  .stApp {{ background: #F7F9FC; }}

  /* Cabeçalho institucional */
  .fg-cabecalho {{
    background: linear-gradient(90deg, {MARINHO} 0%, {MARINHO_CLARO} 100%);
    color: #FFFFFF;
    padding: 18px 26px;
    border-radius: 12px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
  }}
  .fg-cabecalho h1 {{
    font-size: 1.55rem; margin: 0; font-weight: 700; letter-spacing: .3px;
  }}
  .fg-cabecalho p {{ margin: 2px 0 0; opacity: .85; font-size: .92rem; }}
  .fg-status {{
    font-size: .82rem; padding: 6px 12px; border-radius: 999px;
    background: rgba(255,255,255,.14); white-space: nowrap;
  }}
  .fg-status b {{ font-weight: 600; }}

  /* Cartões */
  .fg-card {{
    background: #FFFFFF;
    border: 1px solid {BORDA};
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(11,37,69,.06);
    margin-bottom: 18px;
  }}
  .fg-card h3 {{
    margin: 0 0 4px; font-size: .78rem; font-weight: 700;
    letter-spacing: .09em; text-transform: uppercase; color: {MARINHO};
  }}
  .fg-card .fg-sub {{ color: {CINZA_TEXTO}; font-size: .88rem; margin-bottom: 10px; }}

  /* Veredito */
  .fg-veredito {{
    border-radius: 12px; padding: 22px 24px; margin-bottom: 18px;
    border-left: 6px solid transparent;
  }}
  .fg-veredito .fg-titulo {{ font-size: 1.32rem; font-weight: 700; margin: 0; }}
  .fg-veredito .fg-detalhe {{ font-size: .9rem; margin: 6px 0 0; opacity: .88; }}
  .fg-suspeita {{
    background: {VERMELHO_FUNDO}; border-left-color: {VERMELHO}; color: {VERMELHO};
  }}
  .fg-limpa {{
    background: {VERDE_FUNDO}; border-left-color: {VERDE}; color: {VERDE};
  }}

  /* Barra de probabilidade */
  .fg-barra-fundo {{
    background: {AZUL_SUAVE}; border-radius: 999px; height: 13px;
    overflow: hidden; position: relative; margin: 8px 0 4px;
  }}
  .fg-barra-preenchida {{ height: 100%; border-radius: 999px; }}
  .fg-barra-limiar {{
    position: absolute; top: -3px; width: 2px; height: 19px; background: {MARINHO};
  }}
  .fg-legenda {{ font-size: .78rem; color: {CINZA_TEXTO}; }}

  /* Fatores SHAP */
  .fg-fator {{ margin-bottom: 12px; }}
  .fg-fator-topo {{
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: .88rem; margin-bottom: 4px; gap: 10px;
  }}
  .fg-fator-nome {{ font-weight: 600; color: {MARINHO}; word-break: break-all; }}
  .fg-fator-valor {{ color: {CINZA_TEXTO}; font-size: .8rem; white-space: nowrap; }}
  .fg-fator-trilho {{
    display: flex; align-items: center; height: 15px;
    background: {AZUL_SUAVE}; border-radius: 4px; overflow: hidden;
  }}
  .fg-fator-metade {{ width: 50%; display: flex; height: 100%; }}
  .fg-fator-esq {{ justify-content: flex-end; }}
  .fg-fator-barra {{ height: 100%; }}
  .fg-sobe {{ background: {VERMELHO}; }}
  .fg-desce {{ background: {VERDE}; }}

  /* Explicação da IA */
  .fg-explicacao {{
    background: #FFFFFF; border: 1px solid {BORDA}; border-left: 5px solid {MARINHO};
    border-radius: 10px; padding: 18px 22px; line-height: 1.62; font-size: .97rem;
    white-space: pre-wrap;
  }}

  /* Avisos e rodapé */
  .fg-aviso {{
    background: #FFF8E6; border: 1px solid #F0D9A0; border-radius: 10px;
    padding: 14px 18px; font-size: .88rem; color: #6B5310; margin-bottom: 18px;
  }}
  .fg-rodape {{
    color: {CINZA_TEXTO}; font-size: .78rem; text-align: center;
    margin-top: 28px; padding-top: 14px; border-top: 1px solid {BORDA};
  }}

  section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {BORDA}; }}
  div[data-testid="stMetricValue"] {{ color: {MARINHO}; font-size: 1.5rem; }}
</style>
"""
