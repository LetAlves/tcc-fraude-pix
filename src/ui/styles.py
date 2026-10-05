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
  :root {{ --fg-azul: #1473E6; --fg-azul-claro: #EAF3FF; }}
  .stApp {{ background: #F4F7FB; color: #101828; }}
  .block-container {{ max-width: 1420px; padding-top: 2rem; padding-bottom: 2rem; }}

  /* Cabeçalho limpo, alinhado ao mockup aprovado */
  .fg-cabecalho {{
    color: #101828; padding: 2px 2px 12px; margin-bottom: 8px;
    display: flex; align-items: flex-start; justify-content: space-between;
    flex-wrap: wrap; gap: 12px;
  }}
  .fg-cabecalho h1 {{
    font-size: clamp(1.65rem, 3vw, 2.35rem); margin: 0; font-weight: 760;
    letter-spacing: -.035em; line-height: 1.12;
  }}
  .fg-cabecalho p {{ margin: 7px 0 0; color: #667085; font-size: 1rem; }}
  .fg-status {{
    font-size: .78rem; padding: 7px 12px; border-radius: 999px;
    background: #FFFFFF; border: 1px solid {BORDA}; white-space: nowrap;
    box-shadow: 0 1px 2px rgba(16,24,40,.04);
  }}
  .fg-status b {{ font-weight: 650; }}

  /* Identidade e navegação lateral */
  section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {MARINHO} 0%, #071C35 100%);
    border-right: 0;
  }}
  section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding-top: 1.25rem; }}
  section[data-testid="stSidebar"] h4,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stCaption {{ color: #E8F0FA !important; }}
  section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.16); }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    border-radius: 9px; padding: 7px 8px; transition: background .15s ease;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background: rgba(255,255,255,.09);
  }}
  .fg-marca-sidebar {{ display: flex; align-items: center; gap: 12px; margin: 2px 0 10px; }}
  .fg-marca-icone {{
    width: 44px; height: 44px; display: grid; place-items: center; border-radius: 10px;
    background: linear-gradient(145deg, #3994FF, #0864D7); color: white;
    font-size: 1.55rem; box-shadow: 0 8px 18px rgba(0,73,167,.35);
  }}
  .fg-marca-sidebar strong {{ color: #FFFFFF; font-size: 1rem; display: block; }}
  .fg-marca-sidebar span {{ color: #AFC3DD; font-size: .76rem; display: block; }}
  .fg-sidebar-intro {{ color: #B9CAE0 !important; font-size: .82rem; line-height: 1.45; }}

  /* Cartões e containers */
  .fg-card,
  div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #FFFFFF; border-color: {BORDA} !important; border-radius: 12px !important;
    box-shadow: 0 5px 18px rgba(11,37,69,.045);
  }}
  .fg-card {{ padding: 20px 22px; margin-bottom: 18px; border: 1px solid {BORDA}; }}
  .fg-card h3 {{
    margin: 0 0 4px; font-size: .8rem; font-weight: 750;
    letter-spacing: .075em; text-transform: uppercase; color: {MARINHO};
  }}
  .fg-card .fg-sub {{ color: {CINZA_TEXTO}; font-size: .88rem; margin-bottom: 10px; }}
  div[data-testid="stVerticalBlockBorderWrapper"] h4 {{ color: {MARINHO}; margin-top: 0; }}

  /* Resumo da decisão */
  .fg-metrica {{
    min-height: 98px; box-sizing: border-box; display: flex; align-items: center;
    gap: 15px; padding: 20px; background: #FFFFFF; border: 1px solid {BORDA};
    border-radius: 12px; box-shadow: 0 5px 18px rgba(11,37,69,.045); margin-bottom: 16px;
  }}
  .fg-metrica-alerta {{ background: linear-gradient(100deg, #FFF2F1, #FFFFFF); border-color: #FFD3CF; }}
  .fg-metrica-segura {{ background: linear-gradient(100deg, #ECF8F0, #FFFFFF); border-color: #CDEBD7; }}
  .fg-metrica-icone {{
    width: 48px; height: 48px; border-radius: 50%; display: grid; place-items: center;
    flex: 0 0 auto; background: #FFE0DD; color: {VERMELHO}; font-size: 1.5rem; font-weight: 800;
  }}
  .fg-metrica-segura .fg-metrica-icone {{ background: #DDF3E4; color: {VERDE}; }}
  .fg-icone-azul {{ background: var(--fg-azul-claro); color: var(--fg-azul); }}
  .fg-metrica small {{ color: #475467; font-size: .77rem; display: block; margin-bottom: 2px; }}
  .fg-metrica strong {{ color: {MARINHO}; font-size: 1.55rem; line-height: 1.1; display: block; }}
  .fg-metrica-alerta strong {{ color: {VERMELHO}; }}

  /* Veredito e barra preservados para outras páginas */
  .fg-veredito {{ border-radius: 12px; padding: 22px 24px; margin-bottom: 18px; border-left: 6px solid transparent; }}
  .fg-veredito .fg-titulo {{ font-size: 1.32rem; font-weight: 700; margin: 0; }}
  .fg-veredito .fg-detalhe {{ font-size: .9rem; margin: 6px 0 0; opacity: .88; }}
  .fg-suspeita {{ background: {VERMELHO_FUNDO}; border-left-color: {VERMELHO}; color: {VERMELHO}; }}
  .fg-limpa {{ background: {VERDE_FUNDO}; border-left-color: {VERDE}; color: {VERDE}; }}
  .fg-barra-fundo {{ background: {AZUL_SUAVE}; border-radius: 999px; height: 13px; overflow: hidden; position: relative; margin: 8px 0 4px; }}
  .fg-barra-preenchida {{ height: 100%; border-radius: 999px; }}
  .fg-barra-limiar {{ position: absolute; top: -3px; width: 2px; height: 19px; background: {MARINHO}; }}
  .fg-legenda {{ font-size: .78rem; color: {CINZA_TEXTO}; }}

  /* Gráfico SHAP */
  .fg-fator {{ margin-bottom: 13px; }}
  .fg-fator-topo {{ display: flex; justify-content: space-between; align-items: baseline; font-size: .86rem; margin-bottom: 5px; gap: 10px; }}
  .fg-fator-nome {{ font-weight: 650; color: {MARINHO}; overflow-wrap: anywhere; }}
  .fg-fator-valor {{ color: {CINZA_TEXTO}; font-size: .76rem; white-space: nowrap; }}
  .fg-fator-trilho {{
    display: flex; align-items: center; height: 14px; background: linear-gradient(90deg, #EDF3FA 49.7%, #CDD7E5 49.7%, #CDD7E5 50.3%, #EDF3FA 50.3%);
    border-radius: 4px; overflow: hidden;
  }}
  .fg-fator-metade {{ width: 50%; display: flex; height: 100%; }}
  .fg-fator-esq {{ justify-content: flex-end; }}
  .fg-fator-barra {{ height: 100%; border-radius: 3px; }}
  .fg-sobe {{ background: linear-gradient(90deg, #FF7770, {VERMELHO}); }}
  .fg-desce {{ background: linear-gradient(90deg, #176DCF, #65A9F2); }}
  .fg-legenda-shap {{ display: flex; justify-content: center; gap: 24px; color: #667085; font-size: .77rem; margin-top: 15px; }}
  .fg-bolinha {{ width: 9px; height: 9px; display: inline-block; border-radius: 50%; margin-right: 6px; }}
  .fg-bolinha-sobe {{ background: {VERMELHO}; }}
  .fg-bolinha-desce {{ background: #2478D2; }}

  /* Explicação do fator selecionado */
  .fg-detalhe-fator {{ font-size: .84rem; }}
  .fg-detalhe-linha {{ display: grid; grid-template-columns: minmax(88px, .75fr) 1.25fr; gap: 10px; margin: 7px 0; }}
  .fg-detalhe-linha span {{ color: #667085; }}
  .fg-detalhe-linha strong {{ color: #1D2939; font-weight: 650; overflow-wrap: anywhere; }}
  .fg-impacto-sobe {{ color: {VERMELHO} !important; }}
  .fg-impacto-desce {{ color: #176DCF !important; }}
  .fg-caixa-informacao {{ background: #EAF4FF; border: 1px solid #CFE4FA; border-radius: 9px; padding: 12px 14px; margin-top: 15px; color: #163A63; }}
  .fg-caixa-informacao b {{ display: block; margin-bottom: 4px; }}
  .fg-caixa-informacao p {{ margin: 0; line-height: 1.42; }}
  .fg-limitacao {{ color: #667085; font-size: .75rem; line-height: 1.4; margin: 10px 2px 0; }}

  /* Explicação e avisos */
  .fg-explicacao {{
    background: linear-gradient(100deg, #F5F9FF, #EAF4FF); border: 1px solid #D5E6F7;
    border-left: 5px solid var(--fg-azul); border-radius: 10px; padding: 18px 22px;
    line-height: 1.62; font-size: .94rem; white-space: pre-wrap;
  }}
  .fg-aviso {{ background: #FFF8E6; border: 1px solid #F0D9A0; border-radius: 10px; padding: 14px 18px; font-size: .88rem; color: #6B5310; margin-bottom: 18px; }}
  .fg-rodape {{ color: {CINZA_TEXTO}; font-size: .75rem; text-align: center; margin-top: 28px; padding-top: 14px; border-top: 1px solid {BORDA}; }}
  div[data-testid="stMetricValue"] {{ color: {MARINHO}; font-size: 1.5rem; }}

  @media (max-width: 780px) {{
    .fg-metrica {{ min-height: 82px; }}
    .fg-metrica strong {{ font-size: 1.25rem; }}
    .fg-fator-topo {{ align-items: flex-start; flex-direction: column; gap: 2px; }}
    .fg-fator-valor {{ white-space: normal; }}
  }}
</style>
"""
