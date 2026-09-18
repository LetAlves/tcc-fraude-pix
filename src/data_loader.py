"""
Módulo de carregamento de dados para detecção de fraudes em cartão de crédito.

Estratégia de carregamento (em ordem de prioridade):
    1. CSVs já existentes em data/raw/ → evita downloads desnecessários.
    2. API do Kaggle              → download automático do dataset IEEE-CIS.

O dataset IEEE-CIS tem dois arquivos separados:
    - train_transaction.csv  (transações com ~590k linhas, ~394 colunas)
    - train_identity.csv     (informações de identidade, ~144k linhas)
Os dois são unidos pela coluna TransactionID.
"""

import os
import zipfile
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Carrega variáveis de ambiente (credenciais Kaggle do arquivo .env)
load_dotenv()

# Configuração básica de log — facilita diagnóstico sem poluir o notebook
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Caminhos base do projeto. ``TCC_DATA_RAW`` permite manter os CSVs grandes
# fora de pastas sincronizadas (como OneDrive) sem alterar codigo ou copiar o
# dataset. Quando ausente, o comportamento historico ``data/raw`` e preservado.
PROJETO_RAIZ = Path(__file__).resolve().parent.parent
DATA_RAW = Path(
    os.getenv("TCC_DATA_RAW", PROJETO_RAIZ / "data" / "raw")
).expanduser().resolve()

# Nome da competição Kaggle
KAGGLE_COMPETICAO = "ieee-fraud-detection"

# Nomes esperados dos arquivos locais
ARQUIVO_TRANSACOES = DATA_RAW / "train_transaction.csv"
ARQUIVO_IDENTIDADE = DATA_RAW / "train_identity.csv"


# ─── Funções auxiliares ────────────────────────────────────────────────────────

def _arquivos_existem() -> bool:
    """Verifica se os dois CSVs já foram baixados."""
    return ARQUIVO_TRANSACOES.exists() and ARQUIVO_IDENTIDADE.exists()


def _baixar_kaggle() -> None:
    """
    Faz download do dataset via API oficial do Kaggle.

    Pré-requisitos (qualquer uma das opções de autenticação):
        - Pacote `kaggle` instalado (v2+).
        - Token novo: KAGGLE_API_TOKEN no .env, ou arquivo ~/.kaggle/access_token.
        - Ou credenciais legadas: KAGGLE_USERNAME e KAGGLE_KEY no .env
          (ou em ~/.kaggle/kaggle.json).
        - Aceitar os termos da competição no site do Kaggle.

    Raises:
        EnvironmentError: se nenhuma credencial Kaggle estiver configurada.
        RuntimeError: se o download falhar por qualquer outro motivo.
    """
    tem_token_novo = bool(os.getenv("KAGGLE_API_TOKEN")) or (Path.home() / ".kaggle" / "access_token").exists()
    tem_credencial_legada = bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))

    if not tem_token_novo and not tem_credencial_legada:
        raise EnvironmentError(
            "Credenciais Kaggle não encontradas.\n"
            "Opção 1 (token novo): defina KAGGLE_API_TOKEN no .env, ou salve o token em "
            "~/.kaggle/access_token.\n"
            "Opção 2 (legado): crie um .env na raiz do projeto com:\n"
            "  KAGGLE_USERNAME=seu_usuario\n"
            "  KAGGLE_KEY=sua_chave_api\n"
            "Gere em: https://www.kaggle.com/settings → API"
        )

    # Importação tardia: evita falha de import se kaggle não estiver instalado
    try:
        from kaggle import KaggleApi
    except ImportError as e:
        raise ImportError("Instale o pacote kaggle: pip install kaggle") from e

    logger.info("Autenticando na API do Kaggle...")
    api = KaggleApi()
    api.authenticate()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    logger.info("Baixando dataset '%s'... (pode demorar alguns minutos)", KAGGLE_COMPETICAO)

    # force=False evita re-download se o zip já existir
    api.competition_download_files(
        competition=KAGGLE_COMPETICAO,
        path=str(DATA_RAW),
        force=False,
        quiet=False,
    )

    # O Kaggle baixa um único .zip — precisamos extrair
    zip_path = DATA_RAW / f"{KAGGLE_COMPETICAO}.zip"
    if zip_path.exists():
        logger.info("Extraindo arquivos do zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_RAW)
        zip_path.unlink()  # remove o zip após extração para economizar espaço
        logger.info("Extração concluída.")
    else:
        logger.warning("Arquivo zip não encontrado após download. Verifique manualmente em %s", DATA_RAW)


# ─── Função principal ──────────────────────────────────────────────────────────

def carregar_dados(
    baixar_se_ausente: bool = True,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega os dados de treino do dataset IEEE-CIS Fraud Detection.

    Fluxo:
        1. Se os CSVs já existem em data/raw/, lê direto.
        2. Se não existem e baixar_se_ausente=True, faz download via Kaggle.
        3. Retorna transações, identidade e o DataFrame mesclado (join left).

    Args:
        baixar_se_ausente: Se True (padrão), tenta baixar automaticamente
            quando os arquivos não forem encontrados localmente.
        nrows: Número de linhas a carregar (útil para testes rápidos).
            None carrega o dataset completo.

    Returns:
        Tuple com três DataFrames:
            - df_transacoes: dados de transação (variável alvo = isFraud)
            - df_identidade: dados de identidade do dispositivo/usuário
            - df_merged: junção left de transacoes + identidade por TransactionID

    Raises:
        FileNotFoundError: se os arquivos não existem e baixar_se_ausente=False.
        EnvironmentError: se as credenciais Kaggle estiverem ausentes ao tentar download.
    """
    if not _arquivos_existem():
        if not baixar_se_ausente:
            raise FileNotFoundError(
                f"Arquivos não encontrados em {DATA_RAW}.\n"
                "Execute com baixar_se_ausente=True ou baixe manualmente."
            )
        logger.info("Arquivos locais não encontrados. Iniciando download...")
        _baixar_kaggle()

    logger.info("Carregando train_transaction.csv...")
    df_transacoes = pd.read_csv(ARQUIVO_TRANSACOES, nrows=nrows)

    logger.info("Carregando train_identity.csv...")
    df_identidade = pd.read_csv(ARQUIVO_IDENTIDADE, nrows=nrows)

    logger.info(
        "Dados carregados — transações: %s | identidade: %s",
        df_transacoes.shape,
        df_identidade.shape,
    )

    # Left join: mantém todas as transações (nem toda transação tem registro de identidade)
    logger.info("Mesclando DataFrames por TransactionID (left join)...")
    df_merged = df_transacoes.merge(df_identidade, on="TransactionID", how="left")

    logger.info("DataFrame final: %s", df_merged.shape)
    return df_transacoes, df_identidade, df_merged


# ─── Execução direta (smoke test) ─────────────────────────────────────────────

if __name__ == "__main__":
    # Carrega apenas 1000 linhas para validação rápida
    trans, ident, merged = carregar_dados(nrows=1_000)
    print("\nTransações:", trans.shape)
    print("Identidade:", ident.shape)
    print("Merged:    ", merged.shape)
    print("\nPrimeiras colunas:", list(merged.columns[:10]))
