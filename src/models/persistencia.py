"""
Persistência dos modelos treinados e do pré-processador ajustado.

Sem isto, cada experimento retreina do zero — dois minutos e meio de preparação
de dados mais o tempo de treino — e não existe artefato para o SHAP explicar,
para a interface de demonstração carregar, nem para o pipeline de agosto usar.

**O pré-processador é salvo junto com o modelo, e não é opcional.** O modelo
recebe uma matriz de 460 colunas construída por um `ColumnTransformer` ajustado
no treino: mediana de cada coluna numérica, categorias vistas no one-hot,
frequências de cada categoria. Uma transação nova só vira entrada válida
passando por *aquele* ajuste. Salvar o modelo sozinho produz um arquivo inútil,
e o erro só aparece na primeira previsão.

## Sobre formatos e pickle

O projeto adota, na camada RAG, a convenção de não usar pickle — carregar um
arquivo pickle executa código arbitrário, então um artefato de origem duvidosa
vira vetor de ataque. Aqui a convenção é seguida onde existe alternativa:

- **XGBoost** é gravado no formato nativo (`save_model`, JSON), sem pickle e
  portável entre versões da biblioteca;
- **pré-processador e modelos scikit-learn** usam `joblib`, que é baseado em
  pickle, porque não há formato nativo para `ColumnTransformer` nem para
  `RandomForestClassifier`. A consequência prática: **nunca carregue um
  artefato destes vindo de fonte que você não controla.**

Um `manifest.json` acompanha cada artefato com versões de biblioteca, hashes
dos arquivos e os metadados do treino, espelhando o padrão de
`src/rag/vector_store.py`. Ele é legível sem carregar o modelo — o que importa
quando a pergunta é "que execução gerou isto?" e não "quanto isto prevê?".
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

logger = logging.getLogger(__name__)

ARQUIVO_PREPROCESSADOR = "preprocessador.joblib"
ARQUIVO_MANIFESTO = "manifest.json"
ARQUIVO_MODELO_NATIVO = "modelo.json"
ARQUIVO_MODELO_JOBLIB = "modelo.joblib"


def _sha256(caminho: Path) -> str:
    digestor = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digestor.update(bloco)
    return digestor.hexdigest()


def _versoes() -> dict:
    versoes = {"numpy": np.__version__, "scikit-learn": sklearn.__version__}
    try:
        import xgboost
        versoes["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    return versoes


def _e_xgboost(modelo: Any) -> bool:
    return type(modelo).__module__.startswith("xgboost")


def salvar(
    diretorio: Path | str,
    preprocessador: Any,
    modelo: Any,
    metadados: dict | None = None,
) -> dict:
    """
    Grava pré-processador, modelo e manifesto em `diretorio`.

    `metadados` deve descrever a execução que produziu o modelo — linhas de
    treino, frações do split, tratamento de desbalanceamento, métricas obtidas.
    Sem isso, o artefato não responde de onde veio, e um modelo cuja origem não
    se sabe não pode ser citado em uma monografia.
    """
    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessador, diretorio / ARQUIVO_PREPROCESSADOR)

    if _e_xgboost(modelo):
        nome_modelo = ARQUIVO_MODELO_NATIVO
        modelo.save_model(diretorio / nome_modelo)
        formato = "xgboost_nativo"
    else:
        nome_modelo = ARQUIVO_MODELO_JOBLIB
        joblib.dump(modelo, diretorio / nome_modelo)
        formato = "joblib"

    manifesto = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tipo_do_modelo": type(modelo).__name__,
        "formato_do_modelo": formato,
        "arquivo_do_modelo": nome_modelo,
        "versoes": _versoes(),
        "hashes": {
            ARQUIVO_PREPROCESSADOR: _sha256(diretorio / ARQUIVO_PREPROCESSADOR),
            nome_modelo: _sha256(diretorio / nome_modelo),
        },
        "metadados": metadados or {},
    }
    (diretorio / ARQUIVO_MANIFESTO).write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Modelo %s gravado em %s (formato: %s)",
                manifesto["tipo_do_modelo"], diretorio, formato)
    return manifesto


def carregar(diretorio: Path | str, verificar_hashes: bool = True) -> tuple[Any, Any, dict]:
    """
    Carrega pré-processador, modelo e manifesto.

    Verifica os hashes por padrão: arquivo corrompido ou trocado depois da
    gravação produz erro aqui, e não uma previsão silenciosamente errada.
    Divergência de versão de biblioteca vira aviso, não erro — costuma
    funcionar, mas é a primeira suspeita quando um número não reproduz.
    """
    diretorio = Path(diretorio)
    manifesto = json.loads((diretorio / ARQUIVO_MANIFESTO).read_text(encoding="utf-8"))

    if verificar_hashes:
        for nome, esperado in manifesto["hashes"].items():
            obtido = _sha256(diretorio / nome)
            if obtido != esperado:
                raise ValueError(
                    f"hash de {nome} não confere com o manifesto: o arquivo foi "
                    f"alterado ou corrompido depois da gravação."
                )

    atuais = _versoes()
    for biblioteca, versao_gravada in manifesto["versoes"].items():
        if atuais.get(biblioteca) != versao_gravada:
            logger.warning(
                "%s foi gravado com a versão %s e está sendo carregado com %s",
                biblioteca, versao_gravada, atuais.get(biblioteca),
            )

    preprocessador = joblib.load(diretorio / ARQUIVO_PREPROCESSADOR)

    if manifesto["formato_do_modelo"] == "xgboost_nativo":
        from xgboost import XGBClassifier
        modelo = XGBClassifier()
        modelo.load_model(diretorio / manifesto["arquivo_do_modelo"])
    else:
        modelo = joblib.load(diretorio / manifesto["arquivo_do_modelo"])

    return preprocessador, modelo, manifesto


def prever(preprocessador: Any, modelo: Any, X) -> np.ndarray:
    """
    Probabilidade de fraude para transações novas.

    Existe para que o caminho correto — transformar com o pré-processador
    ajustado e só então prever — seja o caminho mais curto. Chamar
    `modelo.predict_proba` direto sobre um DataFrame cru é o erro natural, e
    ele não levanta exceção óbvia: produz números errados.
    """
    return modelo.predict_proba(preprocessador.transform(X))[:, 1]
