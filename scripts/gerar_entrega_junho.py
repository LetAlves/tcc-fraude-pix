"""Gera a entrega parcial de junho com estatísticas reais e baixo uso de memória."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_REGISTRY = PROJECT_ROOT / "config" / "pix_feature_registry.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "pessoa_2"
    / "junho"
    / "04_entrega_parcial_eda_features.md"
)

TRANSACTION_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "ProductCD",
]
IDENTITY_COLUMNS = ["TransactionID", "DeviceType", "DeviceInfo"]


def calcular_split_temporal(transactions: pd.DataFrame) -> dict[str, object]:
    """Resume o corte 70/15/15 sem separar transações com o mesmo instante."""

    required_columns = {"TransactionID", "TransactionDT", "isFraud"}
    missing_columns = required_columns - set(transactions.columns)
    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias para o split temporal ausentes: "
            + ", ".join(sorted(missing_columns))
        )
    if transactions.empty:
        raise ValueError("O split temporal exige pelo menos uma transação.")
    if transactions["TransactionDT"].isna().any():
        raise ValueError("TransactionDT não pode conter nulos no split temporal.")

    ordered = transactions.sort_values(
        ["TransactionDT", "TransactionID"], kind="mergesort"
    ).reset_index(drop=True)
    timestamps = ordered["TransactionDT"].to_numpy()
    total = len(ordered)

    def advance_past_ties(boundary: int) -> int:
        while 0 < boundary < total and timestamps[boundary] == timestamps[boundary - 1]:
            boundary += 1
        return boundary

    train_end = advance_past_ties(int(total * 0.70))
    validation_end = advance_past_ties(max(train_end, int(total * 0.85)))
    frames = {
        "treino": ordered.iloc[:train_end],
        "validação": ordered.iloc[train_end:validation_end],
        "teste": ordered.iloc[validation_end:],
    }
    if any(frame.empty for frame in frames.values()):
        raise ValueError(
            "O dataset não permite formar três partições temporais não vazias sem "
            "separar timestamps empatados."
        )

    partitions = []
    for name, frame in frames.items():
        frauds = int(frame["isFraud"].eq(1).sum())
        partitions.append(
            {
                "name": name,
                "rows": int(len(frame)),
                "frauds": frauds,
                "fraud_rate": frauds / len(frame),
            }
        )

    return {
        "partitions": partitions,
        "repeated_timestamp_rate": float(
            ordered["TransactionDT"].duplicated(keep="first").mean()
        ),
        "rows_in_repeated_timestamp_groups_rate": float(
            ordered["TransactionDT"].duplicated(keep=False).mean()
        ),
        "boundaries_preserve_timestamps": bool(
            frames["treino"]["TransactionDT"].max()
            < frames["validação"]["TransactionDT"].min()
            and frames["validação"]["TransactionDT"].max()
            < frames["teste"]["TransactionDT"].min()
        ),
    }


def calcular_estatisticas(data_dir: Path) -> dict[str, object]:
    """Calcula somente as estatísticas necessárias para a entrega parcial."""

    transaction_path = data_dir / "train_transaction.csv"
    identity_path = data_dir / "train_identity.csv"
    missing = [path for path in (transaction_path, identity_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Arquivos obrigatórios ausentes: " + ", ".join(str(path) for path in missing)
        )

    transaction_column_count = len(pd.read_csv(transaction_path, nrows=0).columns)
    identity_column_count = len(pd.read_csv(identity_path, nrows=0).columns)

    transactions = pd.read_csv(transaction_path, usecols=TRANSACTION_COLUMNS)
    identities = pd.read_csv(identity_path, usecols=IDENTITY_COLUMNS)

    target_counts = transactions["isFraud"].value_counts().sort_index()
    total = int(len(transactions))
    frauds = int(target_counts.get(1, 0))
    legitimate = int(target_counts.get(0, 0))
    identity_ids = set(identities["TransactionID"].dropna().astype("int64"))
    identity_coverage = transactions["TransactionID"].isin(identity_ids).mean()

    amount_by_class = (
        transactions.groupby("isFraud")["TransactionAmt"]
        .agg(["median", "mean", "max"])
        .round(4)
        .to_dict(orient="index")
    )
    device_type_counts = (
        identities["DeviceType"]
        .fillna("__MISSING__")
        .value_counts()
        .to_dict()
    )

    return {
        "transaction_rows": total,
        "transaction_columns": transaction_column_count,
        "identity_rows": int(len(identities)),
        "identity_columns": identity_column_count,
        "legitimate": legitimate,
        "frauds": frauds,
        "fraud_rate": frauds / total if total else 0.0,
        "imbalance_ratio": legitimate / frauds if frauds else None,
        "identity_coverage": float(identity_coverage),
        "transaction_span_days": float(
            (transactions["TransactionDT"].max() - transactions["TransactionDT"].min())
            / 86_400
        ),
        "transaction_amount_by_class": amount_by_class,
        "card1_unique": int(transactions["card1"].nunique(dropna=True)),
        "product_counts": transactions["ProductCD"].value_counts().to_dict(),
        "device_type_counts": device_type_counts,
        "device_info_unique": int(identities["DeviceInfo"].nunique(dropna=True)),
        "temporal_split": calcular_split_temporal(transactions),
    }


def carregar_registro(registry_path: Path) -> dict[str, object]:
    with registry_path.open(encoding="utf-8") as registry_file:
        registry = json.load(registry_file)
    if registry.get("decision_status") != "approved_and_implemented":
        raise ValueError(
            "O registro deve corresponder às features aprovadas e implementadas pela dupla."
        )
    return registry


def _format_count(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def render_report(stats: dict[str, object], registry: dict[str, object]) -> str:
    amount_by_class = stats["transaction_amount_by_class"]
    legitimate_amount = amount_by_class.get(0, amount_by_class.get("0", {}))
    fraud_amount = amount_by_class.get(1, amount_by_class.get("1", {}))
    features = registry["features"]
    temporal_split = stats["temporal_split"]

    feature_rows = "\n".join(
        "| `{feature_id}` | {label} | {sources} | {warning} |".format(
            feature_id=feature["feature_id"],
            label=feature["concept_label"],
            sources=", ".join(f"`{column}`" for column in feature["source_columns"]),
            warning=feature["warning"],
        )
        for feature in features
    )
    split_rows = "\n".join(
        "| {name} | {rows} | {frauds} | {fraud_rate:.3%} |".format(
            name=partition["name"],
            rows=_format_count(partition["rows"]),
            frauds=_format_count(partition["frauds"]),
            fraud_rate=partition["fraud_rate"],
        )
        for partition in temporal_split["partitions"]
    )
    rates = [
        partition["fraud_rate"] for partition in temporal_split["partitions"]
    ]
    max_rate_difference_pp = (max(rates) - min(rates)) * 100

    return f"""# Entrega parcial 1 — EDA e features Pix simuladas

Data de geração: **18/08/2026**.  
Fonte dos números: CSVs locais do treino IEEE-CIS, lidos por `scripts/gerar_entrega_junho.py`.

## Estado da entrega

Este documento consolida a **EDA executada e as features aprovadas pela dupla em 16/08/2026**. O IEEE-CIS não contém transações Pix: os atributos abaixo são proxies analíticos, implementados em `src/features/pix_features.py`, e não equivalências de campos Pix reais. O envio desta entrega parcial ao orientador ainda deve ser registrado.

## Evidência reproduzida dos dados

| Medida | Resultado |
|---|---:|
| `train_transaction.csv` | {_format_count(stats['transaction_rows'])} linhas × {stats['transaction_columns']} colunas |
| `train_identity.csv` | {_format_count(stats['identity_rows'])} linhas × {stats['identity_columns']} colunas |
| transações legítimas | {_format_count(stats['legitimate'])} |
| fraudes marcadas | {_format_count(stats['frauds'])} |
| prevalência de `isFraud=1` | {stats['fraud_rate']:.3%} |
| razão legítima:fraude | {stats['imbalance_ratio']:.1f}:1 |
| cobertura de identidade no treino | {stats['identity_coverage']:.3%} |
| extensão relativa de `TransactionDT` | {stats['transaction_span_days']:.1f} dias |
| valores distintos de `card1` | {_format_count(stats['card1_unique'])} |
| valores distintos não nulos de `DeviceInfo` | {_format_count(stats['device_info_unique'])} |

`TransactionAmt` apresenta mediana de **{legitimate_amount.get('median', float('nan')):.2f}** nas linhas legítimas e **{fraud_amount.get('median', float('nan')):.2f}** nas linhas marcadas como fraude. Essa diferença descritiva não estabelece causalidade nem define sozinha um limiar de risco.

## Implicações para o baseline

1. A prevalência próxima de 3,5% torna acurácia inadequada como métrica principal; AUC-PR, precisão, recall, F1 e matriz de confusão devem ser relatados.
2. SMOTE, se comparado, deve ser ajustado **somente nos folds/partição de treino** por meio de pipeline do `imbalanced-learn`.
3. `TransactionDT` preserva ordem relativa, mas não revela data, fuso ou hora civil; a avaliação deve respeitar a ordem sem chamar o ciclo de “horário Pix”.
4. A identidade cobre apenas parte das transações; ausência de `DeviceInfo` precisa ser tratada explicitamente e não pode ser interpretada como fraude.
5. Estatísticas históricas por `card1` devem usar apenas eventos anteriores. Cálculo global antes da divisão de treino/validação/teste causaria vazamento.

## Protocolo temporal aprovado

A avaliação principal usa corte temporal 70/15/15, com as transações mais antigas no treino e as mais recentes no teste. As fronteiras são deslocadas quando necessário para manter todas as transações com o mesmo `TransactionDT` na mesma partição.

| Conjunto | Linhas | Fraudes | Taxa de fraude |
|---|---:|---:|---:|
{split_rows}

A diferença máxima entre as taxas é de **{max_rate_difference_pp:.3f} ponto percentual**. Isso representa proporções semelhantes entre as partições, não balanceamento entre as classes: fraude continua sendo rara. Ocorrências repetidas de `TransactionDT`, desconsiderando a primeira de cada valor, correspondem a **{temporal_split['repeated_timestamp_rate']:.1%}** das linhas; ao contar todas as linhas pertencentes a grupos de timestamps repetidos, a proporção é **{temporal_split['rows_in_repeated_timestamp_groups_rate']:.1%}**. Fronteiras sem timestamps compartilhados: **{'sim' if temporal_split['boundaries_preserve_timestamps'] else 'não'}**. Qualquer divisão aleatória estratificada será apenas uma análise complementar.

## Registro aprovado de features

Status do registro: `{registry['decision_status']}`.

| ID | Conceito controlado | Colunas-fonte | Advertência obrigatória |
|---|---|---|---|
{feature_rows}

O arquivo-fonte completo, incluindo fórmula, janela, tratamento de nulos, controle de vazamento e tags do RAG, está em `config/pix_feature_registry.json`.

## O que esta entrega não afirma

- não afirma que `card1` é conta, chave ou destinatário Pix;
- não converte `TransactionDT` em horário local;
- não transforma raridade de dispositivo em “dispositivo comprometido”;
- não usa colunas `C*`, `D*`, `M*`, `V*` ou `id_*` como conceitos Pix individuais;
- não considera o baseline concluído antes da execução dos modelos, publicação das métricas e revisão da dupla.

## Pendências para concluir a entrega ao orientador

- [ ] decidir comparação entre peso de classe e SMOTE sem vazamento;
- [x] definir split temporal 70/15/15, preservando empates de `TransactionDT`;
- [ ] registrar seeds do experimento;
- [ ] executar e revisar o baseline da Pessoa 1;
- [ ] registrar a data de apresentação desta entrega ao orientador.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    stats = calcular_estatisticas(args.data_dir)
    registry = carregar_registro(args.registry)
    report = render_report(stats, registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Entrega parcial gerada em: {args.output}")


if __name__ == "__main__":
    main()
