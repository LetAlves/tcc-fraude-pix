"""Testes do corte temporal (`dividir_temporal`).

A propriedade central sob teste: transações que compartilham `TransactionDT`
formam um bloco indivisível e nunca podem ficar em conjuntos diferentes.
Cortar no meio de um bloco é vazamento temporal — o modelo veria, no treino,
eventos do mesmo instante que precisa prever.

No IEEE-CIS, 2,9% das linhas compartilham timestamp com outra transação, e no
recorte 70/15/15 nenhum bloco chega a ser atravessado. Esses testes garantem
que isso não é sorte das frações escolhidas.
"""

import unittest

import numpy as np
import pandas as pd

from src.features.preprocessor import dividir_temporal


def _quadro(tempos, coluna_tempo="TransactionDT"):
    """DataFrame mínimo com um identificador por linha, para rastrear perdas."""
    return pd.DataFrame({coluna_tempo: tempos, "id_linha": range(len(tempos))})


# Blocos de tamanhos irregulares, escolhidos para que o corte ingênuo por
# posição (70 e 85 de 100 linhas) caia DENTRO de um bloco de empate.
TEMPOS_COM_EMPATE_NA_FRONTEIRA = [0] * 13 + [1] * 27 + [2] * 35 + [3] * 15 + [4] * 10


class CorteTemporalTest(unittest.TestCase):

    def test_premissa_do_cenario_o_corte_ingenuo_partiria_um_bloco(self) -> None:
        """Garante que o cenário de teste exercita mesmo o caso problemático."""
        tempos = np.array(TEMPOS_COM_EMPATE_NA_FRONTEIRA)
        self.assertEqual(len(tempos), 100)
        # corte ingênuo de treino cairia em 70, no meio do bloco do instante 2
        self.assertEqual(tempos[69], tempos[70])
        # corte ingênuo de validação cairia em 85, no meio do bloco do instante 3
        self.assertEqual(tempos[84], tempos[85])

    def test_nenhum_instante_aparece_em_dois_conjuntos(self) -> None:
        treino, val, teste = dividir_temporal(_quadro(TEMPOS_COM_EMPATE_NA_FRONTEIRA))

        t_treino = set(treino["TransactionDT"])
        t_val = set(val["TransactionDT"])
        t_teste = set(teste["TransactionDT"])

        self.assertEqual(t_treino & t_val, set(), "instante dividido entre treino e validação")
        self.assertEqual(t_val & t_teste, set(), "instante dividido entre validação e teste")
        self.assertEqual(t_treino & t_teste, set(), "instante dividido entre treino e teste")

    def test_preserva_todas_as_linhas_sem_duplicar(self) -> None:
        quadro = _quadro(TEMPOS_COM_EMPATE_NA_FRONTEIRA)
        treino, val, teste = dividir_temporal(quadro)

        self.assertEqual(len(treino) + len(val) + len(teste), len(quadro))

        recuperados = sorted(
            list(treino["id_linha"]) + list(val["id_linha"]) + list(teste["id_linha"])
        )
        self.assertEqual(recuperados, list(range(len(quadro))))

    def test_ordem_cronologica_entre_os_conjuntos(self) -> None:
        treino, val, teste = dividir_temporal(_quadro(TEMPOS_COM_EMPATE_NA_FRONTEIRA))

        self.assertLess(treino["TransactionDT"].max(), val["TransactionDT"].min())
        self.assertLess(val["TransactionDT"].max(), teste["TransactionDT"].min())

    def test_entrada_desordenada_produz_os_mesmos_conjuntos(self) -> None:
        """A ordem de chegada não muda *quais* linhas caem em cada conjunto.

        A ordem *interna* de cada conjunto pode mudar: o mergesort é estável em
        relação à entrada, então linhas de mesmo instante preservam a ordem em
        que chegaram. Como os cortes caem em fronteiras de instante, o bloco
        inteiro vai para o mesmo lado de qualquer forma.
        """
        quadro = _quadro(TEMPOS_COM_EMPATE_NA_FRONTEIRA)
        embaralhado = quadro.sample(frac=1.0, random_state=7)

        esperado = dividir_temporal(quadro)
        obtido = dividir_temporal(embaralhado)

        for parte_esperada, parte_obtida in zip(esperado, obtido):
            self.assertEqual(
                set(parte_esperada["id_linha"]), set(parte_obtida["id_linha"])
            )

    def test_fracoes_sao_aproximadas_e_o_desvio_cabe_no_bloco(self) -> None:
        """Com muitos instantes distintos, as proporções ficam próximas do pedido."""
        # 1000 linhas, blocos de 4 linhas por instante
        treino, val, teste = dividir_temporal(_quadro(np.repeat(np.arange(250), 4)))
        n = len(treino) + len(val) + len(teste)

        self.assertAlmostEqual(len(treino) / n, 0.70, delta=0.01)
        self.assertAlmostEqual(len(val) / n, 0.15, delta=0.01)
        self.assertAlmostEqual(len(teste) / n, 0.15, delta=0.01)

    def test_um_unico_instante_nao_forma_tres_periodos(self) -> None:
        with self.assertRaises(ValueError) as contexto:
            dividir_temporal(_quadro([7] * 100))
        self.assertIn("três períodos", str(contexto.exception))

    def test_instantes_insuficientes_para_as_fracoes(self) -> None:
        """Dois instantes não bastam para três períodos, por mais linhas que haja."""
        with self.assertRaises(ValueError):
            dividir_temporal(_quadro([1] * 50 + [2] * 50))

    def test_fracoes_invalidas(self) -> None:
        with self.assertRaises(ValueError):
            dividir_temporal(_quadro(np.arange(100)), frac_treino=0.9, frac_val=0.2)


if __name__ == "__main__":
    unittest.main()
