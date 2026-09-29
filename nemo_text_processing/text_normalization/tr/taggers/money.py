# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Callable

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.tr.graph_utils import (
    NEMO_DIGIT,
    NEMO_SPACE,
    TR_COMMA_WORD,
    TR_DECIMAL_SEPARATOR,
    TR_MINUS_WORD,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.tr.taggers.decimal import get_quantity
from nemo_text_processing.text_normalization.tr.utils import get_abs_path, load_labels

Field = Callable[[str, "pynini.FstLike"], "pynini.FstLike"]


def _labelled(key: str, graph: "pynini.FstLike") -> "pynini.FstLike":
    return pynutil.insert(f"{key}: \"") + graph + pynutil.insert("\"")


def _bare(key: str, graph: "pynini.FstLike") -> "pynini.FstLike":
    return graph


def _attach_currency(amount: "pynini.FstLike", symbols: list, codes: list) -> "pynini.FstLike":
    """
    Attaches a currency's written forms to its amount graph: a symbol directly before
    or after the amount, or a code after it following one space.

    Args:
        amount: amount graph of the currency
        symbols: currency symbols, e.g. ["€"]
        codes: currency codes, e.g. ["EUR"]

    Returns a pynini.FstLike
    """
    symbol = pynutil.delete(pynini.union(*symbols))
    code = pynutil.delete(NEMO_SPACE + pynini.union(*codes))
    return pynini.union(symbol + amount, amount + pynini.union(symbol, code))


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying Turkish money, e.g.
        "100 TL" -> money { integer_part: "yüz" currency_maj: "lira" preserve_order: true }
        "€100", "100€", "100 EUR" -> money { integer_part: "yüz" currency_maj: "avro" preserve_order: true }
        "12,50₺" -> money { integer_part: "on iki" currency_maj: "lira" fractional_part: "elli"
            currency_min: "kuruş" preserve_order: true }
        "0,05 TL" -> money { fractional_part: "beş" currency_min: "kuruş" preserve_order: true }
        "12,345€" -> money { integer_part: "on iki" fractional_part: "üç yüz kırk beş" currency_maj: "avro"
            preserve_order: true }
        "1,5 milyon TL" -> money { integer_part: "bir" fractional_part: "beş" quantity: "milyon"
            currency_maj: "lira" preserve_order: true }

    Supported currencies are in data/money, keyed by ISO code: Turkish lira (₺, TL,
    TRY), US dollar ($, USD), euro (€, EUR) and pound sterling (£, GBP), read "lira",
    "dolar", "avro" and "sterlin", with minor units "kuruş", "sent", "sent" and "peni".
    A symbol attaches directly before or after the amount ("€100", "100€"); a code
    follows the amount after one space ("100 EUR"). All three give the same token.

    The fields follow the en and hu money schema and are always emitted in Turkish
    speaking order, number then currency, with ``preserve_order``, whichever side the
    symbol was written on. The currency is therefore known before the number is read,
    so the amount graph is built once per currency and wrapped with the three
    placements.

    Amounts:
        - integers are read by CardinalFst, grouping included: "1.000 TL" -> "bin lira";
        - one or two fractional digits are minor units, one digit being tenths:
          "12,5 TL" and "12,50 TL" -> "on iki lira elli kuruş", "12,05 TL" -> "on iki
          lira beş kuruş". Zeros after the first two digits change nothing, so
          "12,0500 TL" is "on iki lira beş kuruş";
        - a zero minor amount is dropped, "12,00 TL" -> "on iki lira", and a zero major
          amount is, "0,50 TL" -> "elli kuruş"; "0,00 TL" is "sıfır lira";
        - a fraction with a non-zero digit after the second is not a minor unit amount:
          it is read exactly as DecimalFst reads it, followed by the major unit, "12,345
          TL" -> "on iki virgül üç yüz kırk beş lira", with nothing rounded or dropped;
        - an amount with a quantity word is read by DecimalFst's quantity grammar,
          "1,5 milyon TL" -> "bir virgül beş milyon lira".

    A leading minus is read "eksi", before a prefix symbol or before the amount: "-€5",
    "-5€", "-5 EUR". Turkish does not pluralise the currency after a number.

    Not accepted: a space between a symbol and the amount ("€ 100", "100 €"), codes
    before the amount or attached to it ("EUR 100", "100EUR"), lower case codes,
    currency names ("100 lira"), other currencies, ranges, per unit forms and
    suffixed forms such as "100 TL'ye".

    For the classifier: "€100", "100€" and "100 EUR" are money, while "12,50" is a
    decimal, "%12,50" a percentage, "14.30" a time, "29.09.2026" a date and "3/4" a
    fraction; none of those is accepted here. "100 €" with a space is common in
    running text and is left for corpus driven evaluation.

    Args:
        cardinal: CardinalFst, supplies integer readings
        decimal: DecimalFst, supplies decimal and quantity readings
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        majors = dict(load_labels(get_abs_path("data/money/currency_major.tsv")))
        minors = dict(load_labels(get_abs_path("data/money/currency_minor.tsv")))
        symbols = defaultdict(list)
        for symbol, iso in load_labels(get_abs_path("data/money/currency_symbols.tsv")):
            symbols[iso].append(symbol)
        codes = defaultdict(list)
        for code, iso in load_labels(get_abs_path("data/money/currency_codes.tsv")):
            codes[iso].append(code)

        # "12", "1.234" -> reading. The only zero is "0": CardinalFst rejects "00".
        self.integer_graph = cardinal.graph
        integer_digits = pynini.project(cardinal.graph, "input")
        self.non_zero_integer_graph = pynini.compose(pynini.difference(integer_digits, "0"), cardinal.graph).optimize()

        # The written fraction, as the two digit minor amount it stands for: "5" -> "50",
        # "05" -> "05", "0500" -> "05". Zeros after the second digit are dropped.
        non_zero_digit = NEMO_DIGIT - "0"
        minor_digits = pynini.union(
            NEMO_DIGIT + NEMO_DIGIT + pynutil.delete(pynini.closure("0")),
            NEMO_DIGIT + pynutil.insert("0"),
        )
        two_digit_words = pynini.string_map([(f"{n:02d}", str(n)) for n in range(1, 100)]) @ cardinal.graph
        # "50" -> "elli", "05" -> "beş"; a fraction that is zero as a minor amount is
        # not in this graph.
        self.minor_graph = pynini.compose(minor_digits, two_digit_words).optimize()
        self.zero_minor_digits = pynini.project(pynini.compose(minor_digits, pynini.accep("00")), "input")

        # A fraction with a non-zero digit after the second one: not a minor amount.
        literal_fraction = (
            NEMO_DIGIT + NEMO_DIGIT + pynini.closure(NEMO_DIGIT) + non_zero_digit + pynini.closure(NEMO_DIGIT)
        )
        literal_inputs = integer_digits + TR_DECIMAL_SEPARATOR + literal_fraction

        # Token fields: the decimal and quantity grammars of DecimalFst.
        labelled_literal = pynini.compose(literal_inputs, decimal.final_graph_wo_sign)
        labelled_quantity = get_quantity(decimal.final_graph_wo_sign, decimal.cardinal_one_to_three_digits)

        # Bare readings, joining the same graphs DecimalFst joins.
        quantities = pynini.string_file(get_abs_path("data/numbers/quantities.tsv"))
        decimal_reading = cardinal.graph + pynini.cross(TR_DECIMAL_SEPARATOR, f" {TR_COMMA_WORD} ") + decimal.graph
        bare_literal = pynini.compose(literal_inputs, decimal_reading)
        bare_quantity = (
            pynini.union(decimal.cardinal_one_to_three_digits, decimal_reading) + pynini.accep(NEMO_SPACE) + quantities
        )

        self.final_graph = pynini.union(
            *[
                _attach_currency(
                    self._amount(_labelled, labelled_literal, labelled_quantity, majors[iso], minors[iso]),
                    symbols[iso],
                    codes[iso],
                )
                for iso in majors
            ]
        )
        negative = pynutil.insert("negative: ") + pynini.cross("-", "\"true\" ")
        self.final_graph = (
            pynini.closure(negative, 0, 1) + self.final_graph + pynutil.insert(" preserve_order: true")
        ).optimize()

        # "€12,50", "12,50€", "12,50 EUR" -> "on iki avro elli sent", the reading without
        # token fields.
        self.graph = pynini.union(
            *[
                _attach_currency(
                    self._amount(_bare, bare_literal, bare_quantity, majors[iso], minors[iso]),
                    symbols[iso],
                    codes[iso],
                )
                for iso in majors
            ]
        )
        self.graph = (pynini.closure(pynini.cross("-", f"{TR_MINUS_WORD} "), 0, 1) + self.graph).optimize()

        self.fst = self.add_tokens(self.final_graph).optimize()

    def _amount(
        self,
        field: Field,
        literal: "pynini.FstLike",
        quantity: "pynini.FstLike",
        major: str,
        minor: str,
    ) -> "pynini.FstLike":
        """
        Builds the amount graph of one currency: written amount -> fields in speaking
        order, each wrapped by ``field``.

        Args:
            field: takes a field name and a graph, returns the graph to emit
            literal: amounts read as a decimal, e.g. "12,345", in the form ``field`` emits
            quantity: amounts with a quantity word, e.g. "1,5 milyon", in the form ``field`` emits
            major: major unit word
            minor: minor unit word

        Returns a pynini.FstLike
        """
        delete_separator = pynutil.delete(TR_DECIMAL_SEPARATOR)
        major_unit = insert_space + field("currency_maj", pynutil.insert(major))
        minor_unit = insert_space + field("currency_min", pynutil.insert(minor))

        # 12, 12,00 -> integer_part currency_maj
        whole = field("integer_part", self.integer_graph) + pynini.closure(
            delete_separator + pynutil.delete(self.zero_minor_digits), 0, 1
        )
        # 12,50 -> integer_part currency_maj fractional_part currency_min
        major_and_minor = (
            field("integer_part", self.non_zero_integer_graph)
            + major_unit
            + delete_separator
            + insert_space
            + field("fractional_part", self.minor_graph)
            + minor_unit
        )
        # 0,50 -> fractional_part currency_min
        minor_only = pynutil.delete("0") + delete_separator + field("fractional_part", self.minor_graph) + minor_unit

        amount = pynini.union(
            whole + major_unit,
            major_and_minor,
            minor_only,
            literal + major_unit,
            quantity + major_unit,
        )
        return amount.optimize()
