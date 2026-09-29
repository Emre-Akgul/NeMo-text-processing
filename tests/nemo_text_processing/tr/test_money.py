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

"""
Direct tests for the Turkish money tagger and verbalizer.

Every money token carries ``preserve_order``. ``_readings`` still runs the tagged token
through the token parser and ``Normalizer._permute``, so these tests exercise the path
the Normalizer will.
"""

import pynini
import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.token_parser import TokenParser
from nemo_text_processing.text_normalization.tr.taggers.cardinal import CardinalFst as CardinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.date import DateFst as DateTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.decimal import DecimalFst as DecimalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.fraction import FractionFst as FractionTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.money import MoneyFst as MoneyTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.ordinal import OrdinalFst as OrdinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.percentage import PercentageFst as PercentageTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.time import TimeFst as TimeTaggerFst
from nemo_text_processing.text_normalization.tr.verbalizers.cardinal import CardinalFst as CardinalVerbalizerFst
from nemo_text_processing.text_normalization.tr.verbalizers.decimal import DecimalFst as DecimalVerbalizerFst
from nemo_text_processing.text_normalization.tr.verbalizers.money import MoneyFst as MoneyVerbalizerFst

from ..utils import parse_test_case_file

_TEST_CASES = 'tr/data_text_normalization/test_cases_money.txt'

# symbol, codes, major unit, minor unit
_CURRENCIES = [
    ("₺", ["TL", "TRY"], "lira", "kuruş"),
    ("$", ["USD"], "dolar", "sent"),
    ("€", ["EUR"], "avro", "sent"),
    ("£", ["GBP"], "sterlin", "peni"),
]

# Amounts every written form of every currency must agree on.
_AMOUNTS = ["0", "1", "25", "100", "1.234", "12,50", "12,5", "12,05", "12,00", "0,05", "0,50", "0,00"]
_AMOUNTS += ["12,0500", "12,345", "12,005", "1,2345", "1 milyon", "1,5 milyon"]

_INVALID = [
    ("",),
    ("€",),
    ("$",),
    ("₺",),
    ("TL",),
    ("USD",),
    ("€ 25",),  # a symbol attaches directly
    ("25 €",),
    ("₺ 100",),
    ("100 ₺",),
    ("1 milyon ₺",),
    ("EUR 25",),  # a code follows the amount after one space
    ("EUR25",),
    ("25EUR",),
    ("100TL",),
    ("TL 100",),
    ("100  EUR",),
    ("100 TL ",),
    ("25 eur",),  # codes are upper case
    ("25 usd",),
    ("25 tl",),
    ("25 Tl",),
    ("€05",),  # leading zeros, as CardinalFst
    ("05€",),
    ("05 EUR",),
    ("€00",),
    ("€1.23",),  # malformed grouping, as CardinalFst
    ("1.23€",),
    ("1..000€",),
    ("01.000 TL",),
    ("€1,",),
    ("€,50",),
    ("€1,,5",),
    ("€1,2.3",),
    ("12.50€",),  # English decimal point
    ("¥100",),  # unsupported currencies
    ("100¥",),
    ("100 JPY",),
    ("100 lira",),  # currency names are not symbols
    ("100 dolar",),
    ("100 avro",),
    ("100 sterlin",),
    ("100€'ya",),  # suffixed forms are deferred
    ("100 TL'ye",),
    ("50 TL'den",),
    ("20 USD'ye",),
    ("10 EUR'luk",),
    ("25€'ya",),
    ("100₺'ye",),
    ("€-100",),  # only a leading minus
    ("100-€",),
    ("100€-",),
    ("+100€",),
    ("--100€",),
    ("-€-100",),
    ("€100€",),
    ("€100 EUR",),
    ("$100€",),
    ("10-20 TL",),
    ("100 TL/kg",),
    ("12,50",),
    ("100",),
    ("%25",),
    ("€%25",),
    ("€1 bin milyon",),
    ("€1000 milyon",),  # quantity integers are one to three digits, as DecimalFst
]


class TestMoney:

    cardinal = CardinalTaggerFst()
    decimal = DecimalTaggerFst(cardinal=cardinal)
    tagger = MoneyTaggerFst(cardinal=cardinal, decimal=decimal)
    cardinal_verbalizer = CardinalVerbalizerFst()
    decimal_verbalizer = DecimalVerbalizerFst(cardinal=cardinal_verbalizer)
    verbalizer = MoneyVerbalizerFst(decimal=decimal_verbalizer)
    tagger_fst = tagger.fst
    verbalizer_fst = verbalizer.fst

    class _Permuter:
        _permute = Normalizer._permute

    _permuter = _Permuter()

    @classmethod
    def _readings(cls, written, tagger_fst=None, verbalizer_fst=None):
        """Every reading the tagger plus the verbalizer admit, via the token parser and
        the Normalizer's own field permutation."""
        tagger_fst = tagger_fst or cls.tagger_fst
        verbalizer_fst = verbalizer_fst or cls.verbalizer_fst
        tagged = rewrite.top_rewrite(written, tagger_fst)
        parser = TokenParser()
        parser(f"tokens {{ {tagged} }}")
        readings = set()
        for serialized in cls._permuter._permute(parser.parse()[0]["tokens"]):
            try:
                readings.add(rewrite.top_rewrite(serialized.strip(), verbalizer_fst))
            except rewrite.Error:
                pass
        return readings

    def _normalize(self, written):
        readings = self._readings(written)
        assert len(readings) == 1, f"input: {written} produced {readings}"
        return readings.pop()

    def _tag(self, written):
        tags = rewrite.rewrites(written, self.tagger_fst)
        assert len(tags) == 1, f"input: {written} produced {tags}"
        return tags[0]

    def _decimal_reading(self, number):
        return self._readings(number, self.decimal.fst, self.decimal_verbalizer.fst).pop()

    @parameterized.expand(parse_test_case_file(_TEST_CASES))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        assert self._normalize(test_input) == expected, f"input: {test_input}"

    @parameterized.expand(parse_test_case_file(_TEST_CASES))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_deterministic_single_transduction(self, test_input, expected):
        self._tag(test_input)
        assert self._readings(test_input) == {expected}, f"input: {test_input}"

    @parameterized.expand(parse_test_case_file(_TEST_CASES))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_bare_graph_matches_tagger_and_verbalizer(self, test_input, expected):
        assert rewrite.rewrites(test_input, self.tagger.graph) == [expected]

    @parameterized.expand([(amount,) for amount in _AMOUNTS])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_prefix_postfix_and_code_give_the_same_token(self, amount):
        for symbol, codes, _, _ in _CURRENCIES:
            written = [f"{symbol}{amount}", f"{amount}{symbol}"] + [f"{amount} {code}" for code in codes]
            tokens = {self._tag(w) for w in written}
            assert len(tokens) == 1, tokens
            readings = {rewrite.top_rewrite(w, self.tagger.graph) for w in written}
            assert readings == {self._normalize(written[0])}, readings
            negative = {self._tag(f"-{w}") for w in written}
            assert len(negative) == 1, negative

    @parameterized.expand(_CURRENCIES)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_currency_words(self, symbol, codes, major, minor):
        assert self._normalize(f"{symbol}25") == f"yirmi beş {major}"
        assert self._normalize(f"{symbol}12,50") == f"on iki {major} elli {minor}"
        assert self._normalize(f"{symbol}0,01") == f"bir {minor}"
        # no plural switching
        assert self._normalize(f"{symbol}1") == f"bir {major}"
        assert self._normalize(f"{symbol}2") == f"iki {major}"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_one_fractional_digit_is_tenths(self):
        for digit in range(1, 10):
            assert self._tag(f"12,{digit}€") == self._tag(f"12,{digit}0€")
        assert self._normalize("12,5€") == self._normalize("12,50€") == "on iki avro elli sent"
        assert self._normalize("12,05€") != self._normalize("12,50€")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_zero_minor_is_dropped(self):
        for written in ["12€", "12,0€", "12,00€", "12,000€", "12,0000€"]:
            assert self._tag(written) == 'money { integer_part: "on iki" currency_maj: "avro" preserve_order: true }'

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_zero_major_is_dropped(self):
        assert self._tag("0,50 TL") == 'money { fractional_part: "elli" currency_min: "kuruş" preserve_order: true }'
        assert self._normalize("0,5 TL") == "elli kuruş"
        assert self._normalize("0,0 TL") == "sıfır lira"
        assert self._normalize("0 TL") == "sıfır lira"

    @parameterized.expand([("12,345",), ("12,005",), ("1,2345",), ("0,001",), ("12,3450",), ("1.234,567",)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_more_than_two_digits_read_as_decimal(self, number):
        """Nothing is rounded, trimmed or turned into a minor unit: the amount reads
        exactly as DecimalFst reads the number."""
        for symbol, codes, major, _ in _CURRENCIES:
            expected = f"{self._decimal_reading(number)} {major}"
            assert self._normalize(f"{symbol}{number}") == expected
            assert self._normalize(f"{number} {codes[0]}") == expected

    @parameterized.expand([("1 milyon",), ("2 milyar",), ("1,5 milyon",), ("12,25 milyon",), ("150 bin",)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_quantity_reads_as_decimal(self, number):
        for symbol, codes, major, _ in _CURRENCIES:
            expected = f"{self._decimal_reading(number)} {major}"
            for written in [f"{symbol}{number}", f"{number}{symbol}", f"{number} {codes[0]}"]:
                assert self._normalize(written) == expected, written

    @parameterized.expand(_INVALID)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_invalid_input_is_rejected(self, test_input):
        with pytest.raises(rewrite.Error):
            rewrite.top_rewrite(test_input, self.tagger_fst)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_token_schema(self):
        assert self._tag("25 TL") == 'money { integer_part: "yirmi beş" currency_maj: "lira" preserve_order: true }'
        assert self._tag("12,50 TL") == (
            'money { integer_part: "on iki" currency_maj: "lira" fractional_part: "elli" currency_min: "kuruş"'
            ' preserve_order: true }'
        )
        assert self._tag("0,50 TL") == 'money { fractional_part: "elli" currency_min: "kuruş" preserve_order: true }'
        assert self._tag("12,345 TL") == (
            'money { integer_part: "on iki" fractional_part: "üç yüz kırk beş" currency_maj: "lira"'
            ' preserve_order: true }'
        )
        assert self._tag("1,5 milyon TL") == (
            'money { integer_part: "bir" fractional_part: "beş" quantity: "milyon" currency_maj: "lira"'
            ' preserve_order: true }'
        )
        assert self._tag("-5 TL") == (
            'money { negative: "true" integer_part: "beş" currency_maj: "lira" preserve_order: true }'
        )
        assert self._tag("€12,50") == self._tag("12,50€") == self._tag("12,50 EUR")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_reusable_subgraphs_are_exposed(self):
        assert rewrite.top_rewrite("1.234", self.tagger.integer_graph) == "bin iki yüz otuz dört"
        assert rewrite.top_rewrite("5", self.tagger.minor_graph) == "elli"
        assert rewrite.top_rewrite("05", self.tagger.minor_graph) == "beş"
        assert rewrite.top_rewrite("0500", self.tagger.minor_graph) == "beş"
        assert rewrite.top_rewrite("25€", self.tagger.final_graph) == (
            'integer_part: "yirmi beş" currency_maj: "avro" preserve_order: true'
        )
        assert rewrite.top_rewrite("25€", self.tagger.graph) == "yirmi beş avro"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_generated_cross_check(self):
        """Every minor amount over representative major amounts, in every written form."""
        words = {n: rewrite.top_rewrite(str(n), self.cardinal.graph) for n in range(0, 100)}
        for symbol, codes, major, minor in _CURRENCIES:
            for integer in [0, 1, 12]:
                for cents in range(0, 100):
                    if cents == 0:
                        expected = f"{words[integer]} {major}"
                    elif integer == 0:
                        expected = f"{words[cents]} {minor}"
                    else:
                        expected = f"{words[integer]} {major} {words[cents]} {minor}"
                    amount = f"{integer},{cents:02d}"
                    for written in [f"{symbol}{amount}", f"{amount}{symbol}", f"{amount} {codes[0]}"]:
                        assert rewrite.rewrites(written, self.tagger.graph) == [expected], written
                    assert rewrite.top_rewrite(
                        rewrite.top_rewrite(f"{symbol}{amount}", self.tagger_fst), self.verbalizer_fst
                    ) == (expected)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_graph_has_no_state_explosion(self):
        assert self.tagger_fst.num_states() < 25000, self.tagger_fst.num_states()
        assert self.verbalizer_fst.num_states() < 500, self.verbalizer_fst.num_states()


class TestMoneySeparation:
    """Money against every earlier Turkish grammar."""

    cardinal = CardinalTaggerFst()
    decimal = DecimalTaggerFst(cardinal=cardinal)
    money_fst = MoneyTaggerFst(cardinal=cardinal, decimal=decimal).fst
    others = {
        "cardinal": cardinal.fst,
        "decimal": decimal.fst,
        "fraction": FractionTaggerFst(cardinal=cardinal).fst,
        "date": DateTaggerFst(cardinal=cardinal).fst,
        "time": TimeTaggerFst(cardinal=cardinal).fst,
        "ordinal": OrdinalTaggerFst(cardinal=cardinal).fst,
        "percentage": PercentageTaggerFst(cardinal=cardinal, decimal=decimal).fst,
    }

    @staticmethod
    def _accepts(fst, written):
        try:
            rewrite.top_rewrite(written, fst)
            return True
        except rewrite.Error:
            return False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_no_input_is_shared_with_another_grammar(self):
        """Exact, over all inputs: the input languages do not intersect."""
        money_inputs = pynini.project(self.money_fst, "input").optimize()
        for name, fst in self.others.items():
            shared = pynini.intersect(money_inputs, pynini.project(fst, "input").optimize()).optimize()
            assert shared.num_states() == 0, name

    @parameterized.expand([("€100",), ("100€",), ("100 EUR",), ("12,50€",), ("€14,30",), ("14,30€",)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_money_belongs_to_no_other_grammar(self, test_input):
        assert self._accepts(self.money_fst, test_input)
        for name, fst in self.others.items():
            assert not self._accepts(fst, test_input), name

    @parameterized.expand(
        [("100", "cardinal"), ("12,50", "decimal"), ("%12,50", "percentage"), ("%25", "percentage")]
        + [("14.30", "time"), ("29.09.2026", "date"), ("3/4", "fraction"), ("14.", "ordinal")]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_other_classes_are_not_money(self, test_input, owner):
        assert self._accepts(self.others[owner], test_input)
        assert not self._accepts(self.money_fst, test_input)
