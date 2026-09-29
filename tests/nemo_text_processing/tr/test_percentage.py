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
Direct tests for the Turkish percentage tagger and verbalizer.

The number inside a percentage is a nested cardinal or decimal token, and the
Normalizer permutes the fields of a nested token like any other. ``_readings`` runs
every tagged token through the token parser and ``Normalizer._permute``, so these tests
exercise the path the Normalizer will.
"""

import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.token_parser import TokenParser
from nemo_text_processing.text_normalization.tr.morphology import LOCATIVE_SUFFIX
from nemo_text_processing.text_normalization.tr.taggers.cardinal import CardinalFst as CardinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.date import DateFst as DateTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.decimal import DecimalFst as DecimalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.fraction import FractionFst as FractionTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.ordinal import OrdinalFst as OrdinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.percentage import PercentageFst as PercentageTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.time import TimeFst as TimeTaggerFst
from nemo_text_processing.text_normalization.tr.verbalizers.cardinal import CardinalFst as CardinalVerbalizerFst
from nemo_text_processing.text_normalization.tr.verbalizers.decimal import DecimalFst as DecimalVerbalizerFst
from nemo_text_processing.text_normalization.tr.verbalizers.fraction import FractionFst as FractionVerbalizerFst
from nemo_text_processing.text_normalization.tr.verbalizers.percentage import PercentageFst as PercentageVerbalizerFst

from ..utils import parse_test_case_file

_TEST_CASES = 'tr/data_text_normalization/test_cases_percentage.txt'

_INTEGERS = [
    ("%0", "yüzde sıfır"),
    ("%1", "yüzde bir"),
    ("%5", "yüzde beş"),
    ("%10", "yüzde on"),
    ("%25", "yüzde yirmi beş"),
    ("%50", "yüzde elli"),
    ("%99", "yüzde doksan dokuz"),
    ("%100", "yüzde yüz"),
    ("%101", "yüzde yüz bir"),
    ("%250", "yüzde iki yüz elli"),
    ("%1000", "yüzde bin"),
    ("%1000000", "yüzde bir milyon"),
]

_GROUPED = [
    ("%1.000", "yüzde bin"),
    ("%10.000", "yüzde on bin"),
    ("%1.234", "yüzde bin iki yüz otuz dört"),
    ("%1.234,5", "yüzde bin iki yüz otuz dört virgül beş"),
]

_DECIMALS = [
    ("%0,1", "yüzde sıfır virgül bir"),
    ("%0,01", "yüzde sıfır virgül sıfır bir"),
    ("%0,05", "yüzde sıfır virgül sıfır beş"),
    ("%0,5", "yüzde sıfır virgül beş"),
    ("%1,5", "yüzde bir virgül beş"),
    ("%1,05", "yüzde bir virgül sıfır beş"),
    ("%1,25", "yüzde bir virgül yirmi beş"),
    ("%12,5", "yüzde on iki virgül beş"),
    ("%99,99", "yüzde doksan dokuz virgül doksan dokuz"),
    ("%100,5", "yüzde yüz virgül beş"),
]

_MALFORMED = [
    ("",),
    ("%",),
    ("%%",),
    ("%%25",),
    ("% 25",),
    ("%  25",),
    ("25%",),
    ("25 %",),
    ("25",),
    ("%05",),
    ("%00",),
    ("%,5",),
    ("%1,",),
    ("%1,,5",),
    ("%1.2",),
    ("%1.23",),
    ("%12.34",),
    ("%1..234",),
    ("%1.2345",),
    ("%1,2.3",),
    ("%1,5 milyon",),  # quantities are not percentages
    ("%2 milyon",),
    ("%10 bin",),
    ("%-5",),  # signed percentages are deferred
    ("-%5",),
    ("+%5",),
    ("%+5",),
    ("%-1,5",),
    ("‰50",),  # permille is deferred
    ("%25'i",),  # suffixed forms are deferred
    ("%25'te",),
    ("%10'dan",),
    ("%50'ye",),
    ("%100'ü",),
    ("%25.",),
    ("%25 ",),
    (" %25",),
    ("yüzde 25",),
    ("yüzde yirmi beş",),
    ("%abc",),
    ("%1/2",),
    ("%14.30",),
]


class TestPercentage:

    cardinal = CardinalTaggerFst()
    decimal = DecimalTaggerFst(cardinal=cardinal)
    tagger = PercentageTaggerFst(cardinal=cardinal, decimal=decimal)
    cardinal_verbalizer = CardinalVerbalizerFst()
    decimal_verbalizer = DecimalVerbalizerFst(cardinal=cardinal_verbalizer)
    verbalizer = PercentageVerbalizerFst(cardinal=cardinal_verbalizer, decimal=decimal_verbalizer)
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

    def _assert_rejected(self, written):
        with pytest.raises(rewrite.Error):
            rewrite.top_rewrite(written, self.tagger_fst)

    @parameterized.expand(parse_test_case_file(_TEST_CASES))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        assert self._normalize(test_input) == expected, f"input: {test_input}"

    @parameterized.expand(_INTEGERS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_integers(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_GROUPED)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_grouped_numbers(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_DECIMALS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_decimals(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_MALFORMED)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_malformed_input_is_rejected(self, test_input):
        self._assert_rejected(test_input)

    @parameterized.expand(parse_test_case_file(_TEST_CASES))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_deterministic_single_transduction(self, test_input, expected):
        assert list(rewrite.rewrites(test_input, self.tagger_fst)) == [
            rewrite.top_rewrite(test_input, self.tagger_fst)
        ], f"tagger is ambiguous for {test_input}"
        assert self._readings(test_input) == {expected}, f"input: {test_input}"

    @parameterized.expand(parse_test_case_file(_TEST_CASES))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_number_reads_as_the_number_grammar_reads_it(self, test_input, expected):
        """The part after "yüzde" is exactly what CardinalFst or DecimalFst gives the
        number without the sign."""
        number = test_input[1:]
        fst, verbalizer = (
            (self.decimal.fst, self.decimal_verbalizer.fst)
            if "," in number
            else (self.cardinal.fst, self.cardinal_verbalizer.fst)
        )
        assert self._readings(number, fst, verbalizer) == {expected[len("yüzde ") :]}

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_token_schema(self):
        assert rewrite.top_rewrite("%25", self.tagger_fst) == 'percentage { cardinal { integer: "yirmi beş" } }'
        assert rewrite.top_rewrite("%1.000", self.tagger_fst) == 'percentage { cardinal { integer: "bin" } }'
        assert rewrite.top_rewrite("%12,5", self.tagger_fst) == (
            'percentage { decimal { integer_part: "on iki" fractional_part: "beş" } }'
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_verbalizer(self):
        v = self.verbalizer_fst
        assert rewrite.top_rewrite('percentage { cardinal { integer: "yirmi beş" } }', v) == "yüzde yirmi beş"
        assert rewrite.top_rewrite('percentage { decimal { integer_part: "on iki" fractional_part: "beş" } }', v) == (
            "yüzde on iki virgül beş"
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_value_is_not_simplified(self):
        assert self._normalize("%25") == "yüzde yirmi beş"
        assert self._normalize("%50") == "yüzde elli"
        assert self._normalize("%75") == "yüzde yetmiş beş"
        nondeterministic = PercentageTaggerFst(cardinal=self.cardinal, decimal=self.decimal, deterministic=False)
        assert rewrite.rewrites("%50", nondeterministic.graph) == ["yüzde elli"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_reusable_subgraphs_are_exposed(self):
        assert rewrite.top_rewrite("%25", self.tagger.cardinal_graph) == 'cardinal { integer: "yirmi beş" }'
        assert rewrite.top_rewrite("%12,5", self.tagger.decimal_graph) == (
            'decimal { integer_part: "on iki" fractional_part: "beş" }'
        )
        assert rewrite.top_rewrite("%25", self.tagger.final_graph) == 'cardinal { integer: "yirmi beş" }'
        assert rewrite.top_rewrite("%25", self.tagger.graph) == "yüzde yirmi beş"
        assert rewrite.top_rewrite("%12,05", self.tagger.graph) == "yüzde on iki virgül sıfır beş"

    @parameterized.expand(parse_test_case_file(_TEST_CASES))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_bare_graph_matches_tagger_and_verbalizer(self, test_input, expected):
        assert rewrite.rewrites(test_input, self.tagger.graph) == [expected]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_bare_reading_takes_the_locative(self):
        """What the later suffix layer needs: the last spoken word is inflected."""
        locative = self.tagger.graph @ LOCATIVE_SUFFIX
        assert rewrite.rewrites("%25", locative) == ["yüzde yirmi beşte"]
        assert rewrite.rewrites("%20", locative) == ["yüzde yirmide"]
        assert rewrite.rewrites("%12,5", locative) == ["yüzde on iki virgül beşte"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_generated_cross_check(self):
        for n in range(0, 1001):
            expected = "yüzde " + rewrite.top_rewrite(str(n), self.cardinal.graph)
            assert rewrite.rewrites(f"%{n}", self.tagger.graph) == [expected], n
        for integer in ["0", "1", "12", "100", "1.234"]:
            integer_words = rewrite.top_rewrite(integer, self.cardinal.graph)
            for fraction in ["0", "5", "05", "50", "005", "500", "105", "0105", "999"]:
                fraction_words = rewrite.top_rewrite(fraction, self.decimal.graph)
                expected = f"yüzde {integer_words} virgül {fraction_words}"
                assert self._readings(f"%{integer},{fraction}") == {expected}

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_graph_has_no_state_explosion(self):
        assert self.tagger_fst.num_states() < 6000, self.tagger_fst.num_states()
        assert self.verbalizer_fst.num_states() < 500, self.verbalizer_fst.num_states()


class TestPercentageSeparation:
    """Percentage against the cardinal, decimal, fraction, date, time and ordinal grammars."""

    cardinal = CardinalTaggerFst()
    decimal = DecimalTaggerFst(cardinal=cardinal)
    percentage_fst = PercentageTaggerFst(cardinal=cardinal, decimal=decimal).fst
    others = {
        "cardinal": cardinal.fst,
        "decimal": decimal.fst,
        "fraction": FractionTaggerFst(cardinal=cardinal).fst,
        "date": DateTaggerFst(cardinal=cardinal).fst,
        "time": TimeTaggerFst(cardinal=cardinal).fst,
        "ordinal": OrdinalTaggerFst(cardinal=cardinal).fst,
    }

    @staticmethod
    def _accepts(fst, written):
        try:
            rewrite.top_rewrite(written, fst)
            return True
        except rewrite.Error:
            return False

    @parameterized.expand([("%25",), ("%14",), ("%12,5",), ("%1.000",), ("%0",)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_percentages_belong_to_no_other_grammar(self, test_input):
        assert self._accepts(self.percentage_fst, test_input)
        for name, fst in self.others.items():
            assert not self._accepts(fst, test_input), name

    @parameterized.expand(
        [("25", "cardinal"), ("12,5", "decimal"), ("25/100", "fraction"), ("29.09.2026", "date")]
        + [("14.30", "time"), ("14.", "ordinal")]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_other_classes_are_not_percentages(self, test_input, owner):
        assert self._accepts(self.others[owner], test_input)
        assert not self._accepts(self.percentage_fst, test_input)

    @parameterized.expand([("%25", "25/100", "yüzde yirmi beş"), ("%1", "1/100", "yüzde bir")])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_percentage_and_fraction_read_the_same(self, percentage, fraction, expected):
        """Intended: two written classes, one reading."""
        cardinal_verbalizer = CardinalVerbalizerFst()
        percentage_verbalizer = PercentageVerbalizerFst(
            cardinal=cardinal_verbalizer, decimal=DecimalVerbalizerFst(cardinal=cardinal_verbalizer)
        ).fst
        assert TestPercentage._readings(percentage, self.percentage_fst, percentage_verbalizer) == {expected}
        assert TestPercentage._readings(fraction, self.others["fraction"], FractionVerbalizerFst().fst) == {expected}
