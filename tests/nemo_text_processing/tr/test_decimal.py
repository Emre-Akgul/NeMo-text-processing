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
Direct tests for the Turkish decimal tagger and verbalizer.

Turkish is not registered with ``Normalizer`` yet, so these tests drive the tagger and
the verbalizer directly instead of going through ``normalize()``.
"""

import pynini
import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.tr.taggers.cardinal import CardinalFst as CardinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.decimal import DecimalFst as DecimalTaggerFst
from nemo_text_processing.text_normalization.tr.verbalizers.cardinal import CardinalFst as CardinalVerbalizerFst
from nemo_text_processing.text_normalization.tr.verbalizers.decimal import DecimalFst as DecimalVerbalizerFst

from ..utils import parse_test_case_file

_BASIC = [
    ("0,1", "sıfır virgül bir"),
    ("1,5", "bir virgül beş"),
    ("3,14", "üç virgül on dört"),
    ("12,25", "on iki virgül yirmi beş"),
    ("99,99", "doksan dokuz virgül doksan dokuz"),
]

# A leading zero in the fractional part is spoken, one "sıfır" per zero.
_LEADING_ZEROS = [
    ("1,01", "bir virgül sıfır bir"),
    ("1,001", "bir virgül sıfır sıfır bir"),
    ("12,05", "on iki virgül sıfır beş"),
    ("0,005", "sıfır virgül sıfır sıfır beş"),
    ("3,04", "üç virgül sıfır dört"),
]

# A zero inside the fractional part is carried by the cardinal reading itself.
_INTERNAL_ZEROS = [
    ("1,101", "bir virgül yüz bir"),
    ("1,105", "bir virgül yüz beş"),
    ("1,205", "bir virgül iki yüz beş"),
]

# A trailing zero changes the magnitude of the cardinal reading, so it survives.
_TRAILING_ZEROS = [
    ("1,0", "bir virgül sıfır"),
    ("1,00", "bir virgül sıfır sıfır"),
    ("1,10", "bir virgül on"),
    ("1,20", "bir virgül yirmi"),
    ("1,100", "bir virgül yüz"),
    ("1,140", "bir virgül yüz kırk"),
]

_INTEGER_BOUNDARIES = [
    ("0,5", "sıfır virgül beş"),
    ("9,5", "dokuz virgül beş"),
    ("10,5", "on virgül beş"),
    ("99,5", "doksan dokuz virgül beş"),
    ("100,5", "yüz virgül beş"),
    ("999,5", "dokuz yüz doksan dokuz virgül beş"),
    ("1000,5", "bin virgül beş"),
    ("1000000,5", "bir milyon virgül beş"),
]

_GROUPED_INTEGERS = [
    ("1.000,5", "bin virgül beş"),
    ("1.234,5", "bin iki yüz otuz dört virgül beş"),
    ("12.345,67", "on iki bin üç yüz kırk beş virgül altmış yedi"),
    ("1.234.567,89", "bir milyon iki yüz otuz dört bin beş yüz altmış yedi virgül seksen dokuz"),
    ("1.000.000,25", "bir milyon virgül yirmi beş"),
]

_NEGATIVES = [
    ("-0,5", "eksi sıfır virgül beş"),
    ("-1,5", "eksi bir virgül beş"),
    ("-12,05", "eksi on iki virgül sıfır beş"),
    ("-0,25", "eksi sıfır virgül yirmi beş"),
]

_QUANTITIES = [
    ("1,5 milyon", "bir virgül beş milyon"),
    ("2,25 milyar", "iki virgül yirmi beş milyar"),
    ("2 milyon", "iki milyon"),
    ("150 milyon", "yüz elli milyon"),
]

# Written forms that differ only in zeros must not share a reading.
_PRECISION_GROUPS = [
    ["1,1", "1,10", "1,100", "1,1000"],
    ["1,01", "1,010", "1,0100"],
    ["1,5", "1,05", "1,005", "1,050", "1,500"],
    ["3,14", "3,140", "3,014"],
    ["1,0", "1,00", "1,000"],
]

_INVALID_INPUTS = [
    ("",),
    ("1",),  # bare cardinal
    ("123",),  # bare cardinal
    ("1.",),  # ordinal marker
    (",5",),  # Turkish always writes the integer part
    ("1,",),  # no fractional part
    ("1,,5",),
    ("1.2",),  # full stop is not a decimal separator
    ("1.23",),
    ("1.234",),  # grouped CARDINAL, not a decimal
    ("1.23,45",),  # malformed grouping
    ("12.34,5",),  # malformed grouping
    ("1.2345,6",),  # malformed grouping
    ("1..234,5",),  # malformed grouping
    ("1.234,5,6",),  # two separators
    ("01,5",),  # leading zero in the integer part
    ("1,5milyon",),  # quantity needs a separating space
    ("1,5 bin",),  # "bin" is not a quantity word
    ("1,5 milyonuncu",),  # ordinal suffixation, not this grammar
    ("bir virgül beş",),  # already verbalized
    ("1,2.345",),  # separator in the fractional part
]


class TestDecimal:

    cardinal = CardinalTaggerFst()
    tagger = DecimalTaggerFst(cardinal=cardinal)
    verbalizer = DecimalVerbalizerFst(cardinal=CardinalVerbalizerFst())
    tagger_fst = tagger.fst
    verbalizer_fst = verbalizer.fst
    decimal_fst = pynini.compose(tagger_fst, verbalizer_fst).optimize()

    def _normalize(self, written: str) -> str:
        return rewrite.top_rewrite(written, self.decimal_fst)

    @parameterized.expand(parse_test_case_file('tr/data_text_normalization/test_cases_decimal.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        assert self._normalize(test_input) == expected, f"input: {test_input}"

    @parameterized.expand(_BASIC)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_basic(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_LEADING_ZEROS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_leading_zeros_are_spoken_one_by_one(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_INTERNAL_ZEROS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_internal_zeros(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_TRAILING_ZEROS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_trailing_zeros_are_preserved(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_INTEGER_BOUNDARIES)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_integer_part_boundaries(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_GROUPED_INTEGERS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_grouped_integer_part(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_NEGATIVES)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_negative(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_QUANTITIES)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_quantity(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand([(group,) for group in _PRECISION_GROUPS])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_written_precision_is_not_lost(self, group):
        """Distinct written decimals must never share a spoken form."""
        readings = [self._normalize(written) for written in group]
        assert len(set(readings)) == len(group), dict(zip(group, readings))

    @parameterized.expand(_INVALID_INPUTS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_invalid_input_is_rejected(self, test_input):
        with pytest.raises(rewrite.Error):
            self._normalize(test_input)

    @parameterized.expand(parse_test_case_file('tr/data_text_normalization/test_cases_decimal.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_deterministic_single_transduction(self, test_input, expected):
        outputs = list(rewrite.rewrites(test_input, self.decimal_fst))
        assert outputs == [expected], f"input: {test_input} produced {outputs}"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_token_schema(self):
        assert rewrite.top_rewrite("12,5", self.tagger_fst) == (
            'decimal { integer_part: "on iki" fractional_part: "beş" }'
        )
        assert rewrite.top_rewrite("-12,05", self.tagger_fst) == (
            'decimal { negative: "true" integer_part: "on iki" fractional_part: "sıfır beş" }'
        )
        assert rewrite.top_rewrite("1,5 milyon", self.tagger_fst) == (
            'decimal { integer_part: "bir" fractional_part: "beş" quantity: "milyon" }'
        )
        assert rewrite.top_rewrite("2 milyon", self.tagger_fst) == (
            'decimal { integer_part: "iki" quantity: "milyon" }'
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_reusable_subgraphs_are_exposed(self):
        """Money, measure and percentage will compose these."""
        assert rewrite.top_rewrite("05", self.tagger.graph) == "sıfır beş"
        assert rewrite.top_rewrite("14", self.tagger.graph) == "on dört"
        assert rewrite.top_rewrite("12,5", self.tagger.final_graph_wo_sign) == (
            'integer_part: "on iki" fractional_part: "beş"'
        )
        assert rewrite.top_rewrite("1,5 milyon", self.tagger.final_graph_wo_negative) == (
            'integer_part: "bir" fractional_part: "beş" quantity: "milyon"'
        )
        assert rewrite.top_rewrite("150", self.tagger.cardinal_up_to_thousand) == "yüz elli"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_graph_has_no_state_explosion(self):
        assert self.tagger_fst.num_states() < 20000, self.tagger_fst.num_states()
        assert self.decimal_fst.num_states() < 20000, self.decimal_fst.num_states()
