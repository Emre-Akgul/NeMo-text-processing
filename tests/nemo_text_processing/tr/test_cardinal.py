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
Direct tests for the Turkish cardinal tagger and verbalizer.

Turkish is not registered with ``Normalizer`` yet, so these tests drive the tagger and
the verbalizer directly instead of going through ``normalize()``.
"""

import pynini
import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.tr.taggers.cardinal import CardinalFst as CardinalTaggerFst
from nemo_text_processing.text_normalization.tr.verbalizers.cardinal import CardinalFst as CardinalVerbalizerFst

from ..utils import parse_test_case_file


def _powers_of_ten():
    return [(str(10**exponent),) for exponent in range(0, 21)]


def _invalid_inputs():
    return [
        ("",),  # empty
        ("abc",),  # not a number
        ("bir",),  # already verbalized
        ("01",),  # leading zero
        ("007",),  # leading zeros
        ("1,5",),  # decimal, belongs to DecimalFst
        ("1.23",),  # malformed thousands group
        ("1.2345",),  # malformed thousands group
        ("12.34.567",),  # malformed thousands group
        ("-",),  # bare sign
        ("--1",),  # doubled sign
        ("1-",),  # trailing sign
        ("1 2",),  # two numbers
        ("1'inci",),  # ordinal, not handled in this phase
        ("9" * 22,),  # beyond the supported magnitude
    ]


class TestCardinal:

    tagger = CardinalTaggerFst()
    verbalizer = CardinalVerbalizerFst()
    tagger_fst = tagger.fst
    verbalizer_fst = verbalizer.fst
    cardinal_fst = pynini.compose(tagger_fst, verbalizer_fst).optimize()

    def _normalize(self, written: str) -> str:
        return rewrite.top_rewrite(written, self.cardinal_fst)

    @parameterized.expand(parse_test_case_file('tr/data_text_normalization/test_cases_cardinal.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        assert self._normalize(test_input) == expected, f"input: {test_input}"

    @parameterized.expand([(str(number),) for number in range(0, 21)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_zero_to_twenty_is_accepted(self, test_input):
        assert self._normalize(test_input) != ""

    @parameterized.expand([(str(number),) for number in range(10, 100, 10)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_multiples_of_ten_are_single_words(self, test_input):
        assert " " not in self._normalize(test_input), f"input: {test_input}"

    @parameterized.expand(_powers_of_ten())
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_powers_of_ten(self, test_input):
        assert self._normalize(test_input) != ""

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_one_is_not_spoken_before_yuz_and_bin(self):
        """100 is "yüz", not "bir yüz"; 1000 is "bin", not "bir bin"."""
        for written in ["100", "1000", "1100", "100000", "1001"]:
            assert not self._normalize(written).startswith("bir "), f"input: {written}"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_one_is_spoken_from_milyon_upwards(self):
        """Unlike "yüz" and "bin", magnitudes from 10^6 up keep their multiplier."""
        assert self._normalize("1000000") == "bir milyon"
        assert self._normalize("1000000000") == "bir milyar"
        assert self._normalize("1000000000000") == "bir trilyon"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_magnitude_transitions(self):
        assert self._normalize("999") == "dokuz yüz doksan dokuz"
        assert self._normalize("1000") == "bin"
        assert self._normalize("999999") == "dokuz yüz doksan dokuz bin dokuz yüz doksan dokuz"
        assert self._normalize("1000000") == "bir milyon"
        assert (
            self._normalize("999999999")
            == "dokuz yüz doksan dokuz milyon dokuz yüz doksan dokuz bin dokuz yüz doksan dokuz"
        )
        assert self._normalize("1000000000") == "bir milyar"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_largest_supported_magnitude(self):
        assert self._normalize("9" * 21).startswith("dokuz yüz doksan dokuz kentilyon")

    @parameterized.expand(_invalid_inputs())
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_invalid_input_is_rejected(self, test_input):
        with pytest.raises(rewrite.Error):
            self._normalize(test_input)

    @parameterized.expand(parse_test_case_file('tr/data_text_normalization/test_cases_cardinal.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_deterministic_single_transduction(self, test_input, expected):
        outputs = list(rewrite.rewrites(test_input, self.cardinal_fst))
        assert outputs == [expected], f"input: {test_input} produced {outputs}"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_token_schema(self):
        """The tagger must emit the token schema the verbalizer and NeMo expect."""
        assert rewrite.top_rewrite("100", self.tagger_fst) == 'cardinal { integer: "yüz" }'
        assert rewrite.top_rewrite("-100", self.tagger_fst) == 'cardinal { negative: "true" integer: "yüz" }'

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_reusable_subgraphs_are_exposed(self):
        assert rewrite.top_rewrite("7", self.tagger.digit) == "yedi"
        assert rewrite.top_rewrite("0", self.tagger.zero) == "sıfır"
        assert rewrite.top_rewrite("42", self.tagger.two_digit_non_zero) == "kırk iki"
        assert rewrite.top_rewrite("305", self.tagger.hundreds) == "üç yüz beş"
        assert rewrite.top_rewrite("1905", self.tagger.graph) == "bin dokuz yüz beş"
        assert rewrite.top_rewrite("205", self.tagger.single_digits_graph) == "iki sıfır beş"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_graph_has_no_state_explosion(self):
        """Guards against an enumerating rewrite of the grammar."""
        assert self.tagger_fst.num_states() < 10000, self.tagger_fst.num_states()
        assert self.cardinal_fst.num_states() < 10000, self.cardinal_fst.num_states()
