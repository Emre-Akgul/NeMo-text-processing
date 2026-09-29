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
Direct tests for the Turkish fraction tagger, verbalizer and the locative morphology
they are built on.

Turkish speaks the denominator first, so unlike the other Turkish grammars the tagger
and the verbalizer cannot simply be composed: the tagger emits the fields in written
order and the verbalizer reads them in spoken order, with normalize.py's field
permutation in between. ``_verbalize`` below reproduces exactly that step, so these
tests exercise the same path the Normalizer will.
"""

import itertools

import pynini
import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.token_parser import PRESERVE_ORDER_KEY, TokenParser
from nemo_text_processing.text_normalization.tr.morphology import LOCATIVE_SUFFIX
from nemo_text_processing.text_normalization.tr.taggers.cardinal import CardinalFst as CardinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.fraction import FractionFst as FractionTaggerFst
from nemo_text_processing.text_normalization.tr.verbalizers.fraction import FractionFst as FractionVerbalizerFst

from ..utils import parse_test_case_file

_BASIC = [
    ("1/2", "ikide bir"),
    ("1/3", "üçte bir"),
    ("2/3", "üçte iki"),
    ("3/4", "dörtte üç"),
    ("1/4", "dörtte bir"),
    ("5/8", "sekizde beş"),
    ("7/10", "onda yedi"),
    ("9/10", "onda dokuz"),
]

# 21-29 walks every final digit of the denominator, which between them cover both
# harmony classes, vowel final stems, and voiced and voiceless consonant final stems.
_EVERY_FINAL_DIGIT = [
    ("1/21", "yirmi birde bir"),
    ("1/22", "yirmi ikide bir"),
    ("1/23", "yirmi üçte bir"),
    ("1/24", "yirmi dörtte bir"),
    ("1/25", "yirmi beşte bir"),
    ("1/26", "yirmi altıda bir"),
    ("1/27", "yirmi yedide bir"),
    ("1/28", "yirmi sekizde bir"),
    ("1/29", "yirmi dokuzda bir"),
]

_TENS = [
    ("1/10", "onda bir"),
    ("1/20", "yirmide bir"),
    ("1/30", "otuzda bir"),
    ("1/40", "kırkta bir"),
    ("1/50", "ellide bir"),
    ("1/60", "altmışta bir"),
    ("1/70", "yetmişte bir"),
    ("1/80", "seksende bir"),
    ("1/90", "doksanda bir"),
]

_MAGNITUDES = [
    ("1/100", "yüzde bir"),
    ("1/101", "yüz birde bir"),
    ("1/104", "yüz dörtte bir"),
    ("1/120", "yüz yirmide bir"),
    ("1/1000", "binde bir"),
    ("1/1001", "bin birde bir"),
    ("1/1000000", "bir milyonda bir"),
    ("1/1000000000", "bir milyarda bir"),
]

_LARGER_NUMERATORS = [
    ("12/25", "yirmi beşte on iki"),
    ("25/100", "yüzde yirmi beş"),
    ("101/1000", "binde yüz bir"),
    ("999/1000", "binde dokuz yüz doksan dokuz"),
    ("3/42", "kırk ikide üç"),
    ("5/104", "yüz dörtte beş"),
    ("7/120", "yüz yirmide yedi"),
    ("11/2024", "iki bin yirmi dörtte on bir"),
]

_ZERO_NUMERATOR = [
    ("0/2", "ikide sıfır"),
    ("0/5", "beşte sıfır"),
    ("0/100", "yüzde sıfır"),
]

_NEGATIVE = [
    ("-1/2", "eksi ikide bir"),
    ("-3/4", "eksi dörtte üç"),
    ("-11/2024", "eksi iki bin yirmi dörtte on bir"),
]

_ZERO_DENOMINATOR = [("1/0",), ("0/0",), ("100/0",), ("1/00",), ("7/000",)]

_MALFORMED = [
    ("",),
    ("/",),
    ("1/",),
    (" /2",),
    ("1//2",),
    ("1/2/3",),
    ("abc/2",),
    ("1/abc",),
    ("1,5/2",),  # decimal numerator
    ("1/2,5",),  # decimal denominator
    ("1.000/2",),  # grouped numerator, deliberately not accepted
    ("1/1.000",),  # grouped denominator, deliberately not accepted
    ("1/-2",),  # sign on the denominator is not supported
    ("1/05",),  # leading zero
    ("3 / 4",),  # spaces around the slash
    ("½",),  # vulgar fraction character
    ("29/09/2026",),  # date, not a fraction
    ("1/2/",),
]

_LOCATIVE_STEMS = [
    # consonant final, voiced -> d
    ("bir", "birde"),
    ("on", "onda"),
    ("sekiz", "sekizde"),
    ("dokuz", "dokuzda"),
    ("otuz", "otuzda"),
    ("seksen", "seksende"),
    ("doksan", "doksanda"),
    ("yüz", "yüzde"),
    ("bin", "binde"),
    ("milyon", "milyonda"),
    ("milyar", "milyarda"),
    # consonant final, voiceless -> t
    ("üç", "üçte"),
    ("dört", "dörtte"),
    ("beş", "beşte"),
    ("kırk", "kırkta"),
    ("altmış", "altmışta"),
    ("yetmiş", "yetmişte"),
    # vowel final -> d
    ("iki", "ikide"),
    ("altı", "altıda"),
    ("yedi", "yedide"),
    ("yirmi", "yirmide"),
    ("elli", "ellide"),
    # only the final spoken word controls the suffix
    ("yirmi dört", "yirmi dörtte"),
    ("kırk iki", "kırk ikide"),
    ("yüz dört", "yüz dörtte"),
    ("iki bin yirmi dört", "iki bin yirmi dörtte"),
    ("bir milyon", "bir milyonda"),
]


class TestFraction:

    cardinal = CardinalTaggerFst()
    tagger = FractionTaggerFst(cardinal=cardinal)
    verbalizer = FractionVerbalizerFst()
    tagger_fst = tagger.fst
    verbalizer_fst = verbalizer.fst
    _parser = TokenParser()

    @classmethod
    def _field_permutations(cls, fields):
        """Mirrors Normalizer._permute for a single token."""
        orders = [fields.items()] if PRESERVE_ORDER_KEY in fields else itertools.permutations(fields.items())
        return ["".join(f'{key}: "{value}" ' for key, value in order) for order in orders]

    @classmethod
    def _readings(cls, written):
        """Every reading the tagger plus the verbalizer admit, via the token parser."""
        tagged = rewrite.top_rewrite(written, cls.tagger_fst)
        cls._parser(f"tokens {{ {tagged} }}")
        fields = cls._parser.parse()[0]["tokens"]["fraction"]
        readings = set()
        for body in cls._field_permutations(fields):
            try:
                readings.add(rewrite.top_rewrite(f"fraction {{ {body}}}", cls.verbalizer_fst))
            except rewrite.Error:
                pass
        return readings

    def _normalize(self, written):
        readings = self._readings(written)
        assert len(readings) == 1, f"input: {written} produced {readings}"
        return readings.pop()

    @parameterized.expand(parse_test_case_file('tr/data_text_normalization/test_cases_fraction.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        assert self._normalize(test_input) == expected, f"input: {test_input}"

    @parameterized.expand(_BASIC)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_basic(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_EVERY_FINAL_DIGIT)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_every_denominator_final_digit(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_TENS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_tens(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_MAGNITUDES)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_hundreds_thousands_and_magnitudes(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_LARGER_NUMERATORS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_larger_numerators(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_ZERO_NUMERATOR)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_zero_numerator_is_supported(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_NEGATIVE)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_negative(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_ZERO_DENOMINATOR)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_zero_denominator_is_rejected(self, test_input):
        with pytest.raises(rewrite.Error):
            rewrite.top_rewrite(test_input, self.tagger_fst)

    @parameterized.expand(_MALFORMED)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_malformed_input_is_rejected(self, test_input):
        with pytest.raises(rewrite.Error):
            rewrite.top_rewrite(test_input, self.tagger_fst)

    @parameterized.expand(parse_test_case_file('tr/data_text_normalization/test_cases_fraction.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_deterministic_single_transduction(self, test_input, expected):
        assert list(rewrite.rewrites(test_input, self.tagger_fst)) == [
            rewrite.top_rewrite(test_input, self.tagger_fst)
        ], f"tagger is ambiguous for {test_input}"
        assert self._readings(test_input) == {expected}, f"input: {test_input}"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_token_schema(self):
        """Fields keep the repository's names and the written order; the denominator
        carries the locative inflection, as in the Hungarian grammar."""
        assert rewrite.top_rewrite("3/4", self.tagger_fst) == 'fraction { numerator: "üç" denominator: "dörtte" }'
        assert rewrite.top_rewrite("-1/2", self.tagger_fst) == (
            'fraction { negative: "true" numerator: "bir" denominator: "ikide" }'
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_verbalizer_speaks_denominator_first(self):
        assert rewrite.top_rewrite('fraction { denominator: "dörtte" numerator: "üç" }', self.verbalizer_fst) == (
            "dörtte üç"
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_reusable_subgraphs_are_exposed(self):
        assert rewrite.top_rewrite("4", self.tagger.denominator_graph) == "dörtte"
        assert rewrite.top_rewrite("3", self.tagger.numerator_graph) == "üç"
        assert rewrite.top_rewrite("3/4", self.tagger.final_graph_wo_negative) == (
            'numerator: "üç" denominator: "dörtte"'
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_graph_has_no_state_explosion(self):
        assert self.tagger_fst.num_states() < 20000, self.tagger_fst.num_states()
        assert self.verbalizer_fst.num_states() < 20000, self.verbalizer_fst.num_states()


class TestLocativeMorphology:
    """The reusable locative helper, independent of how a number is read."""

    @parameterized.expand(_LOCATIVE_STEMS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_locative_suffix(self, stem, expected):
        assert list(rewrite.rewrites(stem, LOCATIVE_SUFFIX)) == [expected]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_two_way_vowel_harmony(self):
        """Back vowel stems take a, front vowel stems take e."""
        for stem in ["doksan", "kırk", "otuz", "altı"]:  # a, ı, o, u
            assert rewrite.top_rewrite(stem, LOCATIVE_SUFFIX).endswith("a"), stem
        for stem in ["seksen", "bin", "dört", "yüz"]:  # e, i, ö, ü
            assert rewrite.top_rewrite(stem, LOCATIVE_SUFFIX).endswith("e"), stem

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_d_t_assimilation(self):
        """t after a voiceless consonant, d after anything else."""
        for stem in ["üç", "dört", "beş", "kırk", "altmış", "yetmiş"]:
            assert rewrite.top_rewrite(stem, LOCATIVE_SUFFIX)[-2] == "t", stem
        for stem in ["bir", "on", "yüz", "bin", "sekiz", "iki", "altı", "yirmi"]:
            assert rewrite.top_rewrite(stem, LOCATIVE_SUFFIX)[-2] == "d", stem

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_only_the_final_word_is_suffixed(self):
        for stem in ["yirmi dört", "iki bin yirmi dört", "bir milyon", "yüz yirmi"]:
            suffixed = rewrite.top_rewrite(stem, LOCATIVE_SUFFIX)
            assert suffixed.startswith(stem), suffixed
            assert len(suffixed) == len(stem) + 2, suffixed

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_locative_does_not_trigger_the_ordinal_stem_alternation(self):
        """ "dört" voices to "dörd" only before a vowel initial suffix."""
        assert rewrite.top_rewrite("dört", LOCATIVE_SUFFIX) == "dörtte"
