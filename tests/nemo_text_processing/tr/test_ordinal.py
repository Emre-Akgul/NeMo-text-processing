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
Direct tests for the Turkish ordinal tagger, verbalizer and the suffix morphology
they are built on.

Turkish is not registered with ``Normalizer`` yet, so these tests drive the tagger and
the verbalizer directly instead of going through ``normalize()``.
"""

import pynini
import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.tr.morphology import ORDINAL_MORPHOLOGY, harmonic_suffix
from nemo_text_processing.text_normalization.tr.taggers.cardinal import CardinalFst as CardinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.ordinal import OrdinalFst as OrdinalTaggerFst
from nemo_text_processing.text_normalization.tr.verbalizers.ordinal import OrdinalFst as OrdinalVerbalizerFst

from ..utils import parse_test_case_file

# One ordinal per final digit, so every harmony class and both the vowel final and
# consonant final suffix shapes are covered by name.
_UNIT_ORDINALS = [
    ("1.", "birinci"),
    ("2.", "ikinci"),
    ("3.", "üçüncü"),
    ("4.", "dördüncü"),
    ("5.", "beşinci"),
    ("6.", "altıncı"),
    ("7.", "yedinci"),
    ("8.", "sekizinci"),
    ("9.", "dokuzuncu"),
    ("10.", "onuncu"),
]

_TENS_ORDINALS = [
    ("20.", "yirminci"),
    ("30.", "otuzuncu"),
    ("40.", "kırkıncı"),
    ("50.", "ellinci"),
    ("60.", "altmışıncı"),
    ("70.", "yetmişinci"),
    ("80.", "sekseninci"),
    ("90.", "doksanıncı"),
]

# "dört" is the one stem that alternates (dört -> dörd) and only when the suffix lands
# on it, so it is checked in final and non final position across magnitudes.
_DORT_CASES = [
    ("4.", "dördüncü"),
    ("14.", "on dördüncü"),
    ("24.", "yirmi dördüncü"),
    ("44.", "kırk dördüncü"),
    ("94.", "doksan dördüncü"),
    ("104.", "yüz dördüncü"),
    ("404.", "dört yüz dördüncü"),
    ("1004.", "bin dördüncü"),
    ("2024.", "iki bin yirmi dördüncü"),
    ("1000004.", "bir milyon dördüncü"),
    # non final "dört" must not alternate
    ("40.", "kırkıncı"),
    ("400.", "dört yüzüncü"),
    ("4000.", "dört bininci"),
    ("4000000.", "dört milyonuncu"),
    ("4400.", "dört bin dört yüzüncü"),
]

_INVALID_INPUTS = [
    ("",),
    (".",),
    ("1",),  # cardinal, no ordinal marker
    ("1..",),
    ("abc.",),
    ("bir.",),  # already verbalized
    ("01.",),  # leading zero
    ("1,5.",),  # decimal
    ("1.23.",),  # malformed thousands group
    (".1",),  # misplaced marker
    ("1'inci",),  # apostrophe suffixation, deferred
    ("1.'inci",),  # apostrophe suffixation, deferred
    ("-1.",),  # negative ordinals are not a Turkish form
    ("9" * 22 + ".",),  # beyond the supported magnitude
]


class TestOrdinal:

    cardinal = CardinalTaggerFst()
    tagger = OrdinalTaggerFst(cardinal=cardinal)
    verbalizer = OrdinalVerbalizerFst()
    tagger_fst = tagger.fst
    verbalizer_fst = verbalizer.fst
    ordinal_fst = pynini.compose(tagger_fst, verbalizer_fst).optimize()

    def _normalize(self, written: str) -> str:
        return rewrite.top_rewrite(written, self.ordinal_fst)

    @parameterized.expand(parse_test_case_file('tr/data_text_normalization/test_cases_ordinal.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        assert self._normalize(test_input) == expected, f"input: {test_input}"

    @parameterized.expand(_UNIT_ORDINALS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_units(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_TENS_ORDINALS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_multiples_of_ten(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand([(f"{number}.",) for number in range(1, 21)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_one_to_twenty_is_accepted(self, test_input):
        assert self._normalize(test_input) != ""

    @parameterized.expand([(f"{100 + digit}.",) for digit in range(0, 10)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_every_final_digit_in_the_hundreds(self, test_input):
        assert self._normalize(test_input).startswith("yüz ") or self._normalize(test_input) == "yüzüncü"

    @parameterized.expand(_DORT_CASES)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_dort_stem_alternation(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_hundreds_and_thousands(self):
        assert self._normalize("100.") == "yüzüncü"
        assert self._normalize("101.") == "yüz birinci"
        assert self._normalize("110.") == "yüz onuncu"
        assert self._normalize("120.") == "yüz yirminci"
        assert self._normalize("1000.") == "bininci"
        assert self._normalize("1001.") == "bin birinci"
        assert self._normalize("2026.") == "iki bin yirmi altıncı"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_magnitude_final(self):
        assert self._normalize("1000000.") == "bir milyonuncu"
        assert self._normalize("2000000.") == "iki milyonuncu"
        assert self._normalize("1000000000.") == "bir milyarıncı"
        assert self._normalize("1000000000000.") == "bir trilyonuncu"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_thousands_separator_is_accepted(self):
        assert self._normalize("1.000.") == "bininci"
        assert self._normalize("1.234.") == "bin iki yüz otuz dördüncü"

    @parameterized.expand(_INVALID_INPUTS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_invalid_input_is_rejected(self, test_input):
        with pytest.raises(rewrite.Error):
            self._normalize(test_input)

    @parameterized.expand(parse_test_case_file('tr/data_text_normalization/test_cases_ordinal.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_deterministic_single_transduction(self, test_input, expected):
        outputs = list(rewrite.rewrites(test_input, self.ordinal_fst))
        assert outputs == [expected], f"input: {test_input} produced {outputs}"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_token_schema(self):
        assert rewrite.top_rewrite("1.", self.tagger_fst) == 'ordinal { integer: "birinci" }'
        assert rewrite.top_rewrite("4.", self.tagger_fst) == 'ordinal { integer: "dördüncü" }'

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_bare_ordinals_subgraph_is_exposed(self):
        """Ordinal words without the written full stop, for grammars that add their own."""
        assert rewrite.top_rewrite("3", self.tagger.bare_ordinals) == "üçüncü"
        assert rewrite.top_rewrite("20", self.tagger.bare_ordinals) == "yirminci"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_graph_has_no_state_explosion(self):
        assert self.tagger_fst.num_states() < 10000, self.tagger_fst.num_states()
        assert self.ordinal_fst.num_states() < 10000, self.ordinal_fst.num_states()


class TestOrdinalMorphology:
    """The reusable suffix helper, independent of how a number is read."""

    @parameterized.expand(
        [
            # consonant final stems, one per harmony class
            ("doksan", "doksanıncı"),  # last vowel a -> ı
            ("seksen", "sekseninci"),  # last vowel e -> i
            ("kırk", "kırkıncı"),  # last vowel ı -> ı
            ("bin", "bininci"),  # last vowel i -> i
            ("on", "onuncu"),  # last vowel o -> u
            ("dokuz", "dokuzuncu"),  # last vowel u -> u
            ("yüz", "yüzüncü"),  # last vowel ü -> ü
            ("milyon", "milyonuncu"),
            ("milyar", "milyarıncı"),
            ("sıfır", "sıfırıncı"),
            # vowel final stems take the short suffix
            ("iki", "ikinci"),
            ("altı", "altıncı"),
            ("yedi", "yedinci"),
            ("yirmi", "yirminci"),
            ("elli", "ellinci"),
            # the suffix lands on the last word of a compound
            ("on bir", "on birinci"),
            ("iki bin yirmi altı", "iki bin yirmi altıncı"),
            # lexical stem alternation
            ("dört", "dördüncü"),
            ("yüz dört", "yüz dördüncü"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ordinal_suffix(self, stem, expected):
        assert list(rewrite.rewrites(stem, ORDINAL_MORPHOLOGY)) == [expected]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ö_is_harmonized_from_the_last_vowel_not_the_first(self):
        """ "dördüncü" harmonizes with ö, the last vowel, even though ü follows in output."""
        assert rewrite.top_rewrite("dörd", ORDINAL_MORPHOLOGY) == "dördüncü"
        assert rewrite.top_rewrite("altmış", ORDINAL_MORPHOLOGY) == "altmışıncı"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_harmonic_suffix_is_reusable_for_other_suffixes(self):
        """The helper is parameterised by template, not hard wired to the ordinal."""
        # locative -de/-da is a two way harmony, expressed as a high vowel template it
        # is not; this checks a different high vowel suffix, the possessive -(s)X.
        possessive = harmonic_suffix(after_consonant="{high}", after_vowel="s{high}")
        assert rewrite.top_rewrite("on", possessive) == "onu"
        assert rewrite.top_rewrite("iki", possessive) == "ikisi"
        assert rewrite.top_rewrite("yüz", possessive) == "yüzü"
        assert rewrite.top_rewrite("kırk", possessive) == "kırkı"
