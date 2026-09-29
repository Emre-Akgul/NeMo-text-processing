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
Direct tests for the Turkish abbreviation tagger, verbalized with the shared
abbreviation verbalizer, which reads the single ``value`` field.
"""

import itertools

import pynini
import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.en.verbalizers.abbreviation import (
    AbbreviationFst as AbbreviationVerbalizerFst,
)
from nemo_text_processing.text_normalization.tr.taggers.abbreviation import AbbreviationFst
from nemo_text_processing.text_normalization.tr.taggers.cardinal import CardinalFst as CardinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.date import DateFst as DateTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.decimal import DecimalFst as DecimalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.electronic import ElectronicFst as ElectronicTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.fraction import FractionFst as FractionTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.measure import MeasureFst as MeasureTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.money import MoneyFst as MoneyTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.ordinal import OrdinalFst as OrdinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.percentage import PercentageFst as PercentageTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.telephone import TelephoneFst as TelephoneTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.time import TimeFst as TimeTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.tr.utils import get_abs_path, load_labels

from ..utils import parse_test_case_file

_TEST_CASES = 'tr/data_text_normalization/test_cases_abbreviation.txt'

# The letter names, written out independently of the data file: the 29 letters of the
# Turkish alphabet, then the conventional names of the foreign Q, W and X.
_LETTERS = {
    "A": "a", "B": "be", "C": "ce", "Ç": "çe", "D": "de", "E": "e", "F": "fe", "G": "ge", "Ğ": "yumuşak ge",
    "H": "he", "I": "ı", "İ": "i", "J": "je", "K": "ke", "L": "le", "M": "me", "N": "ne", "O": "o", "Ö": "ö",
    "P": "pe", "R": "re", "S": "se", "Ş": "şe", "T": "te", "U": "u", "Ü": "ü", "V": "ve", "Y": "ye", "Z": "ze",
    "Q": "ku", "W": "ve", "X": "iks",
}  # fmt: skip

_ACRONYMS = load_labels(get_abs_path("data/abbreviation/acronym_readings.tsv"))
_WHITELISTED = [written for written, _ in load_labels(get_abs_path("data/whitelist.tsv"))]

_INVALID = [
    ("",),
    ("A",),  # single letters
    ("T",),
    ("K",),
    ("tdk",),  # capitals only
    ("Tdk",),
    ("pkk",),
    ("Pkk",),
    ("Nato",),
    ("nato",),
    ("Tübitak",),
    ("Bmw",),
    ("tDK",),
    ("TD-K",),  # punctuation and spaces
    ("T_D_K",),
    ("TD.K",),
    ("T D K",),
    (" TDK",),
    ("TDK ",),
    ("T.D.K.",),  # dotted forms other than T.C.
    ("P.K.K.",),
    ("A.B.C.",),
    ("A.Ş.",),
    ("M.Ö.",),
    ("T.",),
    ("T.C",),
    ("TC.",),
    ("TDK2",),  # digits
    ("2FA",),
    ("4K",),
    ("5G",),
    ("B2B",),
    ("TDK'den",),  # suffixed forms
    ("PKK'ya",),
    ("THY'de",),
    ("TRT'den",),
    ("BMW'de",),
    ("NATO'dan",),
    ("ASELSAN'da",),
    ("BOTAŞ'ın",),
    ("TÜBİTAK'ın",),
    ("UNESCO'ya",),
]


def _spell(written):
    return " ".join(_LETTERS[c] for c in written)


class TestAbbreviation:

    whitelist = WhiteListFst()
    tagger = AbbreviationFst(whitelist=whitelist)
    verbalizer = AbbreviationVerbalizerFst()
    tagger_fst = tagger.fst
    verbalizer_fst = verbalizer.fst

    def _accepts(self, fst, written):
        try:
            rewrite.top_rewrite(written, fst)
            return True
        except rewrite.Error:
            return False

    def _normalize(self, written):
        tags = rewrite.rewrites(written, self.tagger_fst)
        assert len(tags) == 1, f"input: {written} produced {tags}"
        readings = rewrite.rewrites(tags[0], self.verbalizer_fst)
        assert len(readings) == 1, readings
        return readings[0]

    @parameterized.expand(parse_test_case_file(_TEST_CASES))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(parse_test_case_file(_TEST_CASES))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_bare_graph(self, test_input, expected):
        assert rewrite.rewrites(test_input, self.tagger.graph) == [expected]

    @parameterized.expand(list(_LETTERS.items()))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_letter_names(self, letter, name):
        assert rewrite.rewrites(letter, self.tagger.letter_graph) == [name]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_letter_names_are_the_turkish_ones(self):
        """TDK's names: "he" not "ha", "ke" not "ka", "se" not "es"; ı and i differ."""
        letters = dict(load_labels(get_abs_path("data/abbreviation/letters.tsv")))
        assert letters == _LETTERS
        assert (letters["H"], letters["K"], letters["S"]) == ("he", "ke", "se")
        assert (letters["I"], letters["İ"], letters["Ğ"]) == ("ı", "i", "yumuşak ge")
        assert (letters["Q"], letters["W"], letters["X"]) == ("ku", "ve", "iks")

    @parameterized.expand(_ACRONYMS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_table_entries_have_one_reading(self, written, spoken):
        assert rewrite.rewrites(written, self.tagger.graph) == [spoken]
        assert rewrite.rewrites(written, self.tagger_fst) == [f'abbreviation {{ value: "{spoken}" }}']

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_pkk_is_pe_ka_ka(self):
        assert rewrite.rewrites("PKK", self.tagger.graph) == ["pe ka ka"]
        assert "pe ke ke" not in rewrite.rewrites("PKK", self.tagger.graph)
        # the letter by letter reading exists, but not for PKK
        assert rewrite.rewrites("PKK", self.tagger.initialism_graph) == ["pe ke ke"]
        assert rewrite.rewrites("PK", self.tagger.graph) == ["pe ke"]

    @parameterized.expand([("CPU", "si pi yu", "ce pe u"), ("GPU", "ci pi yu", "ge pe u")])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_processor_acronyms(self, written, spoken, letter_names):
        assert rewrite.rewrites(written, self.tagger.graph) == [spoken]
        assert letter_names not in rewrite.rewrites(written, self.tagger.graph)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_bmw_is_be_me_ve(self):
        assert rewrite.rewrites("BMW", self.tagger.graph) == ["be me ve"]
        assert all("dabılyu" not in r for r in rewrite.rewrites("BMW", self.tagger.graph))

    @parameterized.expand([(written,) for written in ["AŞ", "MÖ", "MS"]])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_whitelist_entries_are_left_to_the_whitelist(self, written):
        assert not self._accepts(self.tagger_fst, written)
        assert self._accepts(self.whitelist.fst, written)

    @parameterized.expand(_INVALID)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_invalid_input_is_rejected(self, test_input):
        assert not self._accepts(self.tagger_fst, test_input)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_no_organisation_names(self):
        for written, name in [("TDK", "Türk Dil Kurumu"), ("TBMM", "Türkiye Büyük Millet Meclisi")]:
            assert name.lower() not in self._normalize(written)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_foreign_letters(self):
        """Q, W and X have conventional names; W is "ve", like V."""
        assert rewrite.rewrites("WWW", self.tagger.graph) == ["ve ve ve"]
        assert rewrite.rewrites("TWX", self.tagger.graph) == ["te ve iks"]
        assert rewrite.rewrites("QA", self.tagger.graph) == ["ku a"]
        assert rewrite.rewrites("XL", self.tagger.graph) == ["iks le"]
        assert self._normalize("VW") == self._normalize("WV") == "ve ve"

    @parameterized.expand([("TDK", "TKD"), ("THY", "TYH"), ("Iİ", "İI"), ("TBMM", "TBM"), ("ÖO", "OÖ")])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_distinct_abbreviations_read_differently(self, first, second):
        assert self._normalize(first) != self._normalize(second)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_every_two_letter_initialism(self):
        """All 29 x 29 two letter strings, against the independent letter names."""
        special = {w for w, _ in _ACRONYMS} | set(_WHITELISTED)
        for pair in itertools.product(_LETTERS, repeat=2):
            written = "".join(pair)
            if written in special:
                continue
            assert rewrite.rewrites(written, self.tagger.graph) == [_spell(written)], written

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_token_schema(self):
        assert rewrite.top_rewrite("TDK", self.tagger_fst) == 'abbreviation { value: "te de ke" }'
        assert rewrite.top_rewrite("NATO", self.tagger_fst) == 'abbreviation { value: "nato" }'

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_no_input_is_shared_with_another_grammar(self):
        """Exact, over all inputs: the input languages do not intersect."""
        cardinal = CardinalTaggerFst()
        decimal = DecimalTaggerFst(cardinal=cardinal)
        fraction = FractionTaggerFst(cardinal=cardinal)
        others = {
            "whitelist": self.whitelist.fst,
            "cardinal": cardinal.fst,
            "ordinal": OrdinalTaggerFst(cardinal=cardinal).fst,
            "decimal": decimal.fst,
            "fraction": fraction.fst,
            "date": DateTaggerFst(cardinal=cardinal).fst,
            "time": TimeTaggerFst(cardinal=cardinal).fst,
            "percentage": PercentageTaggerFst(cardinal=cardinal, decimal=decimal).fst,
            "money": MoneyTaggerFst(cardinal=cardinal, decimal=decimal).fst,
            "measure": MeasureTaggerFst(cardinal=cardinal, decimal=decimal, fraction=fraction).fst,
            "telephone": TelephoneTaggerFst(cardinal=cardinal).fst,
            "electronic": ElectronicTaggerFst().fst,
        }
        abbreviation_inputs = pynini.project(self.tagger_fst, "input").optimize()
        for name, fst in others.items():
            shared = pynini.intersect(abbreviation_inputs, pynini.project(fst, "input").optimize()).optimize()
            assert shared.num_states() == 0, name

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_graph_has_no_state_explosion(self):
        assert self.tagger_fst.num_states() < 500, self.tagger_fst.num_states()
        assert self.verbalizer_fst.num_states() < 100, self.verbalizer_fst.num_states()
