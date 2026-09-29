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
Direct tests for the Turkish telephone tagger and verbalizer.

Every telephone token carries ``preserve_order``. ``_readings`` still runs the tagged
token through the token parser and ``Normalizer._permute``, so these tests exercise the
path the Normalizer will.
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
from nemo_text_processing.text_normalization.tr.taggers.measure import MeasureFst as MeasureTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.money import MoneyFst as MoneyTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.ordinal import OrdinalFst as OrdinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.percentage import PercentageFst as PercentageTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.telephone import TelephoneFst as TelephoneTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.time import TimeFst as TimeTaggerFst
from nemo_text_processing.text_normalization.tr.verbalizers.telephone import TelephoneFst as TelephoneVerbalizerFst

from ..utils import parse_test_case_file

_TEST_CASES = 'tr/data_text_normalization/test_cases_telephone.txt'

# Written forms of one number that must give the same token.
_EQUIVALENT = [
    ["0532 123 45 67", "0532-123-45-67", "0532 123 4567", "0532-123-4567", "05321234567"]
    + ["0 532 123 45 67", "0 532 123 4567", "0 (532) 123 45 67", "(0532) 123 45 67", "0532 123-45-67"],
    ["+90 532 123 45 67", "+90 532 123 4567", "+90-532-123-45-67", "+905321234567", "+90 (532) 123 45 67"],
    ["532 123 45 67", "532 123 4567", "532-123-45-67", "532-123-4567"],
    ["234 56 78", "234-56-78", "234 5678", "234-5678"],
    ["444 12 34", "444-12-34", "444 1234", "444-1234"],
]

_INVALID = [
    ("",),
    ("0532 123 45",),  # too short
    ("0532 123 45 678",),  # too long
    ("0532 123 45 67 8",),
    ("0612 234 56 78",),  # no such code family
    ("0712 234 56 78",),
    ("0112 234 56 78",),
    ("0012 234 56 78",),
    ("0312 123 45 67",),  # a geographic subscriber number starts with 2-9
    ("0312 023 45 67",),
    ("123 45 67",),
    ("+90 0532 123 45 67",),  # no trunk zero after the country code
    ("+9005321234567",),
    ("+1 202 555 01 23",),  # other countries
    ("+44 20 1234 5678",),
    ("+49 532 123 45 67",),
    ("0090 532 123 45 67",),  # deferred
    ("0532/123/45/67",),  # separators
    ("0532,123,45,67",),
    ("0532::123",),
    ("0532_123_45_67",),
    ("0532.123.45.67",),
    ("0532  123 45 67",),
    ("0532 123  45 67",),
    (" 0532 123 45 67",),
    ("0532 123 45 67 ",),
    ("0 (53) 123 45 67",),  # parentheses
    ("0 (532 123 45 67",),
    ("0 532) 123 45 67",),
    ("0(532) 123 45 67",),
    ("0 (532)123 45 67",),
    ("(532) 123 45 67",),
    ("5321234567",),  # ungrouped numbers CardinalFst also reads
    ("2345678",),
    ("4441234",),
    ("53212345 67",),
    ("112",),  # short numbers
    ("155",),
    ("11811",),
    ("Tel: 0532 123 45 67",),  # prompts, extensions and suffixes
    ("Telefon: 0532 123 45 67",),
    ("0532 123 45 67 dahili 123",),
    ("0532 123 45 67 ext. 123",),
    ("0532 123 45 67 / 123",),
    ("112'yi",),
    ("0532 123 45 67'yi",),
    ("100",),  # other classes
    ("12,50",),
    ("3/4",),
    ("14.30",),
    ("14:30",),
    ("29.09.2026",),
    ("%25",),
    ("100€",),
    ("5 kg",),
]


class TestTelephone:

    cardinal = CardinalTaggerFst()
    tagger = TelephoneTaggerFst(cardinal=cardinal)
    verbalizer = TelephoneVerbalizerFst()
    tagger_fst = tagger.fst
    verbalizer_fst = verbalizer.fst

    class _Permuter:
        _permute = Normalizer._permute

    _permuter = _Permuter()

    @classmethod
    def _readings(cls, written):
        """Every reading the tagger plus the verbalizer admit, via the token parser and
        the Normalizer's own field permutation."""
        tagged = rewrite.top_rewrite(written, cls.tagger_fst)
        parser = TokenParser()
        parser(f"tokens {{ {tagged} }}")
        readings = set()
        for serialized in cls._permuter._permute(parser.parse()[0]["tokens"]):
            try:
                readings.add(rewrite.top_rewrite(serialized.strip(), cls.verbalizer_fst))
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

    @parameterized.expand([(forms,) for forms in _EQUIVALENT])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_written_forms_give_the_same_token(self, forms):
        assert len({self._tag(form) for form in forms}) == 1, forms

    @parameterized.expand(
        [("00", "sıfır sıfır"), ("01", "sıfır bir"), ("05", "sıfır beş"), ("09", "sıfır dokuz")]
        + [("10", "on"), ("45", "kırk beş"), ("99", "doksan dokuz")]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_two_digit_group(self, digits, expected):
        assert rewrite.rewrites(digits, self.tagger.two_digit_group) == [expected]

    @parameterized.expand(
        [("000", "sıfır sıfır sıfır"), ("001", "sıfır sıfır bir"), ("005", "sıfır sıfır beş")]
        + [("009", "sıfır sıfır dokuz"), ("010", "sıfır on"), ("045", "sıfır kırk beş"), ("099", "sıfır doksan dokuz")]
        + [("100", "yüz"), ("123", "yüz yirmi üç"), ("532", "beş yüz otuz iki"), ("999", "dokuz yüz doksan dokuz")]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_three_digit_group(self, digits, expected):
        assert rewrite.rewrites(digits, self.tagger.three_digit_group) == [expected]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_groups_reject_other_lengths(self):
        for digits in ["", "5", "123"]:
            with pytest.raises(rewrite.Error):
                rewrite.top_rewrite(digits, self.tagger.two_digit_group)
        for digits in ["12", "1234"]:
            with pytest.raises(rewrite.Error):
                rewrite.top_rewrite(digits, self.tagger.three_digit_group)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_last_four_digits_are_two_groups(self):
        assert self._normalize("0532 123 4567").endswith("kırk beş altmış yedi")
        assert "dört bin" not in self._normalize("444 4567")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_leading_zeros_survive(self):
        assert self._normalize("0532 005 05 00") == "sıfır beş yüz otuz iki sıfır sıfır beş sıfır beş sıfır sıfır"
        assert self._normalize("+90 505 000 00 00") == (
            "artı doksan beş yüz beş sıfır sıfır sıfır sıfır sıfır sıfır sıfır"
        )

    @parameterized.expand([("2",), ("3",), ("4",), ("5",), ("8",), ("9",)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_code_families(self, first):
        assert self._normalize(f"0{first}12 234 56 78").startswith("sıfır")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_subscriber_first_digit(self):
        """Geographic subscriber numbers start with 2-9; mobile and service ones may
        start with anything."""
        for first in "01":
            for code in ["212", "312", "412"]:
                with pytest.raises(rewrite.Error):
                    rewrite.top_rewrite(f"0{code} {first}23 45 67", self.tagger_fst)
            for code in ["532", "850", "900"]:
                assert self._tag(f"0{code} {first}23 45 67")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_token_schema(self):
        assert self._tag("0532 123 45 67") == (
            'telephone { number_part: "sıfır beş yüz otuz iki yüz yirmi üç kırk beş altmış yedi"'
            ' preserve_order: true }'
        )
        assert self._tag("+90 532 123 45 67") == (
            'telephone { country_code: "artı doksan"'
            ' number_part: "beş yüz otuz iki yüz yirmi üç kırk beş altmış yedi" preserve_order: true }'
        )
        assert self._tag("532 123 45 67") == (
            'telephone { number_part: "beş yüz otuz iki yüz yirmi üç kırk beş altmış yedi" preserve_order: true }'
        )

    @parameterized.expand(_INVALID)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_invalid_input_is_rejected(self, test_input):
        with pytest.raises(rewrite.Error):
            rewrite.top_rewrite(test_input, self.tagger_fst)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_reusable_subgraphs_are_exposed(self):
        reading = "yüz yirmi üç kırk beş altmış yedi"
        assert rewrite.top_rewrite("123 45 67", self.tagger.subscriber_graph) == reading
        assert rewrite.top_rewrite("123 4567", self.tagger.subscriber_graph) == reading
        assert (
            rewrite.top_rewrite("05321234567", self.tagger.national_number_graph)
            == f"sıfır beş yüz otuz iki {reading}"
        )
        assert rewrite.top_rewrite(" 532 123 45 67", self.tagger.international_number_graph) == (
            f"beş yüz otuz iki {reading}"
        )
        assert rewrite.top_rewrite("532 123 45 67", self.tagger.unprefixed_number_graph) == (
            f"beş yüz otuz iki {reading}"
        )
        assert (
            rewrite.top_rewrite("444 12 34", self.tagger.local_number_graph) == "dört yüz kırk dört on iki otuz dört"
        )
        assert rewrite.top_rewrite("+905321234567", self.tagger.graph) == f"artı doksan beş yüz otuz iki {reading}"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_generated_cross_check(self):
        """Every code family and boundary group value, in every written form."""
        three = {"000", "001", "009", "010", "099", "100", "123", "999"}
        two = ["00", "01", "09", "10", "45", "99"]
        for code in ["212", "312", "432", "532", "850", "900"]:
            geographic = code[0] in "234"
            for first in sorted(three):
                if geographic and first[0] in "01":
                    continue
                for a, b in [(x, y) for x in two for y in two]:
                    spoken = " ".join(
                        rewrite.top_rewrite(
                            g, self.tagger.three_digit_group if len(g) == 3 else self.tagger.two_digit_group
                        )
                        for g in [code, first, a, b]
                    )
                    domestic = {
                        f"0{code} {first} {a} {b}",
                        f"0{code}-{first}-{a}-{b}",
                        f"0{code} {first} {a}{b}",
                        f"0{code}{first}{a}{b}",
                        f"0 {code} {first} {a} {b}",
                        f"0 ({code}) {first} {a} {b}",
                    }
                    international = {f"+90 {code} {first} {a} {b}", f"+90{code}{first}{a}{b}"}
                    for forms, expected in [
                        (domestic, f"sıfır {spoken}"),
                        (international, f"artı doksan {spoken}"),
                        ({f"{code} {first} {a} {b}"}, spoken),
                    ]:
                        assert len({self._tag(w) for w in forms}) == 1, forms
                        for w in forms:
                            assert rewrite.rewrites(w, self.tagger.graph) == [expected], w

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_graph_has_no_state_explosion(self):
        assert self.tagger_fst.num_states() < 2500, self.tagger_fst.num_states()
        assert self.verbalizer_fst.num_states() < 200, self.verbalizer_fst.num_states()


class TestTelephoneSeparation:
    """Telephone against every earlier Turkish grammar."""

    cardinal = CardinalTaggerFst()
    decimal = DecimalTaggerFst(cardinal=cardinal)
    fraction = FractionTaggerFst(cardinal=cardinal)
    telephone_fst = TelephoneTaggerFst(cardinal=cardinal).fst
    others = {
        "cardinal": cardinal.fst,
        "ordinal": OrdinalTaggerFst(cardinal=cardinal).fst,
        "decimal": decimal.fst,
        "fraction": fraction.fst,
        "date": DateTaggerFst(cardinal=cardinal).fst,
        "time": TimeTaggerFst(cardinal=cardinal).fst,
        "percentage": PercentageTaggerFst(cardinal=cardinal, decimal=decimal).fst,
        "money": MoneyTaggerFst(cardinal=cardinal, decimal=decimal).fst,
        "measure": MeasureTaggerFst(cardinal=cardinal, decimal=decimal, fraction=fraction).fst,
    }

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_no_input_is_shared_with_another_grammar(self):
        """Exact, over all inputs: the input languages do not intersect."""
        telephone_inputs = pynini.project(self.telephone_fst, "input").optimize()
        for name, fst in self.others.items():
            shared = pynini.intersect(telephone_inputs, pynini.project(fst, "input").optimize()).optimize()
            assert shared.num_states() == 0, name
