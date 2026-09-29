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
Direct tests for the Turkish measure tagger and verbalizer.

The number inside a measurement is a nested cardinal, decimal or fraction token. The
measure token carries ``preserve_order``, but normalize.py still permutes the fields of
a nested token, which is what puts a fraction's denominator first. ``_readings`` runs
every tagged token through the token parser and ``Normalizer._permute``, so these tests
exercise the path the Normalizer will.
"""

import re

import pynini
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
from nemo_text_processing.text_normalization.tr.taggers.measure import MeasureFst as MeasureTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.money import MoneyFst as MoneyTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.ordinal import OrdinalFst as OrdinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.percentage import PercentageFst as PercentageTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.time import TimeFst as TimeTaggerFst
from nemo_text_processing.text_normalization.tr.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.tr.verbalizers.cardinal import CardinalFst as CardinalVerbalizerFst
from nemo_text_processing.text_normalization.tr.verbalizers.decimal import DecimalFst as DecimalVerbalizerFst
from nemo_text_processing.text_normalization.tr.verbalizers.fraction import FractionFst as FractionVerbalizerFst
from nemo_text_processing.text_normalization.tr.verbalizers.measure import MeasureFst as MeasureVerbalizerFst

from ..utils import parse_test_case_file

_TEST_CASES = 'tr/data_text_normalization/test_cases_measure.txt'

_UNITS = load_labels(get_abs_path("data/measure/unit.tsv"))
_UNIT_ABBREVIATIONS = load_labels(get_abs_path("data/measure/unit_abbreviations.tsv"))

# Spellings of one unit that must give the same token.
_ALIASES = [
    ("L", "l"),
    ("mL", "ml"),
    ("h", "sa", "sa."),
    ("min", "dk", "dk."),
    ("s", "sn", "sn."),
    ("m²", "m2"),
    ("cm²", "cm2"),
    ("km²", "km2"),
    ("m³", "m3"),
    ("cm³", "cm3"),
    ("s²", "s2"),
]

_INVALID = [
    ("",),
    ("kg",),  # a unit needs a number
    ("m",),
    ("°C",),
    ("km/h",),
    ("5  kg",),  # one space at most
    ("5   m",),
    (" 5 kg",),
    ("5 kg ",),
    ("05 kg",),  # leading zeros and grouping as CardinalFst and DecimalFst
    ("05kg",),
    ("01.000 kg",),
    ("1,,5 kg",),
    ("1.5 kg",),  # the decimal separator is a comma
    ("14.30 s",),
    ("5 kg//m",),  # one slash in a unit, and no spaces around it
    ("5 kg/m/s",),
    ("5 m/s/s",),
    ("5 km / h",),
    ("5 km/ h",),
    ("5 km /h",),
    ("5 /h",),
    ("5 kg/",),
    ("5 sa./km",),  # the dotted abbreviations end the unit
    ("5 KG",),  # units are case sensitive
    ("5 Kg",),
    ("5 KM",),
    ("5 M",),
    ("5 k",),
    ("5 hz",),
    ("5 mhz",),
    ("20 C",),
    ("20 °c",),
    ("20 ° C",),
    ("5 m^2",),  # other power notations
    ("5 m**2",),
    ("5 m 2",),
    ("5-10 kg",),  # ranges and dimensions
    ("5–10 kg",),
    ("3x4 m",),
    ("5×10 cm",),
    ("100€",),  # other classes
    ("100 TL",),
    ("%25",),
    ("%25 kg",),
    ("29.09.2026",),
    ("14.30",),
    ("3/4",),
    ("5.",),
    ("5",),
    ("1,5",),
    ("5 kg'dan",),  # suffixed forms are deferred
    ("10 km'ye",),
    ("20 m'de",),
    ("5 cm'nin",),
    ("20 °C'de",),
    ("5 kilogram",),  # unit names are words, not units
    ("10 metre",),
    ("+5 kg",),  # only a leading minus
    ("kg -5",),
    ("5- kg",),
    ("--5 kg",),
    ("5 Pa",),  # deferred units
    ("5 kPa",),
    ("5 mm²",),
    ("5 N·m",),
    ("5 kg m/s²",),
    ("5 kg.",),
]


class TestMeasure:

    cardinal = CardinalTaggerFst()
    decimal = DecimalTaggerFst(cardinal=cardinal)
    fraction = FractionTaggerFst(cardinal=cardinal)
    tagger = MeasureTaggerFst(cardinal=cardinal, decimal=decimal, fraction=fraction)
    cardinal_verbalizer = CardinalVerbalizerFst()
    decimal_verbalizer = DecimalVerbalizerFst(cardinal=cardinal_verbalizer)
    fraction_verbalizer = FractionVerbalizerFst()
    verbalizer = MeasureVerbalizerFst(
        cardinal=cardinal_verbalizer, decimal=decimal_verbalizer, fraction=fraction_verbalizer
    )
    tagger_fst = tagger.fst
    verbalizer_fst = verbalizer.fst

    class _Permuter:
        _permute = Normalizer._permute

    _permuter = _Permuter()

    @classmethod
    def _serializations(cls, written, tagger_fst=None):
        tagged = rewrite.top_rewrite(written, tagger_fst or cls.tagger_fst)
        parser = TokenParser()
        parser(f"tokens {{ {tagged} }}")
        return [s.strip() for s in cls._permuter._permute(parser.parse()[0]["tokens"])]

    @classmethod
    def _readings(cls, written, tagger_fst=None, verbalizer_fst=None):
        """Every reading the tagger plus the verbalizer admit, via the token parser and
        the Normalizer's own field permutation."""
        readings = set()
        for serialized in cls._serializations(written, tagger_fst):
            try:
                readings.add(rewrite.top_rewrite(serialized, verbalizer_fst or cls.verbalizer_fst))
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

    def _number_reading(self, number):
        """The reading the number's own grammar gives it."""
        if "/" in number:
            fst, verbalizer = self.fraction.fst, self.fraction_verbalizer.fst
        elif "," in number or " " in number:
            fst, verbalizer = self.decimal.fst, self.decimal_verbalizer.fst
        else:
            fst, verbalizer = self.cardinal.fst, self.cardinal_verbalizer.fst
        readings = self._readings(number, fst, verbalizer)
        assert len(readings) == 1, readings
        return readings.pop()

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
        """Every cardinal and decimal measurement; fractions are documented as absent."""
        if re.match(r"-?\d+/\d", test_input):
            with pytest.raises(rewrite.Error):
                rewrite.rewrites(test_input, self.tagger.graph)
        else:
            assert rewrite.rewrites(test_input, self.tagger.graph) == [expected]

    @parameterized.expand([(symbol, word) for symbol, word in _UNITS + _UNIT_ABBREVIATIONS])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_every_unit(self, symbol, word):
        assert self._normalize(f"5 {symbol}") == f"beş {word}"
        assert self._tag(f"5{symbol}") == self._tag(f"5 {symbol}")

    @parameterized.expand([(symbol, word) for symbol, word in _UNITS])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_unit_in_a_compound(self, symbol, word):
        assert self._normalize(f"5 km/{symbol}") == f"beş kilometre bölü {word}"
        assert self._normalize(f"5 {symbol}/sa") == f"beş {word} bölü saat"

    @parameterized.expand([("5",), ("-5",), ("1.234",), ("1,5",), ("-12,05",), ("3/4",), ("-1/2",), ("1 milyon",)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_spaced_and_attached_units_give_the_same_token(self, number):
        for unit in ["kg", "km", "mL", "°C", "°", "m²", "km/sa", "kg/m³"]:
            assert self._tag(f"{number} {unit}") == self._tag(f"{number}{unit}"), unit

    @parameterized.expand(_ALIASES)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_aliases_give_the_same_token(self, *aliases):
        assert len({self._tag(f"5 {alias}") for alias in aliases}) == 1

    @parameterized.expand(
        [("0",), ("1",), ("2",), ("100",), ("1.000",), ("1.234.567",), ("-5",)]
        + [("1,5",), ("0,5",), ("12,05",), ("1.234,5",), ("-1,5",), ("3,140",)]
        + [("1/2",), ("3/4",), ("2/3",), ("25/100",), ("-3/4",), ("0/5",)]
        + [("1 milyon",), ("1,5 milyon",), ("150 bin",), ("-2 milyar",)]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_number_reads_as_its_own_grammar_reads_it(self, number):
        for unit, word in [("kg", "kilogram"), ("°C", "santigrat derece"), ("km/sa", "kilometre bölü saat")]:
            assert self._normalize(f"{number} {unit}") == f"{self._number_reading(number)} {word}"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_nested_fraction_is_permuted_under_preserve_order(self):
        """The measure token keeps its order; the nested fraction is still permuted, and
        exactly one of its orders verbalizes."""
        tagged = self._tag("3/4 kg")
        assert tagged == (
            'measure { fraction { numerator: "üç" denominator: "dörtte" } units: "kilogram" preserve_order: true }'
        )
        with pytest.raises(rewrite.Error):
            rewrite.top_rewrite(tagged, self.verbalizer_fst)
        serializations = self._serializations("3/4 kg")
        assert len(serializations) == 2
        verbalized = []
        for s in serializations:
            try:
                verbalized.append(rewrite.top_rewrite(s, self.verbalizer_fst))
            except rewrite.Error:
                pass
        assert verbalized == ["dörtte üç kilogram"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_fraction_slash_and_unit_slash(self):
        assert self._normalize("3/4 kg") == "dörtte üç kilogram"
        assert self._normalize("90 km/h") == "doksan kilometre bölü saat"
        assert self._normalize("3/4 kg/m³") == "dörtte üç kilogram bölü metreküp"
        assert '"kilogram bölü metreküp"' in self._tag("3/4 kg/m³")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_degree_celsius_is_not_degree_then_c(self):
        assert self._normalize("20°C") == "yirmi santigrat derece"
        assert self._normalize("20°") == "yirmi derece"

    @parameterized.expand(_INVALID)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_invalid_input_is_rejected(self, test_input):
        with pytest.raises(rewrite.Error):
            rewrite.top_rewrite(test_input, self.tagger_fst)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_token_schema(self):
        assert self._tag("5 kg") == 'measure { cardinal { integer: "beş" } units: "kilogram" preserve_order: true }'
        assert self._tag("1,5 kg") == (
            'measure { decimal { integer_part: "bir" fractional_part: "beş" } units: "kilogram" preserve_order: true }'
        )
        assert self._tag("-5 °C") == (
            'measure { cardinal { negative: "true" integer: "beş" } units: "santigrat derece" preserve_order: true }'
        )
        assert self._tag("1,5 milyon km") == (
            'measure { decimal { integer_part: "bir" fractional_part: "beş" quantity: "milyon" }'
            ' units: "kilometre" preserve_order: true }'
        )
        assert self._tag("90 km/sa") == (
            'measure { cardinal { integer: "doksan" } units: "kilometre bölü saat" preserve_order: true }'
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_reusable_subgraphs_are_exposed(self):
        assert rewrite.top_rewrite("kg", self.tagger.unit_graph) == "kilogram"
        assert rewrite.top_rewrite("km/sa", self.tagger.unit_graph) == "kilometre bölü saat"
        assert rewrite.top_rewrite("5 kg", self.tagger.final_graph) == (
            'cardinal { integer: "beş" } units: "kilogram" preserve_order: true'
        )
        assert rewrite.top_rewrite("5 kg", self.tagger.graph) == "beş kilogram"
        assert rewrite.top_rewrite("-1,5 km", self.tagger.graph) == "eksi bir virgül beş kilometre"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_bare_reading_takes_the_locative(self):
        """What the later suffix layer needs: the spoken unit, not the symbol, takes
        the suffix."""
        locative = self.tagger.graph @ LOCATIVE_SUFFIX
        assert rewrite.rewrites("5 kg", locative) == ["beş kilogramda"]
        assert rewrite.rewrites("20 m", locative) == ["yirmi metrede"]
        assert rewrite.rewrites("5 m²", locative) == ["beş metrekarede"]
        assert rewrite.rewrites("20 °C", locative) == ["yirmi santigrat derecede"]
        assert rewrite.rewrites("3 s", locative) == ["üç saniyede"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_generated_cross_check(self):
        for n in range(0, 201):
            words = rewrite.top_rewrite(str(n), self.cardinal.graph)
            for unit, word in [("kg", "kilogram"), ("m²", "metrekare"), ("km/sa", "kilometre bölü saat")]:
                for written in [f"{n} {unit}", f"{n}{unit}"]:
                    assert rewrite.rewrites(written, self.tagger.graph) == [f"{words} {word}"], written
                    assert rewrite.rewrites(rewrite.top_rewrite(written, self.tagger_fst), self.verbalizer_fst) == [
                        f"{words} {word}"
                    ]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_graph_has_no_state_explosion(self):
        assert self.tagger_fst.num_states() < 8000, self.tagger_fst.num_states()
        assert self.verbalizer_fst.num_states() < 500, self.verbalizer_fst.num_states()


class TestMeasureSeparation:
    """Measure against every earlier Turkish grammar."""

    cardinal = CardinalTaggerFst()
    decimal = DecimalTaggerFst(cardinal=cardinal)
    fraction = FractionTaggerFst(cardinal=cardinal)
    measure_fst = MeasureTaggerFst(cardinal=cardinal, decimal=decimal, fraction=fraction).fst
    others = {
        "cardinal": cardinal.fst,
        "decimal": decimal.fst,
        "fraction": fraction.fst,
        "date": DateTaggerFst(cardinal=cardinal).fst,
        "time": TimeTaggerFst(cardinal=cardinal).fst,
        "ordinal": OrdinalTaggerFst(cardinal=cardinal).fst,
        "percentage": PercentageTaggerFst(cardinal=cardinal, decimal=decimal).fst,
        "money": MoneyTaggerFst(cardinal=cardinal, decimal=decimal).fst,
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
        measure_inputs = pynini.project(self.measure_fst, "input").optimize()
        for name, fst in self.others.items():
            shared = pynini.intersect(measure_inputs, pynini.project(fst, "input").optimize()).optimize()
            assert shared.num_states() == 0, name

    @parameterized.expand(
        [("3/4", "fraction"), ("14.30", "time"), ("29.09.2026", "date"), ("12,50", "decimal"), ("%25", "percentage")]
        + [("100€", "money"), ("100 TL", "money"), ("5.", "ordinal"), ("5", "cardinal")]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_other_classes_are_not_measurements(self, test_input, owner):
        assert self._accepts(self.others[owner], test_input)
        assert not self._accepts(self.measure_fst, test_input)

    @parameterized.expand([("3/4 kg",), ("14,30 s",), ("5 m",), ("100 kg",), ("20 °C",)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_measurements_belong_to_no_other_grammar(self, test_input):
        assert self._accepts(self.measure_fst, test_input)
        for name, fst in self.others.items():
            assert not self._accepts(fst, test_input), name
