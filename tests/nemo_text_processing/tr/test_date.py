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
Direct tests for the Turkish date tagger and verbalizer.

Year first dates such as "2026-09-29" are tagged in written order without
``preserve_order`` and only reach the verbalizer in day/month/year order through
normalize.py's field permutation. ``_readings`` below runs every tagged token through
the token parser and ``Normalizer._permute`` itself, so these tests exercise the same
path the Normalizer will, for year first and day first dates alike.
"""

import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.token_parser import PRESERVE_ORDER_KEY, TokenParser
from nemo_text_processing.text_normalization.tr.taggers.cardinal import CardinalFst as CardinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.date import DateFst as DateTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.fraction import FractionFst as FractionTaggerFst
from nemo_text_processing.text_normalization.tr.verbalizers.date import DateFst as DateVerbalizerFst

from ..utils import parse_test_case_file

_TEST_CASES = 'tr/data_text_normalization/test_cases_date.txt'

_MONTHS = ["ocak", "şubat", "mart", "nisan", "mayıs", "haziran"]
_MONTHS += ["temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık"]

_EYLUL_2026 = "yirmi dokuz eylül iki bin yirmi altı"


def _tr_upper(word):
    """Turkish upper casing for building test inputs; str.upper() maps i to I."""
    return "".join({"i": "İ", "ı": "I"}.get(c, c.upper()) for c in word)


_TEXTUAL = [
    ("1 Ocak 2026", "bir ocak iki bin yirmi altı"),
    ("29 Eylül 2026", _EYLUL_2026),
    ("23 Nisan 1920", "yirmi üç nisan bin dokuz yüz yirmi"),
    ("29 Ekim 1923", "yirmi dokuz ekim bin dokuz yüz yirmi üç"),
    ("31 Aralık 1999", "otuz bir aralık bin dokuz yüz doksan dokuz"),
]

# Every month in capitals. NİSAN, EKİM and HAZİRAN need the dotted capital İ, and
# MAYIS, KASIM and ARALIK the dotless I, which ASCII casing gets the wrong way round.
_CAPITALS = [
    ("1 OCAK 2026", "bir ocak iki bin yirmi altı"),
    ("1 ŞUBAT 2026", "bir şubat iki bin yirmi altı"),
    ("1 MART 2026", "bir mart iki bin yirmi altı"),
    ("1 NİSAN 2026", "bir nisan iki bin yirmi altı"),
    ("1 MAYIS 2026", "bir mayıs iki bin yirmi altı"),
    ("1 HAZİRAN 2026", "bir haziran iki bin yirmi altı"),
    ("1 TEMMUZ 2026", "bir temmuz iki bin yirmi altı"),
    ("1 AĞUSTOS 2026", "bir ağustos iki bin yirmi altı"),
    ("1 EYLÜL 2026", "bir eylül iki bin yirmi altı"),
    ("1 EKİM 2026", "bir ekim iki bin yirmi altı"),
    ("1 KASIM 2026", "bir kasım iki bin yirmi altı"),
    ("1 ARALIK 2026", "bir aralık iki bin yirmi altı"),
]

# The CLDR Turkish abbreviated month names, with and without a full stop.
_ABBREVIATIONS = [
    ("oca", "ocak"),
    ("şub", "şubat"),
    ("mar", "mart"),
    ("nis", "nisan"),
    ("may", "mayıs"),
    ("haz", "haziran"),
    ("tem", "temmuz"),
    ("ağu", "ağustos"),
    ("eyl", "eylül"),
    ("eki", "ekim"),
    ("kas", "kasım"),
    ("ara", "aralık"),
]

_PARTIAL = [
    ("29 Eylül", "yirmi dokuz eylül"),
    ("1 Ocak", "bir ocak"),
    ("Eylül 2026", "eylül iki bin yirmi altı"),
    ("Ocak 2026", "ocak iki bin yirmi altı"),
]

_NON_PADDED = [
    ("1/1/2026", "bir ocak iki bin yirmi altı"),
    ("3/4/2026", "üç nisan iki bin yirmi altı"),
    ("29/9/2026", _EYLUL_2026),
    ("9/9/2026", "dokuz eylül iki bin yirmi altı"),
]

_ISO = [
    ("2026-01-01", "bir ocak iki bin yirmi altı"),
    ("2026-09-29", _EYLUL_2026),
    ("1923-10-29", "yirmi dokuz ekim bin dokuz yüz yirmi üç"),
    ("2026/09/29", _EYLUL_2026),
    ("2026.09.29", _EYLUL_2026),
]

_YEARS = [
    ("1920", "bin dokuz yüz yirmi"),
    ("1923", "bin dokuz yüz yirmi üç"),
    ("1999", "bin dokuz yüz doksan dokuz"),
    ("2000", "iki bin"),
    ("2006", "iki bin altı"),
    ("2024", "iki bin yirmi dört"),
    ("2026", "iki bin yirmi altı"),
    ("2099", "iki bin doksan dokuz"),
]

_OUT_OF_RANGE = [
    ("00/01/2026",),
    ("0/01/2026",),
    ("32/01/2026",),
    ("99/01/2026",),
    ("001/01/2026",),
    ("01/00/2026",),
    ("01/0/2026",),
    ("01/13/2026",),
    ("01/012/2026",),
    ("0 Ocak 2026",),
    ("32 Ocak 2026",),
    ("2026-00-29",),
    ("2026-13-29",),
    ("2026-09-00",),
    ("2026-09-32",),
]

_MALFORMED = [
    ("",),
    ("29//09/2026",),
    ("29/09//2026",),
    ("29/09/2026/1",),
    ("29/09-",),
    ("/09/2026",),
    ("29//2026",),
    ("abc/09/2026",),
    ("29/abc/2026",),
    ("29/09/abc",),
    ("29/09/",),
    ("29 / 09 / 2026",),
    ("29 09 2026",),
    ("29.09.2026.",),  # sentence final full stop is punctuation, not part of the date
    ("29 Eylül 2026.",),
    ("29 Eylül.",),
    ("29  Eylül 2026",),
    ("29Eylül2026",),
    ("29 Eyl.. 2026",),
    ("Eylül",),
    ("29",),
    ("2026",),
]

_MIXED_SEPARATORS = [
    ("29/09-2026",),
    ("29.09/2026",),
    ("29-09.2026",),
    ("29/09.2026",),
    ("29.09-2026",),
    ("29-09/2026",),
    ("2026-09/29",),
    ("2026.09-29",),
    ("2026/09.29",),
]

# Short and malformed years: read as written or not at all, never with an invented
# century, and four digit years only for now.
_YEAR_POLICY = [
    ("29/09/26",),
    ("29.09.26",),
    ("29-09-26",),
    ("29/09/202",),
    ("29/09/0226",),
    ("29/09/02026",),
    ("29/09/20260",),
    ("29 Eylül 26",),
    ("29 Eylül 02026",),
    ("26-09-29",),
]

# Orders Turkish does not write, and month names Turkish casing does not produce.
_UNSUPPORTED_TEXTUAL = [
    ("Eylül 29 2026",),
    ("Eylül 29",),
    ("2026 Eylül 29",),
    ("2026 Eylül",),
    ("1 NISAN 2026",),  # ASCII capital I lower cases to ı: "nısan"
    ("29 EKIM 1923",),
    ("1 HAZIRAN 2026",),
    ("1 nİsan 2026",),  # mixed case
    ("29 EyLüL 2026",),
    ("29 Eylul 2026",),  # ASCII folded
    ("14 Subat 2000",),
    ("29 Sep 2026",),  # English
    ("29 Eylü 2026",),  # not an abbreviation
    ("1 Ock. 2026",),
    ("29 Ey. 2026",),
]

_NOT_A_DATE_FRACTIONS = [("1/2",), ("3/4",), ("12/25",), ("29/09",), ("3-4",), ("29.09",)]

_DATES_THAT_LOOK_LIKE_FRACTIONS = [
    ("1/2/2026", "bir şubat iki bin yirmi altı"),
    ("3/4/2026", "üç nisan iki bin yirmi altı"),
    ("12/5/2026", "on iki mayıs iki bin yirmi altı"),
]


class TestDate:

    cardinal = CardinalTaggerFst()
    tagger = DateTaggerFst(cardinal=cardinal)
    verbalizer = DateVerbalizerFst()
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
        token = parser.parse()[0]["tokens"]
        readings = set()
        for serialized in cls._permuter._permute(token):
            try:
                readings.add(rewrite.top_rewrite(serialized.strip(), cls.verbalizer_fst))
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

    @parameterized.expand(_TEXTUAL)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_textual(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand([(".",), ("/",), ("-",)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_numeric_day_month_year(self, sep):
        assert self._normalize(f"01{sep}01{sep}2026") == "bir ocak iki bin yirmi altı"
        assert self._normalize(f"29{sep}09{sep}2026") == _EYLUL_2026
        assert self._normalize(f"31{sep}12{sep}1999") == "otuz bir aralık bin dokuz yüz doksan dokuz"

    @parameterized.expand(_NON_PADDED)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_non_padded(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_ISO)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_year_first(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_YEARS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_year_is_an_ordinary_cardinal(self, year, expected):
        assert self._normalize(f"1 Ocak {year}") == f"bir ocak {expected}"
        assert rewrite.top_rewrite(year, self.tagger.year_graph) == expected

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_leading_zeros_are_not_spoken(self):
        assert self._normalize("03/04/2026") == "üç nisan iki bin yirmi altı"
        for day in range(1, 10):
            for month in range(1, 10):
                expected = self._normalize(f"{day}/{month}/2026")
                assert "sıfır" not in expected
                for written in [f"0{day}/{month}/2026", f"{day}/0{month}/2026", f"0{day}/0{month}/2026"]:
                    assert self._normalize(written) == expected, written

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_day_month_year_is_never_month_day_year(self):
        assert self._normalize("03/04/2026") == "üç nisan iki bin yirmi altı"
        assert self._normalize("04/03/2026") == "dört mart iki bin yirmi altı"
        self._assert_rejected("12/25/2026")

    @parameterized.expand(_CAPITALS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_month_in_capitals(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand([(month,) for month in _MONTHS])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_month_casing_variants_agree(self, month):
        capitalised = _tr_upper(month[0]) + month[1:]
        capitals = _tr_upper(month)
        for written in [month, capitalised, capitals]:
            assert rewrite.rewrites(written, self.tagger.textual_month_graph) == [month], written

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_turkish_casing_not_ascii_casing(self):
        """The same date in the three cases reads the same, and the ASCII capital I is
        the capital of ı, not of i."""
        readings = {self._normalize(s) for s in ["29 eylül 2026", "29 Eylül 2026", "29 EYLÜL 2026"]}
        assert readings == {_EYLUL_2026}
        assert rewrite.top_rewrite("NİSAN", self.tagger.textual_month_graph) == "nisan"
        assert rewrite.top_rewrite("KASIM", self.tagger.textual_month_graph) == "kasım"
        for ascii_cased in ["NISAN", "EKIM", "HAZIRAN", "KASİM", "ARALİK", "MAYİS"]:
            with pytest.raises(rewrite.Error):
                rewrite.top_rewrite(ascii_cased, self.tagger.textual_month_graph)

    @parameterized.expand(_ABBREVIATIONS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_month_abbreviations(self, abbr, month):
        capitalised = _tr_upper(abbr[0]) + abbr[1:]
        capitals = _tr_upper(abbr)
        for written in [abbr, capitalised, capitals]:
            for dot in ["", "."]:
                assert self._normalize(f"1 {written}{dot} 2026") == f"bir {month} iki bin yirmi altı"
                assert rewrite.rewrites(written + dot, self.tagger.textual_month_graph) == [month]

    @parameterized.expand(_PARTIAL)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_partial_textual(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_OUT_OF_RANGE)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_out_of_range_is_rejected(self, test_input):
        self._assert_rejected(test_input)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_boundaries(self):
        assert self._normalize("1/1/2026") == "bir ocak iki bin yirmi altı"
        assert self._normalize("31/12/2026") == "otuz bir aralık iki bin yirmi altı"
        assert self._normalize("01/12/2026") == "bir aralık iki bin yirmi altı"

    @parameterized.expand(_MALFORMED)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_malformed_input_is_rejected(self, test_input):
        self._assert_rejected(test_input)

    @parameterized.expand(_MIXED_SEPARATORS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_mixed_separators_are_rejected(self, test_input):
        self._assert_rejected(test_input)

    @parameterized.expand(_YEAR_POLICY)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_only_four_digit_years(self, test_input):
        self._assert_rejected(test_input)

    @parameterized.expand(_UNSUPPORTED_TEXTUAL)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_unsupported_textual_forms_are_rejected(self, test_input):
        self._assert_rejected(test_input)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_calendar_combinations_are_not_validated(self):
        """Documented policy: each part's range is checked, the combination is not."""
        assert self._normalize("31/04/2026") == "otuz bir nisan iki bin yirmi altı"
        assert self._normalize("29/02/2023") == "yirmi dokuz şubat iki bin yirmi üç"

    @parameterized.expand(parse_test_case_file(_TEST_CASES))
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
        """Day first and textual dates are written in speaking order and keep it; year
        first dates keep the written order and leave the reordering to the parser."""
        day_first = 'date { day: "yirmi dokuz" month: "eylül" year: "iki bin yirmi altı" preserve_order: true }'
        assert rewrite.top_rewrite("29/09/2026", self.tagger_fst) == day_first
        assert rewrite.top_rewrite("29 Eylül 2026", self.tagger_fst) == day_first
        assert rewrite.top_rewrite("29 Eyl. 2026", self.tagger_fst) == day_first
        assert rewrite.top_rewrite("29 Eylül", self.tagger_fst) == (
            'date { day: "yirmi dokuz" month: "eylül" preserve_order: true }'
        )
        assert rewrite.top_rewrite("Eylül 2026", self.tagger_fst) == (
            'date { month: "eylül" year: "iki bin yirmi altı" preserve_order: true }'
        )
        assert rewrite.top_rewrite("2026-09-29", self.tagger_fst) == (
            'date { year: "iki bin yirmi altı" month: "eylül" day: "yirmi dokuz" }'
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_year_first_needs_the_field_permutation(self):
        """Composed directly, the year first token does not verbalize; the Normalizer's
        permutation supplies the one order the verbalizer reads."""
        tagged = rewrite.top_rewrite("2026-09-29", self.tagger_fst)
        assert PRESERVE_ORDER_KEY not in tagged
        with pytest.raises(rewrite.Error):
            rewrite.top_rewrite(tagged, self.verbalizer_fst)
        parser = TokenParser()
        parser(f"tokens {{ {tagged} }}")
        token = parser.parse()[0]["tokens"]
        serialized = self._permuter._permute(token)
        assert len(serialized) == 6
        verbalized = []
        for s in serialized:
            try:
                verbalized.append(rewrite.top_rewrite(s.strip(), self.verbalizer_fst))
            except rewrite.Error:
                pass
        assert verbalized == [_EYLUL_2026]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_verbalizer(self):
        v = self.verbalizer_fst
        assert rewrite.top_rewrite(
            'date { day: "yirmi dokuz" month: "eylül" year: "iki bin yirmi altı" preserve_order: true }', v
        ) == (_EYLUL_2026)
        assert rewrite.top_rewrite('date { day: "bir" month: "ocak" }', v) == "bir ocak"
        assert rewrite.top_rewrite('date { month: "ocak" year: "iki bin" }', v) == "ocak iki bin"
        # no year first or month first reading, and nothing inserted between fields
        for tagged in [
            'date { year: "iki bin" month: "ocak" day: "bir" }',
            'date { month: "ocak" day: "bir" }',
            'date { day: "bir" year: "iki bin" }',
        ]:
            with pytest.raises(rewrite.Error):
                rewrite.top_rewrite(tagged, v)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_reusable_subgraphs_are_exposed(self):
        assert rewrite.top_rewrite("09", self.tagger.day_graph) == "dokuz"
        assert rewrite.top_rewrite("31", self.tagger.day_graph) == "otuz bir"
        assert rewrite.top_rewrite("09", self.tagger.numeric_month_graph) == "eylül"
        assert rewrite.top_rewrite("EYLÜL", self.tagger.textual_month_graph) == "eylül"
        assert rewrite.top_rewrite("Eyl.", self.tagger.textual_month_graph) == "eylül"
        assert rewrite.top_rewrite("9", self.tagger.month_graph) == "eylül"
        assert rewrite.top_rewrite("Eylül", self.tagger.month_graph) == "eylül"
        assert rewrite.top_rewrite("2026", self.tagger.year_graph) == "iki bin yirmi altı"
        assert rewrite.top_rewrite("29.09.2026", self.tagger.final_graph) == (
            'day: "yirmi dokuz" month: "eylül" year: "iki bin yirmi altı" preserve_order: true'
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_generated_cross_check(self):
        """Every separator, padding and order of the same date reads the same, and
        the reading is the day, month name and year read as cardinals."""
        day_words = {d: rewrite.top_rewrite(str(d), self.cardinal.graph) for d in range(1, 32)}
        for year in ["1923", "2026"]:
            year_words = rewrite.top_rewrite(year, self.cardinal.graph)
            for month in range(1, 13):
                for day in range(1, 32):
                    expected = f"{day_words[day]} {_MONTHS[month - 1]} {year_words}"
                    written = {f"{day}/{month}/{year}", f"{day:02d}.{month:02d}.{year}"}
                    written |= {f"{day:02d}-{month}-{year}", f"{year}-{month:02d}-{day:02d}"}
                    written |= {f"{day} {_MONTHS[month - 1]} {year}"}
                    for w in written:
                        assert self._readings(w) == {expected}, w

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_graph_has_no_state_explosion(self):
        assert self.tagger_fst.num_states() < 5000, self.tagger_fst.num_states()
        assert self.verbalizer_fst.num_states() < 1000, self.verbalizer_fst.num_states()


class TestDateFractionSeparation:
    """Slash expressions with two parts are fractions; with three they are dates."""

    cardinal = CardinalTaggerFst()
    date_fst = DateTaggerFst(cardinal=cardinal).fst
    fraction_fst = FractionTaggerFst(cardinal=cardinal).fst

    @parameterized.expand(_NOT_A_DATE_FRACTIONS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_date_rejects_two_part_expressions(self, test_input):
        with pytest.raises(rewrite.Error):
            rewrite.top_rewrite(test_input, self.date_fst)

    @parameterized.expand(_DATES_THAT_LOOK_LIKE_FRACTIONS + [("29/09/2026", _EYLUL_2026)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_three_part_expressions_are_dates_not_fractions(self, test_input, expected):
        assert rewrite.top_rewrite(test_input, self.date_fst).startswith("date {")
        with pytest.raises(rewrite.Error):
            rewrite.top_rewrite(test_input, self.fraction_fst)

    @parameterized.expand([("1/2", "ikide"), ("3/4", "dörtte")])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_fraction_still_accepts_two_part_expressions(self, test_input, denominator):
        assert f'denominator: "{denominator}"' in rewrite.top_rewrite(test_input, self.fraction_fst)
