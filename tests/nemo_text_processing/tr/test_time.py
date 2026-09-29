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
Direct tests for the Turkish time tagger and verbalizer.

Every time token carries ``preserve_order``, so the Normalizer's field permutation
yields exactly one serialization. ``_readings`` still runs the tagged token through the
token parser and ``Normalizer._permute``, so these tests exercise the path the
Normalizer will take.
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
from nemo_text_processing.text_normalization.tr.taggers.time import TimeFst as TimeTaggerFst
from nemo_text_processing.text_normalization.tr.verbalizers.time import TimeFst as TimeVerbalizerFst

from ..utils import parse_test_case_file

_TEST_CASES = 'tr/data_text_normalization/test_cases_time.txt'

# Hours and minutes, written with a colon; each is also checked with a full stop.
_HM = [
    ("0:00", "sıfır"),
    ("00:00", "sıfır"),
    ("0:01", "sıfır sıfır bir"),
    ("00:05", "sıfır sıfır beş"),
    ("1:00", "bir"),
    ("01:00", "bir"),
    ("8:05", "sekiz sıfır beş"),
    ("08:05", "sekiz sıfır beş"),
    ("09:15", "dokuz on beş"),
    ("13:00", "on üç"),
    ("14:30", "on dört otuz"),
    ("17:30", "on yedi otuz"),
    ("17:45", "on yedi kırk beş"),
    ("23:59", "yirmi üç elli dokuz"),
]

_LEADING_ZERO_HOURS = [
    ("0:30", "sıfır otuz"),
    ("00:30", "sıfır otuz"),
    ("8:30", "sekiz otuz"),
    ("08:30", "sekiz otuz"),
    ("9:59", "dokuz elli dokuz"),
    ("09:59", "dokuz elli dokuz"),
]

# 01-09 keep the written zero, 10-59 are cardinals.
_MINUTES = [
    ("8:01", "sekiz sıfır bir"),
    ("8:05", "sekiz sıfır beş"),
    ("8:09", "sekiz sıfır dokuz"),
    ("14:09", "on dört sıfır dokuz"),
    ("14:10", "on dört on"),
    ("8:15", "sekiz on beş"),
    ("14:25", "on dört yirmi beş"),
    ("8:30", "sekiz otuz"),
    ("14:59", "on dört elli dokuz"),
]

_ZERO_MINUTES = [
    ("14:00", "on dört"),
    ("08:00", "sekiz"),
    ("8:00", "sekiz"),
    ("00:00", "sıfır"),
    ("0:00", "sıfır"),
    ("13.00", "on üç"),
    ("23.00", "yirmi üç"),
]

_HOUR_BOUNDARIES = [
    ("0:30", "sıfır otuz"),
    ("9:30", "dokuz otuz"),
    ("10:30", "on otuz"),
    ("19:30", "on dokuz otuz"),
    ("20:30", "yirmi otuz"),
    ("23:30", "yirmi üç otuz"),
]

_HMS = [
    ("00:00:00", "sıfır sıfır sıfır sıfır sıfır"),
    ("08:05:09", "sekiz sıfır beş sıfır dokuz"),
    ("14:30:05", "on dört otuz sıfır beş"),
    ("23:59:59", "yirmi üç elli dokuz elli dokuz"),
]

# Seconds make every written component audible.
_HMS_ZEROS = [
    ("14:00:05", "on dört sıfır sıfır sıfır beş"),
    ("14:30:00", "on dört otuz sıfır sıfır"),
    ("14:00:00", "on dört sıfır sıfır sıfır sıfır"),
    ("0:00:00", "sıfır sıfır sıfır sıfır sıfır"),
    ("8:00:10", "sekiz sıfır sıfır on"),
]

_MALFORMED = [
    ("",),
    ("14",),
    ("08",),
    ("14:",),
    ("14.",),
    (":30",),
    (".30",),
    ("14:5",),
    ("14.5",),
    ("8:0",),
    ("8:5",),
    ("014:30",),
    ("008:05",),
    ("14:005",),
    ("14:300",),
    ("14.:30",),
    ("14:.30",),
    ("14..30",),
    ("14::30",),
    ("14,30",),
    ("14 : 30",),
    ("14. 30",),
    ("14 :30",),
    ("14: 30",),
    (" 14:30",),
    ("14:30 ",),
    ("14:30.",),  # sentence final full stop is punctuation
    ("14.30.",),
    ("14h30",),
    ("14-30",),
    ("14/30",),
    ("Saat 14.30",),
    ("17.30'da",),  # suffixed forms are deferred
    ("09.15'te",),
    ("13.00'te",),
    ("abc",),
    ("ab:cd",),
    ("1a:30",),
]

_OUT_OF_RANGE = [
    ("24:30",),
    ("24:00",),
    ("24.00",),
    ("25:00",),
    ("99:00",),
    ("14:60",),
    ("14:99",),
    ("8:60",),
    ("8:99",),
    ("-1:30",),
]

_HMS_REJECTED = [
    ("14:30:60",),
    ("14:60:00",),
    ("24:00:00",),
    ("14:30:5",),
    ("14:5:05",),
    ("014:30:05",),
    ("14:30:005",),
    ("14.30.05",),  # dotted seconds are deferred
    ("14.30:05",),
    ("14:30.05",),
    ("14:30:05:00",),
    ("14::05",),
]


class TestTime:

    cardinal = CardinalTaggerFst()
    tagger = TimeTaggerFst(cardinal=cardinal)
    verbalizer = TimeVerbalizerFst()
    tagger_fst = tagger.fst
    verbalizer_fst = verbalizer.fst

    class _Permuter:
        _permute = Normalizer._permute

    _permuter = _Permuter()

    @classmethod
    def _serializations(cls, written):
        tagged = rewrite.top_rewrite(written, cls.tagger_fst)
        parser = TokenParser()
        parser(f"tokens {{ {tagged} }}")
        return [s.strip() for s in cls._permuter._permute(parser.parse()[0]["tokens"])]

    @classmethod
    def _readings(cls, written):
        """Every reading the tagger plus the verbalizer admit, via the token parser and
        the Normalizer's own field permutation."""
        readings = set()
        for serialized in cls._serializations(written):
            try:
                readings.add(rewrite.top_rewrite(serialized, cls.verbalizer_fst))
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

    @parameterized.expand(_HM)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_colon_and_full_stop_read_the_same(self, colon, expected):
        dotted = colon.replace(":", ".")
        assert self._normalize(colon) == expected
        assert self._normalize(dotted) == expected
        assert rewrite.top_rewrite(colon, self.tagger_fst) == rewrite.top_rewrite(dotted, self.tagger_fst)

    @parameterized.expand(_LEADING_ZERO_HOURS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_leading_zero_hours(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_MINUTES)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_leading_zero_minutes(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_ZERO_MINUTES)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_zero_minutes_are_dropped_without_seconds(self, test_input, expected):
        assert self._normalize(test_input) == expected
        assert "minutes" not in rewrite.top_rewrite(test_input, self.tagger_fst)

    @parameterized.expand(_HOUR_BOUNDARIES)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_hour_boundaries(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_minute_boundaries(self):
        assert self._normalize("8:00") == "sekiz"
        assert self._normalize("8:01") == "sekiz sıfır bir"
        assert self._normalize("8:09") == "sekiz sıfır dokuz"
        assert self._normalize("8:10") == "sekiz on"
        assert self._normalize("8:59") == "sekiz elli dokuz"
        self._assert_rejected("8:60")

    @parameterized.expand(_HMS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_hours_minutes_seconds(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_HMS_ZEROS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_seconds_keep_zero_components(self, test_input, expected):
        assert self._normalize(test_input) == expected

    @parameterized.expand(_MALFORMED)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_malformed_input_is_rejected(self, test_input):
        self._assert_rejected(test_input)

    @parameterized.expand(_OUT_OF_RANGE)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_out_of_range_is_rejected(self, test_input):
        self._assert_rejected(test_input)

    @parameterized.expand(_HMS_REJECTED)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_malformed_seconds_are_rejected(self, test_input):
        self._assert_rejected(test_input)

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
        assert rewrite.top_rewrite("14:30", self.tagger_fst) == (
            'time { hours: "on dört" minutes: "otuz" preserve_order: true }'
        )
        assert rewrite.top_rewrite("14.30", self.tagger_fst) == (
            'time { hours: "on dört" minutes: "otuz" preserve_order: true }'
        )
        assert rewrite.top_rewrite("08:05", self.tagger_fst) == (
            'time { hours: "sekiz" minutes: "sıfır beş" preserve_order: true }'
        )
        assert rewrite.top_rewrite("14:00", self.tagger_fst) == 'time { hours: "on dört" preserve_order: true }'
        assert rewrite.top_rewrite("14:30:05", self.tagger_fst) == (
            'time { hours: "on dört" minutes: "otuz" seconds: "sıfır beş" preserve_order: true }'
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_no_field_permutation_is_needed(self):
        """preserve_order leaves one serialization, which is also what direct
        composition of tagger and verbalizer reads."""
        for written in ["14:30", "14:00", "14:30:05"]:
            serializations = self._serializations(written)
            assert len(serializations) == 1, serializations
            direct = rewrite.top_rewrite(rewrite.top_rewrite(written, self.tagger_fst), self.verbalizer_fst)
            assert direct == self._normalize(written)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_verbalizer_inserts_nothing(self):
        v = self.verbalizer_fst
        assert rewrite.top_rewrite('time { hours: "on dört" minutes: "otuz" preserve_order: true }', v) == (
            "on dört otuz"
        )
        assert rewrite.top_rewrite('time { hours: "on dört" }', v) == "on dört"
        for tagged in ['time { minutes: "otuz" hours: "on dört" }', 'time { hours: "on dört" seconds: "beş" }']:
            with pytest.raises(rewrite.Error):
                rewrite.top_rewrite(tagged, v)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_reusable_subgraphs_are_exposed(self):
        assert rewrite.top_rewrite("08", self.tagger.hour_graph) == "sekiz"
        assert rewrite.top_rewrite("8", self.tagger.hour_graph) == "sekiz"
        assert rewrite.top_rewrite("00", self.tagger.hour_graph) == "sıfır"
        assert rewrite.top_rewrite("00", self.tagger.minute_graph) == "sıfır sıfır"
        assert rewrite.top_rewrite("05", self.tagger.minute_graph) == "sıfır beş"
        assert rewrite.top_rewrite("45", self.tagger.minute_graph) == "kırk beş"
        assert rewrite.top_rewrite("05", self.tagger.second_graph) == "sıfır beş"
        assert rewrite.top_rewrite("14.30", self.tagger.final_graph) == (
            'hours: "on dört" minutes: "otuz" preserve_order: true'
        )
        assert rewrite.top_rewrite("14.30", self.tagger.graph) == "on dört otuz"
        assert rewrite.top_rewrite("14:00:05", self.tagger.graph) == "on dört sıfır sıfır sıfır beş"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_bare_reading_takes_the_locative(self):
        """What the later suffix layer needs: 17.30'da, 09.15'te, 13.00'te."""
        locative = self.tagger.graph @ LOCATIVE_SUFFIX
        assert rewrite.rewrites("17.30", locative) == ["on yedi otuzda"]
        assert rewrite.rewrites("09.15", locative) == ["dokuz on beşte"]
        assert rewrite.rewrites("13.00", locative) == ["on üçte"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_generated_cross_check(self):
        """All 24 x 60 clock values in all four hour/minute spellings."""
        hours = {h: rewrite.top_rewrite(str(h), self.cardinal.graph) for h in range(24)}
        minutes = {m: rewrite.top_rewrite(str(m), self.cardinal.graph) for m in range(10, 60)}
        minutes.update({m: "sıfır " + rewrite.top_rewrite(str(m), self.cardinal.graph) for m in range(1, 10)})
        for h in range(24):
            for m in range(60):
                expected = hours[h] if m == 0 else f"{hours[h]} {minutes[m]}"
                for written in {f"{h}:{m:02d}", f"{h:02d}:{m:02d}", f"{h}.{m:02d}", f"{h:02d}.{m:02d}"}:
                    tags = rewrite.rewrites(written, self.tagger_fst)
                    assert len(tags) == 1, written
                    assert rewrite.rewrites(tags[0], self.verbalizer_fst) == [expected], written

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_graph_has_no_state_explosion(self):
        assert self.tagger_fst.num_states() < 1000, self.tagger_fst.num_states()
        assert self.verbalizer_fst.num_states() < 500, self.verbalizer_fst.num_states()


class TestTimeSeparation:
    """Time against the date, decimal, fraction and ordinal grammars."""

    cardinal = CardinalTaggerFst()
    time_fst = TimeTaggerFst(cardinal=cardinal).fst
    date_fst = DateTaggerFst(cardinal=cardinal).fst
    decimal_fst = DecimalTaggerFst(cardinal=cardinal).fst
    fraction_fst = FractionTaggerFst(cardinal=cardinal).fst
    ordinal_fst = OrdinalTaggerFst(cardinal=cardinal).fst

    @staticmethod
    def _accepts(fst, written):
        try:
            rewrite.top_rewrite(written, fst)
            return True
        except rewrite.Error:
            return False

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_date(self):
        assert self._accepts(self.time_fst, "09.15")
        assert not self._accepts(self.date_fst, "09.15")
        for date in ["29.09.2026", "14.30.2026", "14/30/2026", "29/09/2026", "2026-09-29"]:
            assert not self._accepts(self.time_fst, date), date
        assert self._accepts(self.date_fst, "29.09.2026")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_decimal(self):
        assert self._accepts(self.decimal_fst, "14,30")
        assert not self._accepts(self.time_fst, "14,30")
        for time in ["14.30", "14:30", "09.15", "13.00"]:
            assert self._accepts(self.time_fst, time), time
            assert not self._accepts(self.decimal_fst, time), time

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_fraction(self):
        for fraction in ["3/4", "09/15"]:
            assert not self._accepts(self.time_fst, fraction), fraction
        assert self._accepts(self.fraction_fst, "3/4")
        assert not self._accepts(self.fraction_fst, "14.30")
        assert not self._accepts(self.fraction_fst, "14:30")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ordinal(self):
        assert self._accepts(self.ordinal_fst, "14.")
        assert not self._accepts(self.time_fst, "14.")
        assert not self._accepts(self.ordinal_fst, "14.30")
