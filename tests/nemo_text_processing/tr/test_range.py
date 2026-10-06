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
Direct tests for the Turkish range tagger and verbalizer.
"""

import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.tr.taggers.tokenize_and_classify import ClassifyFst
from nemo_text_processing.text_normalization.tr.verbalizers.range import RangeFst as RangeVerbalizerFst

from ..utils import parse_test_case_file

_TEST_CASES = 'tr/data_text_normalization/test_cases_range.txt'

_CLASSIFIER = ClassifyFst()

# Not a range: a dash at either end, two dashes, spaces around the dash, a word or a
# code at either end, and a date, whose dashes are its own.
_NOT_RANGES = ["-5", "5-", "2-5-7", "2 - 5", "2 -5", "COVID-19", "HJR-3", "a-5", "5-a", "2026-09-29", "05-06"]


class TestRange:

    tagger = _CLASSIFIER.range
    verbalizer = RangeVerbalizerFst()

    def _accepts(self, written):
        try:
            rewrite.top_rewrite(written, self.tagger.fst)
            return True
        except rewrite.Error:
            return False

    @parameterized.expand(parse_test_case_file(_TEST_CASES))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        assert rewrite.rewrites(test_input, self.tagger.graph) == [expected]
        tagged = rewrite.top_rewrite(test_input, self.tagger.fst)
        assert tagged == f'range {{ value: "{expected}" }}'
        assert rewrite.rewrites(tagged, self.verbalizer.fst) == [expected]

    @parameterized.expand([(w,) for w in _NOT_RANGES])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_not_a_range(self, written):
        assert not self._accepts(written)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_suffix_on_the_second_end_is_checked(self):
        """The suffix follows the second end's reading, "96" -> "doksan altı"."""
        assert self._accepts("1995-96'da")
        assert not self._accepts("1995-96'de")
        assert not self._accepts("1995-96'daler")
