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
Sentence level regression tests for Turkish through the public Normalizer.

test_cases_sentences_realistic.txt holds realistic sentences from news, economy,
sports, weather, health, education, transport, addresses, the web and daily life,
each with its intended spoken form, written independently of the grammar's output.

test_cases_sentences_fleurs.txt holds sentences of the Turkish FLEURS transcriptions
(CC BY 4.0, see FLEURS_ATTRIBUTION.txt) whose reading was checked after the corpus
evaluation, with the spoken form the grammar gives them.

known_limitations_sentences.txt holds sentences the grammar does not yet read as
intended, one per line as "category~written~intended". They are strict expected
failures: when one starts to pass, it fails here and is moved to the golden file.
"""

import os

import pytest
from parameterized import parameterized

from nemo_text_processing.text_normalization.normalize import Normalizer

from ..utils import parse_test_case_file

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_text_normalization")

_NORMALIZER = Normalizer(input_case="cased", lang="tr")

_REALISTIC = parse_test_case_file("tr/data_text_normalization/test_cases_sentences_realistic.txt")
_FLEURS = parse_test_case_file("tr/data_text_normalization/test_cases_sentences_fleurs.txt")


def _limitations():
    with open(os.path.join(_DATA, "known_limitations_sentences.txt"), encoding="utf-8") as f:
        return [tuple(line.rstrip("\n").split("~")) for line in f if line.strip()]


_LIMITATIONS = _limitations()


@parameterized.expand(_REALISTIC)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_realistic_sentence(written, spoken):
    assert _NORMALIZER.normalize(written) == spoken


@parameterized.expand(_FLEURS)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_fleurs_sentence(written, spoken):
    assert _NORMALIZER.normalize(written) == spoken


@parameterized.expand(_LIMITATIONS)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
@pytest.mark.xfail(strict=True, reason="known limitation, see known_limitations_sentences.txt")
def test_known_limitation(category, written, intended):
    assert _NORMALIZER.normalize(written) == intended


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_files_are_well_formed():
    realistic = [written for written, _ in _REALISTIC + _FLEURS]
    limited = [written for _, written, _ in _LIMITATIONS]
    assert len(realistic) == len(set(realistic)) and len(limited) == len(set(limited))
    assert not set(realistic) & set(limited)
    assert all(len(line) == 3 and all(line) for line in _LIMITATIONS)
    for written, spoken in _REALISTIC + _FLEURS:
        assert isinstance(spoken, str) and spoken and "  " not in spoken and spoken == spoken.strip()
