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
Tests for the Turkish aggregate verbalizer, VerbalizeFst, and the unwrapping of single
tokens by VerbalizeFinalFst.

Tokens here are written in the field order the verbalizers read. The order the taggers
write, and the token parser permutation between the two, are tested in
test_verbalize_final.py and below for fractions and year first dates.
"""

import pynini
import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.token_parser import TokenParser
from nemo_text_processing.text_normalization.tr.verbalizers.verbalize import VerbalizeFst
from nemo_text_processing.text_normalization.tr.verbalizers.verbalize_final import VerbalizeFinalFst

_VERBALIZE = VerbalizeFst(deterministic=True)
_FINAL = VerbalizeFinalFst(deterministic=True)

# One or more tokens of every semantic class, in the verbalizers' field order.
_SEMANTIC = [
    ('cardinal { integer: "iki bin yirmi altı" }', "iki bin yirmi altı"),
    ('cardinal { negative: "true" integer: "beş" }', "eksi beş"),
    ('ordinal { integer: "sekizinci" }', "sekizinci"),
    ('decimal { integer_part: "bir" fractional_part: "beş" }', "bir virgül beş"),
    (
        'decimal { negative: "true" integer_part: "bir" fractional_part: "beş" quantity: "milyon" }',
        "eksi bir virgül beş milyon",
    ),
    ('fraction { denominator: "dörtte" numerator: "üç" }', "dörtte üç"),
    ('fraction { negative: "true" denominator: "dörtte" numerator: "üç" }', "eksi dörtte üç"),
    (
        'date { day: "yirmi dokuz" month: "eylül" year: "iki bin yirmi altı" preserve_order: true }',
        "yirmi dokuz eylül iki bin yirmi altı",
    ),
    ('date { day: "yirmi dokuz" month: "eylül" preserve_order: true }', "yirmi dokuz eylül"),
    ('date { month: "eylül" year: "iki bin yirmi altı" preserve_order: true }', "eylül iki bin yirmi altı"),
    ('time { hours: "on dört" minutes: "otuz" preserve_order: true }', "on dört otuz"),
    ('time { hours: "on dört" minutes: "otuz" seconds: "sıfır beş" preserve_order: true }', "on dört otuz sıfır beş"),
    ('percentage { cardinal { integer: "yirmi beş" } }', "yüzde yirmi beş"),
    ('percentage { decimal { integer_part: "on iki" fractional_part: "beş" } }', "yüzde on iki virgül beş"),
    ('money { integer_part: "yüz" currency_maj: "lira" preserve_order: true }', "yüz lira"),
    (
        'money { integer_part: "on iki" currency_maj: "avro" fractional_part: "elli" currency_min: "sent" '
        'preserve_order: true }',
        "on iki avro elli sent",
    ),
    ('measure { cardinal { integer: "beş" } units: "kilogram" preserve_order: true }', "beş kilogram"),
    (
        'measure { decimal { integer_part: "iki" fractional_part: "beş" } units: "kilogram" preserve_order: true }',
        "iki virgül beş kilogram",
    ),
    (
        'measure { fraction { denominator: "dörtte" numerator: "üç" } units: "kilogram" preserve_order: true }',
        "dörtte üç kilogram",
    ),
    (
        'telephone { country_code: "artı doksan" number_part: "beş yüz otuz iki yüz yirmi üç kırk beş altmış yedi" '
        'preserve_order: true }',
        "artı doksan beş yüz otuz iki yüz yirmi üç kırk beş altmış yedi",
    ),
    (
        'electronic { username: "user" domain: "example.com" preserve_order: true }',
        "user et example nokta kom",
    ),
    (
        'electronic { protocol: "https://www." domain: "example.com/v2" preserve_order: true }',
        "he te te pe se iki nokta eğik çizgi eğik çizgi dabılyu dabılyu dabılyu nokta example nokta kom eğik çizgi v iki",
    ),
    ('suffix { value: "iki bin yirmi altıda" }', "iki bin yirmi altıda"),
    ('suffix { value: "te de keden" }', "te de keden"),
    ('suffix { value: "Almancadan" }', "Almancadan"),
    ('abbreviation { value: "pe ka ka" }', "pe ka ka"),
    ('abbreviation { value: "be me ve" }', "be me ve"),
    ('abbreviation { value: "nato" }', "nato"),
    # Readings of AbbreviationFst that the classifier leaves to words, see test_verbalize_final.py.
    ('abbreviation { value: "pe te te" }', "pe te te"),
    ('abbreviation { value: "yumuşak ge ü ö" }', "yumuşak ge ü ö"),
]

# Ordinary words, whitelist readings and punctuation are all "name" tokens.
_NAMES = [
    ("Ankara", "Ankara"),
    ("merhaba", "merhaba"),
    ("Çayyolu", "Çayyolu"),
    ("Yılmaz'a", "Yılmaz'a"),
    ("IŞIK", "IŞIK"),
    ("ılık", "ılık"),
    ("İzmir", "İzmir"),
    ("iğne", "iğne"),
    ("ĞÖŞÜÇ", "ĞÖŞÜÇ"),
    ("Quartz", "Quartz"),
    ("Wien", "Wien"),
    ("Xerox", "Xerox"),
    ("E-posta", "E-posta"),
    ("doktor", "doktor"),
    ("milattan önce", "milattan önce"),
    ("milattan sonra", "milattan sonra"),
    ("anonim şirket", "anonim şirket"),
    (".", "."),
    (",", ","),
    ("?!", "?!"),
    ("\"", "\""),
    ("…", "…"),
]


def _outputs(fst: "pynini.Fst", tagged: str) -> list:
    try:
        return rewrite.rewrites(pynini.escape(tagged), fst)
    except rewrite.Error:
        return []


def _only(fst: "pynini.Fst", tagged: str) -> str:
    outputs = _outputs(fst, tagged)
    assert len(outputs) == 1, (tagged, outputs)
    return outputs[0]


@parameterized.expand(_SEMANTIC)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_semantic_token(tagged, expected):
    assert _only(_VERBALIZE.fst, tagged) == expected
    assert _only(_FINAL.fst, f"tokens {{ {tagged} }}") == expected


@parameterized.expand(_NAMES)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_name_token(name, expected):
    assert _only(_FINAL.fst, f'tokens {{ name: "{name}" }}') == expected


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_names_are_not_semantic_tokens():
    assert _outputs(_VERBALIZE.fst, 'name: "Ankara"') == []


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_non_breaking_space_becomes_a_space():
    tagged = 'tokens { name: "milattan önce" }'
    assert " " in tagged
    output = _only(_FINAL.fst, tagged)
    assert output == "milattan önce"
    assert " " not in output and output[8] == " "


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_non_breaking_space_is_kept_only_as_a_space():
    # The rewrite is limited to U+00A0: other characters of a name are untouched.
    tagged = 'tokens { name: "a b c d" }'
    assert _only(_FINAL.fst, tagged) == "a b c d"


@parameterized.expand(
    [
        ('fraction { numerator: "üç" denominator: "dörtte" }', "dörtte üç"),
        (
            'date { year: "iki bin yirmi altı" month: "eylül" day: "yirmi dokuz" }',
            "yirmi dokuz eylül iki bin yirmi altı",
        ),
        (
            'measure { fraction { numerator: "üç" denominator: "dörtte" } units: "kilogram" preserve_order: true }',
            "dörtte üç kilogram",
        ),
    ]
)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_written_order_needs_the_token_parser(tagged, expected):
    # The taggers write these fields in written order, which the verbalizers do not
    # read; the Normalizer's permutation of the parsed token supplies the spoken order.
    tokens = f"tokens {{ {tagged} }}"
    assert _outputs(_FINAL.fst, tokens) == []

    parser = TokenParser()
    parser(tokens)
    permutations = Normalizer.__new__(Normalizer)._permute(parser.parse()[0])
    readings = {output for permutation in permutations for output in _outputs(_FINAL.fst, permutation)}
    assert readings == {expected}


@parameterized.expand(
    [
        ('tokens { cardinal { integer: "yüz" } }', "yüz"),
        ('tokens{cardinal{integer:"yüz"}}', "yüz"),
        ('  tokens  {  cardinal  {  integer: "yüz"  }  }  ', "yüz"),
        ('tokens { cardinal { integer: "yüz" } } tokens { name: "kişi" }', "yüz kişi"),
        ('tokens { cardinal { integer: "yüz" } }    tokens { name: "kişi" }', "yüz kişi"),
        (
            'tokens { name: "Toplantı" } tokens { suffix { value: "on dört otuzda" } } tokens { name: "." }',
            "Toplantı on dört otuzda .",
        ),
    ]
)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_token_wrappers_and_separators(tagged, expected):
    assert _only(_FINAL.fst, tagged) == expected


@parameterized.expand(
    [
        ('name: "Ankara"',),
        ('tokens { name: "Ankara" ',),
        ('tokens { unknown { value: "x" } }',),
        ('tokens { fraction { denominator: "dörtte" } }',),
        ('tokens { percentage { integer: "beş" } }',),
    ]
)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_malformed_tokens_are_rejected(tagged):
    assert _outputs(_FINAL.fst, tagged) == []


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_non_deterministic_smoke():
    verbalize = VerbalizeFst(deterministic=False)
    final = VerbalizeFinalFst(deterministic=False)
    for tagged, expected in _SEMANTIC:
        assert expected in _outputs(verbalize.fst, tagged)
        assert expected in _outputs(final.fst, f"tokens {{ {tagged} }}")
