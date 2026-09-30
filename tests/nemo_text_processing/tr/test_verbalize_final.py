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
End to end tests for the Turkish sentence classifier and final verbalizer.

``lang="tr"`` is not registered in normalize.py yet, so ``_normalizer`` sets up a
Normalizer with the Turkish grammars without its constructor. ``Normalizer.normalize``
then runs unchanged: the classifier, the token parser, the field permutations that
respect ``preserve_order``, and the final verbalizer.
"""

import glob
import os
import time

import pynini
import pytest
from parameterized import parameterized
from pynini.lib import rewrite
from sacremoses import MosesDetokenizer

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.token_parser import TokenParser
from nemo_text_processing.text_normalization.tr.taggers.tokenize_and_classify import ClassifyFst
from nemo_text_processing.text_normalization.tr.verbalizers.verbalize_final import VerbalizeFinalFst

from ..utils import parse_test_case_file

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_text_normalization")

# All capital strings that AbbreviationFst reads letter by letter but the classifier
# keeps as words, since they are neither in the acronym table nor in
# known_initialisms.tsv (Phase 16 policy).
_UNATTESTED_ABBREVIATIONS = {"BM", "TV", "PTT", "DSİ", "KHK", "TC", "AI", "ÇŞ", "ĞÜÖ", "IİI", "İTÜ"}


def _normalizer(deterministic: bool = True) -> Normalizer:
    normalizer = Normalizer.__new__(Normalizer)
    normalizer.input_case = "cased"
    normalizer.lang = "tr"
    normalizer.post_processor = None
    normalizer.tagger = ClassifyFst(deterministic=deterministic)
    normalizer.verbalizer = VerbalizeFinalFst(deterministic=deterministic)
    normalizer.parser = TokenParser()
    normalizer.max_number_of_permutations_per_split = 729
    normalizer.moses_detokenizer = MosesDetokenizer(lang="tr")
    return normalizer


_NORMALIZER = _normalizer()


def _raw(text: str) -> str:
    """The final verbalizer's own output for a sentence, before normalize() removes duplicate spaces."""
    tagged = Normalizer.select_tag(_NORMALIZER.find_tags(pynini.escape(text)))
    _NORMALIZER.parser(tagged)
    for permutation in _NORMALIZER.generate_permutations(_NORMALIZER.parser.parse()):
        lattice = _NORMALIZER.find_verbalizer(pynini.escape(permutation))
        if lattice.num_states() != 0:
            return Normalizer.select_verbalizer(lattice)
    raise AssertionError(f"no verbalization of {tagged}")


def _golden_cases():
    cases = []
    for path in sorted(glob.glob(os.path.join(_DATA, "test_cases_*.txt"))):
        name = os.path.basename(path)
        if name == "test_cases_sentences.txt":
            continue
        for written, spoken in parse_test_case_file(f"tr/data_text_normalization/{name}"):
            if name == "test_cases_abbreviation.txt" and written in _UNATTESTED_ABBREVIATIONS:
                spoken = written
            cases.append((written, spoken))
    return cases


_GOLDEN = _golden_cases()
_SENTENCES = parse_test_case_file("tr/data_text_normalization/test_cases_sentences.txt")


@parameterized.expand(_GOLDEN)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_every_class_golden_case(written, spoken):
    # Every standalone golden case of Phases 1-15, through the whole pipeline.
    expected = spoken if isinstance(spoken, list) else [spoken]
    assert _NORMALIZER.normalize(written) in expected


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_golden_cases_cover_every_class():
    files = {os.path.basename(path) for path in glob.glob(os.path.join(_DATA, "test_cases_*.txt"))}
    classes = {name[len("test_cases_") : -len(".txt")] for name in files}
    assert {
        "cardinal",
        "ordinal",
        "decimal",
        "fraction",
        "date",
        "time",
        "percentage",
        "money",
        "measure",
        "telephone",
        "electronic",
        "whitelist",
        "suffix",
        "abbreviation",
    } <= classes
    assert len(_GOLDEN) > 900


@parameterized.expand(_SENTENCES)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_sentence(written, spoken):
    assert _NORMALIZER.normalize(written) == spoken
    assert _raw(written) == spoken


@parameterized.expand(
    [
        ("Toplantı 29.09.2026'da saat 14.30'da başlayacak.", ["suffix"]),
        ("ASELSAN'ın hissesi %5 arttı.", ["suffix", "percentage", "cardinal"]),
        ("5 kg ürün 100 TL'ye satıldı.", ["measure", "cardinal", "suffix"]),
        ("TDK'den Dr. Kaya geldi.", ["suffix"]),
        ("E-posta user@example.com adresine gönderildi.", ["electronic"]),
        ("Ürün 2,5 kg ve 3/4 kg paketlerde satılıyor.", ["measure", "measure"]),
        ("BMW 2026'da 3/4 oranında büyüdü.", ["abbreviation", "suffix", "fraction"]),
        ("Toplantı 2026-09-29 tarihinde, 09.00'da.", ["date", "suffix"]),
    ]
)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_sentence_uses_the_intended_classes(written, classes):
    tagged = Normalizer.select_tag(_NORMALIZER.find_tags(pynini.escape(written)))
    for semantic_class in classes:
        assert f"{semantic_class} {{" in tagged, (semantic_class, tagged)


@parameterized.expand(
    [
        ("Merhaba, dünya!", "Merhaba , dünya !", "Merhaba, dünya!"),
        ("Bugün hava çok güzel.", "Bugün hava çok güzel .", "Bugün hava çok güzel."),
        ("TDK'den Dr. Kaya geldi.", "te de keden doktor Kaya geldi .", "te de keden doktor Kaya geldi."),
        (
            "MÖ 500 yılında kurulan şehir, MS 1453'te fethedildi.",
            "milattan önce beş yüz yılında kurulan şehir , milattan sonra bin dört yüz elli üçte fethedildi .",
            "milattan önce beş yüz yılında kurulan şehir, milattan sonra bin dört yüz elli üçte fethedildi.",
        ),
    ]
)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_punctuation_is_a_separate_token(written, raw, detokenized):
    # The final verbalizer separates punctuation like any token; attaching it is the
    # Normalizer's optional punctuation post-processing, as for the other languages.
    assert _raw(written) == raw
    assert _NORMALIZER.normalize(written) == raw
    assert _NORMALIZER.normalize(written, punct_post_process=True) == detokenized


@parameterized.expand(
    [
        ("  Bugün hava çok güzel.  ", "Bugün hava çok güzel ."),
        ("Bugün\thava  çok\ngüzel.", "Bugün hava çok güzel ."),
        ("\t5 kg\t\türün\n", "beş kilogram ürün"),
        ("MÖ 500", "milattan önce beş yüz"),
        ("29.09.2026 ve 3/4", "yirmi dokuz eylül iki bin yirmi altı ve dörtte üç"),
    ]
)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_no_whitespace_artifacts(written, spoken):
    raw = _raw(written)
    assert raw == spoken
    assert raw == raw.strip()
    assert "  " not in raw
    assert " " not in raw and "\t" not in raw and "\n" not in raw


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_non_deterministic_smoke():
    normalizer = _normalizer(deterministic=False)
    for written, spoken in [
        ("2026", "iki bin yirmi altı"),
        ("3/4", "dörtte üç"),
        ("29.09.2026", "yirmi dokuz eylül iki bin yirmi altı"),
        ("%25", "yüzde yirmi beş"),
        ("100 TL", "yüz lira"),
        ("5 kg", "beş kilogram"),
        ("PKK", "pe ka ka"),
        ("Dr.", "doktor"),
        ("TDK'den Dr. Kaya geldi.", "te de keden doktor Kaya geldi ."),
    ]:
        assert normalizer.normalize(written) == spoken


_CACHE_SAMPLES = [
    'tokens { name: "Toplantı" } tokens { suffix { value: "on dört otuzda" } } tokens { name: "." }',
    'tokens { measure { cardinal { integer: "beş" } units: "kilogram" preserve_order: true } }',
    'tokens { fraction { denominator: "dörtte" numerator: "üç" } }',
    'tokens { name: "milattan önce" }',
]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_cache(tmp_path):
    fresh = VerbalizeFinalFst(deterministic=True)

    cache_dir = str(tmp_path / "cache")
    built = VerbalizeFinalFst(deterministic=True, cache_dir=cache_dir)
    far_file = os.path.join(cache_dir, "tr_tn_True_deterministic_verbalizer.far")
    assert os.listdir(cache_dir) == ["tr_tn_True_deterministic_verbalizer.far"]
    assert pynini.Far(far_file, mode="r")["verbalize"].num_states() == fresh.fst.num_states()

    modified = os.path.getmtime(far_file)
    time.sleep(0.01)
    restored = VerbalizeFinalFst(deterministic=True, cache_dir=cache_dir)
    assert os.path.getmtime(far_file) == modified

    for tagged in _CACHE_SAMPLES:
        expected = rewrite.rewrites(pynini.escape(tagged), fresh.fst)
        assert len(expected) == 1
        assert rewrite.rewrites(pynini.escape(tagged), built.fst) == expected
        assert rewrite.rewrites(pynini.escape(tagged), restored.fst) == expected


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_cache_names(tmp_path):
    cache_dir = str(tmp_path)
    VerbalizeFinalFst(deterministic=False, cache_dir=cache_dir)
    VerbalizeFinalFst(deterministic=True, cache_dir=cache_dir, overwrite_cache=True)
    assert sorted(os.listdir(cache_dir)) == [
        "tr_tn_False_deterministic_verbalizer.far",
        "tr_tn_True_deterministic_verbalizer.far",
    ]
    assert not any(name.endswith("_tr_tn_True_deterministic.far") for name in os.listdir(cache_dir))
