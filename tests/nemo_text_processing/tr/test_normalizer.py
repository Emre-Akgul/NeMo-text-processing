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
Tests for Turkish through NeMo's public API: the Normalizer constructor, its list and
manifest methods, the FAR caches, NormalizerWithAudio and the two command line scripts.
"""

import json
import os
import subprocess
import sys

import pynini
import pytest
from parameterized import parameterized
from pynini.lib import rewrite

import nemo_text_processing.text_normalization.tr.taggers.tokenize_and_classify as tokenize_and_classify
import nemo_text_processing.text_normalization.tr.verbalizers.verbalize_final as verbalize_final
from nemo_text_processing.text_normalization.en.verbalizers.abbreviation import (
    AbbreviationFst as AbbreviationVerbalizer,
)
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio
from nemo_text_processing.text_normalization.tr.taggers.abbreviation import AbbreviationFst as AbbreviationTagger
from nemo_text_processing.text_normalization.tr.taggers.whitelist import WhiteListFst

from ..utils import parse_test_case_file

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_CLASSIFIER_FAR = "_cased_tr_tn_True_deterministic.far"
_VERBALIZER_FAR = "tr_tn_True_deterministic_verbalizer.far"


_SHARED = {}


def _shared() -> Normalizer:
    """One uncached deterministic Normalizer, shared by the tests that only normalize."""
    if "normalizer" not in _SHARED:
        _SHARED["normalizer"] = Normalizer(input_case="cased", lang="tr")
    return _SHARED["normalizer"]


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("tr_cache"))


@pytest.fixture(scope="module")
def normalizer(cache_dir):
    return Normalizer(input_case="cased", lang="tr", deterministic=True, cache_dir=cache_dir)


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_construction(normalizer):
    assert normalizer.lang == "tr"
    assert type(normalizer.tagger).__module__ == tokenize_and_classify.__name__
    assert type(normalizer.verbalizer).__module__ == verbalize_final.__name__
    assert normalizer.normalize("5 kg ürün 100 TL'ye satıldı.") == "beş kilogram ürün yüz liraya satıldı ."
    assert (
        normalizer.normalize("5 kg ürün 100 TL'ye satıldı.", punct_post_process=True)
        == "beş kilogram ürün yüz liraya satıldı."
    )


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_no_post_processing_fst(normalizer, cache_dir):
    # post_process selects a language's WFST post-processor, of which Turkish has none;
    # punctuation post-processing is the separate punct_post_process option of normalize().
    assert normalizer.post_processor is None
    for post_process in [True, False]:
        other = Normalizer(input_case="cased", lang="tr", cache_dir=cache_dir, post_process=post_process)
        assert other.post_processor is None
        assert other.normalize("Merhaba, dünya!") == "Merhaba , dünya !"


@parameterized.expand(
    [
        ("3/4", "dörtte üç"),
        ("-3/4", "eksi dörtte üç"),
        ("29.09.2026", "yirmi dokuz eylül iki bin yirmi altı"),
        ("2026-09-29", "yirmi dokuz eylül iki bin yirmi altı"),
        ("3/4 kg", "dörtte üç kilogram"),
        ("2,5 kg", "iki virgül beş kilogram"),
        ("5 kg", "beş kilogram"),
        ("%12,5", "yüzde on iki virgül beş"),
        ("%25", "yüzde yirmi beş"),
    ]
)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_field_permutations(written, spoken):
    assert _shared().normalize(written) == spoken


@parameterized.expand(
    [
        ("Bugün hava çok güzel.", "Bugün hava çok güzel .", "Bugün hava çok güzel."),
        ("Merhaba, dünya!", "Merhaba , dünya !", "Merhaba, dünya!"),
        ("Ne dedin? Gel!", "Ne dedin ? Gel !", "Ne dedin? Gel!"),
        (
            "Ne dedin? (Hiç) Dr. Kaya'ya sordum!",
            "Ne dedin ? ( Hiç ) doktor Kaya'ya sordum !",
            "Ne dedin? (Hiç) doktor Kaya'ya sordum!",
        ),
        (
            "Ankara'da, İzmir'de; Dr., TDK.",
            "Ankara'da , İzmir'de ; doktor , te de ke .",
            "Ankara'da, İzmir'de; doktor, te de ke.",
        ),
        ("TDK'den Dr. Kaya geldi.", "te de keden doktor Kaya geldi .", "te de keden doktor Kaya geldi."),
        ("Yılmaz'a (5 kg) verildi.", "Yılmaz'a ( beş kilogram ) verildi .", "Yılmaz'a (beş kilogram) verildi."),
    ]
)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_punctuation(written, raw, detokenized):
    normalizer = _shared()
    assert normalizer.normalize(written) == raw
    assert normalizer.normalize(written, punct_post_process=True) == detokenized


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_turkish_letters_are_kept():
    text = "Işık ılık İğne iğne Çağ çağ Öğüt öğüt Şule şule Ünye ünye IŞIK İZMİR"
    assert _shared().normalize(text) == text


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_unattested_abbreviations():
    # AbbreviationFst reads these letter by letter on its own (standalone grammar
    # coverage), but the sentence classifier keeps unattested capitals as words
    # (integrated coverage, Phase 16 policy), so the public API returns them as written.
    tagger = AbbreviationTagger(whitelist=WhiteListFst())
    verbalizer = AbbreviationVerbalizer()
    unattested = {"BM", "TV", "PTT", "DSİ", "KHK", "TC", "AI", "ÇŞ", "ĞÜÖ", "IİI", "İTÜ"}
    golden = dict(parse_test_case_file("tr/data_text_normalization/test_cases_abbreviation.txt"))
    for written in unattested:
        tagged = rewrite.top_rewrite(pynini.escape(written), tagger.fst)
        assert rewrite.top_rewrite(pynini.escape(tagged), verbalizer.fst) == golden[written]
        assert _shared().normalize(written) == written
    for written in ["PKK", "BMW", "NATO", "TDK"]:
        assert _shared().normalize(written) == golden[written]


_WHITELIST = "Hst.\thastane\nGn.Md.\tgenel müdür\n"


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_custom_whitelist(tmp_path):
    path = tmp_path / "tr_whitelist.tsv"
    path.write_text(_WHITELIST, encoding="utf-8")
    cache = str(tmp_path / "cache")
    normalizer = Normalizer(input_case="cased", lang="tr", whitelist=str(path), cache_dir=cache)

    assert rewrite.top_rewrite("Hst.", normalizer.tagger.whitelist.fst) == 'name: "hastane"'
    assert normalizer.normalize("Hst.") == "hastane"
    assert normalizer.normalize("Gn.Md. geldi.") == "genel müdür geldi ."
    # In deterministic mode the file replaces the default entries.
    assert normalizer.normalize("Dr.") != "doktor"
    assert _shared().normalize("Dr.") == "doktor"
    assert _shared().normalize("Hst.") != "hastane"

    assert sorted(os.listdir(cache)) == ["_cased_tr_tn_True_deterministictr_whitelist.tsv.far", _VERBALIZER_FAR]
    restored = Normalizer(input_case="cased", lang="tr", whitelist=str(path), cache_dir=cache)
    assert restored.normalize("Hst. ve Gn.Md.") == "hastane ve genel müdür"


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_lower_cased():
    normalizer = Normalizer(input_case="lower_cased", lang="tr")
    assert normalizer.normalize("dr. kaya geldi.") == "doktor kaya geldi ."
    # No lower case acronym lexicon: "tdk" stays a word.
    assert normalizer.normalize("tdk") == "tdk"
    assert normalizer.normalize("tdk'den") == "tdk'den"
    assert normalizer.normalize("5 kg ürün") == "beş kilogram ürün"
    # The Normalizer does not lower case its input itself.
    assert normalizer.normalize("Işık ılık İzmir") == "Işık ılık İzmir"


_BATCH = [
    "Bugün hava çok güzel.",
    "5 kg ürün aldım.",
    "Fiyat 100 TL oldu.",
    "Toplantı 29.09.2026'da saat 14.30'da başlayacak.",
    "Merhaba, dünya!",
]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_normalize_list():
    normalizer = _shared()
    expected = [normalizer.normalize(text) for text in _BATCH]
    assert expected[:3] == ["Bugün hava çok güzel .", "beş kilogram ürün aldım .", "Fiyat yüz lira oldu ."]
    assert normalizer.normalize_list(_BATCH, n_jobs=1) == expected
    assert normalizer.normalize_list(_BATCH, n_jobs=1, batch_size=2) == expected
    assert normalizer.normalize_list(_BATCH, n_jobs=1, punct_post_process=True) == [
        normalizer.normalize(text, punct_post_process=True) for text in _BATCH
    ]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_normalize_manifest(tmp_path):
    normalizer = _shared()
    manifest = tmp_path / "manifest.json"
    lines = [{"id": i, "text": text, "audio_filepath": f"ses_{i}.wav"} for i, text in enumerate(_BATCH)]
    manifest.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
    output = tmp_path / "normalized.json"

    normalizer.normalize_manifest(
        manifest=str(manifest),
        n_jobs=1,
        punct_pre_process=False,
        punct_post_process=False,
        batch_size=2,
        output_filename=str(output),
        text_field="text",
    )

    results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(results) == len(lines)
    for line, result in zip(lines, results):
        assert {key: result[key] for key in line} == line
        assert result["normalized"] == normalizer.normalize(line["text"])


def _built_far_files(monkeypatch):
    built = []

    def record(file_name, graphs):
        built.append(os.path.basename(file_name))
        return generator_main(file_name, graphs)

    generator_main = tokenize_and_classify.generator_main
    monkeypatch.setattr(tokenize_and_classify, "generator_main", record)
    monkeypatch.setattr(verbalize_final, "generator_main", record)
    return built


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_caches(tmp_path, monkeypatch):
    built = _built_far_files(monkeypatch)
    cache = str(tmp_path)

    first = Normalizer(input_case="cased", lang="tr", cache_dir=cache)
    assert sorted(built) == [_CLASSIFIER_FAR, _VERBALIZER_FAR]
    assert sorted(os.listdir(cache)) == [_CLASSIFIER_FAR, _VERBALIZER_FAR]
    assert pynini.Far(os.path.join(cache, _CLASSIFIER_FAR), mode="r")["tokenize_and_classify"].num_states() > 0
    assert pynini.Far(os.path.join(cache, _VERBALIZER_FAR), mode="r")["verbalize"].num_states() > 0
    assert all("tr_tn" in name for name in os.listdir(cache))

    built.clear()
    restored = Normalizer(input_case="cased", lang="tr", cache_dir=cache)
    assert built == []

    built.clear()
    rebuilt = Normalizer(input_case="cased", lang="tr", cache_dir=cache, overwrite_cache=True)
    assert sorted(built) == [_CLASSIFIER_FAR, _VERBALIZER_FAR]

    uncached = _shared()
    for text in _BATCH:
        expected = uncached.normalize(text)
        assert first.normalize(text) == expected
        assert restored.normalize(text) == expected
        assert rebuilt.normalize(text) == expected


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_non_deterministic(cache_dir):
    normalizer = Normalizer(input_case="cased", lang="tr", deterministic=False, cache_dir=cache_dir)
    for written, spoken in [
        ("3/4", "dörtte üç"),
        ("2026-09-29", "yirmi dokuz eylül iki bin yirmi altı"),
        ("5 kg ürün aldım.", "beş kilogram ürün aldım ."),
        ("TDK'den Dr. Kaya geldi.", "te de keden doktor Kaya geldi ."),
    ]:
        assert normalizer.normalize(written) == spoken


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_normalizer_with_audio(cache_dir):
    normalizer = NormalizerWithAudio(input_case="cased", lang="tr", cache_dir=cache_dir, lm=False)
    assert not normalizer.lm
    # Without an ASR transcript, the options are a set that includes the deterministic form.
    options = normalizer.normalize(text="5 kg ürün aldım.", n_tagged=3, punct_post_process=False)
    assert isinstance(options, set)
    assert "beş kilogram ürün aldım ." in options
    options = normalizer.normalize(text="5 kg ürün aldım.", n_tagged=3)
    assert "beş kilogram ürün aldım." in options


def _run(module: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=_REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run(
        [sys.executable, "-m", module, *args], capture_output=True, text=True, env=env, cwd=_REPO, timeout=600
    )


def _cli_output(result: subprocess.CompletedProcess) -> str:
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    rule = "=" * 40
    start = lines.index(rule)
    return lines[start + 1]


_NORMALIZE = "nemo_text_processing.text_normalization.normalize"


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_cli(normalizer, cache_dir):
    help_text = _run(_NORMALIZE, "--help").stdout
    assert "tr}" in help_text.replace("\n", "")

    for args, expected in [
        (["--text", "2026"], "iki bin yirmi altı"),
        (
            ["--input_case", "cased", "--text", "5 kg ürün 100 TL'ye satıldı."],
            "beş kilogram ürün yüz liraya satıldı .",
        ),
        (["--punct_post_process", "--text", "Merhaba, dünya!"], "Merhaba, dünya!"),
    ]:
        result = _run(_NORMALIZE, "--language", "tr", "--cache_dir", cache_dir, *args)
        assert _cli_output(result) == expected


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_cli_new_cache(tmp_path):
    cache = str(tmp_path / "tr_tn_phase18")
    result = _run(_NORMALIZE, "--language", "tr", "--cache_dir", cache, "--text", "5 kg ürün 100 TL'ye satıldı.")
    assert _cli_output(result) == "beş kilogram ürün yüz liraya satıldı ."
    assert sorted(os.listdir(cache)) == [_CLASSIFIER_FAR, _VERBALIZER_FAR]


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_audio_cli(normalizer, cache_dir):
    result = _run(
        "nemo_text_processing.text_normalization.normalize_with_audio",
        "--language",
        "tr",
        "--cache_dir",
        cache_dir,
        "--n_tagged",
        "3",
        "--text",
        "5 kg ürün aldım.",
    )
    assert result.returncode == 0, result.stderr
    assert "beş kilogram ürün aldım." in result.stdout + result.stderr
