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
Tests for the Turkish sentence classifier, ClassifyFst.

A sentence is classified by its lowest weight path. ``_best`` takes the two best
distinct tokenizations from the lattice with a shortest path search, which is safe on
any input, and checks that the best one is strictly cheaper than the next: a unique
best tokenization. ``_top_two`` returns both without that check, for the known
equal-cost ambiguities, where pynini's shortest path choice is reproducible but not
unique.
"""

import itertools
import os
import random
import re

import pynini
import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.tr.taggers.punctuation import PUNCTUATION_MARKS, PunctuationFst
from nemo_text_processing.text_normalization.tr.taggers.tokenize_and_classify import _WEIGHTS, ClassifyFst
from nemo_text_processing.text_normalization.tr.taggers.word import WordFst

_CLASSIFIER = ClassifyFst(input_case="cased", deterministic=True)
_PUNCTUATION = set(PUNCTUATION_MARKS)

# Single tokens: the input, the class chosen, and a fragment of the token.
_SINGLE = [
    ("Alm.dan", "suffix", 'value: "Almancadan"'),
    ("T.C.de", "suffix", 'value: "te cede"'),
    ("No.lu", "suffix", 'value: "nolu"'),
    ("https://alm.dan", "electronic", 'domain: "alm.dan"'),
    ("user@alm.dan", "electronic", 'username: "user"'),
    ("Dr.", "whitelist", 'name: "doktor"'),
    ("Prof.", "whitelist", 'name: "profesör"'),
    ("Alm.", "whitelist", 'name: "Almanca"'),
    # multi word whitelist readings keep a non-breaking space inside the token
    ("MÖ", "whitelist", 'name: "milattan\u00a0önce"'),
    ("AŞ", "whitelist", 'name: "anonim\u00a0şirket"'),
    ("8.", "ordinal", 'integer: "sekizinci"'),
    ("2026.", "ordinal", "iki bin yirmi altıncı"),
    ("5 kg", "measure", 'units: "kilogram"'),
    ("5 dk.", "measure", 'units: "dakika"'),
    ("5 m²", "measure", 'units: "metrekare"'),
    ("100 TL", "money", 'currency_maj: "lira"'),
    ("100 USD", "money", 'currency_maj: "dolar"'),
    ("₺100", "money", 'currency_maj: "lira"'),
    ("100€", "money", 'currency_maj: "avro"'),
    ("%25", "percentage", 'integer: "yirmi beş"'),
    ("%12,5", "percentage", 'fractional_part: "beş"'),
    ("0532 123 45 67", "telephone", "sıfır beş yüz otuz iki"),
    ("(0532) 123 45 67", "telephone", "sıfır beş yüz otuz iki"),
    ("+90 532 123 45 67", "telephone", 'country_code: "artı doksan"'),
    ("444 12 34", "telephone", "dört yüz kırk dört"),
    ("29.09.2026", "date", 'month: "eylül"'),
    ("03/04/2026", "date", 'month: "nisan"'),
    ("14.30", "time", 'minutes: "otuz"'),
    ("14:30", "time", 'minutes: "otuz"'),
    ("12.05", "time", 'minutes: "sıfır beş"'),
    ("3/4", "fraction", 'denominator: "dörtte"'),
    ("1,5", "decimal", 'fractional_part: "beş"'),
    ("1.000", "cardinal", 'integer: "bin"'),
    ("2026", "cardinal", 'integer: "iki bin yirmi altı"'),
    ("example.com", "electronic", 'domain: "example.com"'),
    ("user@example.com", "electronic", 'username: "user"'),
    ("http://example.com", "electronic", 'protocol: "http://"'),
    ("PKK", "abbreviation", 'value: "pe ka ka"'),
    ("NATO", "abbreviation", 'value: "nato"'),
    ("BMW", "abbreviation", 'value: "be me ve"'),
    ("CPU", "abbreviation", 'value: "si pi yu"'),
    ("TÜBİTAK", "abbreviation", 'value: "tübitak"'),
    ("TDK", "abbreviation", 'value: "te de ke"'),
    ("THY", "abbreviation", 'value: "te he ye"'),
    ("TBMM", "abbreviation", 'value: "te be me me"'),
    ("HTTP", "abbreviation", 'value: "he te te pe"'),
    ("TDK'den", "suffix", 'value: "te de keden"'),
    ("2026'da", "suffix", 'value: "iki bin yirmi altıda"'),
    ("5 kg'dan", "suffix", 'value: "beş kilogramdan"'),
    ("AŞ'de", "suffix", 'value: "anonim şirkette"'),
    ("5 m²ye", "suffix", 'value: "beş metrekareye"'),
    ("100 TL'ye", "suffix", 'value: "yüz liraya"'),
    # upper case words with no evidence of being acronyms stay words
    ("EV", "word", 'name: "EV"'),
    ("OKUL", "word", 'name: "OKUL"'),
    ("ANKARA", "word", 'name: "ANKARA"'),
    ("KARA", "word", 'name: "KARA"'),
    ("ANKARA'da", "word", 'name: "ANKARA\'da"'),
    ("merhaba", "word", 'name: "merhaba"'),
    ("Çayyolu", "word", 'name: "Çayyolu"'),
    ("Yılmaz'a", "word", 'name: "Yılmaz\'a"'),
]

# Sentences: the input and the classes of its tokens in order.
_SENTENCES = [
    ("Toplantı 29.09.2026'da saat 14.30'da başlayacak.", ["word", "suffix", "word", "suffix", "word", "punct"]),
    ("ASELSAN'ın hissesi %5 arttı.", ["suffix", "word", "percentage", "word", "punct"]),
    ("5 kg ürün 100 TL'ye satıldı.", ["measure", "word", "suffix", "word", "punct"]),
    ("TDK'den Dr. Kaya geldi.", ["suffix", "whitelist", "word", "word", "punct"]),
    ("E-posta user@example.com adresine gönderildi.", ["word", "electronic", "word", "word", "punct"]),
    ("Merhaba, dünya!", ["word", "punct", "word", "punct"]),
    ("Bugün hava çok güzel.", ["word", "word", "word", "word", "punct"]),
    ("Çığlık, öğle ve üzüm.", ["word", "punct", "word", "word", "word", "punct"]),
    ("8. sırada", ["ordinal", "word"]),
    ("Dr. geldi.", ["whitelist", "word", "punct"]),
    ("Kat 3/14", ["word", "fraction"]),
    # no class accepts these whole: tokens joined by the punctuation between them
    ("29.09", ["cardinal", "punct", "word"]),
    ("5.5", ["cardinal", "punct", "cardinal"]),
    ("abc.def,ghi", ["electronic", "punct", "word"]),
    # a chain of numbers joined by punctuation: an earlier token takes as much as it can
    ("1,2,3", ["decimal", "punct", "cardinal"]),
    ("3,5,7", ["decimal", "punct", "cardinal"]),
    ("1,2,3,4", ["decimal", "punct", "decimal"]),
    ("3/4/5", ["fraction", "punct", "cardinal"]),
    ("1/2/3", ["fraction", "punct", "cardinal"]),
    ("1/2/3/4", ["fraction", "punct", "fraction"]),
    ("1, 2, 3", ["cardinal", "punct", "cardinal", "punct", "cardinal"]),
]

# Punctuation written against a semantic token stays outside it.
_ADJACENT = [
    ("%25,", ["percentage", "punct"]),
    ("100 TL.", ["money", "punct"]),
    ("29.09.2026.", ["date", "punct"]),
    ("TDK'den,", ["suffix", "punct"]),
    ("example.com.", ["electronic", "punct"]),
    ("user@example.com,", ["electronic", "punct"]),
    ("https://example.com/foo.", ["electronic", "punct"]),
    ("(5 kg)", ["punct", "measure", "punct"]),
    ('"NATO"', ["punct", "abbreviation", "punct"]),
    ("Alm.dan.", ["suffix", "punct"]),
    ("?!", ["punct"]),
    ("...", ["punct"]),
    ("8.,", ["ordinal", "punct"]),
    ("5 dk.,", ["measure", "punct"]),
    ("Dr.,", ["whitelist", "punct"]),
    ("' tek '", ["punct", "word", "punct"]),
]


# Known equal-cost ambiguities, deferred to tokenizer hardening: the input and the
# token classes of its two cheapest tokenizations.
_KNOWN_TIES = [
    ("$3$", ["money", "punct"], ["punct", "money"]),
    ("%4$", ["punct", "money"], ["percentage", "punct"]),
    ("7,$2$$%", ["cardinal", "punct", "money", "punct"], ["cardinal", "punct", "money", "punct"]),
]


def _top_two(sentence, fst=None):
    """The two cheapest distinct tokenizations with their costs, cheapest first."""
    lattice = rewrite.rewrite_lattice(sentence, fst or _CLASSIFIER.fst)
    top = pynini.shortestpath(lattice, nshortest=2, unique=True).optimize()
    return sorted((float(weight), output) for _, output, weight in top.paths().items())


def _is_tie(paths):
    return len(paths) == 2 and paths[1][0] <= paths[0][0] + 1e-6


def _best(sentence, fst=None):
    """The unique best tokenization: checked to be strictly cheaper than the next."""
    paths = _top_two(sentence, fst)
    assert not _is_tie(paths), f"{sentence!r} has an equal-cost ambiguity: {paths}"
    return paths[0][1]


def _classes(tokens):
    """Classes of a token stream; a name token is a word, whitelist or punctuation
    token, told apart by the classifier's own graphs."""
    classes = []
    for token in re.findall(r"tokens \{ (.*?) \}(?= tokens|$)", tokens):
        match = re.match(r"(\w+) \{", token)
        if match:
            classes.append(match.group(1))
            continue
        value = re.match(r'name: "(.*)"$', token).group(1)
        if value and all(c in _PUNCTUATION for c in value):
            classes.append("punct")
        else:
            classes.append("name")
    return classes


def _classify(written):
    """The class of a single token, telling the name classes apart by the input."""
    tokens = _best(written)
    classes = _classes(tokens)
    assert len(classes) == 1, (written, tokens)
    if classes[0] == "name":
        try:
            rewrite.top_rewrite(written, _CLASSIFIER.whitelist.fst)
            return "whitelist", tokens
        except rewrite.Error:
            return "word", tokens
    return classes[0], tokens


def _sentence_classes(sentence):
    """Token classes of a sentence, with name tokens split into word and whitelist by
    comparing the value with the written word."""
    tokens = _best(sentence)
    classes = _classes(tokens)
    values = re.findall(r"tokens \{ (.*?) \}(?= tokens|$)", tokens)
    words = sentence.split()
    result = []
    for cls, value in zip(classes, values):
        if cls == "name":
            name = re.match(r'name: "(.*)"$', value).group(1)
            result.append("word" if any(name in w for w in words) else "whitelist")
        else:
            result.append(cls)
    return result


class TestClassifier:
    @parameterized.expand(_SINGLE)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_single_token(self, written, expected_class, fragment):
        cls, tokens = _classify(written)
        assert cls == expected_class, tokens
        assert fragment in tokens, tokens

    @parameterized.expand(_SENTENCES + _ADJACENT)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_sentence(self, sentence, expected):
        assert _sentence_classes(sentence) == expected, _best(sentence)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_token_stream_shape(self):
        assert _best("Toplantı 29.09.2026'da başlayacak.") == (
            'tokens { name: "Toplantı" } tokens { suffix { value: "yirmi dokuz eylül iki bin yirmi altıda" } }'
            ' tokens { name: "başlayacak" } tokens { name: "." }'
        )
        assert _best("2026") == 'tokens { cardinal { integer: "iki bin yirmi altı" } }'
        assert _best("Dr.") == 'tokens { name: "doktor" }'

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_every_suffix_electronic_overlap_is_a_suffix(self):
        """All 86 distinct input strings both grammars accept classify as suffixed
        abbreviations. The shared acceptor is determinized, so each string has one
        path, and checked acyclic, so the paths can be listed."""
        suffix = pynini.project(_CLASSIFIER.token_graphs["suffix"], "input").optimize()
        electronic = pynini.project(_CLASSIFIER.token_graphs["electronic"], "input").optimize()
        shared = pynini.determinize(pynini.intersect(suffix, electronic).optimize()).optimize()
        assert shared.properties(pynini.ACYCLIC, True) & pynini.ACYCLIC
        written = list(shared.paths().istrings())
        assert len(written) == len(set(written)) == 86
        for form in written:
            assert _classes(_best(form)) == ["suffix"], form

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_semantic_classes_share_no_input_but_the_known_ones(self):
        """Exact over whole tokens: besides the word fallback, the only shared inputs
        are the suffixed abbreviations that are also host names."""
        graphs = dict(_CLASSIFIER.token_graphs, punctuation=_CLASSIFIER.punctuation.fst)
        fallbacks = {"word", "unattested_abbreviation", "unattested_suffix"}
        inputs = {name: pynini.project(graph, "input").optimize() for name, graph in graphs.items()}
        for a, b in itertools.combinations(inputs, 2):
            shared = pynini.intersect(inputs[a], inputs[b]).optimize()
            if {a, b} == {"suffix", "electronic"}:
                assert shared.num_states() > 0
            elif not ({a, b} & fallbacks):
                assert shared.num_states() == 0, (a, b)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_random_input_coverage_and_ties(self):
        """Random strings of letters, digits, signs, punctuation and whitespace: every
        one has a tokenization, and any equal-cost ambiguity is of the known
        synthetic kinds: a currency or percent sign between numbers, or an apostrophe
        word inside a chain of numbers."""
        rng = random.Random(0)
        alphabets = ["aeıioöuüçğşIİÇĞŞTDKLMN0123456789.,:;!?()'’-/%@ \t", "0123456789.,/:'’%₺€$ kgmTL"]
        ties = []
        for i in range(600):
            sentence = "".join(rng.choice(alphabets[i % 2]) for _ in range(rng.randint(1, 14)))
            if not sentence.strip():
                continue
            paths = _top_two(sentence)
            assert paths, f"{sentence!r} has no tokenization"
            if _is_tie(paths):
                ties.append(sentence)
        for sentence in ties:
            assert set(sentence) & set("$%₺€'’"), f"unexpected equal-cost ambiguity: {sentence!r}"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_tie_breaker_survives_long_sentences(self):
        """The per character tie breaker is small; it still decides after a long
        sentence has accumulated a large cost. With a final full stop, "3." and "5."
        are ordinals, the known sentence final reading."""
        cases = [
            ("1,2,3", ["decimal", "punct", "cardinal"]),
            ("1,2,3.", ["decimal", "punct", "ordinal"]),
            ("3/4/5", ["fraction", "punct", "cardinal"]),
            ("3/4/5.", ["fraction", "punct", "ordinal"]),
        ]
        for tail, expected in cases:
            for words in [80]:
                sentence = "bu" + " ve" * words + " " + tail
                assert _classes(_best(sentence))[-3:] == expected, (tail, words)

    @parameterized.expand(_KNOWN_TIES)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_known_equal_cost_ambiguities(self, sentence, first, second):
        """Deferred: a sign between numbers that either side may take. Both
        tokenizations cost the same; the test pins them so a change is noticed."""
        paths = _top_two(sentence)
        assert _is_tie(paths), paths
        assert sorted([_classes(paths[0][1]), _classes(paths[1][1])]) == sorted([first, second])

    @parameterized.expand([("5 kg",), (" 5 kg",), ("5 kg ",), ("  5 kg\t",)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_surrounding_whitespace(self, sentence):
        assert _classes(_best(sentence)) == ["measure"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_whitespace_between_tokens(self):
        assert _best("Bugün  hava\tçok\ngüzel") == (
            'tokens { name: "Bugün" } tokens { name: "hava" } tokens { name: "çok" } tokens { name: "güzel" }'
        )
        # MeasureFst allows one space between number and unit
        assert _classes(_best("5  kg")) == ["cardinal", "name"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_weights_order(self):
        """The reasons in _WEIGHTS: a semantic token beats any split into two tokens,
        suffix beats electronic, attested acronyms beat words, words beat unattested
        ones, punctuation stays cheaper than a word, and a punctuation token costs a
        little per mark, less than a whole token."""
        punctuation = {"punctuation", "punctuation_mark", "joined_token_character"}
        semantic = [
            w for name, w in _WEIGHTS.items() if name not in {"word"} | punctuation and "unattested" not in name
        ]
        assert max(semantic) < 2 * min(semantic)
        assert _WEIGHTS["whitelist"] < _WEIGHTS["suffix"] < _WEIGHTS["electronic"]
        assert _WEIGHTS["abbreviation"] < _WEIGHTS["word"] < _WEIGHTS["unattested_abbreviation"]
        assert _WEIGHTS["word"] < _WEIGHTS["unattested_suffix"]
        assert max(semantic) < _WEIGHTS["punctuation"] < _WEIGHTS["word"]
        assert 0 < _WEIGHTS["punctuation_mark"] < _WEIGHTS["punctuation"]
        # the tie breaker within a chunk stays below the cost of a punctuation mark over
        # a hundred characters of joined tokens
        assert 0 < 100 * _WEIGHTS["joined_token_character"] < _WEIGHTS["punctuation_mark"]


class TestWordAndPunctuation:
    word = WordFst()
    punctuation = PunctuationFst()

    @parameterized.expand(
        [(w,) for w in ["merhaba", "Ankara", "çalışıyor", "şirket", "Çayyolu", "ÇIĞLIK", "İzmir", "ığdır", "öğle"]]
        + [(w,) for w in ["Yılmaz'a", "Yılmaz’a", "e-posta", "B2B", "Quartz", "Wi-Fi", "Xerox"]]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_words_are_kept_as_written(self, word):
        assert rewrite.rewrites(word, self.word.fst) == [f'name: "{word}"']

    @parameterized.expand([(w,) for w in ["merhaba,", "dünya!", "(x", "x)", "'x", "x'", "-x", "x-", ",", "a b", ""]])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_punctuation_is_not_part_of_a_word(self, written):
        with pytest.raises(rewrite.Error):
            rewrite.top_rewrite(written, self.word.fst)

    @parameterized.expand(
        [(p,) for p in [".", ",", ":", ";", "!", "?", "(", ")", '"', "'", "«", "»", "—", "...", "?!"]]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_punctuation(self, mark):
        assert rewrite.rewrites(mark, self.punctuation.fst) == [f'name: "{mark}"']


class TestClassifierBuilds:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_cache(self, tmp_path):
        """Built into a cache, restored from it, with the same results."""
        built = ClassifyFst(input_case="cased", deterministic=True, cache_dir=str(tmp_path))
        far = tmp_path / "_cased_tr_tn_True_deterministic.far"
        assert far.exists() and os.path.getsize(far) > 0
        restored = ClassifyFst(input_case="cased", deterministic=True, cache_dir=str(tmp_path))
        assert not hasattr(restored, "token_graphs")
        for sentence in ["Toplantı 29.09.2026'da başlayacak.", "Alm.dan", "(0532) 123 45 67", "EV"]:
            assert _best(sentence, restored.fst) == _best(sentence, built.fst) == _best(sentence)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_non_deterministic_builds(self):
        classifier = ClassifyFst(input_case="cased", deterministic=False)
        assert "measure" in rewrite.top_rewrite("5 kg", classifier.fst)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_lower_cased_input(self):
        """The whitelist matches lower cased input; acronyms cannot be told apart."""
        classifier = ClassifyFst(input_case="lower_cased", deterministic=True)
        assert rewrite.top_rewrite("dr.", classifier.fst) == 'tokens { name: "doktor" }'
        assert rewrite.top_rewrite("tdk", classifier.fst) == 'tokens { name: "tdk" }'
