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
Direct tests for the Turkish electronic tagger and verbalizer.

Every electronic token carries ``preserve_order``. ``_readings`` still runs the tagged
token through the token parser and ``Normalizer._permute``, so these tests exercise the
path the Normalizer will.
"""

import re

import pynini
import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.token_parser import TokenParser
from nemo_text_processing.text_normalization.tr.taggers.cardinal import CardinalFst as CardinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.date import DateFst as DateTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.decimal import DecimalFst as DecimalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.electronic import ElectronicFst as ElectronicTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.fraction import FractionFst as FractionTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.measure import MeasureFst as MeasureTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.money import MoneyFst as MoneyTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.ordinal import OrdinalFst as OrdinalTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.percentage import PercentageFst as PercentageTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.telephone import TelephoneFst as TelephoneTaggerFst
from nemo_text_processing.text_normalization.tr.taggers.time import TimeFst as TimeTaggerFst
from nemo_text_processing.text_normalization.tr.verbalizers.electronic import ElectronicFst as ElectronicVerbalizerFst

from ..utils import parse_test_case_file

_TEST_CASES = 'tr/data_text_normalization/test_cases_electronic.txt'

_SYMBOLS = [
    (".", "nokta"),
    ("_", "alt çizgi"),
    ("-", "tire"),
    ("/", "eğik çizgi"),
    (":", "iki nokta"),
    ("+", "artı"),
]

_INVALID = [
    ("",),
    ("@example.com",),  # e-mail
    ("user@",),
    ("user@example",),
    ("user@@example.com",),
    ("emre..akgul@example.com",),
    (".emre@example.com",),
    ("emre.@example.com",),
    ("user@example.com/foo",),
    ("user@www.",),
    ('"foo bar"@example.com',),
    (".com",),  # host names
    ("example.",),
    ("example..com",),
    ("-example.com",),
    ("example-.com",),
    ("example.c",),
    ("example.c0m",),
    ("foo_bar.com",),  # "_" is not allowed in a host name
    ("example",),
    ("www.foo",),
    ("http:/example.com",),  # URLs
    ("https:/example.com",),
    ("https//example.com",),
    ("://example.com",),
    ("ftp://example.com",),
    ("file://example.com",),
    ("https://",),
    ("www.",),
    ("https://example.com//foo",),
    ("https://example.com/foo.",),
    ("Http://example.com",),
    ("user @example.com",),  # spaces
    ("user@ example.com",),
    ("example .com",),
    ("https:// example.com",),
    (" example.com",),
    ("example.com ",),
    ("example.com.",),  # sentence punctuation
    ("example.com,",),
    ("(example.com)",),
    ("example.com;",),
    ("https://example.com/search?q=test",),  # deferred
    ("https://example.com/page#section",),
    ("example.com:8080",),
    ("https://example.com/a%20b",),
    ("example.com/~user",),
    ("192.168.1.1",),
    ("@username",),
    ("example.com'da",),
    ("user@example.com'a",),
    ("Google.com'da",),
    ("123",),  # other classes
    ("12,5",),
    ("3/4",),
    ("14.30",),
    ("29.09.2026",),
    ("%25",),
    ("100€",),
    ("5 kg",),
    ("90 km/h",),
    ("05321234567",),
]


# An independent reading, for the generated cross check.
_DIGIT_WORDS = ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_PROTOCOLS = {
    "": "",
    "http://": "he te te pe iki nokta eğik çizgi eğik çizgi",
    "https://": "he te te pe se iki nokta eğik çizgi eğik çizgi",
    "www.": "dabılyu dabılyu dabılyu nokta",
    "http://www.": "he te te pe iki nokta eğik çizgi eğik çizgi dabılyu dabılyu dabılyu nokta",
    "https://www.": "he te te pe se iki nokta eğik çizgi eğik çizgi dabılyu dabılyu dabılyu nokta",
}


def _read_characters(text):
    words = dict(_SYMBOLS)
    parts = re.findall(r"[^\W\d_]+|\d|[^\w]|_", text)
    return " ".join(_DIGIT_WORDS[int(p)] if p.isdigit() else words.get(p, p) for p in parts)


def _read_host(host):
    return " nokta ".join(
        {"com": "kom", "tr": "te re"}.get(label, _read_characters(label)) for label in host.split(".")
    )


def _read_path(path):
    return "".join(
        " eğik çizgi" + (f" {_read_characters(segment)}" if segment else "") for segment in path.split("/")[1:]
    )


class TestElectronic:

    tagger = ElectronicTaggerFst()
    verbalizer = ElectronicVerbalizerFst()
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

    @parameterized.expand(_INVALID)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_invalid_input_is_rejected(self, test_input):
        with pytest.raises(rewrite.Error):
            rewrite.top_rewrite(test_input, self.tagger_fst)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_token_schema(self):
        assert self._tag("emre@example.com") == (
            'electronic { username: "emre" domain: "example.com" preserve_order: true }'
        )
        assert self._tag("example.com") == 'electronic { domain: "example.com" preserve_order: true }'
        assert self._tag("https://example.com") == (
            'electronic { protocol: "https://" domain: "example.com" preserve_order: true }'
        )
        assert self._tag("www.example.com") == (
            'electronic { protocol: "www." domain: "example.com" preserve_order: true }'
        )
        assert self._tag("https://www.example.com/foo") == (
            'electronic { protocol: "https://www." domain: "example.com/foo" preserve_order: true }'
        )

    @parameterized.expand(_SYMBOLS)
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_symbols(self, symbol, word):
        assert rewrite.rewrites(f"a{symbol}b", self.verbalizer.characters_graph) == [f"a {word} b"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_at_is_et_and_url_slash_is_not_division(self):
        assert " et " in self._normalize("a@example.com")
        assert "kuyruklu" not in self._normalize("a@example.com")
        reading = self._normalize("example.com/foo")
        assert reading == "example nokta kom eğik çizgi foo"
        assert "bölü" not in reading

    @parameterized.expand(
        [("http://", "he te te pe iki nokta eğik çizgi eğik çizgi")]
        + [("https://", "he te te pe se iki nokta eğik çizgi eğik çizgi"), ("www.", "dabılyu dabılyu dabılyu nokta")]
        + [("https://www.", "he te te pe se iki nokta eğik çizgi eğik çizgi dabılyu dabılyu dabılyu nokta")]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_protocols(self, protocol, expected):
        assert rewrite.rewrites(protocol, self.verbalizer.protocol_graph) == [expected]

    @parameterized.expand([("http://example.com",), ("https://example.com",), ("www.example.com",)])
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_earlier_protocol_readings_are_gone(self, written):
        """ "h" is "he" and "s" is "se", the Turkish letter names, and "w" is "dabılyu"."""
        words = self._normalize(written).split()
        assert "ha" not in words
        assert "es" not in words
        assert "ve" not in words
        assert self._normalize("https://www.example.com").startswith(
            "he te te pe se iki nokta eğik çizgi eğik çizgi dabılyu dabılyu dabılyu nokta"
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_protocol_and_host_casing(self):
        assert self._tag("HTTPS://WWW.EXAMPLE.COM") == self._tag("https://www.example.com")
        assert self._tag("Example.COM") == self._tag("example.com")
        # host names are lower cased with ASCII rules, paths and user names are kept
        assert '"github.io"' in self._tag("GITHUB.IO")
        assert '"example.com/Foo"' in self._tag("EXAMPLE.com/Foo")
        assert '"Emre"' in self._tag("Emre@example.com")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_digits_are_read_one_by_one(self):
        assert self._normalize("user007@example.com").startswith("user sıfır sıfır yedi et")
        assert self._normalize("abc123.com") == "abc bir iki üç nokta kom"
        assert "yüz" not in self._normalize("https://example.com/123")

    @parameterized.expand(
        [
            ("user1@example.com", "user01@example.com"),
            ("foo-bar.com", "foobar.com"),
            ("foo_bar@example.com", "foobar@example.com"),
        ]
        + [("foo_bar@example.com", "foo-bar@example.com"), ("foo.bar@example.com", "foobar@example.com")]
        + [("example.com/a/b", "example.com/ab"), ("example.com/", "example.com"), ("a.b.com", "ab.com")]
        + [("emre+test@example.com", "emretest@example.com"), ("x7q2.com", "x72q.com")]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_distinct_addresses_read_differently(self, first, second):
        assert self._normalize(first) != self._normalize(second)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_reusable_subgraphs_are_exposed(self):
        assert rewrite.rewrites("emre.akgul", self.tagger.username_graph) == ["emre.akgul"]
        assert rewrite.rewrites("Example.COM/Foo", self.tagger.domain_graph) == ["example.com/Foo"]
        assert rewrite.rewrites("HTTPS://", self.tagger.protocol_graph) == ["https://"]
        assert rewrite.rewrites("emre_7", self.verbalizer.characters_graph) == ["emre alt çizgi yedi"]
        assert rewrite.rewrites("gmail.com.tr", self.verbalizer.domain_graph) == ["gmail nokta kom nokta te re"]
        assert rewrite.rewrites("emre@example.com", self.tagger.graph) == ["emre et example nokta kom"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_generated_cross_check(self):
        """Structural combinations against an independent reading: every character is
        read, nothing is ambiguous, and the bare graph agrees."""
        users = ["emre", "e", "emre7", "7emre", "e7e", "emre.akgul", "emre_akgul", "emre-akgul", "emre+tag", "a.b.c"]
        hosts = ["example.com", "mail.example.com", "a.b.c.example.org", "foo-bar.net", "abc123.com.tr", "x.io"]
        paths = ["", "/", "/foo", "/foo/bar", "/foo/bar/baz", "/v2", "/a-b_c.d", "/007/", "/çağrı"]
        protocols = ["", "http://", "https://", "www.", "http://www.", "https://www."]
        for user in users:
            for host in hosts:
                written = f"{user}@{host}"
                expected = f"{_read_characters(user)} et {_read_host(host)}"
                assert self._normalize(written) == expected, written
                assert rewrite.rewrites(written, self.tagger.graph) == [expected], written
        for protocol in protocols:
            for host in hosts:
                for path in paths:
                    written = f"{protocol}{host}{path}"
                    expected = " ".join(filter(None, [_PROTOCOLS[protocol], _read_host(host) + _read_path(path)]))
                    assert self._normalize(written) == expected, written
                    assert rewrite.rewrites(written, self.tagger.graph) == [expected], written

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_graph_has_no_state_explosion(self):
        assert self.tagger_fst.num_states() < 500, self.tagger_fst.num_states()
        assert self.verbalizer_fst.num_states() < 1000, self.verbalizer_fst.num_states()


class TestElectronicSeparation:
    """Electronic against every earlier Turkish grammar."""

    cardinal = CardinalTaggerFst()
    decimal = DecimalTaggerFst(cardinal=cardinal)
    fraction = FractionTaggerFst(cardinal=cardinal)
    electronic_fst = ElectronicTaggerFst().fst
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
        "telephone": TelephoneTaggerFst(cardinal=cardinal).fst,
    }

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_no_input_is_shared_with_another_grammar(self):
        """Exact, over all inputs: the input languages do not intersect."""
        electronic_inputs = pynini.project(self.electronic_fst, "input").optimize()
        for name, fst in self.others.items():
            shared = pynini.intersect(electronic_inputs, pynini.project(fst, "input").optimize()).optimize()
            assert shared.num_states() == 0, name
