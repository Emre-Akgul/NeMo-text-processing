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

import string

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.tr.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    TR_ALPHA,
    GraphFst,
    delete_preserve_order,
    insert_space,
)
from nemo_text_processing.text_normalization.tr.verbalizers.electronic import ElectronicFst as ElectronicVerbalizer

_ASCII_LOWER = pynini.union(*string.ascii_lowercase).optimize()
_ASCII_UPPER = pynini.union(*string.ascii_uppercase).optimize()
_ASCII_ALNUM = pynini.union(_ASCII_LOWER, _ASCII_UPPER, NEMO_DIGIT).optimize()

# ASCII lower casing. Host names and URL schemes are ASCII and case insensitive, so
# "I" is "i" here, not the Turkish "ı".
_ASCII_TO_LOWER = pynini.cdrewrite(
    pynini.union(*[pynini.cross(u, l) for u, l in zip(string.ascii_uppercase, string.ascii_lowercase)]),
    "",
    "",
    NEMO_SIGMA,
)


def _lowercase_variants(words: list) -> "pynini.FstLike":
    """Accepts each word in lower case or in capitals and emits it in lower case."""
    return pynini.union(*[pynini.cross(word.upper(), word) | pynini.accep(word) for word in words]).optimize()


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying Turkish electronic addresses, e.g.
        "emre.akgul@gmail.com" -> electronic { username: "emre.akgul" domain: "gmail.com" preserve_order: true }
        "example.com.tr" -> electronic { domain: "example.com.tr" preserve_order: true }
        "https://www.example.com/foo" -> electronic { protocol: "https://www." domain: "example.com/foo"
            preserve_order: true }

    Ported from the de and hu grammars: the tagger checks the structure and keeps the
    written strings in the ``username``, ``protocol`` and ``domain`` fields, a path
    being part of ``domain``, and the verbalizer reads them.

    Accepted:
        - e-mail addresses: a user name of letters, digits, "_", "+" and "-", in parts
          joined by single full stops, then "@" and a host name;
        - host names: labels of letters and digits, with hyphens inside a label only,
          joined by full stops, ending in a top level label of two or more letters, so
          "29.09.2026", "14.30" and "192.168.1.1" are not host names;
        - URLs: "http://", "https://", "www." or a scheme followed by "www.", then a host
          name and an optional path; a bare host name with an optional path.
    A path is "/" separated segments of letters, digits, "-", "_" and ".", which may
    include the Turkish letters and may end in "/"; a segment does not end in a full
    stop. User names and host names are ASCII; punycode labels ("xn--...") are
    accepted as written.

    Schemes, "www" and host names are case insensitive and are lower cased with ASCII
    rules; user names and paths are kept as written. A host name beginning with "www."
    is always read with "www." as the protocol.

    Not accepted: a full stop, comma or bracket around the address, which is the
    sentence's ("example.com." is rejected); query strings and fragments, whose "=",
    "&" and "#" have no settled reading; ports; IP addresses; other schemes; quoted
    user names; social media handles ("@emre"); suffixed forms such as
    "example.com'da".

    For the classifier: electronic addresses need letters in the top level label and so
    share no input with the number, date, time, percentage, money, measure or telephone
    grammars. The surrounding punctuation has to be split off first, and a suffix will
    have to agree with how the address is read ("gmail nokta kom'a").

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        # emre.akgul, emre_7, emre+test
        user_part = pynini.closure(_ASCII_ALNUM | "_" | "+" | "-", 1)
        self.username_graph = (user_part + pynini.closure("." + user_part)).optimize()

        # mail.example.com.tr -> lower case
        label = pynini.closure(_ASCII_ALNUM, 1) + pynini.closure(
            pynini.closure("-", 1) + pynini.closure(_ASCII_ALNUM, 1)
        )
        top_level = pynini.closure(_ASCII_LOWER | _ASCII_UPPER, 2)
        host = pynini.closure(label + ".", 1) + top_level
        # "www." belongs to the protocol.
        host_without_www = pynini.difference(host, _lowercase_variants(["www"]).project("input") + "." + NEMO_SIGMA)
        lower_host = pynini.compose(host, _ASCII_TO_LOWER).optimize()
        lower_host_without_www = pynini.compose(host_without_www, _ASCII_TO_LOWER).optimize()

        # /foo/bar_baz.html/
        path_character = _ASCII_ALNUM | (TR_ALPHA - _ASCII_LOWER - _ASCII_UPPER) | "-" | "_" | "."
        segment = pynini.closure(path_character) + pynini.difference(path_character, ".")
        path = pynini.closure("/" + segment) + pynini.closure("/", 0, 1)
        self.domain_graph = (lower_host_without_www + path).optimize()

        # https://, http://www., www. -> lower case
        scheme = _lowercase_variants(["http", "https"]) + "://"
        www = _lowercase_variants(["www"]) + "."
        self.protocol_graph = pynini.union(scheme, www, scheme + www).optimize()

        def field(key: str, graph: "pynini.FstLike") -> "pynini.FstLike":
            return pynutil.insert(f"{key}: \"") + graph + pynutil.insert("\"")

        email = field("username", self.username_graph) + pynini.cross("@", " ") + field("domain", lower_host)
        url = field("protocol", self.protocol_graph) + insert_space + field("domain", self.domain_graph)
        domain = field("domain", self.domain_graph)

        self.final_graph = (pynini.union(email, url, domain) + pynutil.insert(" preserve_order: true")).optimize()

        # "emre@example.com" -> "emre et example nokta kom", the reading without token fields.
        verbalizer = ElectronicVerbalizer(deterministic=deterministic)
        self.graph = pynini.compose(self.final_graph, verbalizer.graph + delete_preserve_order).optimize()

        self.fst = self.add_tokens(self.final_graph).optimize()
