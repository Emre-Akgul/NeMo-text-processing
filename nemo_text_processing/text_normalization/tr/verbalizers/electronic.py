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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.tr.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    TR_ALPHA,
    GraphFst,
    delete_preserve_order,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.tr.utils import get_abs_path, load_labels


def _spaced(words: "pynini.FstLike", letters: "pynini.FstLike" = None) -> "pynini.FstLike":
    """
    Reads a string in which ``letters`` are kept and everything ``words`` maps is
    replaced by its reading, with one space between the parts, e.g. for letters a-z
    and digits:
        "user007" -> "user sıfır sıfır yedi"

    Args:
        words: characters or character sequences to read, with their readings
        letters: characters kept as they are, if any

    Returns a pynini.FstLike
    """
    read = pynutil.insert(NEMO_SPACE) + words + pynutil.insert(NEMO_SPACE)
    spaced = pynini.closure(read if letters is None else pynini.union(letters, read))
    squeeze = pynini.cdrewrite(pynini.cross(NEMO_SPACE * 2, NEMO_SPACE), "", "", NEMO_SIGMA)
    trim = pynini.cdrewrite(pynutil.delete(NEMO_SPACE), "[BOS]", "", NEMO_SIGMA) @ pynini.cdrewrite(
        pynutil.delete(NEMO_SPACE), "", "[EOS]", NEMO_SIGMA
    )
    return (spaced @ squeeze @ trim).optimize()


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing Turkish electronic addresses, e.g.
        electronic { username: "emre.akgul" domain: "gmail.com" preserve_order: true }
            -> emre nokta akgul et gmail nokta kom
        electronic { protocol: "https://" domain: "example.com/v2" preserve_order: true }
            -> ha te te pe es iki nokta eğik çizgi eğik çizgi example nokta kom eğik çizgi v iki

    A run of letters is read as it is written, as a word; digits are read one by one,
    "007" -> "sıfır sıfır yedi"; symbols are read by their Turkish names in
    data/electronic/symbols.tsv. "@" is "et", as an e-mail address is read aloud,
    although TDK names the sign "kuyruklu a"; "/" is "eğik çizgi", TDK's name for the
    sign, not the "bölü" of division. Protocols are read letter by letter
    (data/electronic/protocol.tsv), and domain labels whose written form would be read
    wrongly have their own readings (data/electronic/domain.tsv): "com" is "kom", since a
    Turkish "c" is not a "k", and "tr" is "te re".

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        symbol_words = dict(load_labels(get_abs_path("data/electronic/symbols.tsv")))
        symbols = pynini.string_map(symbol_words.items())
        digits = pynini.union(
            pynini.string_file(get_abs_path("data/numbers/zero.tsv")),
            pynini.string_file(get_abs_path("data/numbers/digit.tsv")),
        )
        characters = pynini.union(digits, symbols)

        # "emre_7" -> "emre alt çizgi yedi"
        self.characters_graph = _spaced(characters, TR_ALPHA)

        # Domain labels: "com" -> "kom", "tr" -> "te re", anything else read as characters.
        domain_labels = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))
        label = pynini.closure(TR_ALPHA | NEMO_DIGIT | "-", 1)
        other_label = pynini.difference(label, pynini.project(domain_labels, "input"))
        label_graph = pynini.union(domain_labels, pynini.compose(other_label, self.characters_graph))
        dot = pynini.cross(".", f" {symbol_words['.']} ")
        slash = pynini.cross("/", f" {symbol_words['/']}")
        host = label_graph + pynini.closure(dot + label_graph)
        path_segment = pynini.compose(
            pynini.closure(TR_ALPHA | NEMO_DIGIT | "-" | "_" | ".", 1), self.characters_graph
        )
        path = pynini.closure(slash + pynini.closure(insert_space + path_segment, 0, 1))
        # "example.com/foo" -> "example nokta kom eğik çizgi foo"
        self.domain_graph = (host + path).optimize()

        # "https://www." -> "ha te te pe es iki nokta eğik çizgi eğik çizgi ve ve ve nokta"
        protocol_words = pynini.string_file(get_abs_path("data/electronic/protocol.tsv"))
        self.protocol_graph = pynini.compose(
            pynini.closure(pynini.union(pynini.project(protocol_words, "input"), ":", "/", "."), 1),
            _spaced(pynini.union(protocol_words, symbols)),
        ).optimize()

        def field(key: str, graph: "pynini.FstLike") -> "pynini.FstLike":
            return pynutil.delete(f"{key}:") + delete_space + pynutil.delete("\"") + graph + pynutil.delete("\"")

        protocol = field("protocol", self.protocol_graph)
        username = field("username", self.characters_graph)
        domain = field("domain", self.domain_graph)
        at = pynutil.insert(f" {symbol_words['@']} ")

        graph = pynini.union(
            protocol + delete_space + insert_space + domain,
            username + delete_space + at + domain,
            domain,
        )

        self.graph = graph
        self.fst = self.delete_tokens(graph + delete_preserve_order).optimize()
