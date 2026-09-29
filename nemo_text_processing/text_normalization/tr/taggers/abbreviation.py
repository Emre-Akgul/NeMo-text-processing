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

from nemo_text_processing.text_normalization.tr.graph_utils import GraphFst, insert_space
from nemo_text_processing.text_normalization.tr.utils import get_abs_path


class AbbreviationFst(GraphFst):
    """
    Finite state transducer for classifying Turkish acronyms and initialisms, e.g.
        "TDK" -> abbreviation { value: "te de ke" }
        "TBMM" -> abbreviation { value: "te be me me" }
        "NATO" -> abbreviation { value: "nato" }
        "PKK" -> abbreviation { value: "pe ka ka" }

    An upper case abbreviation is read in one of two ways, as TDK describes: letter by
    letter ("TDK'den", "THY'de") or as a word ("NATO'dan", "ASELSAN'da"). Which one
    cannot be told from the spelling, so:

        1. data/abbreviation/acronym_readings.tsv lists the readings that are not
           letter by letter with the Turkish letter names: the abbreviations read as
           words ("NATO" -> "nato", "UNESCO" -> "unesko"), and conventional readings
           that differ from the letter names ("PKK" -> "pe ka ka", with the "ka"
           TDK advises against but usage keeps; "BMW" -> "be me ve"; the technical
           "CPU" -> "si pi yu" and "GPU" -> "ci pi yu", with their English letter
           names). It also holds
           the one dotted abbreviation, "T.C." -> "te ce";
        2. any other string of two or more letters of the Turkish alphabet in
           capitals is read letter by letter with the names in
           data/abbreviation/letters.tsv: "H" is "he", "K" is "ke", "S" is "se".

    The table takes priority: its entries are removed from the letter by letter
    reading, so "PKK" has the single reading "pe ka ka". Whitelist entries, which are
    read in full ("AŞ" -> "anonim şirket", "MÖ", "MS"), are removed from both.

    "Q", "W" and "X" are not letters of the Turkish alphabet; letters.tsv gives them
    the conventional names "ku", "ve" and "iks", so "W" reads like "V". These names
    are for abbreviations only: electronic addresses read "www" as "dabılyu dabılyu
    dabılyu". Only capitals are accepted. Other dotted abbreviations ("A.B.C.", "A.Ş."),
    single letters, digits and suffixed forms ("TDK'den", "NATO'dan") are not.

    For the classifier: every string of capitals is accepted, including ordinary words
    written in capitals ("EV", "OKUL"), which a word grammar will have to outrank; the
    whitelist has to outrank this grammar. For the suffix layer, ``graph`` gives the
    reading, which is what a suffix agrees with: "PKK'ya" follows "ka", not "ke".

    Args:
        whitelist: WhiteListFst, whose entries are left to it
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, whitelist: GraphFst, deterministic: bool = True):
        super().__init__(name="abbreviation", kind="classify", deterministic=deterministic)

        # "Ğ" -> "yumuşak ge"
        self.letter_graph = pynini.string_file(get_abs_path("data/abbreviation/letters.tsv")).optimize()
        # "TDK" -> "te de ke"
        self.initialism_graph = (self.letter_graph + pynini.closure(insert_space + self.letter_graph, 1)).optimize()
        # "NATO" -> "nato", "PKK" -> "pe ka ka"
        self.acronym_graph = pynini.string_file(get_abs_path("data/abbreviation/acronym_readings.tsv")).optimize()

        whitelisted = pynini.project(whitelist.graph, "input")
        listed = pynini.project(self.acronym_graph, "input")
        letter_by_letter = pynini.compose(
            pynini.difference(pynini.project(self.initialism_graph, "input"), pynini.union(listed, whitelisted)),
            self.initialism_graph,
        )
        acronyms = pynini.compose(pynini.difference(listed, whitelisted), self.acronym_graph)

        self.graph = pynini.union(acronyms, letter_by_letter).optimize()

        self.fst = self.add_tokens(pynutil.insert("value: \"") + self.graph + pynutil.insert("\"")).optimize()
