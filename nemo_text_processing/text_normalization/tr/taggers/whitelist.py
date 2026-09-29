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

from nemo_text_processing.text_normalization.tr.graph_utils import GraphFst, convert_space
from nemo_text_processing.text_normalization.tr.utils import get_abs_path, load_labels

INPUT_CASED = "cased"
INPUT_LOWER_CASED = "lower_cased"


def _tr_lower(text: str) -> str:
    """Lower cases with Turkish rules: "I" -> "ı", "İ" -> "i"."""
    return text.replace("I", "ı").replace("İ", "i").lower()


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying Turkish whitelisted abbreviations, e.g.
        "Dr." -> name: "doktor"
        "vb." -> name: "ve benzeri"
        "No." -> name: "numara"
        "MÖ" -> name: "milattan önce"

    The whitelist holds abbreviations that are read aloud as their full form, each
    with one reading, from TDK's Kısaltmalar Dizini (data/whitelist.tsv). It is meant
    to have the highest priority among the classifier grammars, above a future
    acronym grammar: "AŞ" is read "anonim şirket", not spelled out.

    Entries are matched exactly as written: case matters ("Sn." is "sayın", while
    "sn." is the unit "saniye" of MeasureFst) and a final full stop is part of the
    abbreviation ("Dr" is not accepted). Abbreviations read as letters or as a word
    ("TDK", "NATO", "T.C.") are left to a separate acronym grammar, and those with
    more than one reading ("vd.", "Mah.", "Hz.") or whose reading depends on the
    address around them ("Cad.", "Sok.") are left out. Unit, currency, month and
    protocol abbreviations belong to their own grammars.

    A suffix on an expanded abbreviation follows the full form and needs no
    apostrophe after a full stop ("Alm.dan", "vb.leri", "No.lu"); these forms are
    not accepted here and are left to the suffix layer.

    Args:
        input_case: accepting either "lower_cased" or "cased" input; lower cased
            input is matched against the entries lower cased with Turkish rules
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        input_file: path to a file with whitelist replacements, which replace the
            default entries in deterministic mode and are added to them otherwise
    """

    def __init__(self, input_case: str = INPUT_CASED, deterministic: bool = True, input_file: str = None):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        def _get_whitelist_graph(file: str) -> "pynini.FstLike":
            whitelist = load_labels(file)
            if input_case == INPUT_LOWER_CASED:
                whitelist = [[_tr_lower(written), spoken] for written, spoken in whitelist]
            return pynini.string_map(whitelist)

        graph = _get_whitelist_graph(get_abs_path("data/whitelist.tsv"))

        if input_file:
            provided = _get_whitelist_graph(input_file)
            graph = provided if deterministic else graph | provided

        # "vb." -> "ve benzeri"
        self.graph = graph.optimize()
        self.final_graph = convert_space(self.graph).optimize()
        self.fst = (pynutil.insert("name: \"") + self.final_graph + pynutil.insert("\"")).optimize()
