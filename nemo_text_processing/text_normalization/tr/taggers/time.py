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

from typing import Callable

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.tr.graph_utils import GraphFst, insert_space

# Hour and minute are separated by a full stop in Turkish orthography ("13.30"), and by
# a colon in the common digital form ("13:30"). Seconds are only accepted after colons.
_HM_SEPARATORS = [".", ":"]
_HMS_SEPARATOR = ":"


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying Turkish digital clock times, e.g.
        "14.30" -> time { hours: "on dört" minutes: "otuz" preserve_order: true }
        "14:30" -> time { hours: "on dört" minutes: "otuz" preserve_order: true }
        "08:05" -> time { hours: "sekiz" minutes: "sıfır beş" preserve_order: true }
        "14:00" -> time { hours: "on dört" preserve_order: true }
        "14:30:05" -> time { hours: "on dört" minutes: "otuz" seconds: "sıfır beş" preserve_order: true }

    A time is read as a digital clock, component by component, with no "saat",
    "dakika" or "saniye": 14.30 is "on dört otuz". The conversational readings ("iki
    buçuk", "ikiyi çeyrek geçiyor") need 12 hour conversion and are not produced.

    Hours are 0-23, optionally with one leading zero, which is formatting and is not
    spoken: 08:30 is "sekiz otuz". "24.00" is not accepted. Minutes and seconds are
    exactly two digits, 00-59. Between 01 and 09 the written zero is spoken, 08:05 is
    "sekiz sıfır beş"; from 10 they are read as cardinals.

    A time written with hours and minutes only drops a zero minute, as the en grammar
    does: 14:00 is "on dört". A time written with seconds keeps every component, zeros
    included, so 14:00:05 is "on dört sıfır sıfır sıfır beş" and 14:30:00 is "on dört
    otuz sıfır sıfır".

    Hours and minutes may be separated by "." (the Turkish orthographic form) or ":";
    both give the same token. Seconds are only accepted in the colon form "14:30:05":
    "14.30.05" is not accepted, since Turkish orthography gives the full stop for
    hours and minutes only and a three part dotted number is a date.

    Written order and spoken order agree, so every token carries ``preserve_order``
    and no field permutation is needed.

    Not handled here: suffixed forms such as "17.30'da", "saat", AM/PM, time zones,
    and the conversational "buçuk", "çeyrek", "geçe" and "kala" readings. For the
    suffix layer, ``graph`` gives the bare reading, "17.30" -> "on yedi otuz", whose
    final word the locative in tr.morphology can inflect.

    For the classifier: "14.30" is a time, "14,30" a decimal, "29.09.2026" a date,
    "3/4" a fraction and "14." an ordinal; none of these inputs is accepted by two of
    those grammars. A dotted time has to be considered as a whole token rather than
    split at its full stop into an ordinal, and "14.30" can still be something other
    than a time, e.g. in arithmetic, which is for context weighting to decide.

    Args:
        cardinal: CardinalFst, supplies the readings of all components
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        # "8" -> "sekiz", "08" -> "sekiz", "00" -> "sıfır"
        hour_digits = [(str(n), str(n)) for n in range(0, 24)] + [(f"0{n}", str(n)) for n in range(0, 10)]
        self.hour_graph = pynini.compose(pynini.string_map(hour_digits), cardinal.graph).optimize()

        # "00" -> "sıfır sıfır", "05" -> "sıfır beş", "30" -> "otuz"
        single_digit = pynini.union(cardinal.zero, cardinal.digit)
        zero_led = cardinal.zero + insert_space + single_digit
        ten_to_fifty_nine = pynini.compose(pynini.union(*[str(n) for n in range(10, 60)]), cardinal.graph)
        self.minute_graph = pynini.union(zero_led, ten_to_fifty_nine).optimize()
        self.second_graph = self.minute_graph

        def labelled(key: str, graph: "pynini.FstLike") -> "pynini.FstLike":
            return pynutil.insert(f"{key}: \"") + graph + pynutil.insert("\"")

        def bare(key: str, graph: "pynini.FstLike") -> "pynini.FstLike":
            return graph

        self.final_graph = (self._clock(labelled) + pynutil.insert(" preserve_order: true")).optimize()

        # "17.30" -> "on yedi otuz", the reading without token fields.
        self.graph = self._clock(bare).optimize()

        self.fst = self.add_tokens(self.final_graph).optimize()

    def _clock(self, field: Callable[[str, "pynini.FstLike"], "pynini.FstLike"]) -> "pynini.FstLike":
        """
        Builds the accepted clock forms, with each component wrapped by ``field``.

        Args:
            field: takes a field name and a component graph, returns the graph to emit

        Returns a pynini.FstLike
        """
        hours = field("hours", self.hour_graph)
        minutes = field("minutes", self.minute_graph)
        seconds = field("seconds", self.second_graph)

        # With hours and minutes only, a zero minute is dropped: 14:00 -> "on dört".
        non_zero_minute_digits = pynini.difference(pynini.project(self.minute_graph, "input"), "00")
        non_zero_minutes = field("minutes", pynini.compose(non_zero_minute_digits, self.minute_graph))
        optional_minutes = pynini.union(pynutil.delete("00"), insert_space + non_zero_minutes)

        hm = pynini.union(*[hours + pynutil.delete(sep) + optional_minutes for sep in _HM_SEPARATORS])

        delete_sep = pynutil.delete(_HMS_SEPARATOR)
        hms = hours + delete_sep + insert_space + minutes + delete_sep + insert_space + seconds

        return pynini.union(hm, hms)
