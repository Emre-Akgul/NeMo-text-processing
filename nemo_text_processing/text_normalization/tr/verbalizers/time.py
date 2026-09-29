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
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
    delete_space,
    insert_space,
)


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing Turkish digital clock times, e.g.
        time { hours: "on dört" minutes: "otuz" preserve_order: true } -> on dört otuz
        time { hours: "on dört" preserve_order: true } -> on dört
        time { hours: "on dört" minutes: "otuz" seconds: "sıfır beş" preserve_order: true } -> on dört otuz sıfır beş

    The components are read in order, hours, minutes, seconds, with nothing inserted
    between them.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        def field(key: str) -> "pynini.FstLike":
            return (
                pynutil.delete(f"{key}:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.closure(NEMO_NOT_QUOTE, 1)
                + pynutil.delete("\"")
            )

        hours = field("hours")
        minutes = field("minutes")
        seconds = field("seconds")
        separator = delete_space + insert_space

        graph = pynini.union(
            hours,
            hours + separator + minutes,
            hours + separator + minutes + separator + seconds,
        )

        self.graph = graph
        self.fst = self.delete_tokens(graph + delete_preserve_order).optimize()
