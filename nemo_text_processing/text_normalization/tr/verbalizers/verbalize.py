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

from nemo_text_processing.text_normalization.en.verbalizers.abbreviation import AbbreviationFst
from nemo_text_processing.text_normalization.tr.graph_utils import GraphFst
from nemo_text_processing.text_normalization.tr.verbalizers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.tr.verbalizers.date import DateFst
from nemo_text_processing.text_normalization.tr.verbalizers.decimal import DecimalFst
from nemo_text_processing.text_normalization.tr.verbalizers.electronic import ElectronicFst
from nemo_text_processing.text_normalization.tr.verbalizers.fraction import FractionFst
from nemo_text_processing.text_normalization.tr.verbalizers.measure import MeasureFst
from nemo_text_processing.text_normalization.tr.verbalizers.money import MoneyFst
from nemo_text_processing.text_normalization.tr.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.tr.verbalizers.percentage import PercentageFst
from nemo_text_processing.text_normalization.tr.verbalizers.suffix import SuffixFst
from nemo_text_processing.text_normalization.tr.verbalizers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.tr.verbalizers.time import TimeFst


class VerbalizeFst(GraphFst):
    """
    Composes the Turkish semantic class verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Whitelist readings and ordinary words are "name" tokens; they are read by the word
    verbalizer in VerbalizeFinalFst, not here. Abbreviations already carry their reading,
    "pe ka ka" or "nato", so the language independent en verbalizer only unwraps them.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)

        cardinal = CardinalFst(deterministic=deterministic)
        decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
        fraction = FractionFst(deterministic=deterministic)
        ordinal = OrdinalFst(deterministic=deterministic)
        date = DateFst(deterministic=deterministic)
        time = TimeFst(deterministic=deterministic)
        percentage = PercentageFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic)
        money = MoneyFst(decimal=decimal, deterministic=deterministic)
        measure = MeasureFst(cardinal=cardinal, decimal=decimal, fraction=fraction, deterministic=deterministic)
        telephone = TelephoneFst(deterministic=deterministic)
        electronic = ElectronicFst(deterministic=deterministic)
        suffix = SuffixFst(deterministic=deterministic)
        abbreviation = AbbreviationFst(deterministic=deterministic)

        graph = (
            cardinal.fst
            | ordinal.fst
            | decimal.fst
            | fraction.fst
            | date.fst
            | time.fst
            | percentage.fst
            | money.fst
            | measure.fst
            | telephone.fst
            | electronic.fst
            | suffix.fst
            | abbreviation.fst
        )
        self.fst = graph.optimize()
