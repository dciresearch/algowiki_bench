from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent, NativeOutput, PromptedOutput

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from .utils import find_code_sections
import hishel
import httpcore
import httpx
from hishel.httpx import AsyncCacheTransport


class DropErrorFilter(hishel.BaseFilter[hishel.Response]):
    def needs_body(self) -> bool:
        """Return True if the filter needs access to the response body."""
        return False

    def apply(self, item: hishel.Response, body: bytes | None) -> bool:
        """
        Return True to cache the response, False to skip caching.

        Args:
            item: The response to filter
            body: The response body (only if needs_body() returns True)
        """
        # Your filtering logic here
        # print(item.status_code)
        return item.status_code < 300


policy = hishel.FilterPolicy(response_filters=[DropErrorFilter()])
policy.use_body_key = True

cache_client = httpx.AsyncClient(
    transport=AsyncCacheTransport(
        next_transport=httpx.AsyncHTTPTransport(),
        storage=hishel.AsyncSqliteStorage(database_path="./cache/pydantic_ai.db"),
        policy=policy
    ),
    timeout=600
)


def setup_pydantic_model(llm_name, api_base, api_key, **gen_kwargs):
    settings = OpenAIChatModelSettings(**gen_kwargs)
    model = OpenAIChatModel(
        model_name=llm_name,
        provider=OpenAIProvider(
            base_url=api_base,
            api_key=api_key, http_client=cache_client
        ),
        settings=settings,
    )
    return model

# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------


class TranslatorOutput(BaseModel):
    code: str = Field("Python implementation")


class AnalyzerOutput(BaseModel):
    proof: str = Field("Step-by-step proof of latex formula")
    latex_formula: str = Field("Latex formula of analyzed feature")
    o_notation_of_formula: str = Field("O-notation of latex formula")


class AlgoWikiAnalysis(BaseModel):
    complexity: str
    inp_size: str
    out_size: str
    form_height: str
    form_width: str
    full_formulae: dict[str, str]
    reasoning: dict[str, str]
    input_desc: str
    scores: list | None = None

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)


class NormalizeOutput(BaseModel):
    normalized_formulae: str = Field("Normalized version of formulae")
    o_notation: str = Field("O-notation of normalized formulae")


class CleanOutput(BaseModel):
    cleaned_input_text: str = Field("Input text with removed data leaks")


# ---------------------------------------------------------------------------
# System prompts (carried over from the original DSPy Signature docstrings)
# ---------------------------------------------------------------------------

_TRANSLATOR_PROMPT = (
    "Translate description to MPI Python code. "
    "Implementation must use no external functions except MPI and math operations."
)

_SERIAL_COMPLEXITY_PROMPT = """\
Serial complexity of algorithm - the number of operations that need to be performed if the algorithm is executed serially.

Read the algorithm description and derive the Serial complexity.
Provide full proof, latex formula and O-notation of latex formula.

Requirements:
- use latex format $$...$$ to write all formulae
- use only single letter variables
- final latex formula must use format $$O(...)$$
"""

_SERIAL_INP_SIZE_PROMPT = """\
Read the algorithm description and derive the input data size.
Provide full proof, latex formula and O-notation of latex formula.

Requirements:
- use latex format $$...$$ to write all formulae
- use only single letter variables
- final latex formula must use format $$O(...)$$
"""

_SERIAL_OUT_SIZE_PROMPT = """\
Read the algorithm description and derive the output data size.
Provide full proof, latex formula and O-notation of latex formula.

Requirements:
- use latex format $$...$$ to write all formulae
- use only single letter variables
- final latex formula must use format $$O(...)$$
"""

_PF_HEIGHT_PROMPT = """\
Parallel Form (PF) is a data dependecy graph in which:
- each vertice represents input element of each algorithm operation
- each vertice layer represents algorithm operation;
- the source vertex of every edge is located in a layer with a lower index than the destination vertex;
- there are no edges between vertices located within the same layer.

PF height is the total number of layers in Parallel Form.

Read the algorithm description and derive the PF height.
Provide full proof, latex formula and O-notation of latex formula.

Requirements:
- use latex format $$...$$ to write all formulae
- use only single letter variables
- final latex formula must use format $$O(...)$$
"""

_PF_WIDTH_PROMPT = """\
Parallel Form (PF) is a data dependecy graph in which:
- each vertice represents input element of each algorithm operation
- each vertice layer represents algorithm operation;
- the source vertex of every edge is located in a layer with a lower index than the destination vertex;
- there are no edges between vertices located within the same layer.

The width of a Parallel Form layer is the number of vertices contained in that layer.
PF Width is the maximum width among all its layers.

Read the algorithm description and derive the PF width formula.
Provide full proof, latex formula and O-notation of latex formula.

Requirements:
- use latex format $$...$$ to write all formulae
- use only single letter variables
"""

_NORMALIZE_PROMPT = """\
Read the latex formulae and normalize.

Rules:
- replace all variables with subscripts with different single letter variable
- replace set size |V| with explicit variable
- replace greek letter variables with english letter variable
- replace all multiplication operations with \\cdot
- remove any rounding operations such as ceil and floor
- remove any O-notation brackets
- put all function arguments in {} brackets
- use \\\\ for all functions

Provide both normalized formulae and their respective O-notation
"""

_CLEAN_PROMPT = """\
You are preparing dataset for machine learning.
Each examples consists of input_text and target that must be derived from the text.
To ensure that there are no data leak you must clean all text fragments that would trivialize target synthesis.

Read the input_text and analyze whether it has fragments that directly overlap with target.
Remove any input_text fragments that contain target answer.

Mark all removed fragments as [REDACTED].
"""


# ---------------------------------------------------------------------------
# Agent factory helpers
# ---------------------------------------------------------------------------


def _make_agent(
    system_prompt: str,
    output_type: type[BaseModel],
    model: OpenAIChatModel | None,
    retries: int = 3,
) -> Agent[None, BaseModel]:
    return Agent(
        model=model,
        output_type=output_type,
        system_prompt=system_prompt,
        retries=retries,
    )


# ---------------------------------------------------------------------------
# Module classes
# ---------------------------------------------------------------------------


class AlgoWikiImplToCode:
    """Translate an algorithm description into MPI Python code."""

    def __init__(self, model: OpenAIChatModel | None = None):
        self._agent = _make_agent(_TRANSLATOR_PROMPT, TranslatorOutput, model)

    def run(self, description: str) -> TranslatorOutput:
        result = self._agent.run_sync(description)
        return result.output

    async def arun(self, description: str) -> TranslatorOutput:
        result = await self._agent.run(description)
        return result.output


class AlgoWikiAnalyzerMM:
    """Run five independent complexity analyses in parallel."""

    def __init__(self, model: OpenAIChatModel | None = None):
        self._complexity = _make_agent(_SERIAL_COMPLEXITY_PROMPT, AnalyzerOutput, model)
        self._inp_size = _make_agent(_SERIAL_INP_SIZE_PROMPT, AnalyzerOutput, model)
        self._out_size = _make_agent(_SERIAL_OUT_SIZE_PROMPT, AnalyzerOutput, model)
        self._pf_height = _make_agent(_PF_HEIGHT_PROMPT, AnalyzerOutput, model)
        self._pf_width = _make_agent(_PF_WIDTH_PROMPT, AnalyzerOutput, model)

    @staticmethod
    def _build_analysis(
        description: str,
        complexity: AnalyzerOutput,
        inp: AnalyzerOutput,
        out: AnalyzerOutput,
        height: AnalyzerOutput,
        width: AnalyzerOutput,
    ) -> AlgoWikiAnalysis:
        return AlgoWikiAnalysis(
            complexity=complexity.o_notation_of_formula,
            inp_size=inp.o_notation_of_formula,
            out_size=out.o_notation_of_formula,
            form_height=height.o_notation_of_formula,
            form_width=width.o_notation_of_formula,
            full_formulae={
                "complexity": complexity.latex_formula,
                "inp_size": inp.latex_formula,
                "out_size": out.latex_formula,
                "form_height": height.latex_formula,
                "form_width": width.latex_formula,
            },
            reasoning={
                "complexity": complexity.proof,
                "inp_size": inp.proof,
                "out_size": out.proof,
                "form_height": height.proof,
                "form_width": width.proof,
            },
            input_desc=description,
        )

    def run(self, description: str) -> AlgoWikiAnalysis:
        complexity = self._complexity.run_sync(description).output
        inp = self._inp_size.run_sync(description).output
        out = self._out_size.run_sync(description).output
        height = self._pf_height.run_sync(description).output
        width = self._pf_width.run_sync(description).output
        return self._build_analysis(description, complexity, inp, out, height, width)

    async def arun(self, description: str) -> AlgoWikiAnalysis:
        complexity_r, inp_r, out_r, height_r, width_r = await asyncio.gather(
            self._complexity.run(description),
            self._inp_size.run(description),
            self._out_size.run(description),
            self._pf_height.run(description),
            self._pf_width.run(description),
        )
        return self._build_analysis(
            description,
            complexity_r.output,
            inp_r.output,
            out_r.output,
            height_r.output,
            width_r.output,
        )


class AlgoWikiCodeAnalyzer:
    """Top-level benchmark pipeline: optionally translate, then analyse."""

    def __init__(self, use_translator: bool = False, model: OpenAIChatModel | None = None):
        self.use_translator = use_translator
        self._translator = AlgoWikiImplToCode(model=model)
        self._analyzer = AlgoWikiAnalyzerMM(model=model)

    async def arun(self, message: str) -> AlgoWikiAnalysis:
        if self.use_translator:
            translation = await self._translator.arun(message)
            if find_code_sections(translation.code):
                message = translation.code
        return await self._analyzer.arun(message)

    def run(self, message: str) -> AlgoWikiAnalysis:
        if self.use_translator:
            translation = self._translator.run(message)
            if find_code_sections(translation.code):
                message = translation.code
        return self._analyzer.run(message)


class AlgoWikiNormalizeGT:
    """Normalise ground-truth LaTeX formulae into canonical form."""

    def __init__(self, model: OpenAIChatModel | None = None):
        self._agent = _make_agent(_NORMALIZE_PROMPT, NormalizeOutput, model)

    async def arun(self, message: str) -> NormalizeOutput:
        result = await self._agent.run(message)
        return result.output


class AlgoWikiCleanData:
    """Redact answer-revealing fragments from training examples."""

    def __init__(self, model: OpenAIChatModel | None = None):
        self._agent = _make_agent(_CLEAN_PROMPT, CleanOutput, model)

    async def arun(self, message: str, target: str) -> CleanOutput:
        prompt = f"input_text:\n{message}\n\ntarget:\n{target}"
        result = await self._agent.run(prompt)
        return result.output
