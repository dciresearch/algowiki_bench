import asyncio
from .utils import find_code_sections
import dspy
from typing import List, Literal


def setup_dspy_model(llm_name, api_base, api_key, **gen_kwargs):
    lm = dspy.LM(f"openai/{llm_name}", api_key=api_key, api_base=api_base, model_type="chat", timeout=600)
    dspy.configure(lm=lm, **gen_kwargs, rollout_id=23, max_retries=10)


class PythonTranslator(dspy.Signature):
    """
    Translate description to MPI Python code.
    Implementation must use no external functions except MPI and math operations.
    """
    description: str = dspy.InputField()
    code: str = dspy.OutputField()


class AlgoWikiImplToCode(dspy.Module):
    def __init__(self):
        self.translator_module = dspy.Predict(PythonTranslator)

    def forward(self, message: str):
        translation = self.translator_module(description=message)

        return dspy.Prediction(
            translation=translation.code
        )

    async def aforward(self, message: str):
        translation = await self.translator_module.acall(description=message)

        return dspy.Prediction(
            translation=translation.code
        )


class AnalyzerSerialComplexity(dspy.Signature):
    """
    Serial complexity of algorithm - the number of operations that need to be performed if the algorithm is executed serially.

    Read the algorithm description and derive the Serial complexity.
    Provide full proof, latex formula and O-notation of latex formula.

    Requirements:
    - use latex format $$...$$ to write all formulae
    - use only single letter variables
    - final latex formula must use format $$O(...)$$
    """
    description: str = dspy.InputField()
    proof: str = dspy.OutputField()
    latex_formula: str = dspy.OutputField()
    o_notation_of_formula: str = dspy.OutputField()


class AnalyzerSerialInpSize(dspy.Signature):
    """
    Read the algorithm description and derive the input data size.
    Provide full proof, latex formula and O-notation of latex formula.

    Requirements:
    - use latex format $$...$$ to write all formulae
    - use only single letter variables
    - final latex formula must use format $$O(...)$$
    """
    description: str = dspy.InputField()
    proof: str = dspy.OutputField()
    latex_formula: str = dspy.OutputField()
    o_notation_of_formula: str = dspy.OutputField()


class AnalyzerSerialOutSize(dspy.Signature):
    """
    Read the algorithm description and derive the output data size.
    Provide full proof, latex formula and O-notation of latex formula.

    Requirements:
    - use latex format $$...$$ to write all formulae
    - use only single letter variables
    - final latex formula must use format $$O(...)$$
    """
    description: str = dspy.InputField()
    proof: str = dspy.OutputField()
    latex_formula: str = dspy.OutputField()
    o_notation_of_formula: str = dspy.OutputField()


class AnalyzerParallelFormHeight(dspy.Signature):
    """
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
    description: str = dspy.InputField()
    proof: str = dspy.OutputField()
    latex_formula: str = dspy.OutputField()
    o_notation_of_formula: str = dspy.OutputField()


class AnalyzerParallelFormWidth(dspy.Signature):
    """
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
    description: str = dspy.InputField()
    proof: str = dspy.OutputField()
    latex_formula: str = dspy.OutputField()
    o_notation_of_formula: str = dspy.OutputField()


class AlgoWikiAnalyzerMM(dspy.Module):
    def __init__(self):
        self.complexity_module = dspy.Predict(AnalyzerSerialComplexity)
        self.inputsize_module = dspy.Predict(AnalyzerSerialInpSize)
        self.outputsize_module = dspy.Predict(AnalyzerSerialOutSize)
        self.formheight_module = dspy.Predict(AnalyzerParallelFormHeight)
        self.formwidth_module = dspy.Predict(AnalyzerParallelFormWidth)

    def forward(self, message: str):
        complexity = self.complexity_module(description=message)
        inputsize = self.inputsize_module(description=message)
        outputsize = self.outputsize_module(description=message)
        formheight = self.formheight_module(description=message)
        formwidth = self.formwidth_module(description=message)

        reasonings = {
            "complexity": complexity.proof,
            "inp_size": inputsize.proof,
            "out_size": outputsize.proof,
            "form_height": formheight.proof,
            "form_width": formwidth.proof,
        }

        full_formulae = {
            "complexity": complexity.latex_formula,
            "inp_size": inputsize.latex_formula,
            "out_size": outputsize.latex_formula,
            "form_height": formheight.latex_formula,
            "form_width": formwidth.latex_formula,
        }

        return dspy.Prediction(
            complexity=complexity.o_notation_of_formula,
            inp_size=inputsize.o_notation_of_formula,
            out_size=outputsize.o_notation_of_formula,
            form_height=formheight.o_notation_of_formula,
            form_width=formwidth.o_notation_of_formula,
            full_formulae=full_formulae,
            reasoning=reasonings,
            input_desc=message
        )

    async def aforward(self, message: str):
        complexity = self.complexity_module(description=message)
        inputsize = self.inputsize_module(description=message)
        outputsize = self.outputsize_module(description=message)
        formheight = self.formheight_module(description=message)
        formwidth = self.formwidth_module(description=message)

        reasonings = {
            "complexity": complexity.proof,
            "inp_size": inputsize.proof,
            "out_size": outputsize.proof,
            "form_height": formheight.proof,
            "form_width": formwidth.proof,
        }

        full_formulae = {
            "complexity": complexity.latex_formula,
            "inp_size": inputsize.latex_formula,
            "out_size": outputsize.latex_formula,
            "form_height": formheight.latex_formula,
            "form_width": formwidth.latex_formula,
        }

        return dspy.Prediction(
            complexity=complexity.o_notation_of_formula,
            inp_size=inputsize.o_notation_of_formula,
            out_size=outputsize.o_notation_of_formula,
            form_height=formheight.o_notation_of_formula,
            form_width=formwidth.o_notation_of_formula,
            full_formulae=full_formulae,
            reasoning=reasonings,
            input_desc=message
        )


class AlgoWikiCodeAnalyzer(dspy.Module):
    def __init__(self, use_translator=False):
        self.use_translator = use_translator
        self.translate_program = AlgoWikiImplToCode()
        self.analyze_program = AlgoWikiAnalyzerMM()

    def forward(self, message: str):
        if self.use_translator:
            res = self.translate_program(message)
            res = res.translation
            if find_code_sections(res):
                message = res
        analysis = self.analyze_program(message)
        return analysis

    async def aforward(self, message: str):
        if self.use_translator:
            res = await self.translate_program.acall(message)
            res = res.translation
            if find_code_sections(res):
                message = res
        analysis = await self.analyze_program.acall(message)
        return analysis


class NormalizeFormula(dspy.Signature):
    """
    Read the latex formulae and normalize.

    Rules:
    - replace all variables with subscripts with different single letter variable
    - replace set size |V| with explicit variable
    - replace greek letter variables with english letter variable
    - replace all multiplication operations with \\cdot
    - remove any rounding operations such as ceil and floor
    - remove any O-notation brackets
    - put all function arguments in {} brackets
    - use \\ for all functions

    Provide both normalized formulae and their respective O-notation.
    """
    original_formulae: str = dspy.InputField()
    normalized_formulae: str = dspy.OutputField()
    O_notation_of_normalized_formulae: str = dspy.OutputField()


class AlgoWikiNormalizeGT(dspy.Module):
    def __init__(self):
        self.normalizer = dspy.ChainOfThought(NormalizeFormula, cache=False)

    async def aforward(self, message: str):
        res = await self.normalizer.acall(original_formulae=message)

        return dspy.Prediction(
            normalized=res.normalized_formulae,
            o_notation=res.O_notation_of_normalized_formulae
        )


class CleanDescription(dspy.Signature):
    """
    You are preparing dataset for machine learning.
    Each examples consists of input_text and target that must be derived from the text.
    To ensure that there are no data leak you must clean all text fragments that would trivialize target synthesis.

    Read the input_text and analyze whether it has fragments that directly overlap with target.
    Remove any input_text fragments that contain target answer.

    Mark all removed fragments as [REDACTED].

    """
    input_text: str = dspy.InputField()
    target: str = dspy.InputField()
    cleaned_input_text: str = dspy.OutputField()


class AlgoWikiCleanData(dspy.Module):
    def __init__(self):
        self.clean = dspy.ChainOfThought(CleanDescription)

    async def aforward(self, message: str, target: str):
        res = await self.clean.acall(input_text=message, target=target)

        return dspy.Prediction(
            cleaned=res.cleaned_input_text
        )
