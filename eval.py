from src.programs_pydantic import setup_pydantic_model
from src.programs import setup_dspy_model
import logging
import sys
from pathlib import Path
import fire
from src.dataset import build_dataset
import os
import dspy
from src.latex_eval import o_notation_score
from src.universal_eval import UniversalEvaluate
import numpy as np


class ONotationSimilarity:
    def __init__(self, target_fields: list):
        self.target_fields = target_fields

    def __call__(self, example, pred, trace=None, pred_name=None, pred_trace=None):
        """
        Computes a score based on agreement between prediction and gold standard.
        Returns the score (float).
        """
        # Compute scores for all modules
        scores = [
            o_notation_score(
                getattr(pred, prop), getattr(example, prop)
            ) for prop in self.target_fields
        ]

        scores = scores
        # Overall score: average of the three accuracies
        total = np.mean(scores)
        pred['scores'] = scores

        return total

    @property
    def target(self):
        return self.target_fields


gen_kwargs = {
    "temperature": 1.0,
    "top_k": 60,
    "top_p": 0.8,
    "repetition_penalty": 1.0,
    "frequency_penalty": 0.1,
    "min_p": 0.2,
}


def setup_program(backend, llm_name, api_url, api_key, gen_kwargs):
    if backend == 'dspy':
        from src.programs import AlgoWikiCodeAnalyzer
        setup_dspy_model(llm_name, api_url, api_key, **gen_kwargs)
        return AlgoWikiCodeAnalyzer()
    if backend == 'pydantic':
        from src.programs_pydantic import AlgoWikiCodeAnalyzer
        model = setup_pydantic_model(llm_name, api_url, api_key, **gen_kwargs)
        return AlgoWikiCodeAnalyzer(model=model)


async def run_eval(
    llm_name: str, api_url: str = '0.0.0.0', api_key: str = "NULL",
    out_path: str = None, limit: int = None, backend='dspy',
    input_format='python_desc'
):
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)
    fp_name = llm_name.replace(" ", "___").replace("/", "---")
    fp_full = out_path / f"{fp_name}.json"

    logging.basicConfig(level=logging.INFO, filename=out_path / f"{fp_name}.log", filemode="w",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logger = logging.getLogger()
    sys.stdout.write = logger.info
    sys.stderr.write = logger.warning

    dspy.disable_litellm_logging()

    dataset = build_dataset(limit=limit, message_field=input_format)
    print(llm_name, out_path.absolute(), len(dataset))
    program = setup_program(backend, llm_name, api_url, api_key, gen_kwargs)

    print("Running", llm_name)

    metric = ONotationSimilarity(['complexity', "inp_size", "out_size", "form_height", "form_width"])

    print(fp_full.absolute())

    evaluate = UniversalEvaluate(
        devset=dataset,
        metric=metric,
        num_threads=10,
        display_progress=True,
        provide_traceback=True,
        save_as_json=fp_full
    )
    print("Starting", llm_name)
    await evaluate.aevaluate(program)
    print("Finished", llm_name)
    return


if __name__ == '__main__':
    fire.Fire(run_eval)
