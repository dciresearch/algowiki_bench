from __future__ import annotations

import asyncio
import inspect
import json
import logging
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import dspy
import tqdm
import tqdm.asyncio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    score: float
    results: list[tuple[dspy.Example, Any, float]]
    num_total: int
    num_correct: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_cell(content) -> str:
    words = str(content).split()
    if len(words) > 25:
        return " ".join(words[:25]) + "..."
    return content


def _prediction_to_dict(prediction: Any) -> dict:
    if isinstance(prediction, dspy.Prediction):
        return prediction.toDict()
    if hasattr(prediction, "model_dump"):
        return prediction.model_dump()
    if hasattr(prediction, "__dict__"):
        return dict(prediction.__dict__)
    return {"prediction": prediction}


def _merge_example_prediction(example: dspy.Example, prediction: Any) -> dict:
    ex = example.toDict() if hasattr(example, "toDict") else dict(example)
    pred = _prediction_to_dict(prediction)
    merged = {}
    for k, v in ex.items():
        if k in pred:
            merged[f"example_{k}"] = v
        else:
            merged[k] = v
    for k, v in pred.items():
        if k in ex:
            merged[f"pred_{k}"] = v
        else:
            merged[k] = v
    return merged


def _convert_old_row(row: dict) -> dict:
    """Convert old flat merged format to new structured format."""
    example = {}
    prediction = {}
    for k, v in row.items():
        if k.startswith("example_"):
            example[k[len("example_"):]] = v
        elif k.startswith("pred_"):
            prediction[k[len("pred_"):]] = v
        else:
            example[k] = v
            prediction[k] = v
    return {"example": example, "prediction": prediction, "score": None}


def _is_dspy_module(program: Any) -> bool:
    return isinstance(program, dspy.Module)


def _has_run_arun(program: Any) -> bool:
    return callable(getattr(program, "run", None)) and callable(getattr(program, "arun", None))


def _extract_inputs(example: dspy.Example) -> dict[str, Any]:
    return example.inputs()


def _call_metric(metric: Callable, example: dspy.Example, prediction: Any) -> float:
    sig = inspect.signature(metric)
    params = list(sig.parameters.keys())
    if len(params) <= 2:
        return metric(example, prediction)
    return metric(example, prediction, trace=None, pred_name=None, pred_trace=None)


# ---------------------------------------------------------------------------
# UniversalEvaluate
# ---------------------------------------------------------------------------


class UniversalEvaluate:
    def __init__(
        self,
        *,
        devset: list[dspy.Example],
        metric: Callable,
        num_threads: int = 5,
        display_progress: bool = True,
        display_table_rows: int = 5,
        max_errors: int | None = None,
        provide_traceback: bool = True,
        failure_score: float = 0.0,
        save_as_json: str | Path | None = None,
    ):
        self.devset = devset
        self.metric = metric
        self.num_threads = num_threads
        self.display_progress = display_progress
        self.display_table_rows = display_table_rows
        self.max_errors = max_errors if max_errors is not None else 100
        self.provide_traceback = provide_traceback
        self.failure_score = failure_score
        self.save_as_json = Path(save_as_json) if save_as_json else None

    def _execute_sync_dspy(self, program: dspy.Module, inputs: dict) -> Any:
        return program(**inputs)

    def _execute_sync_pydantic(self, program: Any, inputs: dict) -> Any:
        message = inputs.get("message", "")
        return program.run(message)

    async def _execute_async_dspy(self, program: dspy.Module, inputs: dict) -> Any:
        return await program.acall(**inputs)

    async def _execute_async_pydantic(self, program: Any, inputs: dict) -> Any:
        message = inputs.get("message", "")
        return await program.arun(message)

    def _resolve_executor_sync(self, program: Any) -> Callable[[dict], Any]:
        if _is_dspy_module(program):
            return lambda inputs: self._execute_sync_dspy(program, inputs)
        if _has_run_arun(program):
            return lambda inputs: self._execute_sync_pydantic(program, inputs)
        raise TypeError(
            f"Unsupported program type {type(program).__name__}. "
            "Expected a dspy.Module or an object with .run()/.arun() methods."
        )

    def _resolve_executor_async(self, program: Any) -> Callable[[dict], Any]:
        if _is_dspy_module(program):
            return lambda inputs: self._execute_async_dspy(program, inputs)
        if _has_run_arun(program):
            return lambda inputs: self._execute_async_pydantic(program, inputs)
        raise TypeError(
            f"Unsupported program type {type(program).__name__}. "
            "Expected a dspy.Module or an object with .run()/.arun() methods."
        )

    def _process_result(
        self, example: dspy.Example, prediction: Any, score: float
    ) -> tuple[dspy.Example, Any, float]:
        return (example, prediction, score)

    def _handle_error(self, idx: int, example: dspy.Example, error: Exception) -> tuple[dspy.Example, Any, float]:
        if self.provide_traceback:
            logger.error(f"Error evaluating example {idx}: {error}", exc_info=True)
        return (example, dspy.Prediction(), self.failure_score)

    def _build_results(
        self, raw_results: list[tuple[dspy.Example, Any, float]]
    ) -> EvaluationResult:
        ncorrect = sum(score for *_, score in raw_results)
        ntotal = len(raw_results)
        score = round(100 * ncorrect / ntotal, 2) if ntotal else 0.0
        return EvaluationResult(
            score=score,
            results=raw_results,
            num_total=ntotal,
            num_correct=ncorrect,
        )

    def _display_table(self, results: list[tuple[dspy.Example, Any, float]]):
        import pandas as pd

        metric_name = (
            self.metric.__name__
            if isinstance(self.metric, types.FunctionType)
            else self.metric.__class__.__name__
        )

        rows = []
        for example, prediction, score in results:
            merged = _merge_example_prediction(example, prediction)
            merged[metric_name] = score
            rows.append(merged)

        df = pd.DataFrame(rows)
        df = df.map(_truncate_cell) if hasattr(df, "map") else df.applymap(_truncate_cell)

        if isinstance(self.display_table_rows, int):
            df = df.head(self.display_table_rows)

        print(f"\nAverage Metric: {self._last_result.num_correct} / "
              f"{self._last_result.num_total} ({self._last_result.score}%)")
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None
        ):  # more options can be specified also
            print(df)

    def _save_json(self, results: list[tuple[dspy.Example, Any, float]]):
        if not self.save_as_json:
            return
        self.save_as_json.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for example, prediction, score in results:
            ex_dict = example.toDict() if hasattr(example, "toDict") else dict(example)
            pred_dict = _prediction_to_dict(prediction)
            rows.append({
                "example": ex_dict,
                "prediction": pred_dict,
                "score": score,
            })

        with open(self.save_as_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2, default=str)

    def _load_saved_results(self) -> list[dict]:
        if not self.save_as_json or not self.save_as_json.exists():
            return []
        with open(self.save_as_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not raw:
            return []
        if "example" in raw[0]:
            return raw
        return [_convert_old_row(row) for row in raw]

    def _match_saved_result(
        self, example: dspy.Example, saved_results: list[dict], metric
    ) -> dict | None:
        inputs = _extract_inputs(example)
        for row in saved_results:
            ex = row.get("example", {})
            if all(ex.get(k) == v for k, v in inputs.items()):
                pred = row.get("prediction", {})
                if all(pred.get(k, None) for k in metric.target):
                    row = dict(row.items())
                    row['prediction'] = {
                        k: v for k, v in row['prediction'].items() if (
                            k not in example or k in metric.target
                        )
                    }
                    return row
                break
        return None

    @staticmethod
    def _reconstruct_prediction(pred_dict: dict) -> dspy.Prediction:
        return dspy.Prediction(**pred_dict)

    def __call__(self, program: Any) -> EvaluationResult:
        executor = self._resolve_executor_sync(program)

        # Step 1-2: Load saved results
        saved_results = self._load_saved_results()

        # Step 3: Match examples, build evaluation_list
        evaluation_list = []
        matched = {}
        for idx, example in enumerate(self.devset):
            match = self._match_saved_result(example, saved_results, self.metric)
            if match is not None:
                matched[idx] = match
            else:
                evaluation_list.append((idx, example))

        # Step 4: Generate predictions for missing examples
        new_predictions = {}
        errors = 0
        iterator = evaluation_list
        if self.display_progress:
            iterator = tqdm.tqdm(iterator, desc="Evaluating", total=len(evaluation_list))

        for idx, example in iterator:
            if errors > self.max_errors:
                logger.warning(f"Stopping early after {errors} errors (max_errors={self.max_errors})")
                break
            inputs = _extract_inputs(example)
            try:
                prediction = executor(inputs)
                new_predictions[idx] = prediction
            except Exception as e:
                errors += 1
                new_predictions[idx] = None

        # Step 5-6: Build raw_results in devset order, re-score ALL
        raw_results = []
        for idx, example in enumerate(self.devset):
            if idx in matched:
                prediction = self._reconstruct_prediction(matched[idx]["prediction"])
            elif idx in new_predictions:
                pred = new_predictions[idx]
                if pred is None:
                    raw_results.append(self._handle_error(idx, example, Exception("Prediction failed")))
                    continue
                prediction = pred
            else:
                raw_results.append(self._handle_error(idx, example, Exception("Skipped")))
                continue

            try:
                score = _call_metric(self.metric, example, prediction)
                raw_results.append(self._process_result(example, prediction, score))
            except Exception as e:
                raw_results.append(self._handle_error(idx, example, e))

        result = self._build_results(raw_results)
        self._last_result = result

        if self.display_table_rows:
            self._display_table(raw_results)

        # Step 7: Save
        self._save_json(raw_results)

        return result

    async def aevaluate(self, program: Any) -> EvaluationResult:
        executor = self._resolve_executor_async(program)
        semaphore = asyncio.Semaphore(self.num_threads)

        # Step 1-2: Load saved results
        saved_results = self._load_saved_results()

        # Step 3: Match examples, build evaluation_list
        evaluation_list = []
        matched = {}
        for idx, example in enumerate(self.devset):
            match = self._match_saved_result(example, saved_results, self.metric)
            if match is not None:
                matched[idx] = match
            else:
                evaluation_list.append((idx, example))

        # Step 4: Generate predictions for missing examples
        new_predictions = {}
        errors = 0

        async def _process_one(idx: int, example: dspy.Example):
            nonlocal errors
            if errors > self.max_errors:
                return idx, None
            inputs = _extract_inputs(example)
            async with semaphore:
                try:
                    prediction = await executor(inputs)
                    return idx, prediction
                except Exception as e:
                    errors += 1
                    return idx, None

        if evaluation_list:
            tasks = [_process_one(idx, ex) for idx, ex in evaluation_list]
            if self.display_progress:
                results = await tqdm.asyncio.tqdm.gather(*tasks, desc="Evaluating (async)")
            else:
                results = await asyncio.gather(*tasks)
            for idx, pred in results:
                new_predictions[idx] = pred

        # Step 5-6: Build raw_results in devset order, re-score ALL
        raw_results = []
        for idx, example in enumerate(self.devset):
            if idx in matched:
                prediction = self._reconstruct_prediction(matched[idx]["prediction"])
            elif idx in new_predictions:
                pred = new_predictions[idx]
                if pred is None:
                    raw_results.append(self._handle_error(idx, example, Exception("Prediction failed")))
                    continue
                prediction = pred
            else:
                raw_results.append(self._handle_error(idx, example, Exception("Skipped")))
                continue

            try:
                score = _call_metric(self.metric, example, prediction)
                raw_results.append(self._process_result(example, prediction, score))
            except Exception as e:
                raw_results.append(self._handle_error(idx, example, e))

        result = self._build_results(raw_results)
        self._last_result = result

        if self.display_table_rows:
            self._display_table(raw_results)

        # Step 7: Save
        self._save_json(raw_results)

        return result
