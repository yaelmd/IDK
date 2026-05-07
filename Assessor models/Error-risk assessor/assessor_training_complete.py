# -*- coding: utf-8 -*-

import os
from typing import Union
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import polars as pl
from datasets import load_dataset

from modeling.experiment import (
    PairedTrainingExperiment,
    AllBenchmarksExperiment
)
from modeling.utils import AutoObjectiveXGBClassifier


DATA_PATH = Path("data")
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"

BENCHMARKS = [
    'addition',
    'anagram',
    'locality',
    'science',
    'logic_puzzles',
    'geometry'
]

ENGINES = [
    "gemini-3-pro-preview",
    "GPT5.2",
    "llama3v3chat-70B",
    "Qwen2.5_72B"
]

FEATURES = [
    'difficulty',
    'embedding',
    'delean',
    'difficulty+embedding',
    'difficulty+delean',
    'embedding+delean',
    'difficulty+embedding+delean'
]

DIFF_COL_MAPPING = {
    'addition': 'carry',
    'anagram': 'letters',
    'locality': 'city_frequency',
}

TEMPLATE_ID_MAPPING = {
    'addition': 12,
    'anagram': 12,
    'locality': 7,
    'science': 7,
    'logic_puzzles': None,
    'geometry': None
}

# Set environment variables for macOS with Homebrew-installed libomp
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib"
os.environ["LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib"
os.environ["CPATH"] = "/opt/homebrew/opt/libomp/include"

# Help prevent thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"         # MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Accelerate (macOS)
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def load_prompts(prompt_data_path):
    dfs = []
    for bench in BENCHMARKS:
        df = pd.read_csv(prompt_data_path / f"{bench}.csv")
        df['benchmark'] = bench
        if bench == 'science':
            df['custom_id'] = df['ori_id'].str.replace("_", "-").str.replace("obqa", "openbookqa")
        else:
            df['custom_id'] = df["benchmark"] + "-" + df["instance_id"].astype(str)
        df = df[["custom_id", "benchmark", "instance_id", 'question']]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).rename(columns={"question": "prompt"})


def load_difficulty(features_path):
    dfs = []
    for bench in BENCHMARKS:
        difficulty_df = pd.read_csv(features_path / f"difficulty/{bench}_difficulty.csv")
        difficulty_df['benchmark'] = bench
        diff_col = DIFF_COL_MAPPING.get(bench, 'difficulty')
        difficulty_df = (
            difficulty_df[['benchmark', 'instance_id', diff_col]]
                .rename(columns={diff_col: 'difficulty'})
        )
        dfs.append(difficulty_df)
    return pd.concat(dfs, ignore_index=True)


def load_cal_diff(features_path):
    dfs = []
    for bench, template_id in TEMPLATE_ID_MAPPING.items():
        if bench in ['logic_puzzles', 'geometry']:
            difficulty_df = pd.read_csv(features_path / f"difficulty/{bench}_calibrated_difficulty.csv")
            difficulty_df.rename(columns={'cal_diff': 'difficulty'}, inplace=True)
        else:
            difficulty_df = load_dataset("lexin-zhou/ReliabilityBench", "science")['test'].to_pandas()
            difficulty_df = difficulty_df[difficulty_df['template_id'] == template_id]
        difficulty_df['benchmark'] = bench
        difficulty_df = difficulty_df[['benchmark', 'instance_id', 'difficulty']]
        dfs.append(difficulty_df)
    return pd.concat(dfs, ignore_index=True)


def load_embeddings(features_path):
    dfs = []
    for bench in BENCHMARKS:
        embedding_df = pd.read_csv(features_path / f"embeddings/{bench}_embedded_text.csv")
        embedding_df['custom_id'] = bench + "-" + embedding_df["instance_id"].astype(str)
        embedding_df['embedding'] = embedding_df['embedding'].apply(lambda x: eval(x))
        embedding_df = embedding_df[["custom_id", "embedding"]]
        dfs.append(embedding_df)
    return pd.concat(dfs, ignore_index=True)
        

def build_prompt_and_features(prompts_path, features_path, out_path, difficulty_type='original'):
    prompts = load_prompts(prompts_path)
    if difficulty_type == 'original':
        difficulty = load_difficulty(features_path)
    elif difficulty_type == 'calibrated':
        difficulty = load_cal_diff(features_path)
    else:
        raise ValueError(f"Invalid difficulty_type: {difficulty_type}")
    embeddings = load_embeddings(features_path)
    delean = pd.read_csv(features_path / "delean/idk-annotations_annotations_wide_cleaned.csv.csv")

    prompt_and_features = (
        prompts
            .merge(
                difficulty,
                on=['benchmark', 'instance_id'],
                how='left'
            )
            .merge(
                embeddings,
                on=['custom_id'],
                how='left'
            )
            .merge(
                delean[list(delean.columns.difference(['benchmark', 'instance_id']))],
                on=['custom_id'],
                how='left'
            )
    )
    out_file = out_path / f"prompt_and_features-{difficulty_type}-diff.parquet"
    prompt_and_features.to_parquet(out_file, index=False)
    return out_file


def load_response_data(
        full_res_path: str,
        engine: str | list | None = None,
        benchmark: str | list | None = None,
        columns: str | list | None = None,
        return_pandas: bool = False,
        lazy: bool = False
    ) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Load and combine response data from benchmark result CSVs.

    Args:
        full_res_path (str): Root path containing folders for each model family.
        return_pandas (bool): If True, return a Pandas DataFrame instead of Polars. Default is False.

    Returns:
        pl.DataFrame, pl.LazyFrame or pd.DataFrame: Combined dataframe with standardized columns.
    """
    if engine is None:
        engine = ["BLOOM", "GPT", "LLAMA"]
    elif isinstance(engine, str):
        engine = [engine]
    elif not isinstance(engine, list):
        raise ValueError("engine must be a string or a list of strings.")
    
    if benchmark is None:
        benchmark = ["addition", "anagram", "locality", "GPQA-fixed", "OpenBookQA", "CommonTransforms"]
    elif isinstance(benchmark, str):
        benchmark = [benchmark]
    elif not isinstance(benchmark, list):
        raise ValueError("benchmark must be a string or a list of strings.")

    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        elif not isinstance(columns, list):
            raise ValueError("columns must be a string or a list of strings.")
    else:
        #columns = ["prompt", "engine", "response", "confidence", "outcome"]
        columns = ["prompt", "engine", "outcome"]

    dfs = []

    for eng in engine:
        for bench in benchmark:
            path = f"{full_res_path}/{eng}/{bench}/formatted.csv"
            if lazy:
                df = pl.scan_csv(path)
                df = df.select(columns)
            else:
                df = pl.read_csv(path, columns=columns)
            
            df = df.with_columns(
                pl.lit(eng).alias("engine"),
                pl.lit(bench).alias("benchmark")
            )

            # Reorder columns: engine + result_features
            ordered_cols = ["engine", "benchmark"] + columns
            df = df.select(ordered_cols)

            dfs.append(df)

    # Concatenate all result DataFrames
    response_data = pl.concat(dfs)

    # Convert to Pandas DataFrame if requested
    if return_pandas:
        if lazy:
            response_data = response_data.collect()
        response_data = response_data.to_pandas()

    return response_data


def load_corpus(prompt_and_features_path, response_data_path):
    prompt_and_features = pl.scan_parquet(prompt_and_features_path)
    response_data = load_response_data(
        full_res_path=response_data_path,
        engine=["gemini-3-pro-preview", "GPT5.2", "llama3v3chat-70B", "Qwen2.5_72B"],
        benchmark=["addition", "anagram", "locality", "science", "logic_puzzles", "geometry"],
        columns=["instance_id", "outcome"],
        lazy=True
    )
    return (
        prompt_and_features
            .join(response_data, on=['benchmark', 'instance_id'], how='inner')
            .rename({'benchmark': 'dataset'})
    )


def build_iid_experiment(
        experiment_name: str,
        corpus: pl.DataFrame,
        features: list[str],
        datasets: list[str],
        engines: list[str],
        results_path: Path,
    ):
    return PairedTrainingExperiment(
        name=experiment_name,
        data=corpus,
        features=features,
        target_col='outcome',
        clfs={
            'XGB': {
                "object": AutoObjectiveXGBClassifier,
                "kwargs": {
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "random_state": 42,
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "eval_metric": "mlogloss",
                    "n_jobs": 1
                }
            }
        },
        datasets=datasets,
        engines=engines,
        training_mode='cv',
        test_size=None,
        results_dir=results_path,
        cv_splits=5,
        cv_verbose=0,
        save_metrics='val',
        save_probas='val',
        random_state=42
    )


def build_ood_experiment(
        experiment_name: str,
        corpus: pl.DataFrame,
        features: list[str],
        engines: list[str],
        results_path: Path,
    ):
    return AllBenchmarksExperiment(
        name=experiment_name,
        data=corpus,
        features=features,
        target_col='outcome',
        clfs={
            'XGB': {
                "object": AutoObjectiveXGBClassifier,
                "kwargs": {
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "random_state": 42,
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "eval_metric": "mlogloss",
                    "n_jobs": 1
                }
            }
        },
        engines=engines,
        training_mode='cv',
        test_size=None,
        results_dir=results_path,
        cv_verbose=0,
        cv_groups='dataset',
        cv_strategy='leave_one_group_out',
        cv_force_standard=True,
        save_metrics='val',
        save_probas='val',
        random_state=42
    )


def save_probas(
        probas: pd.DataFrame,
        corpus: pl.DataFrame | pl.LazyFrame,
        out_path: Path
    ):
    for dataset in sorted(probas['dataset'].unique()):
        for engine in probas[probas['dataset'] == dataset]['engine'].unique():
            for features in probas[(probas['dataset'] == dataset) & (probas['engine'] == engine)]['features'].unique():
                subset = probas[
                    (probas['dataset'] == dataset) &
                    (probas['engine'] == engine) &
                    (probas['features'] == features)
                ]
                outcome = corpus.filter(
                    (corpus['dataset'] == dataset) &
                    (corpus['engine'] == engine)
                ).select(['instance_id', 'outcome'])
                if isinstance(outcome, pl.LazyFrame):
                    outcome = outcome.collect()
                outcome = outcome.to_pandas()
                outcome['dataset'] = dataset
                subset = subset[['custom_id', 'dataset', 'engine', 'features', 'instance_id', 'prob_avoidance', 'prob_correct', 'prob_incorrect']]
                merged = subset.merge(outcome, on=['dataset', 'instance_id'], how='inner')
                merged = merged.rename(columns={'dataset': 'benchmark'}).sort_values(by=['instance_id'])
                path = out_path / engine / dataset
                path.mkdir(parents=True, exist_ok=True)
                merged.to_csv(path / f"{features}.csv", index=False)


def run_experiment(distribution='IID'):
    prompt_and_features_path = build_prompt_and_features(
        prompts_path=RAW_DATA_PATH / "prompts",
        features_path=RAW_DATA_PATH / "features",
        out_path=PROCESSED_DATA_PATH,
        difficulty_type=('original' if distribution == 'IID' else 'calibrated')
    )
    corpus = load_corpus(
        prompt_and_features_path=prompt_and_features_path,
        response_data_path=RAW_DATA_PATH / "outcomes"
    )
    
    exp_name = f"error-risk-assessor-{distribution.lower()}"
    results_path = RAW_DATA_PATH / "experiments"
    exp_builder = build_iid_experiment if distribution == 'IID' else build_ood_experiment
    experiment = exp_builder(
        experiment_name=exp_name,
        corpus=corpus,
        features=FEATURES,
        datasets=BENCHMARKS,
        engines=ENGINES,
        results_path=results_path
    )
    experiment.run(n_jobs=12)
    
    probas = pd.read_csv(results_path / exp_name / "probabilities_val.csv")
    save_probas(probas, corpus, RAW_DATA_PATH / "probabilities" / distribution.lower())


def main():
    parser = ArgumentParser()
    parser.add_argument("--distribution", choices=['IID', 'OOD'], default='IID')
    args = parser.parse_args()

    run_experiment(distribution=args.distribution)


if __name__ == "__main__":
    main()
