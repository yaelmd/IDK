import os
from typing import Union
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import polars as pl
from datasets import load_dataset

from dataops.lexical import (
    LEXICAL_FEATURE_COLUMNS,
    write_lexical_feature_files,
)
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
    'geometry',
    'arithmetic',
    'algebra',
    'combinatory',
    'number_theory',
]

SAMPLED_PROMPT_ROOT = RAW_DATA_PATH / "sampled_prompts"
SUPPORTED_SAMPLE_SIZES = (350, 750)
SAMPLED_PROMPT_PATHS = {
    size: SAMPLED_PROMPT_ROOT / str(size)
    for size in SUPPORTED_SAMPLE_SIZES
}

FEATURES = [
    'difficulty',
    'embedding',
    'delean',
    'difficulty+embedding',
    'embedding+delean',
    'difficulty+delean',
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
    'geometry': None,
    'arithmetic': None,
    'algebra': None,
    'combinatory': None,
    'number_theory': None
}


# Help prevent thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"         # MKL
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Accelerate (macOS)
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def load_prompts(prompt_data_path, expected_rows_per_benchmark: int | None = None):
    dfs = []
    for bench in BENCHMARKS:
        prompt_file = prompt_data_path / f"{bench}.csv"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Missing prompt file for {bench}: {prompt_file}")
        df = pd.read_csv(prompt_file)
        if (
            expected_rows_per_benchmark is not None
            and len(df) != expected_rows_per_benchmark
        ):
            raise ValueError(
                f"Expected {expected_rows_per_benchmark} rows for {bench}, "
                f"but found {len(df)} in {prompt_file}"
            )
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
        if bench in ['logic_puzzles', 'geometry', 'arithmetic', 'algebra', 'combinatory', 'number_theory']:
            difficulty_df = pd.read_csv(features_path / f"difficulty/{bench}_calibrated_difficulty.csv")
            difficulty_df.rename(columns={'cal_diff': 'difficulty'}, inplace=True)
        else:
            difficulty_df = load_dataset(
                "lexin-zhou/ReliabilityBench", bench,
                usecols=['instance_id', 'template_id', 'difficulty'],
            )['test'].to_pandas()
            difficulty_df = difficulty_df[difficulty_df['template_id'] == template_id]
        difficulty_df['benchmark'] = bench
        difficulty_df = difficulty_df[['benchmark', 'instance_id', 'difficulty']]
        dfs.append(difficulty_df)
    return pd.concat(dfs, ignore_index=True)


def load_embeddings(features_path):
    dfs = []
    for bench in BENCHMARKS:
        embedding_df = pd.read_csv(features_path / f"embeddings/{bench}_embedded_text.csv")
        if bench == 'science':
            embedding_df['custom_id'] = (
                embedding_df['ori_id']
                    .str.replace("_", "-", regex=False)
                    .str.replace("obqa", "openbookqa", regex=False)
            )
        else:
            embedding_df['custom_id'] = bench + "-" + embedding_df["instance_id"].astype(str)
        embedding_df['embedding'] = embedding_df['embedding'].apply(lambda x: eval(x))
        embedding_df = embedding_df[["custom_id", "embedding"]]
        dfs.append(embedding_df)
    return pd.concat(dfs, ignore_index=True)


def load_lexical(features_path):
    dfs = []
    for bench in BENCHMARKS:
        lexical_path = features_path / f"lexical/{bench}_lexical.csv"
        if not lexical_path.exists():
            raise FileNotFoundError(
                f"Missing lexical feature file for {bench}: {lexical_path}. "
                "Run write_lexical_feature_files() first."
            )
        lexical_df = pd.read_csv(lexical_path)
        missing = [
            col for col in ["custom_id", *LEXICAL_FEATURE_COLUMNS]
            if col not in lexical_df.columns
        ]
        if missing:
            raise ValueError(
                f"Missing lexical columns in {lexical_path}: {missing}"
            )
        dfs.append(lexical_df[["custom_id", *LEXICAL_FEATURE_COLUMNS]])
    return pd.concat(dfs, ignore_index=True)
        

def build_prompt_and_features(
        prompts_path,
        features_path,
        out_path,
        difficulty_type='original',
        output_suffix: str | None = None,
        expected_rows_per_benchmark: int | None = None
    ):
    prompts = load_prompts(
        prompts_path,
        expected_rows_per_benchmark=expected_rows_per_benchmark
    )
    if difficulty_type == 'original':
        difficulty = load_difficulty(features_path)
    elif difficulty_type == 'calibrated':
        difficulty = load_cal_diff(features_path)
    else:
        raise ValueError(f"Invalid difficulty_type: {difficulty_type}")
    embeddings = load_embeddings(features_path)
    lexical = load_lexical(features_path)
    delean = pd.read_csv(features_path / "delean/idk_annotations.csv")
    delean_benchmark_mask = delean['benchmark'].isin(BENCHMARKS)
    delean.loc[delean_benchmark_mask, 'custom_id'] = (
        delean.loc[delean_benchmark_mask, 'benchmark'] + "-" +
        delean.loc[delean_benchmark_mask, 'instance_id'].astype(str)
    )

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
                lexical,
                on=['custom_id'],
                how='left'
            )
            .merge(
                delean[list(delean.columns.difference(['benchmark', 'instance_id']))],
                on=['custom_id'],
                how='left'
            )
    )
    suffix = f"-{output_suffix}" if output_suffix else ""
    out_file = out_path / f"prompt_and_features-{difficulty_type}-diff{suffix}.parquet"
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


def load_corpus(prompt_and_features_path, response_data_path, model: str):
    prompt_and_features = pl.scan_parquet(prompt_and_features_path)
    response_data = load_response_data(
        full_res_path=response_data_path,
        engine=[model],
        benchmark=BENCHMARKS,
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
                    "importance_type": "gain",
                    "n_jobs": 1,
                    "verbosity": 0
                }
            }
        },
        datasets=datasets,
        engines=engines,
        training_mode='cv',
        test_size=None,
        results_dir=results_path,
        cv_splits=5,
        cv_verbose=2,
        cv_show_progress=False,
        save_metrics='val',
        save_probas='val',
        save_feature_importance=True,
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
                    "importance_type": "gain",
                    "n_jobs": 1,
                    "verbosity": 0
                }
            }
        },
        engines=engines,
        training_mode='cv',
        test_size=None,
        results_dir=results_path,
        cv_verbose=2,
        cv_show_progress=False,
        cv_groups='dataset',
        cv_strategy='leave_one_group_out',
        cv_force_standard=True,
        save_metrics='val',
        save_probas='val',
        save_feature_importance=True,
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
                ].copy()
                benchmark_col = (
                    'holdout_dataset'
                    if 'holdout_dataset' in subset.columns
                    else 'dataset'
                )
                subset['benchmark'] = subset[benchmark_col]
                outcome = corpus.filter(
                    pl.col('engine') == engine
                ).select(['custom_id', 'outcome'])
                if isinstance(outcome, pl.LazyFrame):
                    outcome = outcome.collect()
                outcome = outcome.to_pandas()
                subset = subset[['custom_id', 'benchmark', 'engine', 'features', 'instance_id', 'prob_avoidance', 'prob_correct', 'prob_incorrect']]
                merged = subset.merge(outcome, on=['custom_id'], how='left')
                merged = merged.sort_values(by=['benchmark', 'instance_id'])
                path = out_path / engine / dataset
                path.mkdir(parents=True, exist_ok=True)
                merged.to_csv(path / f"{features}.csv", index=False)


def save_feature_importance(rankings: pd.DataFrame, out_path: Path):
    """Save one ranked feature-importance CSV per trained assessor."""
    rankings = rankings.copy()
    rankings['assessor_dataset'] = rankings['holdout_dataset'].fillna(
        rankings['dataset']
    )
    assessor_cols = [
        'assessor_dataset', 'engine', 'features', 'classifier'
    ]
    for assessor, subset in rankings.groupby(assessor_cols, dropna=False):
        dataset, engine, features, classifier = assessor
        path = out_path / str(engine) / str(dataset)
        path.mkdir(parents=True, exist_ok=True)
        subset = subset.sort_values('rank')[
            [
                'rank', 'feature', 'importance', 'fold',
                'holdout_dataset', 'seed', 'num_instances'
            ]
        ]
        subset.to_csv(
            path / f"{features}__{classifier}.csv",
            index=False
        )


def resolve_prompts_path(sample_size: int | None = None) -> Path:
    if sample_size is None:
        return RAW_DATA_PATH / "prompts"
    if sample_size not in SAMPLED_PROMPT_PATHS:
        valid_sizes = ", ".join(str(size) for size in sorted(SAMPLED_PROMPT_PATHS))
        raise ValueError(
            f"Unsupported sample size: {sample_size}. "
            f"Available sampled prompt sizes: {valid_sizes}"
        )
    return SAMPLED_PROMPT_PATHS[sample_size]


def run_experiment(model: str, distribution='IID', sample_size: int | None = None,
                   features: list[str] | None = None):
    features = FEATURES if features is None else features
    available_models = sorted(
        path.name for path in (RAW_DATA_PATH / "outcomes").iterdir()
        if path.is_dir()
    )
    if model not in available_models:
        raise ValueError(
            f"Unknown model {model!r}. Available models: {', '.join(available_models)}"
        )
    prompts_path = resolve_prompts_path(sample_size)
    run_suffix = f"sampled-{sample_size}" if sample_size is not None else None
    output_dir_name = (
        f"{distribution.lower()}-{run_suffix}"
        if run_suffix
        else distribution.lower()
    )

    output_dir_name = f"{output_dir_name}-{model}"
    exp_name = f"error-risk-assessor-{output_dir_name}"
    results_path = RAW_DATA_PATH / "experiments"
    # Reserve the destination before training so concurrent or repeated runs
    # cannot truncate or mix results from an existing experiment.
    try:
        (results_path / exp_name).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"Results already exist at {results_path / exp_name}. "
            "Move that directory before rerunning."
        ) from None

    print(f"Running experiment for model: {model}, distribution: {distribution}")
    if sample_size is not None:
        print(f"Using sampled prompts: {sample_size} examples per benchmark")
    print(f"Loading prompts from {prompts_path} and features from {RAW_DATA_PATH / 'features'}...")
    write_lexical_feature_files(
        prompt_data_path=RAW_DATA_PATH / "prompts",
        features_path=RAW_DATA_PATH / "features",
        benchmarks=BENCHMARKS,
    )
    prompt_and_features_path = build_prompt_and_features(
        prompts_path=prompts_path,
        features_path=RAW_DATA_PATH / "features",
        out_path=PROCESSED_DATA_PATH,
        difficulty_type=('original' if distribution == 'IID' else 'calibrated'),
        output_suffix=run_suffix,
        expected_rows_per_benchmark=sample_size
    )
    print(f"Loading corpus from {prompt_and_features_path} and response data from {RAW_DATA_PATH / 'outcomes'}...")
    corpus = load_corpus(
        prompt_and_features_path=prompt_and_features_path,
        response_data_path=RAW_DATA_PATH / "outcomes",
        model=model
    )
    print("Corpus loaded. Running experiment...")
    if distribution == 'IID':
        experiment = build_iid_experiment(
            experiment_name=exp_name,
            corpus=corpus,
            features=features,
            datasets=BENCHMARKS,
            engines=[model],
            results_path=results_path
        )
    else:
        experiment = build_ood_experiment(
            experiment_name=exp_name,
            corpus=corpus,
            features=features,
            engines=[model],
            results_path=results_path
        )
    experiment.run(n_jobs=12)
    print("Experiment completed. Saving probabilities and feature importance...")
    probas = pd.read_csv(results_path / exp_name / "probabilities_val.csv")
    save_probas(probas, corpus, RAW_DATA_PATH / "probabilities" / output_dir_name)
    rankings = pd.read_csv(results_path / exp_name / "feature_importance.csv")
    save_feature_importance(
        rankings,
        RAW_DATA_PATH / "feature_importance" / output_dir_name
    )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Model directory name under data/raw/outcomes (e.g. GPT5.2 or gemini-3.1-pro-preview)."
    )
    parser.add_argument("--distribution", choices=['IID', 'OOD'], default='IID')
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURES,
        default=None,
        help="Feature combinations to train. Defaults to all configured combinations."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        choices=sorted(SAMPLED_PROMPT_PATHS),
        default=None,
        help="Use sampled prompt files with this many examples per benchmark."
    )
    args = parser.parse_args()

    run_experiment(model=args.model, distribution=args.distribution,
                   sample_size=args.sample_size, features=args.features)


if __name__ == "__main__":
    main()
