# -*- coding: utf-8 -*-

import gc
import hashlib
import json
import logging
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl
from tqdm.auto import tqdm

from ..dataops.io import clean_corpus
from ..dataops.utils import get_view
from ..utils import make_serializable, mask_path, resolve_n_jobs
from .training import SKLearnTrainer
from .utils import LabelEncoderWithReject, prepare_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
#logging.getLogger("polars").setLevel(logging.WARNING)
#logging.getLogger("pytorch").setLevel(logging.WARNING)
#logging.getLogger("tensorflow").setLevel(logging.WARNING)
#logging.getLogger("JAX").setLevel(logging.WARNING)


class BaseExperiment(ABC):
    """Base class for orchestrating training experiments with incremental CSV saving."""
    
    DEMAND_COLUMNS = [
        'KNs', 'VO', 'KNa', 'KNf', 'KNn', 'MS',
        'SNs', 'KNc', 'AT', 'AS', 'QLl', 'CL',
        'QLq', 'CEc', 'MCr', 'MCu', 'CEe', 'MCt',
    ]

    ENGINES = [ 
        'BLOOM-560m', 'BLOOM-1b1', 'BLOOM-1b7', 'BLOOM-3b', 'BLOOM-7b1', 'BLOOM-176b',
        'BLOOMz-560m', 'BLOOMz-1b1', 'BLOOMz-1b7', 'BLOOMz-3b', 'BLOOMz-7b1', 'BLOOMz-176b',
        'LLaMA-7b', 'LLaMA-13b', 'LLaMA-30b', 'LLaMA-65b',
        'LLaMA-2-7b', 'LLaMA-2-13b', 'LLaMA-2-70b', 
        'LLaMA-2-7b-chat', 'LLaMA-2-13b-chat', 'LLaMA-2-70b-chat',
        'GPT-3 Ada', 'GPT-3 Babbage', 'GPT-3 Curie', 'GPT-3 Davinci',
        'text-davinci-001', 'text-davinci-002', 'text-davinci-003',
        'GPT-3.5-turbo', 'GPT-4 v1', 'GPT-4 v2'
    ]

    DATASETS = ['addition', 'anagram', 'locality', 'science', 'transforms']

    def __init__(
        self,
        name: str,
        data: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
        features: str | list[str],
        target_col: str,
        clfs: dict,
        datasets: str | list[str] | None = None,
        engines: str | list[str] | None = None,
        training_mode: Literal['train_eval', 'cv', 'hp_tuning'] = 'cv',
        results_dir: str | Path | None = None,
        save_metrics: Literal['val', 'test', 'val+test'] = 'val+test',
        save_curves: Literal['val', 'test', 'val+test'] | None = None,
        save_proportions: Literal['val', 'test', 'val+test'] | None = None,
        save_probas: Literal['val', 'test', 'val+test'] | None = None,
        save_detailed_metrics: str | None = None,
        save_errors: bool = True,
        cv_stratify: str | None = None,
        cv_groups: str | None = None,
        cv_force_standard: bool = False,
        test_size: float = 0.3,
        csv_format: Literal['csv', 'parquet'] = 'csv',
        **trainer_kwargs: dict
    ):
        self.name = name
        self.data = data
        self.target_col = target_col
        self.clfs = clfs
        self.training_mode = training_mode
        self.save_metrics = save_metrics
        self.save_curves = save_curves
        self.save_proportions = save_proportions
        self.save_probas = save_probas
        self.save_detailed_metrics = save_detailed_metrics
        if save_detailed_metrics in  ['3p1p1_val_scenarios', '3p1p1_test_scenarios']:
            assert (
                training_mode in ['cv', 'hp_tuning']
                and cv_groups is not None
                and trainer_kwargs.get('cv_strategy', '') == 'leave_one_group_out'
            ), "For detailed metrics '3p1p1_*_scenarios', 'cv_groups' must be defined and 'cv_strategy' must be 'leave_one_group_out'."
        elif save_detailed_metrics == 'test_groups':
            assert (
                training_mode in ['cv', 'hp_tuning']
                and cv_groups is not None
            ), "For detailed metrics 'test_groups', 'cv_groups' must be defined."
        self.save_errors = save_errors
        self.cv_stratify = cv_stratify
        self.cv_groups = cv_groups
        self.cv_force_standard = cv_force_standard
        self.test_size = test_size  # only for 'train_eval' mode
        self.csv_format = csv_format

        # Normalize to list
        self.features = [features] if isinstance(features, str) else features
        self.datasets = datasets or self.DATASETS
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]
        self.engines = engines or self.ENGINES
        if isinstance(self.engines, str):
            self.engines = [self.engines]
        
        # Setup results directory
        self.results_dir = Path(results_dir) if results_dir else Path("experiment_results")
        self.results_dir = self.results_dir / self.name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Store trainer configuration
        self.trainer_kwargs = trainer_kwargs

        # Initialize experiment files
        self._initialize_exp_files()

        # Placeholder for results
        self.results = {}

    def _initialize_exp_files(self):
        """Create experiment files with proper headers at the beginning."""

        self._save_config = {}

        # Metrics file
        self._save_config['metrics'] = {
            'enabled': True,
            'headers': [
                'dataset', 'engine', 'features', 'classifier', 'training_mode', 'num_instances',
                'train_time_sec', 'seed', 'partition', 'metric', 'value', 'margin'
            ],
            'file': self.results_dir / ("metrics_" + (self.save_metrics or '') + "." + self.csv_format)
        }
        
        # Curves file  
        self._save_config['curves'] = {
            'enabled': self.save_curves is not None,
            'headers': [
                'dataset', 'engine', 'features', 'classifier', 'training_mode', 
                'partition', 'rejection_rate', 'threshold',
                'accuracy', 'normalized_accuracy_gain',
                'success_accepted_rate', 'error_rejected_rate',
                'effective_discrimination_score', 'effective_reliability_gain',
            ],
            'file': self.results_dir / ("rejection_curves_" + (self.save_curves or '') + "." + self.csv_format)
        }

        if self.training_mode in ['cv', 'hp_tuning'] and self.save_curves != 'test':
            self._save_config['curves']['headers'].extend([
                'accuracy_margin',
                'normalized_accuracy_gain_margin', 
                'success_accepted_rate_margin',
                'error_rejected_rate_margin',
                'effective_discrimination_score_margin',
                'effective_reliability_gain_margin',
            ])

        # Proportions file
        self._save_config['proportions'] = {
            'enabled': self.save_proportions is not None,
            'headers': [
                'dataset', 'engine', 'features', 'classifier', 'training_mode',
                'partition', 'rejection_rate', 'threshold', 'label', 'proportion',
            ],
            'file': self.results_dir / ("outcome_proportions_" + (self.save_proportions or '') + "." + self.csv_format)
        }
        
        if self.training_mode in ['cv', 'hp_tuning']:
            self._save_config['proportions']['headers'].append('margin')

        # Probabilities file
        self._save_config['probas'] = {
            'enabled': self.save_probas is not None,
            'headers': [
                'dataset', 'engine', 'features', 'classifier', 'training_mode',
                'partition', 'holdout_dataset', 'custom_id', 'instance_id', 
                'prob_avoidance', 'prob_correct', 'prob_incorrect',
            ],
            'file': self.results_dir / ("probabilities_" + (self.save_probas or '') + "." + self.csv_format)
        }

        # Detailed metrics file
        self._save_config['detailed_metrics'] = {
            'enabled': (
                self.save_detailed_metrics
                and self.training_mode in ['cv', 'hp_tuning']
            ),
            'headers': [
                'dataset', 'engine', 'features', 'classifier', 'training_mode',
                'partition', 'fold', 'metric', 'value'
            ],
            'file': self.results_dir / (
                f"detailed_metrics_" + (self.save_detailed_metrics or '') + "." + self.csv_format 
            )
        }

        # Only aggregate group columns if they are enabled AND there are groups
        if self._save_config['detailed_metrics']['enabled'] and self.cv_groups:
            self._save_config['detailed_metrics']['headers'].append('holdout_group')

        # For hp_tuning, add iteration
        if (
            self._save_config['detailed_metrics']['enabled']
            and self.training_mode == 'hp_tuning'
        ):
            # Insert before fold, metric, value (last 3)
            insert_pos = len(self._save_config['detailed_metrics']['headers']) - 3
            self._save_config['detailed_metrics']['headers'].insert(
                insert_pos, 'iteration'
            )

        # Errors file (JSONL)
        self._save_config['errors'] = {
            'enabled': self.save_errors,
            'file': self.results_dir / "errors.jsonl"
        }

        # Summary file (JSON) 
        self._save_config['summary'] = {
            'enabled': True,
            'file': self.results_dir / "summary.json"
        }

        # Create CSV files with headers
        for result_type, config in self._save_config.items():
            if config['enabled'] and 'headers' in config:
                if self.csv_format == 'csv':
                    pd.DataFrame(columns=config['headers']).to_csv(config['file'], index=False)
                elif self.csv_format == 'parquet':
                    pd.DataFrame(columns=config['headers']).to_parquet(config['file'], index=False)
            elif config['enabled'] and result_type in ['errors', 'summary']:
                config['file'].touch()  # Create empty files for JSONL/JSON

    @abstractmethod
    def _get_training_views(self):
        """Return list of training views with metadata."""
        pass

    @abstractmethod
    def _resolve_view_df(self, view_info):
        """Get the DataFrame for a specific view."""
        pass

    def _resolve_test_size(self):
        """Determine test size based on training mode."""
        if (
            self.training_mode in ['cv', 'hp_tuning']
            and self.cv_groups is not None
            and self.trainer_kwargs.get('cv_strategy', '') == 'leave_one_group_out'
        ):
            return None
        return self.test_size

    def _resolve_feature_columns(self, features):
        """Resolve feature string to actual column list."""
        if isinstance(features, str):
            feature_cols = features.split("+")
            if 'demands' in feature_cols:
                feature_cols = [col for col in feature_cols if col != 'demands']
                feature_cols += self.DEMAND_COLUMNS
            return feature_cols
        return features
    
    def _prepare_view_data(self, view_info, features):
        """Prepare data for all datasets of a specific engine."""
        df = self._resolve_view_df(view_info)
        df = clean_corpus(df)
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
            
        label_encoder = LabelEncoderWithReject()
        X_train, y_train, X_test, y_test, stratify, groups = prepare_data(
            df,
            self._resolve_feature_columns(features),
            self.target_col,
            label_encoder,
            test_size=self._resolve_test_size(),
            random_state=self.trainer_kwargs.get('random_state'),
            stratify_col_name=self.cv_stratify,
            group_col_name=self.cv_groups
        )

        return label_encoder, X_train, y_train, X_test, y_test, stratify, groups

    def _build_trainer(self, clf_cfg, label_encoder):
        """Build trainer with experiment-specific configuration."""
        trainer_kwargs = {
            k: v for k, v in self.trainer_kwargs.items() 
            if k not in ('store_curves', 'store_proportions', 'store_metrics')
        }
        return SKLearnTrainer(
            model_cfg=clf_cfg,
            label_encoder=label_encoder,
            store_metrics=self.save_metrics,
            store_curves=self.save_curves,
            store_proportions=self.save_proportions,
            **trainer_kwargs
        )

    def _train_single_configuration(
            self,
            trainer,
            X_train, y_train,
            X_test, y_test,
            stratify, groups,
            context,
            param_space=None
        ):
        """Train a single classifier configuration."""
        try:
            t0 = time.time()
            
            if self.training_mode == 'train_eval':
                metrics, _  = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
                detailed_metrics_df = None
            elif self.training_mode == 'cv':
                metrics, _ = trainer.cross_validate(
                    X_train, y_train, X_test, y_test,
                    stratify, groups, seed=context['seed']
                )
                detailed_metrics_df = (
                    trainer.cv.get_detailed_metrics()
                    if self.save_detailed_metrics else None
                )
            elif self.training_mode == 'hp_tuning':
                if param_space is None:
                    raise ValueError("param_space required for hp_tuning mode")
                metrics, _ = trainer.tune_hyperparameters(
                    X_train, y_train, X_test, y_test, param_space,
                    cv_stratify=stratify, cv_groups=groups, cv_seed=context['seed']
                )
                detailed_metrics_df = (
                    trainer.tuner.get_detailed_metrics()
                    if self.save_detailed_metrics else None
                )
            else:
                raise ValueError(f"Unknown training_mode: {self.training_mode}")
            
            t1 = time.time()
            train_time_sec = t1 - t0

            metrics_df = trainer.get_metrics()
            curves_df = (
                trainer.get_rejection_curves(format='wide')
                if self.save_curves is not None else None
            )
            proportions_df = (
                trainer.get_outcome_proportions(format='long')
                if self.save_proportions is not None else None
            )

            results = {
                'metrics': metrics_df.to_dict(orient='records'),
                'detailed_metrics': detailed_metrics_df.to_dict(orient='records') if detailed_metrics_df else None,
                'curves': curves_df.to_dict(orient='records') if curves_df else None,
                'proportions': proportions_df.to_dict(orient='records') if proportions_df else None,
                'train_time_sec': train_time_sec
            }

            del trainer
            gc.collect()

            return results, None
            
        except Exception as e:
            logging.error(f"Training failed for {context}: {str(e)}")
            tb = traceback.format_exc()
            return {}, {'message': str(e), 'traceback': tb}

    def _save_results_incrementally(self, trainer_results, context, error=None):
        """Save results incrementally to CSV files."""
        base_context = {
            'dataset': context.get('dataset', None),
            'engine': context.get('engine', None), 
            'features': context.get('features', None),
            'classifier': context.get('classifier', None),
            'training_mode': self.training_mode,
            'train_time_sec': context.get('train_time_sec', None),
            'num_instances': context.get('num_instances', None),
            'seed': context.get('seed', None)
        }

        if error is not None:
            self._save_error_row(base_context, error, result_type='train')
            return

        for result_type, config in self._save_config.items():
            if config['enabled'] and 'headers' in config:
                self._save_results_to_csv(
                    trainer_results, base_context,
                    result_type=result_type,
                    headers=config['headers'],
                    file_path=config['file']
                )

    @staticmethod
    def _save_to_parquet_no_headers_mode_append(df, file_path):
        """Save DataFrame to Parquet file without headers, appending if exists."""
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)
        if file_path.exists():
            loaded_df = pl.read_parquet(file_path)
            df = pl.concat([loaded_df, df], how="vertical", rechunk=True)
        df.write_parquet(file_path)

    def _save_results_to_csv(self, trainer_results, base_context, result_type, headers, file_path):
        """Generic function to save results to CSV files."""
        try:
            data = trainer_results.get(result_type)
            if not data:
                return
                
            df = pd.DataFrame(data)
            #print(df.columns)

            # Add context columns
            for key, value in base_context.items():
                df[key] = value

            # Reorder columns to match header
            df = df[headers]

            # Append to CSV
            if self.csv_format == 'csv':
                df.to_csv(file_path, mode='a', header=False, index=False)
            elif self.csv_format == 'parquet':
                self._save_to_parquet_no_headers_mode_append(df, file_path)

        except Exception as e:
            logging.error(f"Failed to save {result_type} at {base_context}: {str(e)}")
            tb = traceback.format_exc()
            self._save_error_row(base_context, {'message': str(e), 'traceback': tb}, result_type)

    def _save_error_row(self, base_context, error_obj, result_type='train'):
        """Append an error record to a central JSONL errors file.

        error_obj may be a string or a dict {'message':..., 'traceback':...}.
        Traceback is truncated to avoid huge lines.
        """
        if not self._save_config['errors']['enabled']:
            return
        
        try:
            if isinstance(error_obj, dict):
                message = error_obj.get('message')
                tb = error_obj.get('traceback')
            else:
                message = str(error_obj)
                tb = None

            # Truncate traceback if very long
            if tb:
                max_tb = 50000
                tb = tb if len(tb) <= max_tb else tb[:max_tb] + "\n... (truncated)"

            record = {
                **base_context,
                'result_type': result_type,
                'error_message': message,
                'traceback': tb,
                'timestamp': datetime.now().isoformat()
            }
            errors_file = self._save_config['errors']['file']
            with open(errors_file, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as write_err:
            logging.error(f"Failed to write to errors file {errors_file}: {write_err}")

    def _save_summary(self):
        """Save a summary JSON file with experiment parameters and results."""
        summary = {
            'name': self.name,
            'training_mode': self.training_mode,
            'results_dir': str(self.results_dir),
            'save_curves': self.save_curves,
            'save_proportions': self.save_proportions,
            'save_detailed_metrics': self.save_detailed_metrics,
            'save_errors': self.save_errors,
            'clfs': make_serializable(deepcopy(self.clfs)),
            'features': self.features,
            'target_col': self.target_col,
            'engines': self.engines,
            'datasets': self.datasets,
            'results': self.results,
            'date': datetime.now().isoformat()
        }
        
        try:
            summary_file = self._save_config['summary']['file']
            with open(summary_file, 'w', encoding='utf-8') as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=4)
        
        except Exception as e:
            logging.error(f"Failed to write summary file {summary_file}: {str(e)}")


    @staticmethod
    def _derive_view_seed(dataset: str, engine: str, features: str | None, base_seed: int = 42) -> int:
        """
        Deriva una semilla estable a partir de (dataset, engine[, features]) y un base_seed.
        Devuelve un entero en el rango [0, 2^31-1] para ser compatible con sklearn/lightgbm.
        """
        key = f"{base_seed}|{dataset}|{engine}|{features or ''}"
        h = hashlib.md5(key.encode('utf-8')).hexdigest()
        # toma 8 bytes y conviértelos a entero; limítalo a 31 bits
        val = int(h[:8], 16) & 0x7FFFFFFF
        return val

    def _process_view(self, view_info):
        results = []
        
        for features in self.features:
            label_encoder, X_train, y_train, X_test, y_test, stratify, groups = \
                self._prepare_view_data(view_info, features)

            base_seed = self.trainer_kwargs.get('random_state', 42)
            seed_view = self._derive_view_seed(
                view_info['dataset'], view_info['engine'], str(features), base_seed
            )

            for clf_name, clf_cfg in self.clfs.items():
                clf_cfg_seeded = {**clf_cfg, 'random_state': seed_view}

                context = {
                    **view_info,
                    'features': features,
                    'classifier': clf_name,
                    'seed': seed_view
                }

                trainer = self._build_trainer(clf_cfg_seeded, label_encoder)

                trainer_result, error = self._train_single_configuration(
                    trainer, X_train, y_train, X_test, y_test, stratify, groups, context,
                    param_space=clf_cfg.get('param_space') if self.training_mode == 'hp_tuning' else None
                )

                context['train_time_sec'] = round(trainer_result.pop('train_time_sec', 0.0), 6)

                results.append((trainer_result, context, error))

            del X_train, y_train, X_test, y_test, stratify, groups, label_encoder
            gc.collect()

        return results
    
    def _run_serial(self, verbose=True):
        """Sequential execution implementation."""        
        training_views = self._get_training_views()
        total_combinations = len(training_views) * len(self.features) * len(self.clfs)
        
        logging.info(f"Total training combinations: {total_combinations}")
        
        completed = 0
        failed = 0

        for view_info in tqdm(training_views, total=len(training_views), desc="Processing views"):
            if verbose:
                logging.info(f"Processing view: {view_info}")
                
            try:
                view_results = self._process_view(view_info)
                
                # Save results incrementally
                for trainer_result, context, error in view_results:
                    self._save_results_incrementally(trainer_result, context, error)
                    
                    if error is None:
                        completed += 1
                    elif error is not None:
                        failed += 1
            
            except Exception as e:
                logging.error(f"Failed to process view {view_info}: {str(e)}")
                continue

        logging.info(f"Experiment completed! Completed: {completed/total_combinations:.2%}, Failed: {failed/total_combinations:.2%}. Results saved to {self.results_dir}")
        return {
            'completed': completed,
            'failed': failed,
            'total': total_combinations,
            'results_dir': str(self.results_dir)
        }
    
    def _run_parallel(self, n_jobs=4, verbose=True):
        """Parallel execution implementation."""
        training_views = self._get_training_views()
        total_combinations = len(training_views) * len(self.features) * len(self.clfs)

        logging.info(f"Total training combinations: {total_combinations}")

        completed = 0
        failed = 0

        # Build constant payload parts once to minimize pickling size
        base_payload = {
            "data": self.data,
            "target_col": self.target_col,
            "features_list": self.features,
            "clfs": self.clfs,
            "training_mode": self.training_mode,
            "trainer_kwargs": self.trainer_kwargs,
            "demand_columns": self.DEMAND_COLUMNS,
            "cv_stratify": getattr(self, 'cv_stratify', None),
            "cv_groups": getattr(self, 'cv_groups', None),
            "cv_force_standard": self.cv_force_standard,
            "test_size": self._resolve_test_size(),   # None en cv/hp_tuning, float en train_eval
            "save_metrics": self.save_metrics,
            "save_curves": self.save_curves,
            "save_proportions": self.save_proportions,
            "save_detailed_metrics": self.save_detailed_metrics,
            "save_probas": self.save_probas,
        }

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {}
            for view_info in training_views:
                payload = dict(base_payload)
                payload["view_info"] = view_info
                futures[executor.submit(process_view_entry, payload)] = view_info

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing views"):
                view_info = futures[future]
                try:
                    results = future.result()
                    for trainer_result, context, error in results:
                        self._save_results_incrementally(trainer_result, context, error)
                        if error is None:
                            completed += 1
                        else:
                            failed += 1
                except Exception as e:
                    raise e
                    logging.error(f"Failed to process view {view_info}: {str(e)}")
                    failed += 1
                    continue

        logging.info(f"Experiment completed! Completed: {completed/total_combinations:.2%}, Failed: {failed/total_combinations:.2%}. Results saved to {mask_path(self.results_dir)}")
        return {
            'completed': completed,
            'failed': failed,
            'total': total_combinations,
            'results_dir': str(self.results_dir)
        }
    
    def run(self, n_jobs=4, verbose=True):
        """
        Run the experiment, automatically choosing serial or parallel execution.
        
        Args:
            n_jobs (int): Number of parallel workers to use.
                 1 = serial execution
                -1 = use all available cores minus one
                >1 = use that many workers
            verbose (bool): Enable verbose logging
        """
        n_jobs = resolve_n_jobs(n_jobs)
        logging.info(f"Starting {self.__class__.__name__} experiment ...")
        
        t0 = time.time()
        if n_jobs == 1:
            self.results = self._run_serial(verbose)
        else:
            self.results = self._run_parallel(n_jobs, verbose)

        t1 = time.time()
        self.results['total_time_sec'] = round(t1 - t0, 6)
        self._save_summary()

        return self.results
    
    def _get_results_from_csv(self, result_type, custom_warning=None):
        """Generic function to load results from result files (CSV or Parquet)."""
        config = self._save_config.get(result_type)
        
        if not config:
            raise ValueError(f"Unknown result type: {result_type}")
        
        if not config['enabled']:
            warning_msg = custom_warning or f"{result_type.title()} file saving is disabled."
            logging.warning(warning_msg)
            return pd.DataFrame(columns=config.get('headers', []))
        
        file_path = config['file']
        
        if file_path.exists():
            if self.csv_format == 'csv':
                return pd.read_csv(file_path)
            elif self.csv_format == 'parquet':
                return pd.read_parquet(file_path)

        else:
            warning_msg = custom_warning or f"{result_type.title()} file does not exist. Make sure to run the experiment first."
            logging.warning(warning_msg)
            return pd.DataFrame(columns=config.get('headers', []))

    def get_metrics(self):
        """Load metrics from the CSV file."""
        return self._get_results_from_csv(
            'metrics',
            "Metrics file does not exist. Make sure to run the experiment first."
        )

    def get_rejection_curves(self):
        """Load rejection curves from the CSV file."""
        msg = (
            "Curves file does not exist or saving is disabled. "
            "Make sure to run the experiment first or set save_curves=True."
        )
        return self._get_results_from_csv('curves', msg)

    def get_outcome_proportions(self):
        """Load outcome proportions from the CSV file."""
        msg = (
            "Proportions file does not exist or saving is disabled. "
            "Make sure to run the experiment first or set save_proportions=True."
        )
        return self._get_results_from_csv('proportions', msg)

    def get_detailed_metrics(self):
        """Load detailed metrics from the CSV file."""
        msg = (
            "Detailed metrics file does not exist or saving is disabled. "
            "Make sure to run the experiment first or set save_detailed_metrics=True."
        )
        return self._get_results_from_csv('detailed_metrics', msg)

    def summarize_classifier_rankings(self, metrics: list[str]) -> pd.DataFrame:
        """Summarize classifier rankings based on multiple metrics.

        Returns
        -------
        pd.DataFrame
            Rows = classifiers
            Columns = MultiIndex (metric, ['mean rank', 'max rank', 'min rank', 'mean±margin'])
        """
        df = self.get_metrics()
        df = df.loc[df['partition'] == 'val', :]
        required_cols = {'dataset', 'engine', 'features', 'classifier', 'metric', 'value', 'margin'}
        if not required_cols.issubset(df.columns):
            raise ValueError("Metrics file is missing required columns.")

        rows = []

        for metric in metrics:
            metric_df = df[df['metric'] == metric].copy()
            metric_df['view_feat'] = list(zip(metric_df['dataset'], metric_df['engine'], metric_df['features']))

            ranked = (
                metric_df
                .groupby('view_feat', group_keys=False)
                .apply(lambda g: g.assign(rank=g['value'].rank(ascending=False, method='min')))
            )

            agg = ranked.groupby('classifier').agg(
                mean_rank=('rank', 'mean'),
                max_rank=('rank', 'max'),
                min_rank=('rank', 'min'),
                mean_value=('value', 'mean'),
                mean_margin=('margin', 'mean')
            )

            # Formato de texto: mean ± margin
            agg['mean_metric'] = agg.apply(
                lambda row: f"{row['mean_value']:.3f} ± {row['mean_margin']:.3f}", axis=1
            )

            # Selección final de columnas y MultiIndex
            final = agg[['mean_rank', 'max_rank', 'min_rank', 'mean_metric']]
            final.columns = pd.MultiIndex.from_product([[metric], final.columns])
            rows.append(final)

        summary = pd.concat(rows, axis=1).sort_index()
        return summary
    
    def summarize_best_models(self, metrics: list[str]) -> pd.DataFrame:
        """Summarize best classifier and metric value±margin per (dataset, engine, features).

        Columns: MultiIndex (features, metric, ['best_clf', 'value±margin'])
        Rows: MultiIndex (dataset, engine)
        """
        df = self.get_metrics()
        df = df.loc[df['partition'] == 'val', :]
        required_cols = {'dataset', 'engine', 'features', 'classifier', 'metric', 'value', 'margin'}
        if not required_cols.issubset(df.columns):
            raise ValueError("Metrics file is missing required columns.")

        records = []

        for metric in metrics:
            df_metric = df[df["metric"] == metric].copy()
            df_metric = df_metric.sort_values("value", ascending=False)

            grouped = df_metric.groupby(["dataset", "engine", "features"], as_index=False).first()
            grouped["metric"] = metric
            grouped["value"] = grouped.apply(
                lambda row: f"{row['value']:.3f} ± {row['margin']:.3f}", axis=1
            )

            records.append(grouped)

        combined = pd.concat(records, axis=0)

        # Pivot: classifier and value_margin
        pivoted = combined.pivot(
            index=["dataset", "engine", "features"],
            columns=["metric"],
            values=["classifier", "value"]
        )

        # Reorganizar el MultiIndex: (features, metric, tipo)
        pivoted.columns = pd.MultiIndex.from_tuples([
            (metric, "best_clf") if col == "classifier" else (metric, "value")
            for col, metric in pivoted.columns
        ])

        return pivoted.sort_index(axis=1)


class PairedTrainingExperiment(BaseExperiment):
    """Train classifiers on individual (dataset, engine) pairs."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _get_training_views(self):
        """Return all (dataset, engine) combinations."""
        views = []
        for engine in self.engines:
            for dataset in self.datasets:
                views.append({
                    'engine': engine,
                    'dataset': dataset
                })
        return views
    
    def _resolve_view_df(self, view_info):
        """Get the DataFrame for a specific dataset and engine."""
        df = get_view(self.data, dataset=view_info['dataset'], engine=view_info['engine'])
        return df


class AllBenchmarksExperiment(BaseExperiment):
    """Train classifiers on all datasets combined for each engine."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _get_training_views(self):
        """Return all engines (with all datasets combined)."""
        views = []
        for engine in self.engines:
            views.append({
                'engine': engine,
                'dataset': 'all'
            })
        return views
    
    def _resolve_view_df(self, view_info):
        """Get the DataFrame for a specific engine (all datasets combined)."""
        df = get_view(self.data, engine=view_info['engine'])
        return df


#####################################################################
#  Static functions for multiprocessing compatibility
#####################################################################

def _resolve_feature_columns_static(features, demand_columns):
    """Resolve feature string to actual column list (module-level for multiprocessing)."""
    if isinstance(features, str):
        feature_cols = features.split("+")
        if 'demands' in feature_cols:
            feature_cols = [col for col in feature_cols if col != 'demands']
            feature_cols += demand_columns
        return feature_cols
    return features

def _derive_view_seed_static(dataset: str, engine: str, features: str | None, base_seed: int = 42) -> int:
    """Seed derivation usable from worker processes (module-level)."""
    key = f"{base_seed}|{dataset}|{engine}|{features or ''}"
    h = hashlib.md5(key.encode('utf-8')).hexdigest()
    val = int(h[:8], 16) & 0x7FFFFFFF
    return val

def process_view_entry(payload):
    """
    Standalone worker entry to reduce pickling of bound methods/self.
    Expects `payload` dict with:
      - data, target_col, features_list, clfs, training_mode, trainer_kwargs
      - view_info (dict with dataset, engine)
      - demand_columns (list)
      - cv_stratify, cv_groups, test_size, save_curves, save_proportions
    Returns: list of tuples (trainer_result, context, error, t_train) like _process_view.
    """
    # Unpack inputs
    data = payload["data"]
    target_col = payload["target_col"]
    features_list = payload["features_list"]
    clfs = payload["clfs"]
    training_mode = payload["training_mode"]
    trainer_kwargs = payload.get("trainer_kwargs", {})
    view_info = payload["view_info"]
    demand_columns = payload["demand_columns"]
    cv_stratify = payload.get("cv_stratify")
    cv_groups = payload.get("cv_groups")
    cv_force_standard = payload['cv_force_standard']
    test_size = payload["test_size"]
    save_metrics = payload['save_metrics']
    save_curves = payload["save_curves"]
    save_proportions = payload["save_proportions"]
    save_detailed_metrics = payload["save_detailed_metrics"]
    save_probas = payload['save_probas']

    results = []

    for features in features_list:
        # Prepare data for this (view, features)
        
        engine = view_info.get('engine')
        dataset = view_info.get('dataset')
        if dataset == 'all':
            df = get_view(data, engine=engine)
        else:
            if dataset is None or engine is None:
                raise ValueError(f"view_info must include dataset and engine; got {view_info}")
            df = get_view(data, dataset=dataset, engine=engine)
        
        df = clean_corpus(df)
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        feature_cols = _resolve_feature_columns_static(features, demand_columns)
        #df = df.drop_nulls(subset=feature_cols)
        #print('\n', engine, target_col, features, feature_cols, cv_stratify, cv_groups)

        label_encoder = LabelEncoderWithReject()
        (
            X_train, y_train, df_idx_train, stratify_train, groups_train,
            X_test, y_test, df_idx_test, stratify_test, groups_test
        ) = prepare_data(
            df,
            feature_cols,
            target_col,
            label_encoder,
            test_size=test_size,
            random_state=trainer_kwargs.get('random_state'),
            stratify_col_name=cv_stratify,
            group_col_name=cv_groups
        )

        # Seed per view (+features)
        base_seed = trainer_kwargs.get('random_state', 42)
        seed_view = _derive_view_seed_static(
            view_info.get('dataset', ''), view_info.get('engine', ''),
            str(features), base_seed
        )

        for clf_name, clf_cfg in clfs.items():
            clf_cfg_seeded = {**clf_cfg, 'random_state': seed_view}

            # Build trainer (mirror of BaseExperiment._build_trainer)
            trainer = SKLearnTrainer(
                model_cfg=clf_cfg_seeded,
                label_encoder=label_encoder,
                store_metrics=save_metrics,
                store_curves=save_curves,
                store_proportions=save_proportions,
                store_probas=save_probas,
                **{k: v for k, v in trainer_kwargs.items() 
                   if k not in ('store_metrics', 'store_curves', 'store_proportions', 'store_detailed_metrics')}
            )

            context = {
                **view_info,
                'features': features,
                'classifier': clf_name,
                'seed': seed_view,
                'num_instances': len(X_train) + len(X_test)
            }

            try:
                t0 = time.time()
                
                if training_mode == 'train_eval':
                    _ = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
                    detailed_metrics_df = None
                elif training_mode == 'cv':
                    _ = trainer.cross_validate(
                        X_train, y_train, X_test, y_test,
                        stratify_train=stratify_train, groups_train=groups_train,
                        groups_test=groups_test, seed=seed_view,
                        force_standard_cv=cv_force_standard
                    )
                    if save_detailed_metrics is not None:
                        detailed_metrics = []
                        for dm in save_detailed_metrics.split(","):
                            dm_df = trainer.cv.get_detailed_metrics(scope=dm)
                            detailed_metrics.append(dm_df)
                        detailed_metrics_df = pd.concat(detailed_metrics, axis=0)
                    else:
                        detailed_metrics_df = None
                elif training_mode == 'hp_tuning':
                    param_space = clf_cfg.get('param_space')
                    if param_space is None:
                        raise ValueError("param_space required for hp_tuning mode")
                    _ = trainer.tune_hyperparameters(
                        X_train, y_train, X_test, y_test, param_space,
                        cv_stratify=stratify_train, cv_groups=groups_train, cv_seed=seed_view
                    )
                    detailed_metrics_df = (
                        trainer.tuner.get_detailed_metrics(scope=save_detailed_metrics)
                        if save_detailed_metrics is not None else None
                    )
                else:
                    raise ValueError(f"Unknown training_mode: {training_mode}")

                time_train_sec = time.time() - t0
                context['train_time_sec'] = round(time_train_sec, 6)
                
                metrics_df = trainer.get_metrics()
                curves_df = (
                    trainer.get_rejection_curves(format='wide')
                    if save_curves is not None else None
                )
                proportions_df = (
                    trainer.get_outcome_proportions(format='long')
                    if save_proportions is not None else None
                )

                probas_df = (
                    trainer.get_probas(format='wide')
                    if save_probas is not None else None
                )
                if probas_df is not None:
                    df = df.to_pandas() if hasattr(df, "to_pandas") else df

                    def _metadata_from_indices(source_indices, partition):
                        if source_indices is None or len(source_indices) == 0:
                            return pd.DataFrame(
                                columns=[
                                    'index', 'partition', 'holdout_dataset',
                                    'custom_id', 'instance_id'
                                ]
                            )
                        metadata_cols = ['custom_id', 'instance_id']
                        holdout_col = None
                        if cv_groups is not None and cv_groups in df.columns:
                            holdout_col = cv_groups
                        elif 'dataset' in df.columns:
                            holdout_col = 'dataset'
                        if holdout_col is not None and holdout_col not in metadata_cols:
                            metadata_cols.append(holdout_col)
                        meta = (
                            df.iloc[source_indices][metadata_cols]
                            .reset_index(drop=True)
                        )
                        # `probas_df["index"]` is compact within X_train/X_test,
                        # not the original row position before feature-specific drops.
                        meta.insert(0, 'index', range(len(meta)))
                        meta['partition'] = partition
                        meta['holdout_dataset'] = (
                            meta[holdout_col] if holdout_col is not None else None
                        )
                        return meta[
                            [
                                'index', 'partition', 'holdout_dataset',
                                'custom_id', 'instance_id'
                            ]
                        ]

                    original = _metadata_from_indices(
                        df_idx_train,
                        'val' if training_mode != 'train_eval' else 'train'
                    )
                    if df_idx_test is not None and len(df_idx_test) > 0:
                        original_test = _metadata_from_indices(df_idx_test, 'test')
                        original = pd.concat([original, original_test], axis=0, ignore_index=True)
                    probas_df = probas_df.merge(
                        original,
                        on=['index', 'partition'],
                        how='left',
                        validate='many_to_one'
                    )
                    missing_metadata = probas_df['custom_id'].isna()
                    if missing_metadata.any():
                        missing_count = int(missing_metadata.sum())
                        total_count = len(probas_df)
                        raise ValueError(
                            "Could not align metadata for "
                            f"{missing_count}/{total_count} probability rows. "
                            "This indicates a mismatch between compact probability "
                            "indices and source row indices."
                        )
                    probas_df = probas_df.drop(columns=['index'])
                
                trainer_results = {
                    'metrics': metrics_df.to_dict(orient='records'),
                    'detailed_metrics': (
                        detailed_metrics_df.to_dict(orient='records')
                        if detailed_metrics_df is not None else None
                    ),
                    'curves': (
                        curves_df.to_dict(orient='records')
                        if curves_df is not None else None
                    ),
                    'proportions': (
                        proportions_df.to_dict(orient='records')
                        if proportions_df is not None else None
                    ),
                    'probas': (
                        probas_df.to_dict(orient='records')
                        if probas_df is not None else None
                    )
                }

                del trainer
                gc.collect()

                results.append((trainer_results, context, None))
            
            except Exception as e:
                raise e
                tb = traceback.format_exc()
                results.append(({}, context, {'message': str(e), 'traceback': tb}))

        del X_train, y_train, X_test, y_test, stratify_train, groups_train, stratify_test, groups_test, label_encoder
        gc.collect()

    return results
