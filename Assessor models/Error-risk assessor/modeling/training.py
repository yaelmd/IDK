# -*- coding: utf-8 -*-

import logging
from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cross_validator import RejectorCrossValidator
from .evaluation import RejectionAnalyzer
from .utils import tempered_class_weight


class BaseTrainer:
    """Base class for all trainers with common functionality."""

    def __init__(self, model_cfg, label_encoder, scaler=None,
                 scale_all=False, random_state=42):
        """
        Initialize base trainer with model configuration and encoding settings.

        Args:
            model_cfg (dict): Dictionary with model configuration (at minimum containing 'object').
            label_encoder (LabelEncoderWithReject): Label encoder with rejection capability.
            scaler (object): Sklearn-compatible scaler for feature normalization.
            scale_all (bool): Whether to apply scaling to all models by default.
            random_state (int): Random seed for reproducibility.
        """
        self.label_encoder = label_encoder
        self.model_cfg = deepcopy(model_cfg)
        self.scaler = scaler or StandardScaler()
        self.scale_all = scale_all
        self.random_state = random_state
        self.model = None

        # Validate model configuration
        if 'object' not in self.model_cfg:
            raise ValueError("model_cfg must contain 'object' key with the classifier class")

    def _build_model(self, model_cfg):
        """Build the model, including scaling if required."""
        base_model = model_cfg['object'](**(model_cfg.get('kwargs') or {}))
        if model_cfg.get('calibrate', False):
            # Wrap the model in a calibrated classifier if specified
            base_model = CalibratedClassifierCV(base_model, method='isotonic')
        if model_cfg.get('scale', False) or self.scale_all:
            return Pipeline([('scaler', self.scaler), ('clf', base_model)])
        return base_model
    
    def _resolve_class_weight(self, y, model_cfg):
        cfg = deepcopy(model_cfg)
        kw = cfg.get('kwargs', {})
        policy = kw.pop('class_weight', None)
        tc = kw.pop("tempered_cfg", {"gamma": 0.5, "w_min": 0.5, "w_max": 6.0})

        if policy is None:
            return cfg
        
        if policy == "tempered":
            cw = tempered_class_weight(y, **tc)   # dict
        elif isinstance(policy, dict) or policy == "balanced":
            cw = policy  # dict o 'balanced'
        else:
            cw = None

        cfg['kwargs'] = {**kw, 'class_weight': cw}
        return cfg

    def fit(self, X_train, y_train):
        """Fit the model to the training data."""
        resolved_cfg = self._resolve_class_weight(y_train, self.model_cfg)
        self.model = self._build_model(resolved_cfg)
        callbacks = resolved_cfg.get('callbacks')
        if callbacks:
            self.model.fit(X_train, y_train, callbacks=callbacks)
        else:
            self.model.fit(X_train, y_train)
        return self.model

    def _align_proba(self, probs, model):
        """
        Aligns predict_proba output to the global class order defined 
        by the label_encoder. Always returns an array of shape 
        (n_samples, n_global_classes).

        Args:
            probs (ndarray): Raw probability matrix from model.predict_proba(X).
            model (sklearn estimator): Trained model (can be Pipeline).

        Returns:
            ndarray: Probability matrix aligned to the global class order.
        """
        # Global class order from the LabelEncoder (e.g., [0, 1, 2])
        all_classes = np.arange(len(self.label_encoder.classes_))

        # Classes seen by the model during training
        if hasattr(model, "seen_classes_"):
            seen_classes = model.seen_classes_
        elif hasattr(model, "classes_"):
            seen_classes = model.classes_
        elif hasattr(model, "named_steps") and "clf" in model.named_steps:
            seen_classes = model.named_steps["clf"].classes_
        else:
            raise AttributeError(
                "Could not find classes_ attribute in the trained model."
            )
        
        # If shapes match, just return probs unchanged ---
        if len(all_classes) == probs.shape[1] and np.array_equal(seen_classes, all_classes):
            return probs

        # Build aligned probability matrix (fill missing classes with 0)
        out = np.zeros((probs.shape[0], len(all_classes)))
        for j, c in enumerate(seen_classes):
            out[:, c] = probs[:, j]

        return out

    def predict_proba(self, X_test, align=True):
        """
        Predict class probabilities for the given data.

        Args:
            X_test: Input features.
            align (bool): If True, align probabilities to the global class order.

        Returns:
            ndarray: Probability matrix with shape (n_samples, n_classes_global).
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        probs = self.model.predict_proba(X_test)
        if align:
            return self._align_proba(probs, self.model)
        return probs

    def predict(self, X_test):
        """Predict class labels for the test data."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X_test)

class SKLearnTrainer(BaseTrainer):
    """Main trainer class that orchestrates training, evaluation, and analysis."""

    def __init__(
            self,
            model_cfg,
            label_encoder,
            scaler=None,
            scale_all: bool = False,
            ra_num_points: int = 100,
            ra_eps: list | None = None,
            cv_strategy: str = 'stratified_kfold',
            cv_splits: int = 5,
            cv_verbose: int = 1,
            hp_iter: int = 50,
            hp_n_jobs: int = -1,
            hp_verbose: int = 1,
            hp_algorithm: str = 'gp',
            hp_algorithm_kwargs: dict | None = None,
            random_state: int = 42,
            store_metrics: Literal['val', 'test', 'val+test'] = 'val+test',
            store_curves: Literal['val', 'test', 'val+test'] | None = None,
            store_proportions: Literal['val', 'test', 'val+test'] | None = None,
            store_probas: Literal['val', 'test', 'val+test'] | None = None,
            refit: bool = True
        ):
        """
        Initialize the SKLearn trainer with integrated rejection analysis.

        Args:
            model_cfg (dict): Dictionary with model configuration (must contain 'object' key).
            label_encoder (LabelEncoderWithReject): Label encoder with rejection capability.
            scaler (object): Sklearn-compatible scaler for feature normalization.
            scale_all (bool): Whether to apply scaling to all models by default.
            num_points (int): Number of points to compute for curves/thresholds.
            eps (list): List of epsilon values for point-wise metrics.
            cv_strategy (str): Cross-validation strategy.
            cv_splits (int): Number of CV folds.
            cv_verbose (int): Verbosity level for cross-validation.
            hp_iter (int): Number of iterations for hyperparameter tuning.
            hp_n_jobs (int): Number of parallel jobs for tuning.
            hp_verbose (int): Verbosity level for hyperparameter tuning.
            hp_algorithm (str): Optimization algorithm for tuning ('gp', 'forest', 'gbrt', 'random').
            hp_algorithm_kwargs (dict): Additional arguments for the optimization algorithm.
            random_state (int): Random seed for reproducibility.
            store_curves (bool): Whether to store rejection curves.
            store_proportions (bool): Whether to store outcome proportions.
            refit (bool): Whether to refit the model after CV (full data) or HP tuning (best params).
        """
        super().__init__(model_cfg, label_encoder, scaler, scale_all, random_state)

        self.analyzer = RejectionAnalyzer(label_encoder, ra_num_points, ra_eps)
        self.cv = RejectorCrossValidator(cv_strategy, cv_splits, random_state, cv_verbose)
        self.tuner = None  # Placeholder for future hyperparameter tuner implementation

        self.store_metrics = store_metrics
        self.store_curves = store_curves
        self.store_proportions = store_proportions
        self.store_probas = store_probas
        self.refit = refit

        # Placeholders
        self.metrics = None
        self.rejection_curves = None
        self.outcome_proportions = None
        self.probas = None
        self.cv_results = None
        self.tuning_results = None

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Train the model and run complete rejection analysis.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            dict: Evaluation metrics.
        """
        logging.info("Training model...")
        self.fit(X_train, y_train)

        logging.info("Evaluating model...")

        probs_train = self.predict_proba(X_train)

        analyzer_train = self.analyzer(y_train, probs_train)
        self.metrics = {
            'train': analyzer_train.evaluate()
        }

        if self.store_curves:
            self.rejection_curves = {
                'train': analyzer_train.get_rejection_curves(format='wide')
            }

        if self.store_proportions:
            self.outcome_proportions = {
                'train': analyzer_train.get_outcome_proportions(format='wide')
            }

        if X_test is not None and X_test.size > 0:
            probs_test = self.predict_proba(X_test)

            analyzer_test = self.analyzer(y_test, probs_test)
            self.metrics = {
                'test': analyzer_test.evaluate()
            }

            if self.store_curves:
                self.rejection_curves = {
                    'test': analyzer_test.get_rejection_curves(format='wide')
                }

            if self.store_proportions:
                self.outcome_proportions = {
                    'test': analyzer_test.get_outcome_proportions(format='wide')
                }

        return self.metrics, None

    def cross_validate(
            self,
            X_train, y_train,
            X_test=None, y_test=None,
            stratify_train=None,
            groups_train=None,
            groups_test=None,
            confidence_level=0.95,
            seed=None,
            save_probas=False,
            force_standard_cv=False
        ):
        """
        Perform cross-validation with rejection analysis.

        Args:
            X: Feature matrix.
            y: Target labels.
            groups: Optional grouping for GroupKFold.
            confidence_level: Confidence level for intervals (default 0.95).

        Returns:
            Tuple: Aggregated cross-validation metrics and detailed cross-validation results.
        """
        # Run cross-validation
        cv_results = self.cv.cross_validate(
            self, X_train, y_train, X_test, y_test,
            stratify_trainval=stratify_train, groups_trainval=groups_train,
            groups_test=groups_test, seed=seed, save_probas=save_probas,
            force_standard_cv=force_standard_cv
        )

        if self.store_curves:
            self.rejection_curves = self.cv.get_rejection_curves(
                partition=self.store_curves,
                confidence_level=confidence_level
            )

        if self.store_proportions:
            self.outcome_proportions = self.cv.get_outcome_proportions(
                partition=self.store_proportions,
                confidence_level=confidence_level
            )

        if self.store_probas:
            self.probas = self.cv.get_probas(
                partition=self.store_probas
            )

        self.metrics = self.cv.get_metrics(
            partition=self.store_metrics,
            confidence_level=confidence_level
        )

        if self.refit:
            # Refit the model on the full training data
            self.fit(X_train, y_train)

        return self.metrics, cv_results

    def tune_hyperparameters(
            self,
            X_train, y_train,
            X_test, y_test,
            param_space,
            cv_stratify=None,
            cv_groups=None,
            primary_metric='FOC-AUC',
            cv_seed=None,
        ):
        """
        Tune hyperparameters using Bayesian optimization with comprehensive tracking.
        
        Args:
            X: Feature matrix.
            y: Target labels.
            param_space: Parameter space for tuning.
            groups: Optional group labels for GroupKFold.
            primary_metric: Primary metric to optimize.
            n_iter: Number of iterations for Bayesian optimization.
            algorithm: Optimization algorithm ('gp', 'forest', 'gbrt', 'random').
            n_jobs: Number of parallel jobs.
            verbose: Verbosity level.
            **algorithm_kwargs: Additional arguments for the optimization algorithm.
            
        Returns:
            dict: Comprehensive tuning results.
        """
        raise NotImplementedError("Hyperparameter tuning is not implemented yet.")

    def get_metrics(self):
        """
        Return a DataFrame with evaluation metrics:

        - If metrics came from simple train and evaluation,
            DataFrame contains only its values from test set.

        - If metrics came from cross-validation, DataFrame contains
            mean across folds and margin of error for each metric.
            For detailed results use `self.cv.get_detailed_metrics()` method.

        - If metrics came from hyperparameter tuning, DataFrame contains
            mean across folds and margin of error for each metric
            for the best hyperparameters found. For detailed results
            use `self.tuner.get_detailed_metrics()` method.
        """
        assert self.metrics is not None, "No metrics available. Run train_and_evaluate(), cross_validate() or tune_hyperparameters() first."
        data = {'partition': [], 'metric': [], 'value': [], 'margin': []}
        for partition, metrics in self.metrics.items():
            for metric, value in metrics.items():
                if metric.endswith('_margin'):
                    continue
                data['partition'].append(partition)
                data['metric'].append(metric)
                data['value'].append(value)
                data['margin'].append(metrics.get(metric + '_margin', None))
        return pd.DataFrame(data)
    
    def get_probas(self, format='wide'):
        assert self.probas is not None, "No probabilities available. Run cross_validate() with save_probas=True."
        dfs = []
        for partition, probas in self.probas.items():
            if partition == 'val':
                idxs = np.hstack([pb['val_idx'] for pb in probas])
                probs = np.concatenate([pb['probs'] for pb in probas])
            elif partition == 'test':
                idxs = np.asarray(list(range(len(probas['probs']))))
                probs = probas['probs'].copy()
            else:
                raise ValueError("Invalid partition in stored probabilities.")
            cols = ['prob_' + str(c).lower() for c in self.label_encoder.classes_]
            probas_df = pd.DataFrame(probs, index=idxs, columns=cols).sort_index()
            probas_df['partition'] = partition
            probas_df = probas_df.reset_index()
            dfs.append(probas_df)
        
        df = pd.concat(dfs, axis=0)

        if format == 'long':
            df = df.melt(id_vars=['index', 'partition'], var_name='class', value_name='probability')
            df['class'] = df['class'].str.replace('prob_', '').str.upper()

        return df

    def get_rejection_curves(self, format='wide'):
        """Return a DataFrame with rejection curves points data:
        rejection_rate, success_accuracy, value_retention, effective_discrimination, ed_f1 and (associated) threshold.

        - If curves came from simple train and evaluation,
            DataFrame contains only curves from test set.

        - If curves came from cross-validation, DataFrame contains
            mean curves across folds and additional margin of error
            columns for each rejection rate.

        - If curves came from hyperparameter tuning, DataFrame contains
            mean curves across folds and additional margin of error
            columns for each rejection rate for the best hyperparameters found.
        """
        assert self.rejection_curves is not None, "No rejection curves available. Run train_and_evaluate(), cross_validate() or tune_hyperparameters() first, or set store_curves=True."
        if format == 'wide':
            dfs = []
            for partition, curves in self.rejection_curves.items():
                #print(partition, curves.keys(), [type(v) for v in curves.values()])
                curves_df = pd.DataFrame(curves)
                curves_df['partition'] = partition
                dfs.append(curves_df)
            return pd.concat(dfs, axis=0)

        elif format == 'long':
            dfs = []
            for partition, rejection_curves in self.rejection_curves.items():
                curves = rejection_curves.copy()
                rejections = curves.pop('rejection_rate')
                thresholds = curves.pop('threshold')

                result = {
                    'rejection_rate': [],
                    'threshold': [],
                    'metric': [],
                    'value': [],
                    'margin': [],
                }

                for k, v in curves.items():
                    if k.endswith('_margin'):
                        continue
                    result['rejection_rate'].extend(rejections)
                    result['threshold'].extend(thresholds)
                    result['metric'].extend([k] * len(rejections))
                    result['value'].extend(v)
                    result['margin'].extend(curves.get(k + '_margin', [None]*len(rejections)))

                for k, v in result.items():
                    result[k] = np.array(v)

                curves_df = pd.DataFrame(result)
                curves_df['partition'] = partition
                dfs.append(curves_df)
            return pd.concat(dfs, axis=0)

        else:
            raise ValueError("Invalid format. Choose 'wide' or 'long'.")

    def get_outcome_proportions(self, format='wide'):
        """
        Return a DataFrame with rejection-based proportion data:
        rejection rate, (associated) threshold, label, and proportion.

        - If proportions came from simple train and evaluation,
            DataFrame contains only its values from test set.

        - If proportions came from cross-validation, DataFrame contains
            mean label proportions across folds and margin of error 
            for each rejection rate.

        - If proportions came from hyperparameter tuning, DataFrame contains
            mean label proportions across folds and margin of error 
            for each rejection rate for the best hyperparameters found.
        """
        assert self.outcome_proportions is not None, "No rejection outcome_proportions available. Run train_and_evaluate(), cross_validate() or tune_hyperparameters() first, or set store_proportions=True."
        if format == 'wide':
            dfs = []
            for partition, proportions in self.outcome_proportions.items():
                proportions_df = pd.DataFrame(proportions)
                proportions_df['partition'] = partition
                dfs.append(proportions_df)
            return pd.concat(dfs, axis=0)
        
        elif format == 'long':
            dfs = []
            for partition, outcome_proportions in self.outcome_proportions.items():
                proportions = outcome_proportions.copy()
                rejections = proportions.pop('rejection_rate')
                thresholds = proportions.pop('threshold')

                result = {
                    'rejection_rate': [],
                    'threshold': [],
                    'label': [],
                    'proportion': [],
                    'margin': [],
                }

                for k, v in proportions.items():
                    if k.endswith('_margin'):
                        continue
                    result['rejection_rate'].extend(rejections)
                    result['threshold'].extend(thresholds)
                    label = k.replace('proportion_', '')
                    result['label'].extend([label] * len(rejections))
                    result['proportion'].extend(v)
                    result['margin'].extend(proportions.get(k + '_margin', [None]*len(rejections)))

                for k, v in result.items():
                    result[k] = np.array(v)
                proportions_df = pd.DataFrame(result)
                proportions_df['partition'] = partition
                dfs.append(proportions_df)
            return pd.concat(dfs, axis=0)
        
        else:
            raise ValueError("Invalid format. Choose 'wide' or 'long'.")
