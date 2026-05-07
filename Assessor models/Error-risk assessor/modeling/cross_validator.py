# -*- coding: utf-8 -*-

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    RepeatedKFold,
    GroupKFold,
    LeaveOneGroupOut
)
from itertools import permutations


class RejectorCrossValidator:
    """Handles cross-validation procedures for rejection-based evaluation."""

    def __init__(self, cv_strategy='stratified_kfold', n_splits=5, random_state=42, verbose=1):
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.cv_results_ = None

    def get_cv_splitter(self, seed=None):
        """Get the appropriate cross-validation splitter based on the strategy."""
        if self.cv_strategy == 'kfold':
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=(seed or self.random_state))
        elif self.cv_strategy == 'stratified_kfold':
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=(seed or self.random_state))
        elif self.cv_strategy == 'repeated_kfold':
            return RepeatedKFold(n_splits=self.n_splits, n_repeats=3, random_state=(seed or self.random_state))
        elif self.cv_strategy == 'group_kfold':
            return GroupKFold(n_splits=self.n_splits)
        elif self.cv_strategy == 'leave_one_group_out':
            return LeaveOneGroupOut()
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")
        
    def _build_stratification_vector(self, y, stratify):
        """Builds a composite stratification label from y and additional stratify info."""
        y = np.asarray(y)
        if stratify is None:
            return y

        stratify = np.asarray(stratify)

        # Ensure both are column vectors
        y = y.reshape(-1, 1)
        stratify = stratify.reshape(-1, 1) if stratify.ndim == 1 else stratify

        # Concatenate and hash to create unique composite labels
        composite = np.concatenate([y, stratify], axis=1)
        _, y_ext = np.unique(composite.astype(str), axis=0, return_inverse=True)
        return y_ext

    def _cross_validate_standard(
            self,
            trainer,
            X_trainval,
            y_trainval,
            X_test=None,
            y_test=None,
            stratify_trainval=None,
            groups_trainval=None,
            groups_test=None,
            seed=None,
            save_probas=False
        ):
        cv_splitter = self.get_cv_splitter(seed=seed)

        if self.verbose > 1:
            logging.info(f"Starting standard CV...")

        cv_metrics = defaultdict(list)
        cv_rejection_curves = []
        cv_outcome_proportions = []
        cv_fold_groups = []
        cv_probas = []

        if stratify_trainval is not None:
            y_ext = self._build_stratification_vector(y_trainval, stratify_trainval)
        else:
            y_ext = y_trainval

        group_metrics_val = defaultdict(lambda: defaultdict(list))

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(
            X_trainval, y_ext, groups_trainval
        )):
            X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
            y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

            fold_groups = None
            if groups_trainval is not None:
                fold_groups = {
                    'train': np.unique(groups_trainval[train_idx]).tolist(),
                    'test': np.unique(groups_trainval[val_idx]).tolist()
                }
            cv_fold_groups.append(fold_groups)

            #logging.info("Training fold %d/%d...", fold_idx + 1, self.n_splits)
            model = trainer.fit(X_train, y_train)
            probs = trainer.predict_proba(X_val)

            cv_probas.append({
                'val_idx': val_idx,
                'y_val': y_val,
                'probs': probs
            })

            split_analyzer = trainer.analyzer(y_val, probs)
            eval_metrics = split_analyzer.evaluate()
            rejection_curves = split_analyzer.get_rejection_curves(format='wide')
            outcome_proportions = split_analyzer.get_outcome_proportions(format='wide')

            for k, v in eval_metrics.items():
                cv_metrics[k].append(v)
            cv_rejection_curves.append(rejection_curves)
            cv_outcome_proportions.append(outcome_proportions)

            if self.verbose > 1:
                logging.info(f"Fold: {fold_idx} | Results: {eval_metrics}")

            if groups_trainval is not None:
                groups = np.unique(groups_trainval[val_idx])
                for group in groups:
                    group_mask = groups_trainval[val_idx] == group
                    y_val_group = y_val[group_mask]
                    probs_group = probs[group_mask]
                    analyzer_group = trainer.analyzer(y_val_group, probs_group)
                    metrics = analyzer_group.evaluate()
                    for mk, mv in metrics.items():
                        group_metrics_val[mk][group].append(mv)
        
        if groups_trainval is not None:
            by_group_metrics_val = {}
            for mk, group_dict in group_metrics_val.items():
                by_group_metrics_val[mk] = {g: np.mean(vs) for g, vs in group_dict.items()}
        else:
            by_group_metrics_val = None

        test_results = None
        test_probas = None
        if (
            (X_test is not None and len(X_test) > 0)
            and (y_test is not None and len(y_test) > 0)
        ):
            model = trainer.fit(X_trainval, y_trainval)
            test_probs = trainer.predict_proba(X_test)

            test_probas = {
                'y_test': y_test,
                'probs': test_probs
            }

            test_analyzer = trainer.analyzer(y_test, test_probs)
            test_results = {
                'metrics': test_analyzer.evaluate(),
                'rejection_curves': test_analyzer.get_rejection_curves(format='wide'),
                'outcome_proportions':  test_analyzer.get_outcome_proportions(format='wide'),
                'probabilities': test_probas
            }
            if groups_test is not None:
                test_results['by_group_metrics'] = defaultdict(dict)
                test_results['groups'] = np.unique(groups_test).tolist()
                for group in test_results['groups']:
                    group_mask = groups_test == group
                    y_test_group = y_test[group_mask]
                    test_probs_group = test_probs[group_mask]
                    analyzer_group = trainer.analyzer(y_test_group, test_probs_group)
                    metrics = analyzer_group.evaluate()
                    for mk, mv in metrics.items():
                        test_results['by_group_metrics'][mk][group] = mv
                
        self.cv_results_ = {
            'val': {
                'metrics': cv_metrics,
                'rejection_curves': cv_rejection_curves,
                'outcome_proportions': cv_outcome_proportions,
                'fold_groups': cv_fold_groups,
                'by_group_metrics': by_group_metrics_val,
                'groups': np.unique(groups_trainval).tolist(),
                'probabilities': cv_probas
            },
            'test': test_results
        }

        if self.verbose > 0:
            logging.info(f"Val Results: {self.aggregate_metrics()}")
            if test_results is not None:
                logging.info(f"Test Results: {test_results['metrics']}")

        return self.cv_results_

    def cross_validate(
            self,
            trainer,
            X_trainval,
            y_trainval,
            X_test=None,
            y_test=None,
            stratify_trainval=None,
            groups_trainval=None,
            groups_test=None,
            seed=None,
            save_probas=False,
            force_standard_cv=False
        ):
        """Public entrypoint. Decides between standard CV and 3+1+1 depending on strategy and groups.
        - If cv_strategy == 'leave_one_group_out' and groups was passed for train and no explicit X_test/y_test passed,
          then run OOD cross-validation.
        - Otherwise, fall back to the standard CV.
        """
        if not force_standard_cv:
            if (self.cv_strategy == 'leave_one_group_out' and groups_trainval is not None):
                if (
                    (X_test is None or len(X_test) == 0)
                    and (y_test is None or len(y_test) == 0)
                ):
                    return self.cross_validate_3p1p1(
                        trainer, X_trainval, y_trainval, groups_trainval
                    )

        return self._cross_validate_standard(
            trainer, X_trainval, y_trainval, X_test=X_test, y_test=y_test,
            stratify_trainval=stratify_trainval,
            groups_trainval=groups_trainval, groups_test=groups_test,
            seed=seed, save_probas=save_probas
        )

    def _aggregate_group_vectors(self, vectors_by_group, group_order, confidence_level=0.95):
        """Given vectors_by_group: dict[group] -> list[float], compute per-group mean and global mean+margin.
        Returns (list_per_group_order, global_mean, global_margin).
        """
        vals = []
        per_group = []
        for g in group_order:
            arr = np.asarray(vectors_by_group.get(g, []), dtype=float)
            m = float(np.mean(arr)) if arr.size else np.nan
            per_group.append(m)
            if arr.size:
                vals.append(m)
        vals = np.asarray(vals, dtype=float)
        n = len(vals)
        if n > 1:
            std = np.std(vals, ddof=1)
            t_value = st.t.ppf((1 + confidence_level) / 2.0, df=n-1)
            margin = float(t_value * std / np.sqrt(n))
        else:
            margin = 0.0
        global_mean = float(np.nanmean(per_group))
        return per_group, global_mean, margin

    def cross_validate_3p1p1(self, trainer, X, y, groups, confidence_level=0.95):
        """Implements the 3+1+1 protocol over groups (benchmarks).
        Returns a dict with the same top-level structure as the standard cross_validate:
        {
          'val':  { 'metrics': dict(metric -> list[5]), 'rejection_curves': list[5], 'outcome_proportions': list[5], 'fold_groups': list[5], ...extras },
          'test': { 'metrics': dict(metric -> float), 'rejection_curves': dict or None, 'outcome_proportions': dict or None, ...extras }
        }
        Extras:
          - val['by_group_lists']: metric -> {group: list[4]}
          - val['by_scenario_val']: list of 20 records (id, val, test, metrics)
          - test['by_group_metrics']: metric -> {group: mean_of_4}
          - test['by_group_lists']: metric -> {group: list[4]}
          - test['by_scenario_test']: list of 20 records (id, val, test, metrics)
        """
        # Prepare groups and ordering
        groups_arr = np.asarray(groups)
        unique_groups = list(np.unique(groups_arr))
        group_order = sorted(unique_groups)

        # Build grouped_data mapping
        grouped_data = {}
        for g in group_order:
            idx = np.where(groups_arr == g)[0]
            grouped_data[g] = (X[idx], y[idx])

        # Containers
        val_lists = defaultdict(lambda: defaultdict(list))   # metric -> group -> [4]
        test_lists = defaultdict(lambda: defaultdict(list))  # metric -> group -> [4]
        curves_val_by_group = defaultdict(list)   # list of dicts (wide) per VAL group
        props_val_by_group = defaultdict(list)
        curves_test_by_group = defaultdict(list)  # list of dicts (wide) per TEST group
        props_test_by_group = defaultdict(list)
        scenarios_val = []
        scenarios_test = []

        # Iterate 20 scenarios (VAL, TEST)
        for val_g, test_g in permutations(group_order, 2):
            id_groups = [g for g in group_order if g not in (val_g, test_g)]
            # assemble train (ID)
            X_tr = np.concatenate([grouped_data[g][0] for g in id_groups], axis=0)
            y_tr = np.concatenate([grouped_data[g][1] for g in id_groups], axis=0)

            # --- VAL OOD ---
            trainer.fit(X_tr, y_tr)
            X_val, y_val = grouped_data[val_g]
            probs_val = trainer.predict_proba(X_val)
            an_val = trainer.analyzer(y_val, probs_val)
            m_val = an_val.evaluate()  # dict metric -> value (compatible con el contrato actual)
            for mk, val in m_val.items():
                # contrato actual: en CV guardas el número crudo, no {'value': ...}
                val_lists[mk][val_g].append(float(val))
            curves_val_by_group[val_g].append(an_val.get_rejection_curves(format='wide'))
            props_val_by_group[val_g].append(an_val.get_outcome_proportions(format='wide'))
            scenarios_val.append({'id': id_groups, 'val': val_g, 'test': test_g, 'metrics': m_val})

            # --- TEST OOD --- (reutilizamos el modelo entrenado)
            X_te, y_te = grouped_data[test_g]
            probs_te = trainer.predict_proba(X_te)
            an_te = trainer.analyzer(y_te, probs_te)
            m_te = an_te.evaluate()
            for mk, val in m_te.items():
                test_lists[mk][test_g].append(float(val))
            curves_test_by_group[test_g].append(an_te.get_rejection_curves(format='wide'))
            props_test_by_group[test_g].append(an_te.get_outcome_proportions(format='wide'))
            scenarios_test.append({'id': id_groups, 'val': val_g, 'test': test_g, 'metrics': m_te})

        # Aggregate per-group means (5) and global (1)
        val_metrics_vec = {}
        test_metrics_scalar = {}
        test_by_group_metrics = {}
        # Determine metric keys from accumulated dicts
        metric_keys = list(val_lists.keys() or test_lists.keys())

        for mk in metric_keys:
            # VAL: per-group means list[5]
            per_group_vec, _, _ = self._aggregate_group_vectors(val_lists[mk], group_order, confidence_level)
            val_metrics_vec[mk] = per_group_vec
            # TEST: per-group means and global scalar
            per_group_vec_test, global_mean, global_margin = self._aggregate_group_vectors(test_lists[mk], group_order, confidence_level)
            test_by_group_metrics.setdefault(mk, {})
            for gi, g in enumerate(group_order):
                test_by_group_metrics[mk][g] = per_group_vec_test[gi]
            test_metrics_scalar[mk] = global_mean
            test_metrics_scalar[f"{mk}_margin"] = global_margin

        # --- Aggregation helpers mimicking standard, do NOT aggregate 'rejection_rate' ---
        def _aggregate_curves_like_standard(dicts_list):
            """Aggregate list of wide dicts like aggregate_rejection_curves():
            - Do NOT aggregate 'rejection_rate' (copy from the first item)
            - Aggregate the rest with aggregate_dict_series
            - Return a flattened dict with '<key>' and '<key>_margin' arrays.
            """
            if not dicts_list:
                return {}
            # Copy fixed rejection_rate from the first dict
            fixed_rr = dicts_list[0].get('rejection_rate', None)
            # Filter out 'rejection_rate' before aggregation
            filtered = [ {k: v for k, v in d.items() if k != 'rejection_rate'} for d in dicts_list ]
            aggregated = self.aggregate_dict_series(filtered, confidence_level=confidence_level)

            result = {}
            # reinstate rejection_rate as fixed series
            if fixed_rr is not None:
                result['rejection_rate'] = fixed_rr
            # keep threshold as in standard (unflattened dict with value/margin)
            if 'threshold' in aggregated:
                result['threshold'] = aggregated.pop('threshold')['value']
            # flatten the rest
            for k, v in aggregated.items():
                result[k] = v['value']
                result[f"{k}_margin"] = v['margin']
            return result

        def _aggregate_props_like_standard(dicts_list):
            # Same logic for outcome_proportions
            return _aggregate_curves_like_standard(dicts_list)

        # Build lists of 5 aggregated dicts (one per group), matching the standard shape
        val_rejection_curves = []
        val_outcome_props = []
        for g in group_order:
            dicts_curves = curves_val_by_group[g]
            dicts_props  = props_val_by_group[g]
            val_rejection_curves.append(_aggregate_curves_like_standard(dicts_curves))
            val_outcome_props.append(_aggregate_props_like_standard(dicts_props))

        # --- Global TEST aggregates (curves/proportions) ---
        all_test_curves = []
        all_test_props  = []
        for g in group_order:
            all_test_curves.extend(curves_test_by_group[g])
            all_test_props.extend(props_test_by_group[g])
        test_rejection_curves_global = _aggregate_curves_like_standard(all_test_curves) if all_test_curves else None
        test_outcome_props_global = _aggregate_props_like_standard(all_test_props) if all_test_props else None

        # Fold groups summary (5 items)
        fold_groups_summary = [
            {'train': [g for g in group_order if g != v], 'test': [v]} for v in group_order
        ]

        # Pack results with same keys as standard
        self.cv_results_ = {
            'val': {
                'metrics': val_metrics_vec,                 # metric -> list[5]
                'rejection_curves': val_rejection_curves,   # list[5] dicts (flattened)
                'outcome_proportions': val_outcome_props,   # list[5] dicts (flattened)
                'fold_groups': fold_groups_summary,
                # extras
                'by_group_lists': {mk: dict(val_lists[mk]) for mk in metric_keys},
                'by_scenario_val': scenarios_val,
                'groups': group_order,
            },
            'test': {
                'metrics': test_metrics_scalar,             # metric -> float (global mean across 5 per-group means)
                'rejection_curves': test_rejection_curves_global,
                'outcome_proportions': test_outcome_props_global,
                # extras
                'by_group_metrics': test_by_group_metrics,  # metric -> {group: value}
                'by_group_lists': {mk: dict(test_lists[mk]) for mk in metric_keys},
                'by_scenario_test': scenarios_test,
                'groups': group_order,
            }
        }
        if self.verbose > 0:
            logging.info("3+1+1 - built results with 5-per-group and global aggregates.")
        return self.cv_results_

    def aggregate_dict_series(self, dict_list, keys=None, confidence_level=0.95):
        if not dict_list:
            return {}

        if keys is None:
            keys = list(dict_list[0].keys())

        n_folds = len(dict_list)
        aggregated = {}

        for key in keys:
            data = np.array([d[key] for d in dict_list])
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0, ddof=1)
            t_value = st.t.ppf((1 + confidence_level) / 2.0, df=n_folds - 1)
            margin = t_value * std / np.sqrt(n_folds)
            aggregated[key] = {'value': mean, 'margin': margin}

        return aggregated

    def aggregate_metrics(self, confidence_level=0.95):
        n_splits = len(next(iter(self.cv_results_['val']['metrics'].values())))
        aggregated = self.aggregate_dict_series(
            [
                {
                    k: v[i]
                    for k, v in self.cv_results_['val']['metrics'].items()
                }
                for i in range(n_splits)
            ],
            confidence_level=confidence_level
        )
        result = {}
        for k, v in aggregated.items():
            result[k] = v['value']
            result[f"{k}_margin"] = v['margin']

        return result
    
    def get_probas(self, partition='val'):
        if partition == 'val':
            return {
                'val': self.cv_results_['val']['probabilities']
            }
        elif partition == 'test':
            return {
                'test': self.cv_results_['test']['probabilities']
            }
        elif partition == 'val+test':
            return {
                'val': self.cv_results_['val']['probabilities'],
                'test': self.cv_results_['test']['probabilities']
            }
        else:
            raise ValueError(f"Invalid partition '{partition}'. Choose 'val' or 'test'.")

    def get_metrics(self, partition='val', confidence_level=0.95):
        if partition == 'val':
            return {
                'val': self.aggregate_metrics(
                    confidence_level=confidence_level
                )
            }
        elif partition == 'test':
            return {
                'test': self.cv_results_['test']['metrics']
            }
        elif partition == 'val+test':
            return {
                'val': self.aggregate_metrics(
                    confidence_level=confidence_level
                ),
                'test': self.cv_results_['test']['metrics']
            }
        else:
            raise ValueError(f"Invalid partition '{partition}'. Choose 'val' or 'test'.")

    def aggregate_rejection_curves(self, confidence_level=0.95):
        filtered = [
            {k: v for k, v in d.items() if k != 'rejection_rate'}
            for d in self.cv_results_['val']['rejection_curves']
        ]
        
        aggregated = self.aggregate_dict_series(filtered, confidence_level=confidence_level)

        result = {
            'rejection_rate': self.cv_results_['val']['rejection_curves'][0]['rejection_rate'],
            'threshold': aggregated.pop('threshold', None)
        }

        for k, v in aggregated.items():
            result[k] = v['value']
            result[f"{k}_margin"] = v['margin']

        return result
    
    def get_rejection_curves(self, partition='val', confidence_level=0.95):
        if partition == 'val':
            return {
                'val': self.aggregate_rejection_curves(
                    confidence_level=confidence_level
                )
            }
        
        elif partition == 'test':
            return {
                'test': self.cv_results_['test']['rejection_curves']
            }
        
        elif partition == 'val+test':
            curves = {
                'val': self.aggregate_rejection_curves(
                    confidence_level=confidence_level
                ),
                'test': self.cv_results_['test']['rejection_curves']
            }
            return curves

        else:
            raise ValueError(f"Invalid partition '{partition}'. Choose 'val', 'test' or 'val+test'.")

    def aggregate_outcome_proportions(self, confidence_level=0.95):
        filtered = [
            {k: v for k, v in d.items() if k != 'rejection_rate'}
            for d in self.cv_results_['val']['outcome_proportions']
        ]

        aggregated = self.aggregate_dict_series(filtered, confidence_level=confidence_level)

        result = {
            'rejection_rate': self.cv_results_['val']['outcome_proportions'][0]['rejection_rate'],
            'threshold': aggregated.pop('threshold', None)
        }

        for k, v in aggregated.items():
            result[k] = v['value']
            result[f"{k}_margin"] = v['margin']

        return result
    
    def get_outcome_proportions(self, partition='val', confidence_level=0.95):
        if partition not in self.cv_results_:
            raise ValueError(f"Partition '{partition}' not found in CV results.")
        if partition == 'val':
            return {
                'val': self.aggregate_outcome_proportions(confidence_level=confidence_level)
            }
        elif partition == 'test':
            return {
                'test': self.cv_results_['test']['outcome_proportions']
            }
        elif partition == 'val+test':
            return {
                'val': self.aggregate_outcome_proportions(
                    confidence_level=confidence_level
                ),
                'test': self.cv_results_['test']['outcome_proportions']
            }
        else:
            raise ValueError(f"Invalid partition '{partition}'. Choose 'val', 'test' or 'val+test'.")

    def get_detailed_metrics(self, format='long', scope='val_folds'):
        """
        Get detailed evaluation metrics in either long or wide format for different scopes.

        Args:
            format (str): 'long' or 'wide'.
            scope (str): one of
                - 'val_folds' (default): standard CV fold metrics
                - 'test_groups': test metrics by benchmark (5 rows before reshaping)
                    - if self.strategy was 'LeaveOneGroupOut' and cross_validate received groups_train,
                        this metrics are aggregated over the time that group (benchmark) was test
                    - if self.strategy was '[Stratified]KFold' and cross_validate received groups_test,
                        this metrics are the specific test metrics for that group (benchmark)
                - '3p1p1_val_scenarios': 3+1+1 validation scenarios (20 rows before reshaping)
                - '3p1p1_test_scenarios': 3+1+1 test scenarios (20 rows before reshaping)
        Returns:
            pd.DataFrame
        """
        scope = scope.lower()
        if format not in {'long', 'wide'}:
            raise ValueError("format must be 'long' or 'wide'")

        # --- Scope: 3p1p1 validation scenarios ---
        if scope == '3p1p1_val_scenarios':
            by_sc = self.cv_results_.get('val', {}).get('by_scenario_val')
            if not by_sc:
                raise ValueError("No 3p1p1 validation scenario data available. Did you run cross_validate_3p1p1?")
            # Expand each scenario into rows and build homogeneous schema
            rows = []
            for fold, sc in enumerate(by_sc):
                val_g = sc.get('val')
                metrics = sc.get('metrics', {})
                for m, v in metrics.items():
                    rows.append({'holdout_group': val_g, 'metric': m, 'value': v, 'fold': fold, 'partition': 'val'})
            df_long2 = pd.DataFrame(rows)
            if format == 'long':
                return df_long2
            # wide format: one row per holdout_group with metrics as columns
            df_wide = df_long2.pivot(index='holdout_group', columns='metric', values='value').reset_index()
            df_wide.columns.name = None
            return df_wide

        # --- Scope: 3p1p1 test scenarios ---
        if scope == '3p1p1_test_scenarios':
            by_sc = self.cv_results_.get('test', {}).get('by_scenario_test')
            if not by_sc:
                raise ValueError("No 3p1p1 test scenario data available. Did you run cross_validate_3p1p1?")
            rows = []
            for fold, sc in enumerate(by_sc):
                test_g = sc.get('test')
                metrics = sc.get('metrics', {})
                for m, v in metrics.items():
                    rows.append({'holdout_group': test_g, 'metric': m, 'value': v, 'fold': fold, 'partition': 'test'})
            df_long2 = pd.DataFrame(rows)
            if format == 'long':
                return df_long2
            df_wide = df_long2.pivot(index='holdout_group', columns='metric', values='value').reset_index()
            df_wide.columns.name = None
            return df_wide

        # --- Scope: 3p1p1 test by groups ---
        if scope == 'test_groups':
            bg = self.cv_results_.get('test', {}).get('by_group_metrics')
            groups = self.cv_results_.get('test', {}).get('groups', [])
            if not bg or not groups:
                raise ValueError("No 3p1p1 test-by-group data available. Did you run cross_validate_3p1p1?")
            metrics = list(bg.keys())
            # build long
            rows = []
            for g in groups:
                for m in metrics:
                    rows.append({'holdout_group': g, 'metric': m, 'value': bg[m].get(g, np.nan), 'fold': None, 'partition': 'test'})
            df_long = pd.DataFrame(rows)
            if format == 'long':
                return df_long
            # wide: one row per holdout_group with metrics as columns
            df_wide = df_long.pivot(index='holdout_group', columns='metric', values='value').reset_index()
            df_wide.columns.name = None
            return df_wide
        
        # --- Scope: val by groups ---
        if scope == 'val_groups':
            bg = self.cv_results_.get('val', {}).get('by_group_metrics')
            groups = self.cv_results_.get('val', {}).get('groups', [])
            if not bg or not groups:
                raise ValueError("No 3p1p1 val-by-group data available. Did you run cross_validate_3p1p1?")
            metrics = list(bg.keys())
            # build long
            rows = []
            for g in groups:
                for m in metrics:
                    rows.append({'holdout_group': g, 'metric': m, 'value': bg[m].get(g, np.nan), 'fold': None, 'partition': 'val'})
            df_long = pd.DataFrame(rows)
            if format == 'long':
                return df_long
            # wide: one row per holdout_group with metrics as columns
            df_wide = df_long.pivot(index='holdout_group', columns='metric', values='value').reset_index()
            df_wide.columns.name = None
            return df_wide

        # --- Default scope: standard CV folds ---
        if scope != 'val_folds':
            raise ValueError(f"Invalid scope '{scope}'. Choose 'val_folds', '3p1p1_val_scenarios', '3p1p1_test_groups', or '3p1p1_test_scenarios'.")

        # Standard CV fold metrics
        cv_metrics = self.cv_results_['val']['metrics']
        cv_fold_groups = self.cv_results_['val']['fold_groups']
        if not cv_metrics:
            raise ValueError("No metrics found in CV results.")

        metric_names = list(cv_metrics.keys())
        n_folds = len(cv_metrics[metric_names[0]])
        has_groups = cv_fold_groups and any(fold_groups is not None for fold_groups in cv_fold_groups)

        rows = []
        for fold_idx in range(n_folds):
            # Determine holdout_group for this fold
            holdout_group = None
            if has_groups:
                fold_groups = cv_fold_groups[fold_idx]
                if fold_groups is not None and fold_groups.get('test'):
                    test_groups = fold_groups['test']
                    if isinstance(test_groups, list):
                        if len(test_groups) == 1:
                            holdout_group = test_groups[0]
                        elif len(test_groups) > 1:
                            holdout_group = '|'.join(str(g) for g in test_groups)
                        else:
                            holdout_group = None
                    else:
                        holdout_group = str(test_groups)
                else:
                    holdout_group = f"fold-{fold_idx+1}"
            else:
                holdout_group = f"fold-{fold_idx+1}"
            for metric_name in metric_names:
                rows.append({
                    'holdout_group': holdout_group,
                    'metric': metric_name,
                    'value': cv_metrics[metric_name][fold_idx],
                    'fold': fold_idx + 1,
                    'partition': 'val'
                })

        df_long2 = pd.DataFrame(rows)
        if format == 'long':
            return df_long2
        # wide: index by holdout_group, columns by metric, values by value
        df_wide = df_long2.pivot(index='holdout_group', columns='metric', values='value').reset_index()
        df_wide.columns.name = None
        return df_wide
