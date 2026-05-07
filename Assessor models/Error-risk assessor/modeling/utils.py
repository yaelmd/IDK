# -*- coding: utf-8 -*-

import logging
import pandas as pd
import polars as pl
import numpy as np
from typing import Tuple, Union
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, Hashable
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


class LabelEncoderWithReject(LabelEncoder):
    def __init__(
            self,
            reject_label="REJECT",
            reject_code=-1,
            incorrect_label="INCORRECT",
            success_labels=["CORRECT", "AVOIDANCE"],
            global_classes=None
        ):
        super().__init__()
        self.reject_label = reject_label
        self.reject_code = reject_code
        self.incorrect_label = incorrect_label
        self.success_labels = success_labels

        self.global_classes = (
            list(global_classes)
            if global_classes is not None
            else [incorrect_label] + list(success_labels)
        )
        
        # These will be set after fit_transform
        self.incorrect_idx = None
        self.success_class_idxs = None

    def fit(self, y):
        """Fit label encoder and compute indices."""
        super().fit(
            self.global_classes
        )
        self._compute_indices()
        return self
    
    def transform(self, y):
        y = np.asarray(y, dtype=object)
        out = np.full(len(y), self.reject_code, dtype=int)

        for i, cls in enumerate(self.classes_):
            out[y == cls] = i

        return out
    
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def _compute_indices(self):
        """Compute class indices after fitting."""
        if not hasattr(self, 'classes_') or len(self.classes_) == 0:
            return
        
        if len(self.classes_) == 1:
            logging.warning("Only one class found in labels during fitting.")
            return

        try:
            # Find incorrect class index
            self.incorrect_idx = np.argwhere(
                self.classes_ == self.incorrect_label
            ).flatten()[0]

            # Find success class indices
            self.success_class_idxs = set(np.argwhere(
                np.isin(self.classes_, self.success_labels)
            ).flatten())
        except (IndexError, ValueError) as e:
            logging.error(f"Some labels not found in encoder classes: {self.classes_}")
            raise e

    def inverse_transform(self, y):
        y = np.array(y)
        output = np.empty_like(y, dtype=object)
        valid_mask = (y != self.reject_code)
        output[valid_mask] = super().inverse_transform(y[valid_mask])
        output[~valid_mask] = self.reject_label
        return output

    def get_incorrect_idx(self):
        """Get the index of the INCORRECT class."""
        if self.incorrect_idx is None:
            raise ValueError("Encoder not fitted or INCORRECT class not found")
        return self.incorrect_idx

    def get_success_idxs(self):
        """Get indices of success classes."""
        if not self.success_class_idxs:
            raise ValueError("Encoder not fitted or success classes not found")
        return self.success_class_idxs


def _extract_X_y_pd(
        df, feature_cols, target_col, stratify_col_name=None, group_col_name=None
    ) -> Tuple[np.array, np.array, np.array, np.array]:
    columns = feature_cols + [target_col]
    if stratify_col_name and stratify_col_name in df.columns:
        columns.append(stratify_col_name)
    if group_col_name and group_col_name in df.columns and not group_col_name in columns:
        columns.append(group_col_name)
    row_positions = pd.Series(np.arange(len(df)), index=df.index)
    df = df.dropna(how='any', subset=columns)
    if len(df) == 0:
        return np.array([]), np.array([]), None, None, None
    df_idx = row_positions.loc[df.index].to_numpy()
    df = df[columns]
    embedding_feature_list = [col for col in feature_cols if "embedding" in col]
    if embedding_feature_list:
        embedding_feature = embedding_feature_list[0]
        stacked_embeddings = np.vstack(df[embedding_feature].values)
        not_embedding_cols = [
            col for col in feature_cols
            if col not in [embedding_feature, stratify_col_name, group_col_name]
        ]
        if not not_embedding_cols:
            # If no non-embedding columns, just use embeddings
            X = stacked_embeddings
        else:
            tabular = df[not_embedding_cols].values
            X = np.hstack((tabular, stacked_embeddings))
    else:
        X = df[feature_cols].values
    y = df[target_col].values.ravel()
    stratify_col = df[stratify_col_name].values if stratify_col_name and stratify_col_name in df.columns else None
    group_col = df[group_col_name].values if group_col_name and group_col_name in df.columns else None
    return X, y, df_idx, stratify_col, group_col


def _extract_X_y_pl(
        df, feature_cols, target_col, stratify_col_name=None, group_col_name=None
    ) -> Tuple[np.array, np.array, np.array, np.array]:
    columns = feature_cols + [target_col]
    if stratify_col_name and stratify_col_name in df.columns:
        columns.append(stratify_col_name)
    if group_col_name and group_col_name in df.columns and not group_col_name in columns:
        columns.append(group_col_name)
    df = df.with_row_index().select(columns+['index']).drop_nans().drop_nulls()
    if len(df) == 0:
        return np.array([]), np.array([]), None, None, None
    df_idx = df["index"].to_numpy()
    df = df.select(columns)
    embedding_feature_list = [col for col in feature_cols if "embedding" in col]
    if embedding_feature_list:
        embedding_feature = embedding_feature_list[0]
        stacked_embeddings = np.vstack(df[embedding_feature].to_numpy())
        not_embedding_cols = [
            col for col in feature_cols 
            if col not in [embedding_feature, stratify_col_name, group_col_name]
        ]
        if not not_embedding_cols:
            # If no non-embedding columns, just use embeddings
            X = stacked_embeddings
        else:
            tabular = df.select(not_embedding_cols).to_numpy()
            X = np.hstack((tabular, stacked_embeddings))
    else:
        X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy().ravel()
    stratify_col = df[stratify_col_name].to_numpy() if stratify_col_name and stratify_col_name in df.columns else None
    group_col = df[group_col_name].to_numpy() if group_col_name and group_col_name in df.columns else None
    return X, y, df_idx, stratify_col, group_col


def prepare_data(
    df: Union[pd.DataFrame, pl.DataFrame],
    feature_cols: list[str],
    target_col: str,
    label_encoder: LabelEncoderWithReject,
    test_size: float = 0.3,
    random_state: int = 123,
    stratify_col_name: str = None,
    group_col_name: str = None,
) -> Tuple[
        np.array, np.array, Union[np.array, None], Union[np.array, None], Union[np.array, None], 
        np.array, np.array, Union[np.array, None], Union[np.array, None], Union[np.array, None]
    ]:
    """
    Prepare data for training and testing.

    Args:
        df (pd.DataFrame or pl.DataFrame): DataFrame containing the data.
        feature_cols (str): List of feature column names.
        target_col (str): Name of the target column.
        label_encoder: LabelEncoderWithReject instance for encoding labels.
        random_state (int): Random state for reproducibility.

    Returns:
        X_train, y_train_enc, df_idx_train, stratify_train, group_train,
        X_test, y_test_enc, df_idx_test, stratify_test, group_test

        Notes:
        - If no split is performed, the corresponding *_test arrays are empty
            except for stratify_train and group_train, which are None.
        - If no stratify/group column was provided, the returned values are always None.
    """
    if isinstance(df, pd.DataFrame):
        X, y, df_idx, stratify_col, group_col = (
            _extract_X_y_pd(
                df,
                feature_cols,
                target_col,
                stratify_col_name,
                group_col_name,
            )
        )
    elif isinstance(df, pl.DataFrame):
        X, y, df_idx, stratify_col, group_col = (
            _extract_X_y_pl(
                df,
                feature_cols,
                target_col,
                stratify_col_name,
                group_col_name,
            )
        )
    else:
        raise TypeError("Unsupported DataFrame type.")
    
    if len(X) == 0:
        # No data: return empty arrays for all outputs
        empty = np.array([])
        return empty, empty, None, None, None, empty, empty, None, None, None

    # Normalize and validate optional meta-columns
    if stratify_col is not None:
        stratify_col = np.asarray(stratify_col)
        if stratify_col.shape[0] != len(y):
            raise ValueError("stratify_col length != number of samples")
    if group_col is not None:
        group_col = np.asarray(group_col)
        if group_col.shape[0] != len(y):
            raise ValueError("group_col length != number of samples")

    if test_size is not None:
        # Work with indices to keep all meta-columns aligned
        idx_all = np.arange(len(y))

        # Build labels used for stratify (combined y+stratify if provided)
        if stratify_col is not None:
            sc = stratify_col
            if sc.ndim == 1:
                combined = np.array(
                    [f"{y[i]}__{sc[i]}" for i in range(len(y))],
                    dtype=object,
                )
            else:
                combined = np.array(
                    [f"{y[i]}__{tuple(sc[i])}" for i in range(len(y))],
                    dtype=object,
                )
            labels_for_counts = combined
        else:
            labels_for_counts = y

        # Detect labels (possibly combined) with only one sample
        # and force those labels into train
        classes, counts = np.unique(labels_for_counts, return_counts=True)
        count_map = {cls: cnt for cls, cnt in zip(classes, counts)}
        is_singleton = np.array([
            count_map[val] == 1 for val in labels_for_counts
        ])

        idx_singleton = np.where(is_singleton)[0]
        idx_rest = np.where(~is_singleton)[0]

        # If no non-singleton remains, put all in train
        if idx_rest.size == 0:
            idx_train = idx_all
            idx_test = np.array([], dtype=int)
        else:
            # Build stratify_subset for the remaining samples
            stratify_subset = labels_for_counts[idx_rest]

            # Stratify only if there are at least 2 groups
            stratify_arg = (
                stratify_subset
                if np.unique(stratify_subset).size >= 2
                else None
            )

            # Split indices
            idx_train_rest, idx_test = train_test_split(
                idx_rest,
                test_size=test_size,
                stratify=stratify_arg,
                random_state=random_state,
            )

            # Combine train indices with singletons (forced to train)
            if idx_singleton.size > 0:
                idx_train = np.concatenate([idx_train_rest, idx_singleton])
                # shuffle train indices to mix appended singletons
                rng = np.random.RandomState(random_state)
                perm = rng.permutation(idx_train.shape[0])
                idx_train = idx_train[perm]
            else:
                idx_train = idx_train_rest

        # Index into original arrays to build split datasets
        X_train = X[idx_train]
        X_test = X[idx_test] if idx_test.size else np.array([])
        y_train = y[idx_train]
        y_test = y[idx_test] if idx_test.size else np.array([])
        df_idx_train = df_idx[idx_train]
        df_idx_test = df_idx[idx_test] if idx_test.size else np.array([])

        # Split stratify and group if provided, otherwise keep None
        if stratify_col is not None:
            stratify_train = stratify_col[idx_train]
            stratify_test = (
                stratify_col[idx_test] if idx_test.size else np.array([])
            )
        else:
            stratify_train, stratify_test = None, None

        if group_col is not None:
            group_train = group_col[idx_train]
            group_test = (
                group_col[idx_test] if idx_test.size else np.array([])
            )
        else:
            group_train, group_test = None, None

        # 5) Encode labels (fit on train, transform test)
        y_train_enc = label_encoder.fit_transform(y_train)
        y_test_enc = (
            label_encoder.transform(y_test) if y_test.size else np.array([])
        )

        return (
            X_train,
            y_train_enc,
            df_idx_train,
            stratify_train,
            group_train,
            X_test,
            y_test_enc,
            df_idx_test,
            stratify_test,
            group_test,
        )
    else:
        # No split requested: everything goes to train
        y_train_enc = label_encoder.fit_transform(y)
        stratify_train = (
            stratify_col if stratify_col is not None else None
        )
        group_train = group_col if group_col is not None else None
        X_test, y_test_enc = np.array([]), np.array([])
        df_idx_test = np.array([])
        stratify_test = None
        group_test = None

        return (
            X,
            y_train_enc,
            df_idx,
            stratify_train,
            group_train,
            X_test,
            y_test_enc,
            df_idx_test,
            stratify_test,
            group_test,
        )


def tempered_class_weight(
    y: np.ndarray,
    gamma: float = 0.5,
    w_min: float = 0.5,
    w_max: float = 6.0
) -> Dict[Hashable, float]:
    """
    Devuelve un dict {clase: peso} con pesos templados por vista.
    - Independiente de priors globales.
    - Normaliza para que la media de pesos sea 1.
    - Aplica clipping global para evitar explosiones por singletons.
    """
    classes, counts = np.unique(y, return_counts=True)
    N, K = len(y), len(classes)

    # Balanced base
    w_bal = N / (K * counts.astype(float))  # evita integer division

    # Templado
    w = np.power(w_bal, float(gamma))

    # Clipping
    if w_min is not None or w_max is not None:
        lo = w_min if w_min is not None else -np.inf
        hi = w_max if w_max is not None else np.inf
        w = np.clip(w, lo, hi)

    # Normaliza media=1 (no cambia la escala global de la loss)
    w *= (K / w.sum())

    return dict(zip(classes.tolist(), w.tolist()))


class AutoObjectiveLGBMClassifier(LGBMClassifier):
    """
    Auto-selects LightGBM objective at fit-time based on y:
      - n_unique==2  -> objective='binary'
      - n_unique>2   -> objective='multiclass'
      - n_unique==1  -> raise ValueError
    If force_auto=False, validates the provided objective and
    raises if incompatible.
    """
    def __init__(self, force_auto=True, clear_num_class=True, **kwargs):
        super().__init__(**kwargs)
        self.force_auto = force_auto
        self.clear_num_class = clear_num_class
        self.last_objective_ = None
        self.last_n_unique_ = None

    def _maybe_clear_num_class(self):
        if self.clear_num_class:
            super().set_params(num_class=None)

    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        y = np.asarray(y)
        classes = np.unique(y)
        n_unique = classes.size
        self.last_n_unique_ = int(n_unique)

        if n_unique == 1:
            raise ValueError(
                "AutoObjectiveLGBMClassifier: y is constant (1 unique class). "
                "This indicates a bad split or data issue."
            )

        desired = "binary" if n_unique == 2 else "multiclass"
        current = getattr(self, "objective", None)

        if self.force_auto:
            # Only set if needed
            if current != desired:
                super().set_params(objective=desired)
            self._maybe_clear_num_class()
            self.last_objective_ = desired

        else:
            if current is None:
                # No objective provided → set the correct one
                super().set_params(objective=desired)
                self.last_objective_ = desired
            else:
                ok = (current == "binary" and n_unique == 2) or \
                     (current == "multiclass" and n_unique > 2)
                if not ok:
                    raise ValueError(
                        (
                            "AutoObjectiveLGBMClassifier: incompatible "
                            "objective='{}' for n_unique={}. "
                            "Expected '{}'. Set force_auto=True."
                        ).format(current, n_unique, desired)
                    )
                self.last_objective_ = current
            self._maybe_clear_num_class()

        return super().fit(X, y, sample_weight=sample_weight, **fit_kwargs)


class AutoObjectiveXGBClassifier(XGBClassifier):
    """
    XGBClassifier wrapper que:
      1) Detecta clases presentes en cada fit.
      2) Re-encodea y a [0..K-1] (requisito de XGBoost en multiclass y útil en binary).
      3) Ajusta objective dinámicamente:
           - K==2  -> binary:logistic (y limpia num_class)
           - K>2   -> multi:softprob + num_class=K
           - K==1  -> error
      4) EXPONE classes_ como las labels originales (globales) vistas en ese fit,
         para que BaseTrainer._align_proba pueda alinear correctamente.
      5) predict() devuelve labels originales.
    """

    def __init__(self, force_auto=True, clear_num_class=True, **kwargs):
        super().__init__(**kwargs)
        self.force_auto = force_auto
        self.clear_num_class = clear_num_class

        self.last_objective_ = None
        self.last_n_unique_ = None

        self.label_to_local_ = None   # dict: original_label -> local_int
        self.local_to_label_ = None   # dict: local_int -> original_label
        self.seen_labels_ = None      # np.array de labels originales (orden estable)

    def _maybe_clear_num_class(self):
        if self.clear_num_class:
            # En binary no debe quedar num_class fijado a 3, etc.
            super().set_params(num_class=None)

    def _infer_desired_objective(self, n_unique):
        if n_unique == 2:
            return "binary:logistic"
        return "multi:softprob"

    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        y = np.asarray(y)
        classes = np.unique(y)
        n_unique = classes.size
        self.last_n_unique_ = int(n_unique)

        if n_unique == 1:
            raise ValueError(
                "AutoObjectiveXGBClassifier: y is constant (1 unique class). "
                "Bad split or data issue."
            )

        # Orden estable (np.unique ya devuelve ordenado)
        self.seen_labels_ = classes

        # Mapping original -> local 0..K-1
        self.label_to_local_ = {lbl: i for i, lbl in enumerate(self.seen_labels_)}
        self.local_to_label_ = {i: lbl for lbl, i in self.label_to_local_.items()}
        y_local = np.array([self.label_to_local_[lbl] for lbl in y], dtype=int)

        desired = self._infer_desired_objective(n_unique)
        current = getattr(self, "objective", None)

        if self.force_auto:
            if current != desired:
                super().set_params(objective=desired)
            if desired.startswith("multi:"):
                super().set_params(num_class=int(n_unique))
            else:
                self._maybe_clear_num_class()
            self.last_objective_ = desired
        else:
            # Valida compatibilidad si el usuario fija objective
            if current is None:
                super().set_params(objective=desired)
                current = desired

            is_binary_obj = str(current).startswith("binary:")
            is_multi_obj  = str(current).startswith("multi:")

            ok = (n_unique == 2 and is_binary_obj) or (n_unique > 2 and is_multi_obj)
            if not ok:
                raise ValueError(
                    f"AutoObjectiveXGBClassifier: incompatible objective='{current}' "
                    f"for n_unique={n_unique}. Expected '{desired}'. "
                    "Set force_auto=True."
                )

            if is_multi_obj:
                super().set_params(num_class=int(n_unique))
            else:
                self._maybe_clear_num_class()

            self.last_objective_ = current

        # Fit real con y_local (0..K-1)
        out = super().fit(X, y_local, sample_weight=sample_weight, **fit_kwargs)

        # SUPER IMPORTANTE:
        # Exponer classes_ como labels originales vistas en este fit.
        # Esto hace que BaseTrainer._align_proba funcione, ya que hace:
        #   for j,c in enumerate(seen_classes): out[:, c] = probs[:, j]
        self.seen_classes_ = self.seen_labels_.copy()

        return out

    def predict(self, X):
        # Predicción en espacio local -> map a label original
        pred_local = super().predict(X)
        pred_local = np.asarray(pred_local, dtype=int)
        return np.array([self.local_to_label_[i] for i in pred_local])

    def predict_proba(self, X):
        # Devuelve probs en el orden local [0..K-1], pero BaseTrainer alineará
        # usando self.classes_ (labels originales) que corresponden a ese mismo orden.
        return super().predict_proba(X)
    
    def get_xgb_params(self):
        params = super().get_xgb_params()
        # quitar params que son solo del wrapper
        params.pop("force_auto", None)
        params.pop("clear_num_class", None)
        return params

    def get_params(self, deep=True):
        # Mantén compatibilidad sklearn: estos deben seguir apareciendo aquí
        params = super().get_params(deep=deep)
        params["force_auto"] = self.force_auto
        params["clear_num_class"] = self.clear_num_class
        return params
