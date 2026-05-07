# -*- coding: utf-8 -*-

from collections import defaultdict
from copy import deepcopy
from typing import Tuple

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


class RejectionAnalyzer:
    """Handles rejection-based analysis and evaluation metrics."""

    __EPS_DEFAULT = [0.05, 0.10, 0.20]

    def __init__(
            self,
            label_encoder,
            num_points=100,
            eps=None
        ): 
        """Initialize the rejection analyzer and compute rejection points."""
        self.label_encoder = label_encoder
        self.num_points = num_points
        self.eps = self.__EPS_DEFAULT if eps is None else eps

        # Get indices from the encoder
        self.incorrect_idx = self.label_encoder.get_incorrect_idx()
        self.success_class_idxs = self.label_encoder.get_success_idxs()
        self.reject_code = self.label_encoder.reject_code
        self.reject_label = self.label_encoder.reject_label

        # Placeholders
        self.y_true = None
        self.probs = None
        self._rejection_points = {}

    def __call__(self, y_true, probs, add_noise=True, interpolate=False, threshold_mode='quantile'):
        """
        Analyze predictions and return a new analyzer instance with results.
        
        Returns:
            RejectionAnalyzer: New instance with computed results
        """
        new_analyzer = RejectionAnalyzer(
            self.label_encoder,
            self.num_points,
            self.eps
        )
        new_analyzer.y_true = y_true
        new_analyzer.probs = probs
        new_analyzer._compute_rejection_points(
            add_noise, interpolate, threshold_mode
        )
        return new_analyzer

    def _compute_rejection_points(self, add_noise=True, interpolate=False, threshold_mode='quantile'):
        """
        Compute rejection metrics and per-class proportions under progressive rejection.

        Returns:
            Dictionary with:
                - rejection_rate
                - threshold
                - success_accuracy
                - value_retention
                - error_discrimination
                - ed_f1
                - proportion_<label_name> (for each class including 'REJECT')
        """
        prob_incorrect = self.probs[:, self.incorrect_idx]

        self._rejection_points = compute_rejection_points(
            y_true=self.y_true,
            probs_rejection=prob_incorrect,
            success_class_labels=self.success_class_idxs,
            error_class_label=self.incorrect_idx,
            label_encoder=self.label_encoder,
            reject_label=self.reject_label,
            num_points=self.num_points,
            add_noise=add_noise,
            return_all_proportions=True,
            interpolate=interpolate,
            threshold_mode=threshold_mode,
        )

    def get_rejection_points(self):
        """Return the full dictionary of rejection points."""
        return self._rejection_points

    def get_outcome_proportions(self, format='long'):
        """
        Return class proportions in long-form dataframe-like structure:
            - rejection_rate
            - threshold
            - label
            - proportion
        """
        proportion_points = {
            k: v for k, v in self._rejection_points.items()
            if k.startswith('proportion_') or k in ['rejection_rate', 'threshold']
        }

        if format == 'wide':
            return proportion_points

        elif format == 'long':
            r = proportion_points['rejection_rate']
            t = proportion_points['threshold']

            proportions = {
                'rejection_rate': [],
                'threshold': [],
                'label': [],
                'proportion': []
            }

            all_labels = list(self.label_encoder.classes_) + [self.reject_label]
            for label in all_labels:
                prop_values = proportion_points.get(f"proportion_{label}", None)
                if prop_values is None:
                    continue
                for ri, ti, pi in zip(r, t, prop_values):
                    proportions['rejection_rate'].append(ri)
                    proportions['threshold'].append(ti)
                    proportions['label'].append(label)
                    proportions['proportion'].append(pi)

            return proportions
        
        else:
            raise ValueError("format must be 'wide' or 'long'")

    def get_rejection_curves(self, format='wide'):
        """Return only rejection metrics (no class proportions)."""
        curve_points = {
            k: v for k, v in self._rejection_points.items()
            if not k.startswith('proportion_')
        }
       
        if format == 'wide':
            return curve_points
        
        elif format == 'long':
            r = curve_points['rejection_rate']
            t = curve_points['threshold']

            curves = {
                'rejection_rate': [],
                'threshold': [],
                'metric': [],
                'value': []
            }

            for key, values in curve_points.items():
                if key in ['rejection_rate', 'threshold']:
                    continue
                for ri, ti, vi in zip(r, t, values):
                    curves['rejection_rate'].append(ri)
                    curves['threshold'].append(ti)
                    curves['metric'].append(key)
                    curves['value'].append(vi)

            return curves
        
        else:
            raise ValueError("format must be 'wide' or 'long'")

    def evaluate(self):
        """
        Compute evaluation metrics for the rejection model.
        Includes standard scoring metrics + ARC-based metrics.
        """
        y_reject_true = (self.y_true == self.incorrect_idx).astype(int)
        y_reject_score = self.probs[:, self.incorrect_idx]

        r = self._rejection_points["rejection_rate"]
        thr = self._rejection_points["threshold"]
        acc = self._rejection_points["accuracy"]
        nag = self._rejection_points["normalized_accuracy_gain"]
        sar = self._rejection_points["success_accepted_rate"]
        err = self._rejection_points["error_rejected_rate"]
        eds = self._rejection_points["effective_discrimination_score"]
        erg = self._rejection_points["effective_reliability_gain"]

        metrics = {
            "original-error": 1 - acc[0],
            "FOC-AUC": compute_foc_auc(err, sar),
            "ERG-AUC": compute_erg_auc(r, erg),
            "MAX-ERG": max(erg),
            "AU-SAC": compute_auvrc(r, sar),
            "AU-ERC": compute_auedc(r, err),
            "AU-ARC": compute_auarc(r, acc),
            #"ROC-AUC": roc_auc_score(y_reject_true, y_reject_score),
            #"ECE": calibration_error(y_reject_true, y_reject_score)[2],
            #"LogLoss": log_loss(y_reject_true, y_reject_score),
            #"BrierScore": brier_score_loss(y_reject_true, y_reject_score),
            #"WinklersScore": winklers_score(y_reject_true, y_reject_score),
        }

        ms_max_erg = metrics_at_max_erg(r, thr, acc, nag, sar, err, eds, erg, eps=1)
        metrics.update({
            **{
                metric.replace("100", "MAX-ERG"): v
                for metric, v in ms_max_erg.items() if metric != 'idx'
            },
        })

        for eps in self.eps:
            metrics_at_eps = metrics_at_operating_point(r, thr, acc, nag, sar, err, eds, erg, eps)
            metrics.update({
                **{k: v for k, v in metrics_at_eps.items() if k != 'idx'},
                f"ERG_max@{int(100 * eps)}": erg_max_at_epsilon(acc, erg, eps)[0],
            })

        return metrics


#=============================================================
# Rejection Curve Metrics
#=============================================================

def compute_rejection_points(
    y_true: np.ndarray,
    probs_rejection: np.ndarray,
    error_class_label: int | set,
    success_class_labels: int | set,
    label_encoder,
    reject_label: str = "REJECT",
    num_points: int = 1000,
    add_noise: bool = True,
    return_all_proportions: bool = True,
    interpolate: bool = False,
    threshold_mode: str = "quantile",  # "quantile" | "interp"
) -> dict:
    """
    Compute rejection curve metrics and per-class proportions under progressive rejection.

    Args:
        y_true: Ground truth labels (encoded as integers).
        probs_rejection: Estimated probability of error per instance.
        error_class_label: Label considered an error (e.g. INCORRECT).
        success_class_labels: Labels considered success (e.g. CORRECT, EVASIVE).
        label_encoder: Fitted LabelEncoder used to decode original class names.
        reject_code: Label name to assign to rejected samples (e.g. "REJECT").
        num_points: Number of rejection levels (default = 1000).
        interpolate: Whether to interpolate over rejection rate space.
        threshold_mode: "quantile" (default) or "interp" for threshold interpolation.
        add_noise: Whether to add small noise to break score ties.
        return_all_proportions: Whether to return proportions of each class (+ REJECT).

    Returns:
        A dictionary with arrays containing:
            - 'rejection_rate': Rejection rates from 0 to 1 (either exact or interpolated)
            - 'success_accuracy': Accuracy over the retained samples, counting only successful predictions
            - 'value_retention': Proportion of total success cases retained at each rejection rate
            - 'effective_discrimination': Proportion of true errors successfully rejected
            - 'ed_f1': Harmonic mean of error discrimination and value retention
            - 'threshold': Corresponding threshold for each rejection rate
            - proportion_<label_name> (one per class including REJECT)
    """

    rejection_levels = np.linspace(0, 1, num_points + 1)
    probs = deepcopy(probs_rejection)

    if add_noise:
        rng = np.random.default_rng(seed=42)
        probs += rng.random(len(probs)) / 1_000_000
        probs = np.clip(probs, 0.0, 1.0)

    results = defaultdict(list)

    for r in rejection_levels:
        threshold = np.quantile(probs, 1 - r)
        
        accepted_mask = probs < threshold

        # Métricas de rechazo
        metrics = compute_metrics_at_mask(y_true, accepted_mask, error_class_label, success_class_labels)
        for metric, value in metrics.items():
            results[metric].append(value)
        results["threshold"].append(threshold)

        # Proporciones por clase
        if return_all_proportions:
            proportions = compute_class_proportions_at_mask(y_true, accepted_mask, label_encoder, reject_label)
            for label, value in proportions.items():
                results[f"proportion_{label}"].append(value)

    # Convertir todo a arrays y ordenar por rejection rate
    for key in results:
        results[key] = np.array(results[key])
    order = np.argsort(results["rejection_rate"])
    for key in results:
        results[key] = results[key][order]

    if not interpolate:
        return results

    target_rs = np.linspace(0, 1, num_points + 1)
    interp_results = interpolate_metrics(results, target_rs)
    interp_results["rejection_rate"] = target_rs

    if threshold_mode == "quantile":
        interp_results["threshold"] = np.quantile(probs, 1 - target_rs)
    elif threshold_mode == "interp":
        f_thresh = interp1d(
            results["rejection_rate"],
            results["threshold"],
            kind="linear",
            bounds_error=False,
            fill_value=(results["threshold"][0], results["threshold"][-1]),
        )
        interp_results["threshold"] = f_thresh(target_rs)
    else:
        raise ValueError("threshold_mode must be 'quantile' or 'interp'")

    return interp_results


def compute_metrics_at_mask(y_true, accepted_mask, error_class_label, success_class_labels):
    """
    Compute rejection metrics over the accepted and rejected subsets.

    Returns:
        Dict with:
            - rejection_rate (proportion of rejected samples)
            - accuracy (over accepted samples, counting only success)
            - reliability_gain_index (relative accuracy improvement over initial accuracy)
            - success_retention (proportion of successful samples preserved)
            - error_rejection (proportion of error samples rejected)
            - effective_discrimination_score (harmonic mean of error rejection and success retention)
            - effective_reliability_gain (reliability gain index * effective discrimination score)
    """
    is_success = np.isin(y_true, list(success_class_labels))
    is_error = np.isin(y_true, error_class_label)

    y_accepted = y_true[accepted_mask]
    y_rejected = y_true[~accepted_mask]

    total_success = is_success.sum()
    total_error = is_error.sum()

    success_accepted = np.isin(y_accepted, list(success_class_labels)).sum()
    error_rejected = np.isin(y_rejected, error_class_label).sum()

    n_accepted = accepted_mask.sum()
    rejection_rate = 1.0 - n_accepted / len(y_true)

    accuracy = success_accepted / n_accepted if n_accepted > 0 else 1.0
    # Baseline accuracy over the full set (r = 0): proportion of successes in all data
    N = len(y_true)
    A0 = total_success / N if N > 0 else 0.0
    normalized_accuracy_gain = (
        (accuracy - A0) / np.maximum(1.0 - A0, 1e-12)
    )

    success_accepted_rate = (
        success_accepted / total_success if total_success > 0 else 0.0
    )
    error_rejected_rate = (
        error_rejected / total_error if total_error > 0 else 0.0
    )

    denom = error_rejected_rate + success_accepted_rate
    effective_discrimination_score = (
        2 * (error_rejected_rate * success_accepted_rate) / denom if denom > 0 else 0.0
    )

    effective_reliability_gain = effective_discrimination_score * normalized_accuracy_gain

    return {
        "rejection_rate": rejection_rate,
        "accuracy": accuracy,
        "normalized_accuracy_gain": normalized_accuracy_gain,
        "success_accepted_rate": success_accepted_rate,
        "error_rejected_rate": error_rejected_rate,
        "effective_discrimination_score": effective_discrimination_score,
        "effective_reliability_gain": effective_reliability_gain,
    }


def compute_class_proportions_at_mask(y_true, accepted_mask, label_encoder, reject_label):
    """
    Compute class proportions after applying rejection.
    Rejected instances are labeled as `reject_code`.

    Returns:
        Dict with:
            - proportion_<label_name>: float
    """
    # Map original int labels to names
    decoded_labels = label_encoder.inverse_transform(y_true)
    decoded_labels = np.where(accepted_mask, decoded_labels, reject_label)

    all_labels = list(label_encoder.classes_) + [reject_label]

    proportions = {}
    for label in all_labels:
        proportions[label] = np.mean(decoded_labels == label)

    return proportions


def interpolate_metrics(data: dict, target_rs: np.ndarray) -> dict:
    """
    Interpolates each array in `data` with respect to rejection_rate over `target_rs`.

    Args:
        data (dict): Dictionary with key → np.array, must include 'rejection_rate'.
        target_rs (np.ndarray): Rejection rates to interpolate to.

    Returns:
        dict: Same keys, interpolated to target_rs.
    """
    interpolated = {"rejection_rate": target_rs}
    r = data["rejection_rate"]

    for key, values in data.items():
        if key == "rejection_rate":
            continue
        values = np.asarray(values)
        if len(np.unique(r)) > 1:
            f_interp = interp1d(r, values, kind="linear", bounds_error=False, fill_value=(values[0], values[-1]))
            interpolated[key] = f_interp(target_rs)
        else:
            interpolated[key] = np.full_like(target_rs, values[0])

    return interpolated


#=============================================================
# Aggregated Evaluation Metrics
#=============================================================

def calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 100,
        strategy: str = 'uniform'
    ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the expected calibration error (ECE) of a set of predictions.
    The ECE is a measure of how well the predicted probabilities of a model
    align with the actual outcomes. A lower ECE indicates better calibration.

    Args:
        y_true (np.ndarray): True binary labels (0 or 1).
        y_prob (np.ndarray): Predicted probabilities of the positive class.
        n_bins (int): Number of bins to use for calibration.
        strategy (str): Strategy for binning. Can be 'uniform' or 'quantile'.

    Returns:
        prob_true (np.ndarray): Average true probability in each bin.
        prob_pred (np.ndarray): Average predicted probability in each bin.
        expected_calibration_error (float): The expected calibration error.
    """
    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    expected_calibration_error = np.sum(bin_total[nonzero] * np.abs(prob_true - prob_pred))/bin_total[nonzero].sum()
    return prob_true, prob_pred, expected_calibration_error


def winklers_score(outs: np.ndarray, preds: np.ndarray) -> float:
    """
    Compute Winkler's Score as described by:
    Winkler, R. L. (1994).
    "Evaluating probabilities: Asymmetric scoring rules."
    Management Science,923 40(11):1395–1405

    Parameters
    ----------
    preds : array-like
        An array of predicted probabilities a(x_i) for each instance i.
        Each element should be in [0,1].
    outs : array-like
        An array of ground-truth binary outcomes v_i (0 or 1).

    Returns
    -------
    float
        The Winkler's Score.
    """
    # Convert inputs to numpy arrays
    preds = np.array(preds, dtype=float)
    outs = np.array(outs, dtype=int)

    # Check dimensions
    if preds.shape[0] != outs.shape[0]:
        raise ValueError("Predictions and ground_truth must have the same length.")

    n = len(preds)

    # Compute c_j as the average success rate (fraction of positives)
    c_j = np.mean(outs)

    # c_j must be in (0,1) for Winkler's score to be well-defined.
    # If c_j = 0 or c_j = 1, the transformation breaks down.
    if c_j <= 0.0 or c_j >= 1.0:
        print("The computed c_j is not strictly between 0 and 1. "
              "Ensure that ground_truth has at least one positive and one negative outcome.")
        return np.nan

    # Indicator for v_i
    v_is_one = (outs == 1).astype(float)
    v_is_zero = (outs == 0).astype(float)

    # Indicators for prediction relative to c_j
    pred_leq_c = (preds <= c_j).astype(float)
    pred_gt_c = (preds > c_j).astype(float)

    # Compute alpha_i:
    # alpha_i = [(1-c_j)^2 - (1-a_i)^2]*1{v_i=1} + (c_j^2 - a_i^2)*1{v_i=0}
    term_v1 = ((1 - c_j) ** 2 - (1 - preds) ** 2) * v_is_one
    term_v0 = (c_j ** 2 - preds ** 2) * v_is_zero
    alpha = term_v1 + term_v0

    # Compute beta_i:
    # beta_i = c_j^2 * 1{a_i ≤ c_j} + (1-c_j)^2 * 1{a_i > c_j}
    beta = c_j ** 2 * pred_leq_c + (1 - c_j) ** 2 * pred_gt_c

    # Compute Winkler's Score = (1/n) * sum(alpha_i / beta_i)
    score = np.mean(alpha / beta)

    return score


def compute_area_under_curve(x: np.ndarray, y: np.ndarray, method: str = "simpson") -> float:
    """
    Compute the area under a curve defined by x and y coordinates.

    Args:
        x (np.ndarray): x-coordinates of the curve.
        y (np.ndarray): y-coordinates of the curve.
        method (str): Integration method: 'trapezoid' or 'simpson'.

    Returns:
        float: Area under the curve.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    x = np.array(x)
    y = np.array(y)
    sorted_idx = np.argsort(x)
    x = x[sorted_idx]
    y = y[sorted_idx]
    if method == "trapezoid":
        return np.trapz(y, x)
    elif method == "simpson":
        return simpson(y, x)
    else:
        raise ValueError("Invalid method. Choose 'trapezoid' or 'simpson'.")


def compute_auarc(r: np.ndarray, acc: np.ndarray, method: str = "trapezoid") -> float:
    """Area Under the Accuracy-Rejection Curve (AU-ARC)."""
    return compute_area_under_curve(r, acc, method=method)


def compute_auvrc(r: np.ndarray, vr: np.ndarray, method: str = "trapezoid") -> float:
    """Area Under the Value Retention Curve (AU-VRC)."""
    return compute_area_under_curve(r, vr, method=method)


def compute_auedc(r: np.ndarray, ed: np.ndarray, method: str = "trapezoid") -> float:
    """Area Under the Error Discrimination Curve (AU-EDC)."""
    return compute_area_under_curve(r, ed, method=method)


def compute_foc_auc(ed: np.ndarray, vr: np.ndarray, method: str = "trapezoid") -> float:
    """
    FOC-AUC (1 - error discrimination vs value retention).
    Represents trade-off between keeping valuable samples and failing to reject errors.
    """
    return compute_area_under_curve(1 - ed, vr, method=method)


def compute_erg_auc(r: np.ndarray, erg: np.ndarray, method: str = "trapezoid") -> float:
    """
    ERG-AUC (rejection rate vs effective reliability gain).
    Represents trade-off between rejecting samples and improving reliability.
    """
    return compute_area_under_curve(r, erg, method=method)


def compute_pvr(r: np.ndarray, acc: np.ndarray, epsilon: float):
    """
    PVR@ε: Max proportion of non-rejected samples (1 - r)
    such that accuracy ≥ 1 - ε.
    Returns (pvr_value, index) where index is the left-most feasible point.
    """
    r = np.asarray(r)
    acc = np.asarray(acc)
    mask = acc >= 1.0 - epsilon
    if not np.any(mask):
        return 0.0, None
    coverage = 1.0 - r
    # masked argmax with tie-break to smallest r (left-most)
    masked = np.where(mask, coverage, -np.inf)
    idx = int(np.argmax(masked))
    return float(coverage[idx]), idx


def compute_pvvr(acc: np.ndarray, sar: np.ndarray, epsilon: float):
    """
    PV&VR@ε: Max success accepted rate such that accuracy ≥ 1 - ε.
    Returns (pvvr_value, index) where index is the left-most feasible point.
    """
    acc = np.asarray(acc)
    sar = np.asarray(sar)
    mask = acc >= 1.0 - epsilon
    if not np.any(mask):
        return 0.0, None
    masked = np.where(mask, sar, -np.inf)
    idx = int(np.argmax(masked))
    return float(sar[idx]), idx


def compute_nag_at_epsilon(acc: np.ndarray, nag: np.ndarray, epsilon: float):
    """
    NAG@ε: Normalized accuracy gain at the left-most index
    such that accuracy ≥ 1 - ε. Returns (nag_value, index).
    """
    acc = np.asarray(acc)
    nag = np.asarray(nag)
    idxs = np.where(acc >= 1.0 - epsilon)[0]
    if len(idxs) == 0:
        return 0.0, None
    idx = int(idxs[0])
    return float(nag[idx]), idx


def compute_erg_at_epsilon(acc: np.ndarray, erg: np.ndarray, epsilon: float):
    """
    ERG@ε: Max effective reliability gain such that accuracy ≥ 1 - ε.
    Returns (erg_value, index) choosing the maximum within the feasible set;
    tie-break goes to the smallest r (left-most index).
    """
    acc = np.asarray(acc)
    erg = np.asarray(erg)
    mask = acc >= 1.0 - epsilon
    if not np.any(mask):
        return 0.0, None
    masked = np.where(mask, erg, -np.inf)
    idx = int(np.argmax(masked))
    return float(erg[idx]), idx


def operating_index_by_risk(acc: np.ndarray, eps: float, _tol: float = 1e-12):
    """
    Returns the first index i such that acc[i] >= 1 - eps - tol.
    If no feasible, returns None.
    """
    idxs = np.where(acc >= 1.0 - eps - _tol)[0]
    return None if len(idxs) == 0 else int(idxs[0])


def metrics_at_operating_point(r, thr, acc, nag, sar, err, eds, erg, eps, _tol=1e-12):
    """
    Read PVR, SAR, ERR, NAG, ERG at the same operating index r*.
    If no feasible, return values 0.0 and idx=None.
    """
    idx = operating_index_by_risk(acc, eps, _tol)
    if idx is None:
        return {
            f"idx@{int(100 * eps)}": None,
            f"r@{int(100 * eps)}": 0.0,
            f"theta@{int(100 * eps)}": 0.0,
            f"PVR@{int(100 * eps)}": 0.0,
            f"SAR@{int(100 * eps)}": 0.0,
            f"ERR@{int(100 * eps)}": 0.0,
            f"NAG@{int(100 * eps)}": 0.0,
            f"EDS@{int(100 * eps)}": 0.0,
            f"ERG@{int(100 * eps)}": 0.0,
        }
    return {
        f"idx@{int(100 * eps)}": idx,
        f"r@{int(100 * eps)}": float(r[idx]),
        f"theta@{int(100 * eps)}": float(thr[idx]),
        f"PVR@{int(100 * eps)}": float(1.0 - r[idx]),
        f"SAR@{int(100 * eps)}": float(sar[idx]),
        f"ERR@{int(100 * eps)}": float(err[idx]),
        f"NAG@{int(100 * eps)}": float(nag[idx]),
        f"EDS@{int(100 * eps)}": float(eds[idx]),
        f"ERG@{int(100 * eps)}": float(erg[idx]),
    }

def metrics_at_max_erg(r, thr, acc, nag, sar, err, eds, erg, eps, _tol=1e-12):
    """
    Read PVR, SAR, ERR, NAG, ERG at the same operating index r*.
    If no feasible, return values 0.0 and idx=None.
    """
    max_erg, idx = erg_max_at_epsilon(acc, erg, eps, _tol=_tol)
    if idx is None:
        return {
            f"idx@{int(100 * eps)}": None,
            f"r@{int(100 * eps)}": 0.0,
            f"theta@{int(100 * eps)}": 0.0,
            f"PVR@{int(100 * eps)}": 0.0,
            f"SAR@{int(100 * eps)}": 0.0,
            f"ERR@{int(100 * eps)}": 0.0,
            f"ACC@{int(100 * eps)}": 0.0,
            f"NAG@{int(100 * eps)}": 0.0,
            f"EDS@{int(100 * eps)}": 0.0,
            f"ERG@{int(100 * eps)}": 0.0,
        }
    return {
        f"idx@{int(100 * eps)}": idx,
        f"r@{int(100 * eps)}": float(r[idx]),
        f"theta@{int(100 * eps)}": float(thr[idx]),
        f"PVR@{int(100 * eps)}": float(1.0 - r[idx]),
        f"SAR@{int(100 * eps)}": float(sar[idx]),
        f"ERR@{int(100 * eps)}": float(err[idx]),
        f"ACC@{int(100 * eps)}": float(acc[idx]),
        f"NAG@{int(100 * eps)}": float(nag[idx]),
        f"EDS@{int(100 * eps)}": float(eds[idx]),
        f"ERG@{int(100 * eps)}": float(erg[idx]),
    }

def max_within_epsilon(acc, y, eps, prefer_left=True, _tol=1e-12):
    """
    Maximum of y in the mask acc >= 1-eps-tol.
    Tie-break to the left-most index if prefer_left=True.
    Returns (max_value, idx) or (0.0, None) if no feasible.
    """
    acc = np.asarray(acc)
    y = np.asarray(y)
    mask = acc >= 1.0 - eps - _tol
    if not np.any(mask):
        return 0.0, None
    masked = np.where(mask, y, -np.inf)
    idx = int(np.argmax(masked))
    # Tie-break: move left if there are other equals
    if prefer_left:
        v = masked[idx]
        # search for the first feasible index with that same value
        idxs = np.where((y == v) & mask)[0]
        if len(idxs) > 0:
            idx = int(idxs[0])
    return float(y[idx]), idx

def sar_max_at_epsilon(acc, sar, eps, _tol=1e-12):
    return max_within_epsilon(acc, sar, eps, _tol=_tol)

def err_max_at_epsilon(acc, err, eps, _tol=1e-12):
    return max_within_epsilon(acc, err, eps, _tol=_tol)

def erg_max_at_epsilon(acc, erg, eps, _tol=1e-12):
    return max_within_epsilon(acc, erg, eps, _tol=_tol)

#############

def compute_acc_f1_at_epsilon(acc: np.ndarray, acc_f1: np.ndarray, epsilon: float) -> float:
    """
    Accuracy-F1@ε: Max F1 (between accuracy and value retention)
    such that success accuracy ≥ 1 - ε.
    """
    mask = acc >= 1 - epsilon
    return np.max(acc_f1[mask]) if np.any(mask) else 0.0


def compute_ed_f1_at_epsilon(acc: np.ndarray, ed_f1: np.ndarray, epsilon: float) -> float:
    """
    ED-F1@ε: Max F1 (between error discrimination and value retention)
    such that success accuracy ≥ 1 - ε.
    """
    mask = acc >= 1 - epsilon
    return np.max(ed_f1[mask]) if np.any(mask) else 0.0
