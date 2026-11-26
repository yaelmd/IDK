import pandas as pd

def apply_avoidance_by_confidence(
    df: pd.DataFrame,
    confidence_col: str = "confidence",
    outcome_col: str = "outcome",
    threshold: float = 90.0
) -> pd.DataFrame:
    """
    If confidence < threshold, set outcome to 'AVOIDANCE'.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    confidence_col : str
        Name of the confidence column.
    outcome_col : str
        Name of the outcome column.
    threshold : float
        Confidence threshold below which outcome becomes 'AVOIDANCE'.

    Returns
    -------
    pd.DataFrame
        Modified copy of the dataset.
    """
    df = df.copy()
    df.loc[df[confidence_col] < threshold, outcome_col] = "AVOIDANCE"
    return df

import os
import ast
import glob
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier


# -----------------------
# Default input paths (overridable through the CLI)
# -----------------------
SCRIPT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = str(SCRIPT_ROOT / "Datasets")
EMBEDDINGS_ROOT = str(SCRIPT_ROOT / "Embeddings")
TASK_DIFFICULTY_ROOT = str(SCRIPT_ROOT / "Difficulty")
ADELE_CSV_PATH = str(SCRIPT_ROOT / "idk_annotations.csv")
ANAGRAM_IDS_PATH = str(SCRIPT_ROOT / "ids_anagram.csv")
DEFAULT_TASKS = ["geometry", "logic_puzzles", "algebra", "arithmetic",
                 "combinatory", "number_theory", "addition", "locality",
                 "anagram", "science"]
DEFAULT_PROMPTS = ["CoT", "multi_step", "self_probing", "top_k", "vanilla"]
CALIBRATED_TASKS = {"geometry", "logic_puzzles", "algebra", "arithmetic",
                    "combinatory", "number_theory"}
TASK_DIFFICULTY_COLUMNS = {"addition": "carry", "anagram": "letters",
                           "locality": "city_frequency", "science": "difficulty"}


RESULTS_ROOT = "assessor_results_ood"


FEATURE_COMBOS = {
    "emb": ("emb",),
    "diff": ("diff",),
    "adele": ("adele",),
    "emb+diff": ("emb", "diff"),
    "emb+adele": ("emb", "adele"),
    "diff+adele": ("diff", "adele"),
    "emb+diff+adele": ("emb","diff","adele"),
}

def normalize_combo_name(combo: str) -> str:
    combo = combo.strip().lower().replace(" ", "")
    combo = combo.replace("difficulty", "diff")
    combo = combo.replace("taskdiff", "diff")
    combo = combo.replace("task_difficulty", "diff")
    combo = combo.replace("embeddings", "emb")

    # Canonicalize ordering if user writes "diff+emb" etc.
    parts = combo.split("+")
    parts = [p for p in parts if p]

    allowed = {"emb", "diff", "adele"}
    if not set(parts).issubset(allowed):
        return combo

    order = {"emb": 0, "diff": 1, "adele": 2}
    parts = sorted(set(parts), key=lambda x: order[x])

    return "+".join(parts)


def embeddings_path_for_task(task_name: str) -> str:
    return os.path.join(EMBEDDINGS_ROOT, f"{task_name}_embedded_text.csv")

def load_embeddings_for_task(task: str) -> pd.DataFrame:
    path = embeddings_path_for_task(task)
    df = pd.read_csv(path)
    if "instance_id" not in df.columns or "embedding" not in df.columns:
        raise ValueError(f"Embeddings file must have columns ['instance_id','embedding']: {path}")
    return df[["instance_id", "embedding"]].copy()

def load_task_difficulty_for_task(task: str, difficulty_type: str = "original") -> pd.DataFrame:
    path = os.path.join(TASK_DIFFICULTY_ROOT, f"{task}_difficulty.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing task difficulty file for '{task}': {path}")

    df = pd.read_csv(path)

    if "instance_id" not in df.columns:
        raise ValueError(f"Task difficulty CSV must contain 'instance_id': {path}")

    if difficulty_type not in {"original", "calibrated"}:
        raise ValueError(f"Unknown difficulty type: {difficulty_type}")
    if difficulty_type == "calibrated" and (task in CALIBRATED_TASKS or "cal_diff" in df.columns):
        diff_col = "cal_diff"
    elif task in CALIBRATED_TASKS:
        diff_col = "diff" if "diff" in df.columns else "difficulty"
    else:
        diff_col = TASK_DIFFICULTY_COLUMNS.get(task, "difficulty")
    if diff_col not in df.columns:
        raise ValueError(f"Missing {difficulty_type} difficulty column '{diff_col}' for '{task}': {path}")

    out = df[["instance_id", diff_col]].copy()
    out = out.rename(columns={diff_col: "task_difficulty"})
    out["task_difficulty"] = pd.to_numeric(out["task_difficulty"], errors="coerce")
    out.attrs["difficulty_column"] = diff_col
    out.attrs["difficulty_type"] = difficulty_type
    return out

def load_adele_for_task(task: str, adele_df_all: pd.DataFrame) -> pd.DataFrame:
    required = {"benchmark", "instance_id"}
    if not required.issubset(set(adele_df_all.columns)):
        raise ValueError(f"ADELE CSV must contain columns {required}. Found: {list(adele_df_all.columns)}")

    if task == "science":
        if "custom_id" not in adele_df_all.columns:
            raise ValueError("Science ADELE annotations must contain 'custom_id'.")
        science_ids = pd.read_csv(
            embeddings_path_for_task(task), usecols=["instance_id", "ori_id"]
        )
        science_ids["custom_id"] = (
            science_ids.pop("ori_id")
            .str.replace("_", "-", regex=False)
            .str.replace("obqa", "openbookqa", regex=False)
        )
        df = science_ids.merge(
            adele_df_all.drop(columns=["instance_id"]),
            on="custom_id", how="left", validate="one_to_one",
        )
        if df["benchmark"].isna().all():
            raise ValueError("No science ADELE annotations matched the original IDs.")
    else:
        df = adele_df_all[adele_df_all["benchmark"].astype(str) == str(task)].copy()


    df = df.drop(columns=["benchmark", "custom_id"])
    if "instance_id" not in df.columns:
        raise ValueError("ADELE filtered df is missing instance_id unexpectedly.")

    feature_cols = [c for c in df.columns if c != "instance_id"]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df 


def exclude_ambiguous_anagrams(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if task != "anagram":
        return df.copy()
    flags = pd.read_csv(ANAGRAM_IDS_PATH)
    excluded = flags.loc[flags["multiple_anagrams"].eq(1), "instance_id"]
    return df.loc[~df["instance_id"].isin(excluded)].copy()


def parse_embedding(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    return np.asarray(value, dtype=np.float32)


def build_feature_matrix(
    prompt_df: pd.DataFrame,
    task: str,
    feature_families: tuple,
    embeddings_df: pd.DataFrame | None = None,
    difficulty_df: pd.DataFrame | None = None,
    adele_df_task: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns:
      X: float32 numpy matrix
      y: string labels array
      merged: merged dataframe aligned with X/y
    """
    merged = prompt_df.copy()


    if "instance_id" not in merged.columns:
        raise ValueError("prompt_df must contain 'instance_id'")
    if "outcome" not in merged.columns:
        raise ValueError("prompt_df must contain 'outcome'")


    merged = exclude_ambiguous_anagrams(merged, task)


    if "emb" in feature_families:
        if embeddings_df is None:
            raise ValueError("embeddings_df is required when using embeddings.")
        merged = merged.merge(embeddings_df[["instance_id", "embedding"]], on="instance_id", how="inner")

    if "diff" in feature_families:
        if difficulty_df is None:
            raise ValueError("difficulty_df is required when using task difficulty.")
        merged = merged.merge(difficulty_df[["instance_id", "task_difficulty"]], on="instance_id", how="inner")

    if "adele" in feature_families:
        if adele_df_task is None:
            raise ValueError("adele_df_task is required when using ADELE.")
        merged = merged.merge(adele_df_task, on="instance_id", how="inner")

    merged = merged.sort_values("instance_id").reset_index(drop=True)

    blocks = []

    if "emb" in feature_families:
        if merged["embedding"].isna().any():
            raise ValueError(f"Missing embeddings after merge for task '{task}'. Check instance_id alignment.")
        X_emb = np.stack(merged["embedding"].apply(parse_embedding).tolist())
        blocks.append(X_emb)

    if "diff" in feature_families:
        x = merged["task_difficulty"].astype(np.float32).to_numpy().reshape(-1, 1)
        blocks.append(x)

    if "adele" in feature_families:
        if adele_df_task is None:
            raise ValueError("adele_df_task is required when using ADELE.")

        adele_feature_cols = [
            c for c in adele_df_task.columns
            if c != "instance_id" and pd.api.types.is_numeric_dtype(adele_df_task[c])
        ]

        if len(adele_feature_cols) == 0:
            raise ValueError(f"No numeric ADELE features found for task '{task}'.")

        X_adele = merged[adele_feature_cols].to_numpy(dtype=np.float32, copy=True)

        if np.isnan(X_adele).any():
            col_means = np.nanmean(X_adele, axis=0)
            inds = np.where(np.isnan(X_adele))
            X_adele[inds] = np.take(col_means, inds[1])

        blocks.append(X_adele)


    if not blocks:
        raise ValueError("No feature families selected.")

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    y = merged["outcome"].astype(str).values
    return X, y, merged

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
import ast
import json
import os
from datetime import datetime

# -------------------------------
# Metrics
# -------------------------------
def calculate_all_metrics(results_df):
    metrics = {}

    metrics['full_classification_report'] = classification_report(
        results_df['CV_True_Label'],
        results_df['CV_Predicted_Label'],
        labels=['INCORRECT', 'CORRECT', 'AVOIDANCE'],
        target_names=['INCORRECT', 'CORRECT', 'AVOIDANCE'],
        output_dict=True,
        zero_division=0
    )

    metrics['overall_accuracy'] = float(np.mean(results_df['CV_True_Label'] == results_df['CV_Predicted_Label']))

    made_definitive = (results_df['CV_Predicted_Label'] != 'AVOIDANCE')
    true_def = results_df['CV_True_Label'][made_definitive]
    pred_def = results_df['CV_Predicted_Label'][made_definitive]

    metrics['conditional_analysis'] = {
        'definitive_predictions_count': int(len(true_def)),
        'total_instances': int(len(results_df)),
        'abstention_rate': float(1 - (len(true_def) / len(results_df))),
        'conditional_accuracy': float(np.mean(true_def == pred_def)) if len(true_def) > 0 else 0.0
    }

    both_def_mask = ((results_df['CV_Predicted_Label'] != 'AVOIDANCE') &
                     (results_df['CV_True_Label']      != 'AVOIDANCE'))
    true_both = results_df['CV_True_Label'][both_def_mask]
    pred_both = results_df['CV_Predicted_Label'][both_def_mask]

    metrics['mutually_definitive_analysis'] = {
        'mutually_definitive_count': int(len(true_both)),
        'mutually_definitive_accuracy': float(np.mean(true_both == pred_both)) if len(true_both) > 0 else 0.0
    }

    is_correct_actual = (results_df['CV_True_Label'] == 'CORRECT')
    prob_correct_pred = results_df['CV_Predicted_Class_1'].astype(float)
    binary_true = is_correct_actual.astype(int)

    if len(np.unique(binary_true)) > 1:
        metrics['binary_metrics'] = {
            'roc_auc': float(roc_auc_score(binary_true, prob_correct_pred)),
            'brier_score': float(brier_score_loss(binary_true, prob_correct_pred))
        }
    else:
        metrics['binary_metrics'] = {'roc_auc': float('nan'), 'brier_score': float('nan')}

    return metrics

# -------------------------------
# Save / Load / Print helpers
# -------------------------------
def save_results(results_dict, save_dir='assessor_results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    prompt_type = results_dict['prompt_type']
    timestamp = results_dict['timestamp'].replace(':', '-').replace(' ', '_')
    base = f"{prompt_type}_{timestamp}"

    # Metrics JSON
    metrics_path = os.path.join(save_dir, f"{base}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results_dict['metrics'], f, indent=2)

    # Results CSV
    df_path = os.path.join(save_dir, f"{base}_results.csv")
    results_dict['results_df'].drop(columns=["embedding"], errors="ignore").to_csv(df_path, index=False)

    print(f"Saved for {prompt_type}:")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Data:    {df_path}")
    return metrics_path, df_path

def print_detailed_metrics(metrics):
    print(f"\n{'='*60}")
    print("DETAILED PERFORMANCE METRICS")
    print(f"{'='*60}")

    print(f"\n1. Overall Accuracy: {metrics['overall_accuracy']:.4f}")

    cond = metrics['conditional_analysis']
    print(f"\n2. Conditional Analysis (excluding 'AVOIDANCE' predictions):")
    print(f"   Definitive predictions: {cond['definitive_predictions_count']} / {cond['total_instances']}")
    print(f"   Abstention Rate: {cond['abstention_rate']:.2%}")
    print(f"   Conditional Accuracy: {cond['conditional_accuracy']:.4f}")

    mutual = metrics['mutually_definitive_analysis']
    print(f"\n3. Mutually Definitive Analysis:")
    print(f"   Both definitive cases: {mutual['mutually_definitive_count']}")
    print(f"   Accuracy on mutually definitive: {mutual['mutually_definitive_accuracy']:.4f}")

    if 'binary_metrics' in metrics:
        bm = metrics['binary_metrics']
        ra = bm.get('roc_auc', np.nan)
        bs = bm.get('brier_score', np.nan)
        if isinstance(ra, float) and isinstance(bs, float):
            print(f"\n4. Binary Metrics ('CORRECT' vs 'NOT CORRECT'):")
            print(f"   ROC-AUC: {ra:.4f}")
            print(f"   Brier :  {bs:.4f}")



def train_assessor_cv_from_X(
    merged_df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    prompt_name: str,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int | None = None,
):
    print(f"\n{'='*60}")
    print(f"TRAINING ASSESSOR FOR PROMPT: {prompt_name}")
    print(f"{'='*60}")

    if len(merged_df) == 0:
        print(f"No data for prompt: {prompt_name}")
        return None

    print(f"Data size for {prompt_name}: {len(merged_df)} instances")

    # Fixed global class order
    class_order = ['INCORRECT', 'CORRECT', 'AVOIDANCE']
    present_classes = [c for c in class_order if c in set(y)]
    if len(present_classes) < 2:
        print(f"Only one class present for {prompt_name}: {present_classes}. Cannot train a classifier.")
        return None

    print(f"Classes present in {prompt_name}: {present_classes}")

    label_mapping = {lbl: i for i, lbl in enumerate(present_classes)}
    y_numeric = np.array([label_mapping[lbl] for lbl in y], dtype=int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_test_predictions = np.zeros((len(merged_df), 3), dtype=np.float32)
    all_test_labels = np.empty(len(merged_df), dtype=object)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_numeric)):
        print(f"Training on Fold {fold+1}/{n_splits}...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train_global = y_numeric[train_idx]
        y_test_str = y[test_idx]

        fold_global_classes = sorted(np.unique(y_train_global).tolist())
        fold_map = {g: i for i, g in enumerate(fold_global_classes)}
        inv_fold_map = {i: g for g, i in fold_map.items()}
        y_train_fold = np.array([fold_map[g] for g in y_train_global], dtype=int)

        ros = RandomOverSampler(random_state=fold)
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train_fold)

        K = len(fold_global_classes)
        if K == 3:
            model = XGBClassifier(
                device="cpu",
                objective="multi:softprob",
                num_class=3,
                n_jobs=n_jobs,
                random_state=random_state,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="mlogloss"
            )
        elif K == 2:
            model = XGBClassifier(
                device="cpu",
                objective="binary:logistic",
                n_jobs=n_jobs,
                random_state=random_state,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss"
            )
        else:
            raise RuntimeError("Fold has <2 classes; cannot train")

        model.fit(X_train_res, y_train_res)
        y_pred_proba = model.predict_proba(X_test)

        probs_full = np.zeros((len(test_idx), 3), dtype=np.float32)
        for col_idx, fold_local_class in enumerate(model.classes_):
            global_numeric = inv_fold_map[int(fold_local_class)]  # back to global numeric (0/1/2)
            global_col = class_order.index(present_classes[global_numeric])
            probs_full[:, global_col] = y_pred_proba[:, col_idx]

        all_test_predictions[test_idx] = probs_full
        all_test_labels[test_idx] = y_test_str

    print("Cross-Validation complete!")

    results_df = merged_df.copy()
    results_df['CV_Predicted_Class_0']   = all_test_predictions[:, 0]  # INCORRECT
    results_df['CV_Predicted_Class_1']   = all_test_predictions[:, 1]  # CORRECT
    results_df['CV_Predicted_Class_IDK'] = all_test_predictions[:, 2]  # AVOIDANCE
    results_df['CV_True_Label']          = all_test_labels

    pred_numeric = np.argmax(all_test_predictions, axis=1)
    reverse_global = {0: 'INCORRECT', 1: 'CORRECT', 2: 'AVOIDANCE'}
    results_df['CV_Predicted_Label'] = [reverse_global[i] for i in pred_numeric]

    metrics = calculate_all_metrics(results_df)

    return {
        'prompt_type': prompt_name,
        'results_df': results_df,
        'metrics': metrics,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'data_size': len(results_df),
        'n_splits': n_splits,
        'present_classes': present_classes
    }


def list_models(datasets_root: str):
    return sorted(
        d for d in os.listdir(datasets_root)
        if os.path.isdir(os.path.join(datasets_root, d))
    )

def load_prompt_dfs_for_task(datasets_root: str, model: str, task: str, prompts_to_run=None) -> dict:
    task_dir = os.path.join(datasets_root, model, task)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"Missing task folder: {task_dir}")
    prompt_names = DEFAULT_PROMPTS if prompts_to_run is None else prompts_to_run
    if isinstance(prompt_names, str):
        prompt_names = [prompt_names]
    out = {}
    for prompt_name in prompt_names:
        fp = os.path.join(task_dir, f"{prompt_name}.csv")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Missing requested prompt: {fp}")
        out[prompt_name] = pd.read_csv(fp)
    return out


def save_training_log(runs, results_root, mode):
    """Persist and print the successfully completed training runs."""
    os.makedirs(results_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    log_path = os.path.join(results_root, f"{mode}_{timestamp}_training_log.json")
    with open(log_path, "w") as f:
        json.dump({"mode": mode, "completed_runs": runs}, f, indent=2)
    print(f"\n{mode.upper()} complete: {len(runs)} assessor runs trained and saved.")
    for run in runs:
        print(
            f"  Model={run['model']} | features={run['feature_combo']} | "
            f"prompt={run['prompt']} | trained on={run['training_tasks']} | "
            f"evaluated on={run['evaluation_tasks']}"
        )
    print(f"Training log: {log_path}")


def run_assessors_for_tasks(
    tasks,
    feature_combo: str | list[str],
    n_splits: int = 5,
    random_state: int = 42,
    model: str | list[str] | None = None,
    models_to_run: str | list[str] | None = None,
    prompts_to_run: str | list[str] | None = None,
    results_root: str | None = None,
    confidence_threshold: float = 90.0,
    n_jobs: int | None = None,
    difficulty_type: str = "original",
):
    """Run IID task CV for selected models and prompt names (CSV stems).

    Each filter accepts a name, a list of names, or None to select all.
    The legacy model argument remains supported; use only one model filter.
    """
    if model is not None and models_to_run is not None:
        raise ValueError("Use either model or models_to_run, not both.")
    selected_models = models_to_run if models_to_run is not None else model
    results_root = RESULTS_ROOT if results_root is None else results_root
    if isinstance(prompts_to_run, str):
        prompts_to_run = [prompts_to_run]

    if isinstance(tasks, str):
        tasks = [tasks]

    if isinstance(feature_combo, str):
        combos = [feature_combo]
    else:
        combos = feature_combo

    combos = [normalize_combo_name(c) for c in combos]

    if len(combos) == 1 and combos[0] == "all":
        combos = list(FEATURE_COMBOS.keys())


    models = list_models(DATASETS_ROOT)
    if not models:
        raise FileNotFoundError(f"No model folders found in: {DATASETS_ROOT}")

    if selected_models is not None:
        requested_models = [selected_models] if isinstance(selected_models, str) else list(selected_models)
        unknown_models = sorted(set(requested_models) - set(models))
        if unknown_models:
            raise ValueError(f"Unknown models: {unknown_models}. Available: {models}")
        models = [name for name in models if name in requested_models]

    completed_runs = []


    adele_all = pd.read_csv(ADELE_CSV_PATH) if any("adele" in FEATURE_COMBOS.get(c, ()) for c in combos) else None

    print("Models discovered:", models)
    print("Tasks requested:", tasks)
    print("Feature combos requested:", combos)
    print("Prompt filter:", prompts_to_run)

    for combo_name in combos:
        if combo_name not in FEATURE_COMBOS:
            raise ValueError(f"Unknown feature_combo '{combo_name}'. Allowed: {list(FEATURE_COMBOS.keys())}")

        families = FEATURE_COMBOS[combo_name]
        print(f"\n\n##############################")
        print(f"FEATURE COMBO: {combo_name} -> {families}")
        print(f"##############################")

        for task in tasks:
            print(f"\n==============================")
            print(f"TASK: {task}")
            print(f"==============================")

            # Load only the sources needed for this combo
            embeddings_df = load_embeddings_for_task(task) if "emb" in families else None
            difficulty_df = load_task_difficulty_for_task(task, difficulty_type) if "diff" in families else None
            adele_task_df = load_adele_for_task(task, adele_all) if "adele" in families else None

            for model in models:
                task_dir = os.path.join(DATASETS_ROOT, model, task)
                prompt_dfs = load_prompt_dfs_for_task(DATASETS_ROOT, model, task, prompts_to_run)

                prompt_dfs = {name: apply_avoidance_by_confidence(df, threshold=confidence_threshold) for name, df in prompt_dfs.items()}

                print(f"\n--- Model: {model} | prompts: {list(prompt_dfs.keys())} ---")

                save_dir = os.path.join(results_root, combo_name, task, model)
                os.makedirs(save_dir, exist_ok=True)

                for prompt_name, df_prompt in prompt_dfs.items():
                    try:
                        # Build X using requested families
                        X, y, merged = build_feature_matrix(
                            prompt_df=df_prompt,
                            task=task,
                            feature_families=families,
                            embeddings_df=embeddings_df,
                            difficulty_df=difficulty_df,
                            adele_df_task=adele_task_df,
                        )

                        # Train + CV
                        res = train_assessor_cv_from_X(
                            merged_df=merged,
                            X=X,
                            y=y,
                            prompt_name=prompt_name,
                            n_splits=n_splits,
                            random_state=random_state,
                            n_jobs=n_jobs,
                        )

                        if res:
                            metrics_path, results_path = save_results(res, save_dir=save_dir)
                            completed_runs.append({
                                "model": model,
                                "feature_combo": combo_name,
                                "prompt": prompt_name,
                                "training_tasks": [task],
                                "evaluation_tasks": [task],
                                "data_size": res["data_size"],
                                "n_splits": n_splits,
                                "metrics_path": metrics_path,
                                "results_path": results_path,
                            })
                            print_detailed_metrics(res["metrics"])
                        else:
                            raise RuntimeError(f"Could not train {combo_name}/{task}/{model}/{prompt_name}")

                    except Exception as e:
                        print(f"Failed assessor for {combo_name}/{task}/{model}/{prompt_name}: {e}")
                        save_training_log(completed_runs, results_root, "cv")
                        raise

    save_training_log(completed_runs, results_root, "cv")
    return completed_runs


def train_assessor_ood_train_test(
    train_merged_df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    test_merged_df: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    prompt_name: str,
    random_state: int = 42,
    n_jobs: int | None = None,
):
    print(f"\n{'='*60}")
    print(f"OOD TRAINING ASSESSOR FOR PROMPT: {prompt_name}")
    print(f"{'='*60}")

    if len(train_merged_df) == 0:
        print("Empty train set.")
        return None
    if len(test_merged_df) == 0:
        print("Empty test set.")
        return None

    print(f"Train size: {len(train_merged_df)}")
    print(f"Test size : {len(test_merged_df)}")

    class_order = ['INCORRECT', 'CORRECT', 'AVOIDANCE']
    present_train_classes = [c for c in class_order if c in set(y_train)]

    if len(present_train_classes) < 2:
        print(f"Only one class in training set: {present_train_classes}. Cannot train.")
        return None

    print(f"Classes present in TRAIN: {present_train_classes}")
    print(f"Classes present in TEST : {[c for c in class_order if c in set(y_test)]}")

    label_mapping = {lbl: i for i, lbl in enumerate(present_train_classes)}
    reverse_mapping = {i: lbl for lbl, i in label_mapping.items()}

    y_train_numeric = np.array([label_mapping[lbl] for lbl in y_train], dtype=int)

    ros = RandomOverSampler(random_state=random_state)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train_numeric)

    K = len(present_train_classes)

    if K == 3:
        model = XGBClassifier(
            device="cpu",
            objective="multi:softprob",
            num_class=3,
            n_jobs=n_jobs,
            random_state=random_state,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss"
        )
    elif K == 2:
        model = XGBClassifier(
            device="cpu",
            objective="binary:logistic",
            n_jobs=n_jobs,
            random_state=random_state,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss"
        )
    else:
        raise RuntimeError("Training set has <2 classes; cannot train")

    model.fit(X_train_res, y_train_res)

    y_pred_proba = model.predict_proba(X_test)

    probs_full = np.zeros((len(test_merged_df), 3), dtype=np.float32)

    for col_idx, local_class in enumerate(model.classes_):
        label_name = reverse_mapping[int(local_class)] 
        global_col = class_order.index(label_name)  
        probs_full[:, global_col] = y_pred_proba[:, col_idx]

    pred_numeric = np.argmax(probs_full, axis=1)
    reverse_global = {0: 'INCORRECT', 1: 'CORRECT', 2: 'AVOIDANCE'}
    pred_labels = np.array([reverse_global[i] for i in pred_numeric], dtype=object)

    results_df = test_merged_df.copy()
    results_df['OOD_Predicted_Class_0']   = probs_full[:, 0]
    results_df['OOD_Predicted_Class_1']   = probs_full[:, 1]
    results_df['OOD_Predicted_Class_IDK'] = probs_full[:, 2]
    results_df['OOD_True_Label']          = y_test
    results_df['OOD_Predicted_Label']     = pred_labels

    metrics_input = results_df.rename(columns={
        'OOD_True_Label': 'CV_True_Label',
        'OOD_Predicted_Label': 'CV_Predicted_Label',
        'OOD_Predicted_Class_0': 'CV_Predicted_Class_0',
        'OOD_Predicted_Class_1': 'CV_Predicted_Class_1',
        'OOD_Predicted_Class_IDK': 'CV_Predicted_Class_IDK',
    }).copy()

    metrics = calculate_all_metrics(metrics_input)

    return {
        'prompt_type': prompt_name,
        'results_df': results_df,
        'metrics': metrics,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'train_size': len(train_merged_df),
        'test_size': len(test_merged_df),
        'train_classes': present_train_classes
    }

def save_ood_results(results_dict, save_dir='assessor_results_ood'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    prompt_type = results_dict['prompt_type']
    timestamp = results_dict['timestamp'].replace(':', '-').replace(' ', '_')
    base = f"{prompt_type}_{timestamp}"

    metrics_path = os.path.join(save_dir, f"{base}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results_dict['metrics'], f, indent=2)

    df_path = os.path.join(save_dir, f"{base}_results.csv")
    results_dict['results_df'].drop(columns=["embedding"], errors="ignore").to_csv(df_path, index=False)

    print(f"Saved OOD for {prompt_type}:")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Data:    {df_path}")
    return metrics_path, df_path

def ood_result_already_exists(save_dir: str, prompt_name: str) -> bool:
    pattern = os.path.join(save_dir, f"{prompt_name}_*_results.csv")
    return len(glob.glob(pattern)) > 0

def run_assessors_leave_one_task_out(
    all_tasks,
    feature_combo: str | list[str],
    tasks_to_run=None,
    models_to_run=None,
    prompts_to_run=None,
    random_state: int = 42,
    results_root: str = "assessor_results_ood",
    confidence_threshold: float = 90.0,
    n_jobs: int | None = None,
    difficulty_type: str = "calibrated",
):
    if isinstance(all_tasks, str):
        all_tasks = [all_tasks]

    if tasks_to_run is None:
        tasks_to_run = list(all_tasks)
    elif isinstance(tasks_to_run, str):
        tasks_to_run = [tasks_to_run]

    if isinstance(feature_combo, str):
        combos = [feature_combo]
    else:
        combos = feature_combo

    combos = [normalize_combo_name(c) for c in combos]

    if len(combos) == 1 and combos[0] == "all":
        combos = list(FEATURE_COMBOS.keys())

    models = list_models(DATASETS_ROOT)
    if not models:
        raise FileNotFoundError(f"No model folders found in: {DATASETS_ROOT}")

    if models_to_run is not None:
        if isinstance(models_to_run, str):
            models_to_run = [models_to_run]
        unknown_models = sorted(set(models_to_run) - set(models))
        if unknown_models:
            raise ValueError(f"Unknown models: {unknown_models}. Available: {models}")
        models = [m for m in models if m in models_to_run]

    adele_all = pd.read_csv(ADELE_CSV_PATH) if any("adele" in FEATURE_COMBOS.get(c, ()) for c in combos) else None

    print("Models discovered:", models)
    completed_runs = []

    print("All tasks:", all_tasks)
    print("Held-out tasks to run now:", tasks_to_run)
    print("Feature combos requested:", combos)
    print("Prompt filter:", prompts_to_run)

    for combo_name in combos:
        if combo_name not in FEATURE_COMBOS:
            raise ValueError(f"Unknown feature_combo '{combo_name}'. Allowed: {list(FEATURE_COMBOS.keys())}")

        families = FEATURE_COMBOS[combo_name]

        for held_out_task in tasks_to_run:
            if held_out_task not in all_tasks:
                raise ValueError(f"tasks_to_run contains '{held_out_task}', which is not in all_tasks.")

            train_tasks = [t for t in all_tasks if t != held_out_task]

            embeddings_by_task = {}
            difficulty_by_task = {}
            adele_by_task = {}

            for task in all_tasks:
                if "emb" in families:
                    embeddings_by_task[task] = load_embeddings_for_task(task)
                if "diff" in families:
                    difficulty_by_task[task] = load_task_difficulty_for_task(task, difficulty_type)
                if "adele" in families:
                    adele_by_task[task] = load_adele_for_task(task, adele_all)

            for model in models:
                print(f"\n--- Model: {model} ---")

                held_out_prompt_dfs = load_prompt_dfs_for_task(DATASETS_ROOT, model, held_out_task, prompts_to_run)

                prompt_names = sorted(held_out_prompt_dfs.keys())

                if prompts_to_run is not None:
                    if isinstance(prompts_to_run, str):
                        prompts_to_run = [prompts_to_run]
                    prompt_names = [p for p in prompt_names if p in prompts_to_run]

                print(f"Prompts to run: {prompt_names}")

                save_dir = os.path.join(results_root, combo_name, f"leave_out_{held_out_task}", model)
                os.makedirs(save_dir, exist_ok=True)

                for prompt_name in prompt_names:
                    try:
                        if ood_result_already_exists(save_dir, prompt_name):
                            print(f"[Skip existing] {combo_name} | leave_out={held_out_task} | {model} | {prompt_name}")
                            continue

                        # -----------------------
                        # TRAIN set
                        # -----------------------
                        train_merged_parts = []
                        train_X_parts = []
                        train_y_parts = []

                        for train_task in train_tasks:
                            task_prompt_dfs = load_prompt_dfs_for_task(DATASETS_ROOT, model, train_task, [prompt_name])

                            df_prompt = apply_avoidance_by_confidence(task_prompt_dfs[prompt_name], threshold=confidence_threshold)

                            X_part, y_part, merged_part = build_feature_matrix(
                                prompt_df=df_prompt,
                                task=train_task,
                                feature_families=families,
                                embeddings_df=embeddings_by_task.get(train_task),
                                difficulty_df=difficulty_by_task.get(train_task),
                                adele_df_task=adele_by_task.get(train_task),
                            )

                            merged_part = merged_part.copy()
                            merged_part["source_task"] = train_task

                            train_merged_parts.append(merged_part)
                            train_X_parts.append(X_part)
                            train_y_parts.append(y_part)

                        if len(train_X_parts) == 0:
                            raise ValueError(f"No training data for prompt '{prompt_name}' with held-out task '{held_out_task}'")

                        train_merged_df = pd.concat(train_merged_parts, ignore_index=True)
                        X_train = np.concatenate(train_X_parts, axis=0).astype(np.float32)
                        y_train = np.concatenate(train_y_parts, axis=0)

                        # -----------------------
                        # TEST set
                        # -----------------------
                        df_test_prompt = apply_avoidance_by_confidence(held_out_prompt_dfs[prompt_name], threshold=confidence_threshold)

                        X_test, y_test, test_merged_df = build_feature_matrix(
                            prompt_df=df_test_prompt,
                            task=held_out_task,
                            feature_families=families,
                            embeddings_df=embeddings_by_task.get(held_out_task),
                            difficulty_df=difficulty_by_task.get(held_out_task),
                            adele_df_task=adele_by_task.get(held_out_task),
                        )

                        test_merged_df = test_merged_df.copy()
                        test_merged_df["source_task"] = held_out_task

                        # -----------------------
                        # Train on train tasks, evaluate on held-out task
                        # -----------------------
                        res = train_assessor_ood_train_test(
                            train_merged_df=train_merged_df,
                            X_train=X_train,
                            y_train=y_train,
                            test_merged_df=test_merged_df,
                            X_test=X_test,
                            y_test=y_test,
                            prompt_name=prompt_name,
                            random_state=random_state,
                            n_jobs=n_jobs,
                        )

                        if res:
                            metrics_path, results_path = save_ood_results(res, save_dir=save_dir)
                            completed_runs.append({
                                "model": model,
                                "feature_combo": combo_name,
                                "prompt": prompt_name,
                                "training_tasks": train_merged_df["source_task"].drop_duplicates().tolist(),
                                "training_rows_by_task": {
                                    task: int(count) for task, count in
                                    train_merged_df["source_task"].value_counts().items()
                                },
                                "evaluation_tasks": [held_out_task],
                                "train_size": res["train_size"],
                                "test_size": res["test_size"],
                                "metrics_path": metrics_path,
                                "results_path": results_path,
                            })
                            print_detailed_metrics(res["metrics"])
                        else:
                            raise RuntimeError(f"Could not train OOD {combo_name}/{held_out_task}/{model}/{prompt_name}")

                    except Exception as e:
                        print(f"Failed OOD assessor for {combo_name} | leave_out={held_out_task} | {model} | {prompt_name}: {e}")
                        save_training_log(completed_runs, results_root, "ood")
                        raise

    save_training_log(completed_runs, results_root, "ood")
    return completed_runs



def build_argument_parser():
    parser = argparse.ArgumentParser(description="Train IID or OOD assessors.")
    parser.add_argument("--mode", choices=["ood", "iid"], required=True)
    parser.add_argument("--models", nargs="+", default=["all"], help="Exact model folder names, or all (default).")
    parser.add_argument("--all-tasks", nargs="+", choices=DEFAULT_TASKS, default=DEFAULT_TASKS,
                        help="Task universe for leave-one-task-out training.")
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS,
                        help="Held-out tasks for OOD; evaluated tasks for IID. Defaults to all tasks.")
    parser.add_argument("--prompts", nargs="+", choices=DEFAULT_PROMPTS, default=DEFAULT_PROMPTS)
    parser.add_argument("--feature-combos", nargs="+", default=["all"], help="all, or emb/diff/adele combinations.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of IID CV folds.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, help="CPU threads per XGBoost fit; set explicitly for parallel processes.")
    parser.add_argument("--confidence-threshold", type=float, default=90.0)
    parser.add_argument("--results-root", type=Path, required=True, help="Output directory.")
    parser.add_argument("--datasets-root", default=DATASETS_ROOT)
    parser.add_argument("--embeddings-root", default=EMBEDDINGS_ROOT)
    parser.add_argument("--difficulty-root", default=TASK_DIFFICULTY_ROOT)
    parser.add_argument("--difficulty-type", choices=["original", "calibrated"],
                        help="Defaults to calibrated for OOD and original for IID; selects cal_diff where supplied.")
    parser.add_argument("--adele-csv", default=ADELE_CSV_PATH)
    parser.add_argument("--anagram-ids", default=ANAGRAM_IDS_PATH)
    return parser


def main(argv=None):
    global DATASETS_ROOT, EMBEDDINGS_ROOT, TASK_DIFFICULTY_ROOT, ADELE_CSV_PATH, ANAGRAM_IDS_PATH
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    DATASETS_ROOT = args.datasets_root
    EMBEDDINGS_ROOT = args.embeddings_root
    TASK_DIFFICULTY_ROOT = args.difficulty_root
    ADELE_CSV_PATH = args.adele_csv
    ANAGRAM_IDS_PATH = args.anagram_ids
    models = list_models(DATASETS_ROOT) if args.models == ["all"] else args.models
    combos = list(FEATURE_COMBOS) if args.feature_combos == ["all"] else args.feature_combos
    tasks = args.tasks if args.tasks is not None else args.all_tasks
    difficulty_type = args.difficulty_type or ("calibrated" if args.mode == "ood" else "original")
    common = dict(
        feature_combo=combos, models_to_run=models, prompts_to_run=args.prompts,
        random_state=args.random_state, results_root=str(args.results_root),
        confidence_threshold=args.confidence_threshold, n_jobs=args.n_jobs,
        difficulty_type=difficulty_type,
    )
    if args.mode == "ood":
        run_assessors_leave_one_task_out(all_tasks=args.all_tasks, tasks_to_run=tasks, **common)
    else:
        run_assessors_for_tasks(tasks=tasks, n_splits=args.n_splits, **common)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
