import os
import json
from pathlib import Path
from statistics import mean
from collections import defaultdict
import csv

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

import json
import math
from statistics import mean

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, roc_curve # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import seaborn as sns # type: ignore

import argparse


# Set visible CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Semantic Energy Calculation"
    )
    parser.add_argument(
        "--model-path",
        default="llama3v3chat-70B",
        type=str,
        help="HF model ID(s) or local path(s) (default: %(default)s)",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Implemented task(s): addition",
    )
    parser.add_argument(
        "--n-responses",
        type=int,
        default=7,
        help="Number of responses to generate (default: %(default)s)",
    )

    return parser.parse_args()


# Helper function to load a JSONL file
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]
    

def sum_normalize(lst):
    """
    Normalize a list so that its elements sum to 1.
    If the total sum is zero, returns a list of zeros.
    """
    total = sum(lst)
    return [x / total if total != 0 else 0 for x in lst]


def cal_cluster_ce(probs, logits, clusters):
    """
    Calculate cluster-level cross entropy and summed logits.

    Args:
        probs (List[float]): List of probabilities for each item.
        logits (List[float]): List of logits for each item.
        clusters (List[List[int]]): Index clusters over which to aggregate values.

    Returns:
        Tuple:
            - probs_se: Summed normalized probabilities per cluster.
            - logits_se: Negative summed logits per cluster.
    """
    probs_se = []
    logits_se = []

    # Normalize probabilities once
    normalized_probs = sum_normalize(probs)


    for cluster in clusters:
        # Sum normalized probabilities within the cluster
        print(cluster)
        cluster_prob_sum = sum(normalized_probs[i] for i in cluster)
        probs_se.append(cluster_prob_sum)

        # Sum logits (negated) within the cluster
        cluster_logit_sum = -sum(logits[i] for i in cluster)
        logits_se.append(cluster_logit_sum)

    return probs_se, logits_se


def cal_probs(probs_list):
    """
    Calculate the product of probabilities for each item list.

    Args:
        probs_list (List[List[float]]): A list of token-level probability lists.

    Returns:
        List[float]: Product of probabilities per item.
    """
    return [math.prod(sublist) for sublist in probs_list]


def fermi_dirac(E, mu, kT):
    """
    Apply the Fermi-Dirac function.

    Args:
        E (float): Energy level (logit value).
        mu (float): Chemical potential.
        kT (float): Thermal energy (default = 1.0).

    Returns:
        float: Fermi-Dirac value.
    """
    return E / (math.exp((E - mu) / kT) + 1)



def cal_boltzmann_logits(logits_list):
    """
    Apply Boltzmann transformation (negated mean of logits).

    Args:
        logits_list (List[List[float]]): List of token-level logits per item.

    Returns:
        List[float]: Transformed logits per item.
    """
    out = []
    for seq in logits_list:
        # keep only finite values
        clean = [x for x in seq if np.isfinite(x)]

        # fallback if everything is invalid
        if len(clean) == 0:
            out.append(0.0)
        else:
            out.append(-mean(clean))
    return out


def cal_flow(question_group, clusters, fermi_mu=None):
    """
    Compute cluster-wise flow using probabilities and logits.

    Args:
        question_group (List[Dict]): Group of model responses for a question.
        clusters (List[List[int]]): Cluster indices.
        fermi_mu (float, optional): If provided, applies Fermi-Dirac transformation.

    Returns:
        Tuple[List[float], List[float]]:
            - Cluster-level summed probabilities
            - Cluster-level weighted logits (Fermi-Dirac or Boltzmann)
    """
    probs_list = [item['probs'] for item in question_group]
    logits_list = [item['logits'] for item in question_group]

    # Compute the product of token probabilities per item
    probs = cal_probs(probs_list)

    # Apply appropriate logit transformation
    if fermi_mu is not None:
        logits = cal_fermi_dirac_logits(logits_list, mu=fermi_mu) #type: ignore
    else:
        logits = cal_boltzmann_logits(logits_list)

    # Return cluster-level cross-entropy and logit scores
    return cal_cluster_ce(probs, logits, clusters)


def top_k_precision(scores, labels, k=1.0):
    if len(scores) == 0:
        return float('nan')
    sorted_indices = np.argsort(scores)[::-1]
    top_k_count = int(len(scores) * k)
    top_k_indices = sorted_indices[:top_k_count]
    top_k_labels = np.array(labels)[top_k_indices]
    return np.sum(top_k_labels) / len(top_k_labels) if len(top_k_labels) > 0 else float('nan')

def safe_roc_auc_score(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return float('nan')

def safe_average_precision_score(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except:
        return float('nan')

def compute_metrics(labels, scores):
    auroc = safe_roc_auc_score(labels, scores)
    aupr = safe_average_precision_score(labels, scores)
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fpr_at_95_tpr = next((fpr[i] for i in range(len(tpr)) if tpr[i] >= 0.95), float('nan'))
    except:
        fpr_at_95_tpr = float('nan')
    precision, recall, f1, _ = precision_recall_fscore_support(labels, [int(s >= 0.5) for s in scores], average='binary', zero_division=0)
    return auroc, aupr, fpr_at_95_tpr, precision, recall, f1
# Set proxy configuration


def main():
    args = parse_args()
    dataset_name = args.task
    model_name = args.model_path
    n_responses = args.n_responses

    # Initialize tokenizer
    if model_name == "llama3v3chat-70B":
        model_checkpoint = "meta-llama/Llama-3.3-70B-Instruct"
    elif model_name == "Qwen2.5-72B-Instruct":
        model_checkpoint = "Qwen/Qwen2.5-72B-Instruct"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)

    base_path = Path(f'./additional_data/{model_name}/{dataset_name}')
    dataset = load_jsonl(f"./datasets/{dataset_name}.jsonl")
    cluster_data = load_jsonl(f'./additional_data/{model_name}/{dataset_name}/semantic_clusters.jsonl')

    # ✅ FIX: map clusters by question_id so missing IDs are safe
    cluster_by_qid = {row["question_id"]: row["clusters"] for row in cluster_data}

    # ---- Load model outputs ----
    all_data = []
    for file_idx in range(1, 8):
        file_path = base_path / f'response_{file_idx}.jsonl'

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())

                try:
                    end_idx = item["token_ids"][2:].index(23) + 2
                except ValueError:
                    end_idx = len(item["token_ids"])

                all_data.append({
                    "question_id": item["question_id"],
                    "answer": item["answer"],
                    "logits": item["logits"][:end_idx],
                    "probs": item["probs"][:end_idx],
                    "token_ids": item["token_ids"][:end_idx],
                })

    # ---- Group responses by REAL question_id ----
    question_groups = defaultdict(list)
    for item in all_data:
        question_groups[item["question_id"]].append(item)

    # ---- Load judgement files (.json) ----
    data_list = []
    for answer_idx in range(1, n_responses + 1):
        path = f'./additional_data/{model_name}/{dataset_name}/judgement_{answer_idx}.json'
        with open(path, 'r', encoding='utf-8') as f:
            data_list.append(json.load(f))

    # ========== Metrics ==========
    sample_scores_logits = []
    sample_labels = []
    semantic_energy_records = []

    for qid in sorted(question_groups.keys()):
        # ✅ FIX: lookup by qid, not qid-1
        clusters = cluster_by_qid.get(qid)
        if not clusters:
            continue

        score_probs_se, score_logits_se = cal_flow(
            question_groups[qid],
            clusters,
            fermi_mu=None
        )

        best_cluster_idx = int(np.argmax(score_logits_se))
        best_cluster = clusters[best_cluster_idx]
        best_score = score_logits_se[best_cluster_idx]

        if not best_cluster:
            continue

        sample_idx = best_cluster[0]
        answer_idx = sample_idx + 1

        try:
            correct_flag = 1 if next(
                x for x in data_list[answer_idx - 1]
                if x["question_idx"] == qid
            )["correct"] else 0
        except StopIteration:
            continue

        sample_scores_logits.append(best_score)
        sample_labels.append(correct_flag)

        semantic_energy_records.append({
            "instance_id": qid,
            "best_cluster_idx": best_cluster_idx,
            "semantic_energy": best_score,
            "correct_flag": correct_flag
        })

    # ---- Normalize semantic energy ----
    scaler = MinMaxScaler()
    normalized_scores = 30 * scaler.fit_transform(
        np.array(sample_scores_logits).reshape(-1, 1)
    ).flatten()

    for rec, norm_score in zip(semantic_energy_records, normalized_scores):
        rec["semantic_energy"] = float(norm_score)

    # ---- Save CSV ----
    csv_filename = f"./results/{model_name}/{dataset_name}/semantic_energy_normalized_scores.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["instance_id", "best_cluster_idx", "semantic_energy", "correct_flag"],
            delimiter=";"
        )
        writer.writeheader()
        writer.writerows(semantic_energy_records)

    print(f"Normalized semantic energy scores saved to '{csv_filename}'")


if __name__ == "__main__":
    main()
