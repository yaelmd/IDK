from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from tqdm import tqdm
import pandas as pd
import argparse

class SemanticAnalyser:
    def __init__(self, model_path="cross-encoder/nli-deberta-v3-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).cuda()

    def semantic_analyse(self, question, ans1, ans2):
        inputs = self.tokenizer(ans1, ans2, return_tensors="pt", truncation=True, padding=True).to("cuda")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
        score = probs[:, 1].item()  # entailment or "semantic match" score
        return "Yes" if score > 0.5 else "No"

def find_semantic_clusters(question, answer_list, analyser):
    def is_semantic_same(i, j):
        return "Yes" in analyser.semantic_analyse(question, answer_list[i], answer_list[j])
    
    n = len(answer_list)
    clusters = []
    visited = [False] * n
    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        
        for j in range(i + 1, n):
            if not visited[j] and is_semantic_same(i, j):
                cluster.append(j)
                visited[j] = True
        clusters.append(tuple(cluster))
    print(len(clusters), f"Clusters: {clusters}")
    return clusters


def process_jsonl_files(response_dir, num_files, question_df, analyser, output_file):
    """Reads response_1.jsonl ... response_n.jsonl, applies clustering per instance_id, and writes results."""
    
    # --- Read all responses in order ---
    response_files = [f"{response_dir}/response_{i}.jsonl" for i in range(1, num_files + 1)]
    data_by_instance = {}

    for fpath in response_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                iid = int(item["question_id"])
                if iid not in data_by_instance:
                    data_by_instance[iid] = []
                data_by_instance[iid].append(item["answer"])

    # --- Apply clustering ---
    results = []
    print(len(data_by_instance), "instances to process")
    print(data_by_instance.keys())
    for iid in tqdm(sorted(data_by_instance.keys())):
        answers = data_by_instance[iid]
        question_row = question_df[question_df["id"] == iid]
        if question_row.empty:
            print(f"WARNING: question_id {iid} not found in dataset, skipping")
            continue
        question = question_row.iloc[0]["question"]

        clusters = find_semantic_clusters(question, answers, analyser)
        results.append({"question_id": iid,"question": question, "clusters": clusters})

    # --- Write output ---
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Done! {len(results)} instances processed → {output_file}")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Judgement creation"
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

def main():

    args = parse_args()
    model = args.model_path
    dataset = args.task
    n_responses = args.n_responses  

    analyser = SemanticAnalyser()

    with open(f"./datasets/{dataset}.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    question_df = pd.DataFrame(data)
    question_df["id"] = question_df["id"].astype(int)

    process_jsonl_files(f"./cache_data/{model}/{dataset}", n_responses, question_df,analyser,f"./cache_data/{model}/{dataset}/semantic_clusters.jsonl")


if __name__ == "__main__":
    main()





