#!/usr/bin/env python3
import os, json
from dataclasses import dataclass
from typing import List, Dict, Any
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# Generation settings (mirrors your snippet)
MAX_NEW_TOKENS = 128
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 50
DO_SAMPLE = True


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

@dataclass
class Item:
    idx: int
    question: str
    meta: Dict[str, Any]


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
        help="Implemented task(s): addition, anagram, locality, openbookqa, gpqa",
    )

    parser.add_argument(
        "--n-responses",
        type=int,
        default=7,
        help="Number of responses to generate (default: %(default)s)",
    )

    return parser.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def build_chat_prompts(tokenizer: AutoTokenizer, questions: List[str]) -> List[str]:
    """Apply the model's chat template to each question (like your snippet)."""
    prompts = []
    for q in questions:
        messages = [{"role": "user", "content": q}]
        generated_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(generated_prompt)
    return prompts

def main():
    args = parse_args()
    model_name = args.model_path
    dataset = args.task

    if  model_name == "llama3v3chat-70B":
        model_checkpoint = "meta-llama/Llama-3.3-70B-Instruct"
    elif model_name == "Qwen/Qwen2.5-72B-Instruct" or model_name == "Qwen2.5-72B-Instruct":
        model_checkpoint = "Qwen/Qwen2.5-72B-Instruct"
    else:
        raise ValueError(f"Unknown model {args.model_path}")


    print(f"Loading {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        dtype=DTYPE,
        device_map="auto"
    )

    rows = read_jsonl(f"./datasets/{dataset}.jsonl")
    items = [Item(idx=i, question=row["question"], meta=row) for i, row in enumerate(rows, start=1)]

    # You looped over 1..7; keeping that behavior
    for response in range(1, args.n_responses + 1):
        output_jsonl = f"./additional_data/{model_name}/{dataset}/response_{response}.jsonl"
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

        with open(output_jsonl, "w", encoding="utf-8") as fout:
            for it in items:
                question = it.question  # assumes items = [Item(idx, question, meta), ...]
                
                messages = [
                    {"role": "user", "content": question}
                ]
                generated_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = tokenizer(generated_prompt, return_tensors='pt')
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                generated_output = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                # Extract the generated token ids (excluding the prompt)
                generated_ids = generated_output.sequences[0][len(inputs["input_ids"][0]):].tolist()
                scores = generated_output.scores  # List[tensor: (batch, vocab_size)], len == num_generated_tokens

                logits_list = []
                probs_list = []
                token_ids = []

                for step_idx, score_tensor in enumerate(scores):
                    """
                    Example: the next-token probability distribution is {'token_id_1': '0.75', 'token_id_2': '0.22', ...},
                    the next-token logit distribution is {'token_id_1': '35', 'token_id_2': '28', ...},
                    suppose the sampled token is 'token_id_2'
                    """
                    logits = score_tensor[0].tolist()  # (vocab_size,)
                    token_id = generated_ids[step_idx]
                    prob = F.softmax(score_tensor[0], dim=-1)[token_id].item()
                    logits_list.append(logits[token_id])  # selected token's logit
                    probs_list.append(prob)               # selected token's probability
                    token_ids.append(token_id)            # selected token id

                # Decode the generated text from the selected tokens
                answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                record = {
                    "question_id": it.idx,
                    "reasoning": "",
                    "answer": answer_text,
                    "token_ids": token_ids,   # list[int]
                    "logits": logits_list,    # list[float]
                    "probs": probs_list,      # list[float]
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

if __name__ == "__main__":
    main()