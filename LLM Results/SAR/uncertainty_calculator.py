import gc
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig, GenerationConfig  # type: ignore
import torch  # type: ignore
import pandas as pd  # type: ignore
import json
import time
import argparse
from pathlib import Path
import sys
sys.setrecursionlimit(10000)

def initial_clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Initial memory cleanup done.")

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a dataset through a model and compute SAR uncertainty."
    )
    parser.add_argument(
        "--model-path",
        default="meta-llama/Llama-3.3-70B-Instruct",
        type=str,
        help="HF model ID or local path (default: %(default)s)",
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable 4-bit quantization (use --no-quantize to disable).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: %(default)s)",
    )
    parser.add_argument(
        "--task",
        type=str,
    )
    return parser.parse_args()

def get_model_name(model_path: str) -> str:
    if "/" in model_path:
        return model_path.split("/")[-1]
    return model_path

def main():
    initial_clean()
    args = parse_args()

    task = args.task
    model_path = args.model_path
    batch_size = args.batch_size
    quant = args.quantize

    model_name = get_model_name(model_path)

    # ======= Config =======
    INPUT_JSONL = Path(f"./datasets/{task}_dataset.jsonl")   # Input dataset
    OUTPUT_JSONL = Path(f"./results/{model_name}/{task}_predictions.jsonl")   # Output results
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = AutoConfig.from_pretrained(
        model_path,
        attn_implementation="eager"
    )

    if quant:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config = None

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=cfg,
        device_map="balanced",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    generation_config = GenerationConfig.from_pretrained(model_path)

    ini_time = time.time()

    def load_questions(jsonl_path: Path):
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                yield {"id": obj["id"], "question": obj["question"]}

    records = list(load_questions(INPUT_JSONL))

    processed_ids = set()
    if OUTPUT_JSONL.exists():
        with OUTPUT_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    processed_ids.add(obj["instance_id"])
                except json.JSONDecodeError:
                    continue

    records = [r for r in records if r["id"] not in processed_ids]

    print(f"Resuming... {len(processed_ids)} done, {len(records)} remaining.")
    if not records:
        print("All records processed. Exiting.")
        return

    messages = [[{"role": "user", "content": rec["question"]}] for rec in records]
    chat_messages = [tokenizer.apply_chat_template(m, tokenize=False) for m in messages]

    print("Number of samples to process now:", len(chat_messages))

    # ======= SAR Setup =======
    from lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator  # type: ignore
    from lm_polygraph.stat_calculators.sample import SamplingGenerationCalculator  # type: ignore
    from lm_polygraph.stat_calculators.cross_encoder_similarity import CrossEncoderSimilarityMatrixCalculator  # type: ignore
    from lm_polygraph.estimators.sar import SAR  # type: ignore
    from lm_polygraph.model_adapters import WhiteboxModel  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore

    model_adapter = WhiteboxModel(base_model, tokenizer, model_path=model_path)
    calc_greedy = GreedyProbsCalculator()
    calc_sampling = SamplingGenerationCalculator(samples_n=5)
    calc_similarity = CrossEncoderSimilarityMatrixCalculator()
    estimator = SAR()

    data_loader = DataLoader(chat_messages, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    with OUTPUT_JSONL.open("a", encoding="utf-8") as fout:
        rec_index = 0
        for i, batch in enumerate(data_loader):
            current_id = records[rec_index]["id"] if rec_index < len(records) else "Unknown"
            logging.info(f"--- Starting Instance {i+1}/{len(data_loader)} (ID: {current_id}) ---")

            try:

                deps = {"input_texts": batch}
                
                logging.info("Running greedy generation...")
                deps.update(calc_greedy(deps, texts=batch, model=model_adapter, max_new_tokens=10))
                
                logging.info("Running sampling calculations...")
                deps.update(calc_sampling(deps, texts=batch, model=model_adapter,max_new_tokens=10))
                
                logging.info("Running similarity calculations...")
                deps.update(calc_similarity(deps, texts=batch, model=model_adapter))

                logging.info("Estimating uncertainty...")
                uncertainty_scores = estimator(deps)
                generated_texts = tokenizer.batch_decode(deps['greedy_tokens'], skip_special_tokens=True)

                # --- Success: Write to file ---
                batch_records = records[rec_index: rec_index + len(batch)]
                for rec, text, ue_score in zip(batch_records, generated_texts, uncertainty_scores):
                    result = {
                        "instance_id": rec["id"],
                        "question": rec["question"],
                        "response": text.strip(),
                        "SAR": float(ue_score),
                    }
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                logging.info(f"Successfully processed ID: {current_id}")

            except torch.cuda.OutOfMemoryError:
                logging.warning(f"OOM ERROR: Skipping ID {current_id}. This instance was too large for VRAM.")
                
                logging.info("oom-clean: del locals")
                if 'deps' in locals(): del deps

                logging.info("oom-clean: del uncertainty_scores")
                if 'uncertainty_scores' in locals(): del uncertainty_scores

            except Exception as e:
                logging.error(f"UNEXPECTED ERROR on ID {current_id}: {str(e)}")
                if 'deps' in locals(): del deps
                torch.cuda.empty_cache()
                gc.collect()
            
            finally:
                logging.info("Update index and continue to next batch.")
                rec_index += len(batch)
                logging.info(f"--- Finished Instance {i+1}/{len(data_loader)} (ID: {current_id}) ---\n")
            

    end_time = time.time()
    print("Total time (s):", end_time - ini_time)


if __name__ == "__main__":
    main()
