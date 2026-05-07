import pandas as pd
import json
import re
import argparse


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
    

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]
    
def extract_numbers(response):
    numbers = re.findall(r"\d+\.\d+|\d+", response)
    return numbers

def correct_addition_response(response, target):
    numbers = extract_numbers(response)
    if len(numbers) == 1:
        try:
            if str(numbers[0]) == str(target):
                return True
            else:
                return False
        except:
            return False
    elif len(numbers) == 3:
        if str(numbers[2]) == str(target):
            return True
        else:
            return False
    return False

def correct_anagram_response(response, target):
    response = response.strip().lower()
    target = target.strip().lower()
    
    return response == target

def correct_locality_response(response, target):
    response = response.strip().lower()
    target = target.strip().lower()
    
    return response == target

def correct_science_response(response, target):
    target = target.strip()
    response = str(response).strip()
    response = re.sub(r'[, ]*\b\d{1,3}\s*%\s*$', '', response).strip()

    m = re.search(r'(?:^|[\s\(\[\{])([A-Z])(?:\s*[\.\,\)\]\}\:\-]|\s+|$)', response)
    if not m:
        return False

    pred = m.group(1).upper()

    return pred == target


def correct_omega_response(response, target):
    response = response.strip().lower().replace(',', '').replace('.', '')
    target = str(target).strip().lower()
    
    return target in response


def create_judgment(dataset_name, response_data):
    dataset = load_jsonl(f"./datasets/{dataset_name}.jsonl")
    if dataset_name == "addition":
        labelling_method = correct_addition_response
    elif dataset_name == "anagram":
        labelling_method = correct_anagram_response
    elif dataset_name == "locality":
        labelling_method = correct_locality_response
    elif dataset_name == "openbookqa" or dataset_name == "gpqa":
        labelling_method = correct_science_response
    elif dataset_name == "geometry" or dataset_name == "logic_puzzles":
        labelling_method = correct_omega_response
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    results = []
    for i in range(len(data)):
        answer = response_data[i]["answer"]
        truth_label = dataset[i]["answer"]
        results.append(
            {
                "question_idx": response_data[i]["question_id"],
                "correct": labelling_method(answer,truth_label),
                "question": dataset[i]["question"],
                "answer": answer,
                "truth_label": truth_label
            }
        )
        
    return results


if __name__ == "__main__":
    args = parse_args()
    n_responses = args.n_responses
    dataset_name = args.task
    model = args.model_path

    for i in range(1,n_responses+1):
        input_path = f"./cache_data/{model}/{dataset_name}"
        data = load_jsonl(f"{input_path}/response_{i}.jsonl")
        with open(f"{input_path}/judgement_{i}.json", "w") as f:
            json.dump(create_judgment(dataset_name,data), f, indent=4)
    
    print("Process completed succesfully!")