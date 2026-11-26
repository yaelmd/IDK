from methods import *

import pandas as pd
import argparse

BLACK_BOX_METHODS = ["vanilla","top_k","self_probing","multi_step","CoT","self_consistency","formatted"]
WHITE_BOX_METHODS = ["SAR", "semantic_energy"]

ASSESSOR_KEXIN = [
    "asr_adele_CoT", "asr_adele_multi_step", "asr_adele_self_probing", "asr_adele_top_k", "asr_adele_vanilla",
    "asr_diff_CoT", "asr_diff_multi_step", "asr_diff_self_probing", "asr_diff_top_k", "asr_diff_vanilla",
    "asr_diff+adele_CoT", "asr_diff+adele_multi_step", "asr_diff+adele_self_probing", "asr_diff+adele_top_k", "asr_diff+adele_vanilla",
    "asr_emb_CoT", "asr_emb_multi_step", "asr_emb_self_probing", "asr_emb_top_k", "asr_emb_vanilla",
    "asr_emb+adele_CoT", "asr_emb+adele_multi_step", "asr_emb+adele_self_probing", "asr_emb+adele_top_k", "asr_emb+adele_vanilla",
    "asr_emb+diff_CoT", "asr_emb+diff_multi_step", "asr_emb+diff_self_probing", "asr_emb+diff_top_k", "asr_emb+diff_vanilla",
    "asr_emb+diff+adele_CoT", "asr_emb+diff+adele_multi_step", "asr_emb+diff+adele_self_probing", "asr_emb+diff+adele_top_k", "asr_emb+diff+adele_vanilla",
]

ASSESSOR_ALVARO = ["asr_difficulty+embedding", "asr_difficulty", "asr_embedding", "asr_demands",
                    "asr_difficulty+demands", "asr_difficulty+embedding+demands", "asr_embedding+demands"]


ASSESSOR_METHODS = ASSESSOR_ALVARO + ASSESSOR_KEXIN

TARGET_DICT = {"addition":"target","anagram":"word", "locality" : "target_city", "science": "correct_option",
               "geometry": "answer", "logic_puzzles": "answer", "algebra":"answer", "arithmetic": "answer",
               "combinatory": "answer", "number_theory": "answer"}
BLACK_BOX_MODELS = ["GPT5.2","gemini-3-pro-preview", "gemini-3.1-pro-preview"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process confidence results"
    )
    parser.add_argument(
        "--model",
        nargs="+",
        type=str,
        default=["Qwen2.5_72B", "gemini-3.1-pro-preview", "GPT5.2", "llama3v3chat-70B"],
        help="HF model ID(s) or local path(s)",
    )
    parser.add_argument(
        "--method",
        nargs="+",
        type=str,
        default=BLACK_BOX_METHODS+WHITE_BOX_METHODS+ASSESSOR_METHODS,
        help="Method(s) to compute confidence",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        type=str,
        default=["addition", "anagram", "locality", "science", "geometry", "logic_puzzles",
                 "arithmetic", "algebra", "combinatory", "number_theory"],
        help=f"Implemented task(s): {', '.join(TARGET_DICT.keys())}",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="If set, run in test mode",
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        default=False,
        help="If set, only apply filters to existing labelled data",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        help="Choose between IID/OOD/OOD-SAMPLED-{N}",
    )
    return parser.parse_args()


def clean_data(df, method, task, model, test_mode=False):
    if task == "locality":
        df_aux = get_target(task)
        df = df.merge(df_aux, on="instance_id", how="left")
    if model in BLACK_BOX_MODELS:
        df["cleaned_response"] = df["response"]
    else:
        if method != "self_consistency":
            if task == "science":
                df = df[["instance_id","prompt", "response", TARGET_DICT[task], "correct_answer"]].copy()
            else:
                df = df[["instance_id","prompt", "response", TARGET_DICT[task]]].copy()

            df["cleaned_response"] = df.apply(lambda x: clean_response(x["prompt"], x["response"], model_type=model), axis=1)

    if test_mode:
        df["original_response"] = df["cleaned_response"].copy() # TO TEST

    if method == "self_consistency":
        df["cleaned_response"] = remove_special_tokens(df["response"])
        df["cleaned_response"] = df["cleaned_response"].apply(lambda x: x.lower().strip())
    else:
        func_name = f"clean_{method.lower() if method != 'self_consistency' else 'formatted'}"
        cleaner = globals().get(func_name)

        if cleaner is None or not callable(cleaner):
            raise ValueError(f"{func_name} not found")

        df["cleaned_response"] = df["cleaned_response"].apply(cleaner)

    return df


def extract_prompting_response_and_confidence(df, method, task, model):
    if method == "self_probing":
        df_aux = get_auxiliary_results(model, task)
        df = df.merge(df_aux, on="instance_id", how="left")
        df["confidence"] = df["cleaned_response"].apply(extract_confidence_probing)
    else:
        if method == "top_k":
            # It gets special treatment because when we clean the response we get the G1: part, and we need the original response for the P1: part
            df["confidence"] = df["response"].apply(extract_confidence_top_k)
        elif method == "formatted":
            df["confidence"] = None
        else:
            df["confidence"] = df["cleaned_response"].apply(extract_confidence)
        
        if task == "science":
            df["outcome"] = df.apply(lambda x: label_outcome(x["cleaned_response"], x[TARGET_DICT[task]], task, confidence=x["confidence"], correct_answer=x["correct_answer"]), axis=1)
            del df["correct_answer"]
        else:
            df["outcome"] = df.apply(lambda x: label_outcome(x["cleaned_response"], x[TARGET_DICT[task]], task, confidence=x["confidence"], prompt_type=method), axis=1)
    
    return df

def extract_consistency_response_and_confidence(df,task,model):

    # Group responses with the same instance_id and sort from most repeated answer to less repeated
    if model in BLACK_BOX_MODELS:
        result = df.groupby(["instance_id", "question", TARGET_DICT[task]]).apply(count_responses)
    else:  
        result = df.groupby(["instance_id", TARGET_DICT[task]]).apply(count_responses)
    result = result.reset_index(name="response_dict")
    result['response_dict'] = result['response_dict'].apply(lambda x: dict(sorted(x.items(), key=lambda item: item[1], reverse=True)))

    result["confidence"] = result["response_dict"].apply(extract_confidence_consistency)

    df_aux = get_auxiliary_results(model, task)
    result = result.merge(df_aux, on=["instance_id"], how="left")

    return result

def extract_whitebox_outcome_and_confidence(df, method, task, model):
    df_aux = get_auxiliary_results(model, task, target=True)

    df = df.merge(df_aux, on="instance_id", how="left")

    if method == "SAR":
        if task == "science":
            df["outcome"] = df.apply(lambda x: label_outcome(x["cleaned_response"], x[TARGET_DICT[task]], task, correct_answer=x["correct_answer"]), axis=1)
            del df["correct_answer"]
        else:
            df["outcome"] = df.apply(lambda x: label_outcome(x["cleaned_response"], x[TARGET_DICT[task]], task), axis=1)
    df["uncertainty"] = df[method]

    return df



def apply_filters(labelled_data, task, method, model):

    df = labelled_data.copy()
    if method in WHITE_BOX_METHODS:
        df["uncertainty"] = normalize_uncertainty(df["uncertainty"])
        df["confidence"] = convert_to_confidence(df["uncertainty"], method)

        df["outcome"] = df.apply(lambda x: drop_confidence_none(x["outcome"], x["confidence"]), axis=1)

    return df


def read_raw(model, task, method):
    if method == "SAR":
        raw_data = load_jsonl(f"raw_results/{model}/{task}/{method}.jsonl")
        raw_data = pd.DataFrame(raw_data)
    else:
        if model == "GPT5.2" or (model=="gemini-3-pro-preview" and method == "self_consistency"):
            raw_data = pd.read_csv(f"raw_results/{model}/{task}/{method}.csv")
        else:
            raw_data = pd.read_csv(f"raw_results/{model}/{task}/{method}.csv", delimiter=";")
        if method != "semantic_energy":
            raw_data["response"] = raw_data["response"].astype(str)
    return raw_data

def main():

    args = parse_args()

    model_list = args.model
    if args.method[0] == "ASSESSOR":
        method_list = ASSESSOR_METHODS
    elif args.method[0] == "BLACK_BOX":
        method_list = BLACK_BOX_METHODS
    elif args.method[0] == "WHITE_BOX":
        method_list = WHITE_BOX_METHODS
    else:
        method_list = args.method
    task_list = args.task

    if args.distribution != "IID" and args.distribution != "OOD":
        #Check if the distribution is OOD-SAMPLED-{N}
        if not args.distribution.startswith("OOD-SAMPLED-") and not args.distribution.startswith("IID-SAMPLED-"):
            raise ValueError("Invalid distribution. Choose between IID or OOD.")

    
    for model in model_list:
        for method in method_list:

            if "OOD" in args.distribution and method not in ASSESSOR_METHODS:
                print(f"Skipping {model} with {method} as it is not an assessor method and we are only processing OOD distribution.")
                continue
            
            if model in BLACK_BOX_MODELS and method in WHITE_BOX_METHODS:
                print(f"Skipping {model} with {method} as it is a white-box method not applicable to black-box models.")
                continue

            for task in task_list:

                print(f"Processing {model} with {method} for task {task} and distribution {args.distribution}...")
                
                try:
                    if not args.filter_only and not method in ASSESSOR_METHODS:

                        if task == "addition" or task == "geometry" or task == "science" or task == "logic_puzzles" or task == "locality" or task == "anagram":
                            print(f"{task} task has already been labelled and revised. Skipping...")
                            continue
                        
                        if task == "science":

                            if method == "self_consistency" and (model in BLACK_BOX_MODELS and model != "gemini-3.1-pro-preview"):
                                raw_data = read_raw(model, "science", method)
                            else:
                                obqa = read_raw(model, "openbookqa", method)
                                obqa = obqa.rename(columns={"answerKey":"correct_option"})
                                if model in BLACK_BOX_MODELS:
                                    obqa = obqa[["instance_id","question", "response"]]
                                elif method != "SAR" and method != "semantic_energy" and method != "self_consistency":
                                    obqa = obqa[["instance_id","prompt", "response"]]
                                obqa["instance_id"] = obqa["instance_id"].apply(lambda x: f"obqa_{x}")

                                gpqa = read_raw(model, "gpqa", method)
                                if model in BLACK_BOX_MODELS:
                                    gpqa = gpqa[["instance_id","question", "response"]]
                                elif method != "SAR" and method != "semantic_energy" and method != "self_consistency":
                                    gpqa = gpqa[["instance_id","prompt", "response"]]
                                gpqa["instance_id"] = gpqa["instance_id"].apply(lambda x: f"gpqa_{x}")
                                
                                raw_data = pd.concat([obqa, gpqa], ignore_index=True)
                                raw_data = raw_data.rename(columns={"instance_id":"ori_id"})

                                # Map the original IDs to the science instance IDs
                                science = pd.read_csv("./original_datasets/science.csv", usecols=["instance_id", "ori_id","correct_option", "correct_answer"])
                                raw_data = raw_data.merge(science, on="ori_id", how="left")
                                raw_data = raw_data.drop(columns=["ori_id"])
                                raw_data = raw_data.dropna(subset=["instance_id"])

                                raw_data = raw_data.sort_values(by=["instance_id"]).reset_index(drop=True)
                                raw_data["instance_id"] = raw_data["instance_id"].astype(int)


                        else:
                            raw_data = read_raw(model, task, method)

                        if task == "locality":
                            # Drop 1427 instance id because it is nan
                            raw_data = raw_data[raw_data["instance_id"] != 1427]

                        # Data Cleaning
                        if method in BLACK_BOX_METHODS:
                            cleaned_data = clean_data(raw_data, method, task, model, test_mode=args.test_mode)
                           
                        elif method == "SAR":
                            cleaned_data = raw_data.copy()
                            cleaned_data["cleaned_response"] = remove_special_tokens(raw_data["response"])
                            if task == "science":
                                cleaned_data = cleaned_data.drop(columns=["correct_option"])
                        else:
                            cleaned_data = raw_data

                        # Outcome and Confidence Extraction
                        if method == "self_consistency":
                            labelled_data = extract_consistency_response_and_confidence(cleaned_data, task, model)
                        elif method in WHITE_BOX_METHODS:
                            labelled_data = extract_whitebox_outcome_and_confidence(cleaned_data,method,task, model)
                        elif method in BLACK_BOX_METHODS:
                            labelled_data = extract_prompting_response_and_confidence(cleaned_data, method, task, model)
                        else:
                            raise ValueError(f"{method} not implemented")
                        
                        if args.test_mode:
                            #Random check of instances
                            check = labelled_data.sample(n=10, random_state=5)

                            for i,r in check.iterrows():
                                print(f"Instance ID: {r['instance_id']}")
                                # print(f"Prompt: {r['prompt']}")
                                if method != "self_consistency":
                                    print(f"Original Response: {r['original_response']}")
                                    print(f"Cleaned Response: {r['cleaned_response']}")
                                print(f"Target: {r[TARGET_DICT[task]]}")
                                print(f"Confidence: {r['confidence']}")
                                print(f"Outcome: {r['outcome']}")
                                print("-"*50)
                        else:   
                            save_csv(labelled_data, f"./labelled_data/{args.distribution}/{model}/{task}/{method}.csv", index=False)

                        print(f"Data labelled for {model}, estimation {method}, task {task}.")
                    else:

                        labelled_data = pd.read_csv(f"./labelled_data/{args.distribution}/{model}/{task}/{method}.csv")

                        if method in ASSESSOR_METHODS:
                            if "confidence" in labelled_data.columns:
                                del labelled_data["confidence"]
                            """The outcome given by the file is not the one that we want,
                            we need the outputs of the base model"""
                            del labelled_data["outcome"]
                            df_aux = get_auxiliary_results(model, task, target=False)
                            labelled_data = labelled_data.merge(df_aux[["instance_id","outcome"]], on="instance_id", how="left")
                            if method in ASSESSOR_KEXIN:
                                labelled_data["confidence"] = labelled_data["assessor_prob_correct"] * 100 # op 1
                            elif method in ASSESSOR_ALVARO:
                                labelled_data["confidence"] = (1 - labelled_data["prob_incorrect"]) * 100 # op 2
                            # labelled_data["confidence"] = (1 - labelled_data["prob_incorrect"]) * 100 # op 2
                            # labelled_data["confidence"] = (1 - 0.7 * labelled_data["prob_incorrect"] - labelled_data["prob_avoidance"]) * 100 # op 3.1
                            # labelled_data["confidence"] = (1 - 0.6 * labelled_data["prob_incorrect"] - labelled_data["prob_avoidance"]) * 100 # op 3.2 esta parece mejor
                            # labelled_data["confidence"] = (1 - 0.5 * labelled_data["prob_incorrect"] - labelled_data["prob_avoidance"]) * 100 # op 3.3
                            # labelled_data["confidence"] = (1 - 0.4 * labelled_data["prob_incorrect"] - labelled_data["prob_avoidance"]) * 100 # op 3.4
                            # labelled_data["confidence"] = (1 - labelled_data["prob_incorrect"] - 0.7 * labelled_data["prob_avoidance"]) * 100 # op 4
                            # labelled_data["confidence"] = 1 - labelled_data["prob_avoidance"] * 100  # op 5
                
                except FileNotFoundError as e:
                    print(e)
                    print(f"Results not found for {model}, estimation {method}, task {task}, distribution {args.distribution}. Skipping...")
                    continue


                if not args.test_mode:
                    if task == "anagram":
                        labelled_data = get_cleaned_anagrams(labelled_data)
                    # Apply threshold
                    final_results =  apply_filters(labelled_data, task, method, model)

                    # if method != "formatted":
                    save_csv(final_results, f"./confidence_results/{args.distribution}/{model}/{task}/{method}.csv", index=False)
                    
                    print(f"Results processed for {model}, estimation {method}, task {task}, distribution {args.distribution}.")


if __name__ == "__main__":
    main()