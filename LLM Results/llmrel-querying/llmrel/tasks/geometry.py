from typing import List, Dict, Any

import pandas as pd

import gc

from llmrel.tasks.task import Task


class GeometryTask(Task):
    def ins2prompts(self, row: pd.Series, prompt_type: str) -> List[str]:

        prompt_list = [row["question"]]

        # Add Confidence estimation sentence
        with open(f"./prompts/{prompt_type}.txt", "r") as f:
            prompt_prefix = f.read()

        
        prompt_suffix = "Now, please answer this question and provide your confidence level."
        cot_suffix = " Let's think it step by step."


        if prompt_type == "self_probing" or prompt_type == "harder_self_probing":
            prev_response = row["prev_response"]
            prompt_list = [f"Question: {prompt}\nPossible Answer: {prev_response}\n{prompt_prefix}" for prompt in prompt_list]
        elif prompt_type == "vanilla" or prompt_type == "harder_vanilla":
            prompt_list = [f"{prompt_prefix}\nQuestion: {prompt}\n{prompt_suffix}" for prompt in prompt_list]
        elif prompt_type == "CoT" or prompt_type == "harder_CoT":
            prompt_list = [f"{prompt_prefix}\nQuestion: {prompt}\n{prompt_suffix}{cot_suffix}" for prompt in prompt_list]
        elif prompt_type == "repetitions" or prompt_type=="formatted" or prompt_type == "vanilla_v2" or prompt_type == "harder_vanilla_v2":
            prompt_list = [f"Question: {prompt}\n{prompt_prefix}" for prompt in prompt_list]
        else :
            prompt_list = [f"{prompt_prefix}\nQuestion: {prompt}" for prompt in prompt_list]
        # fmt: on
        return prompt_list

    def load_data(self, path) -> pd.DataFrame:
        df = pd.read_csv(path, delimiter=";")

        df = df[["problem_id", "question", "answer", "task", "subtask", "difficulty"]]

        df = df.reset_index(drop=True)
        return df  # type: ignore
    
    def load_data_probing(self, path, path_aux) -> pd.DataFrame:
        df = pd.read_csv(path, delimiter=";")
        df_aux = pd.read_csv(path_aux)

        df = df[["problem_id", "question", "answer", "task", "subtask", "difficulty"]]
        df = df.reset_index(drop=True)

        df["prev_response"] = df_aux["response"]

        # Delete the DataFrame
        del df_aux

        # Run garbage collection
        gc.collect()

        return df


    def ins2dict(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "problem_id": row["problem_id"],
            "question": row["question"],
            "answer": row["answer"], 
            "task": row["task"],
            "subtask": row["subtask"],
            "difficulty": row["difficulty"],
        }
    def fieldnames(self) -> List[str]:
        return [
            "problem_id",
            "question",
            "answer",
            "task",
            "subtask",
            "difficulty",
        ]
