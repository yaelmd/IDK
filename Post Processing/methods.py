import re
import json
import os
import string
import pandas as pd

TARGET_DICT = {"addition":"target","anagram":"word", "locality" : "target_city", "science": "correct_option",
               "geometry": "answer", "logic_puzzles": "answer", "algebra":"answer", "arithmetic": "answer",
               "combinatory": "answer", "number_theory": "answer"}

OMEGA_TASKS = ["geometry", "logic_puzzles", "algebra", "arithmetic", "combinatory", "number_theory"]


def clean_response(prompt, response, model_type = None):
    """
    Generic method for initially cleaning every response. It removes the initial prompt and special tokens
    """
    if "llama" in model_type or "Qwen" in model_type:
        
        if "llama" in model_type:
            ASSISTANT_HDR = r"<\|start_header_id\|>assistant<\|end_header_id\|>"
        if "Qwen" in model_type:
            ASSISTANT_HDR = r"<\|im_start\|>assistant"
        
        # Normalize newlines
        response = response.replace("\r\n", "\n")


        m = re.search(ASSISTANT_HDR, response)
        if m:
            response = response[m.end():]
            stop = re.search(
                r"<\|start_header_id\|>.*?<\|end_header_id\|>|<\|end_of_text\|>|<\|eot_id\|>",
                response,
                flags=re.DOTALL
            )
            if stop:
                response = response[:stop.start()]

        # 2) Remove all special tokens like <|eot_id|>, <|end_of_text|>, etc.
        response = re.sub(r"<\|[^|]+?\|>", "", response)

        # Tidy up whitespace
        response = re.sub(r"[ \t]+\n", "\n", response)
        response = re.sub(r"\n{3,}", "\n\n", response)

    
    response = response.strip()

    return response


def remove_special_tokens(response):


    response = response.replace("<|start_header_id|>assistant<|end_header_id|>", '')
    response = response.replace("<|im_start|>assistant", '')
    response = response.replace("<|eot_id|>", '')
    response = response.replace("<|begin_of_text|>", '')
    response = response.replace("<|end_of_text|>", '')
    response = response.replace("<|endoftext|>", '')
    response = response.replace("<|im_end|>", '')

    return response


def extract_numbers(response):
    """
    Auxiliar method to extract numbers from responses
    """
    # numbers = re.findall(r"\d+\.\d+|\d+", response)
    # numbers = re.findall(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", response)
    return re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        response,
    )
    return numbers

def remove_punctuation(response: str) -> str:
    """
    It removes punctuation from a given response
    """
    translator = str.maketrans('', '', string.punctuation)
    return response.translate(translator)

def clean_self_probing(response):
    """
    It extracts the final answer for the prompt type "self_probing"
    """
    response = response.replace("[the probability of the answer to be correct, not the one you think correct, please only include the numerical number]" , '')
    return response


def clean_multi_step(response):
    """
    It extracts the final answer for the prompt type "multi_step"
    """
    alternative_answer = "Final Answer and Overall Confidence (0-100):"
    final_asnwer = "Final Answer and Overall Confidence:"
    if alternative_answer in response:
        response = response.split(alternative_answer)[1]
    elif final_asnwer in response:
        response = response.split(final_asnwer)[1]
    return response


def clean_formatted(response):
    """
    It extracts the final answer for the prompt type "formatted"
    """
    final_asnwer = "Thus, the correct answer is:"
    alternative_asnwer = "The final answer is:" # For some cases in geometry
    if final_asnwer in response:
        response = response.split(final_asnwer)[1].strip()
    elif alternative_asnwer in response:
        response = response.split(alternative_asnwer)[1].strip()
    else:
        return "None"
    return response


def clean_vanilla(response):
    """
    It extracts the final answer for the prompt type "vanilla"
    """
    if "Answer and Confidence (0-100):" in response:
        response = response.split("Answer and Confidence (0-100):")[1].strip()
    return response


def clean_cot(response):
    """
    It extracts the final answer for the prompt type "CoT"
    """
    final_asnwer = "Answer and Confidence (0-100):"
    if final_asnwer in response:
        response = response.split(final_asnwer)[1]
    else:
        alternative_asnwer = "Answer and Confidence:"
        if alternative_asnwer in response:
            response = response.split(alternative_asnwer)[1]

    response = response.split("\n")[0]
    return response



def clean_top_k(response):
    """
    It extracts the final answer for the prompt type "Top K"
    """
    g1_match = re.search(r'G1:\s*(.*?)\s*P1:', response)

    g1_text = g1_match.group(1).strip() if g1_match else None
    return g1_text


def extract_confidence(response): 
    """
    It extracts the confidence in a given response, choosing the LAST number ocurrence
    """

    match = re.findall(r'(\d+(?:\.\d+)?)(?:\s*%)?', response)

    if match:
        confidence_value = match[-1]
        if float(confidence_value) > 100 or float(confidence_value) < 0:
            return None
        else:
            return confidence_value
    else:
        return None
    

def extract_confidence_reverse(response):
    """
    It extracts the confidence in a given response, choosing the FIRST number ocurrence
    """
    match = re.findall(r'\d+(?:\.\d+)?', response)

    if match:
        confidence_value = match[0]
        if float(confidence_value) > 100 or float(confidence_value) < 0:
            return None
        else:
            return confidence_value
    else:
        return None


def extract_confidence_probing(response):
    """
    It extracts the confidence in a self_probing prompt type. 
    It takes into account that the confidence in this cases tends to be the first number to appear in the response 
    and not the last. However, there are some cases where it is the last.
    """
    if "Confidence: " in response:
        response = response.split("Confidence: ")[1]

    temp_confidence = extract_confidence_reverse(response)

    if temp_confidence !=None:
        temp_confidence = float(temp_confidence)
        if temp_confidence <= 1:
            temp_confidence = temp_confidence * 100
        return temp_confidence
    else:
        match = re.findall(r'(\d+(?:\.\d+)?)\s*', response)

        if match:
            confidence_value = float(match[-1])
            if float(confidence_value) > 100 or float(confidence_value) < 0:
                return None
            else:
                if confidence_value <= 1:
                    confidence_value = confidence_value * 100
                return confidence_value
        else:
            return None
    


def extract_confidence_top_k(response):
    """
    It extracts the confidence in a top_k prompt type. 
    Example: T1:25 P1:95%
    """
    p1_match = re.search(r'P1:\s*(\d+)%?', response)

    p1_number = p1_match.group(1) if p1_match else None

    return p1_number

  
def extract_results_vanilla_v2(response):
    """
    Special method to handle this prompt:
    You may employ any reasoning strategy to analyze the task before providing your final answer. 
    However, you must conclude your assessment with the statement: 'Thus, the correct answer is: X.
    My confidence is: Y.', where 'X' is the solution you've determined to be correct and Y is the confidence you have in that solution.
    """  
    if "My confidence is:" in response:
        try:
            answer, confidence_answer = response.split("My confidence is")
        except:
            print("Error in response: ", response)
    else:
        return response, None

    response = answer.strip()
    confidence = extract_confidence(confidence_answer)
    return response, confidence


def extract_confidence_consistency(response_dict):
    """
    Given a dictionary of response -> number of times repeated
    finds the majority response and computes confidence 
    """
    majoritary_response = list(response_dict.keys())[0]
    confidence = (response_dict[majoritary_response]/10)*100

    return confidence

def extract_dollar(text: str) -> str | None:
    # First try to extract \boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]*)\}', text)
    if boxed_match:
        return boxed_match.group(1)

    # Otherwise, extract the content inside $...$
    dollar_match = re.search(r'\$(.*?)\$', text)
    if dollar_match:
        return dollar_match.group(1)

    return None

def label_addition_response(response, target, confidence=None):
    """
    Task: addition
    Once the response has been cleaned, it can tag CORRECT, INCORRECT and AVOIDANCE for every prompt type
    """
    if response is None:
        return "AVOIDANCE"
    numbers = extract_numbers(response)
    if confidence!= None and confidence in numbers:
        numbers.remove(confidence) 
    if len(numbers) == 1:
        try:
            if str(numbers[0]).rstrip(".") == str(target).rstrip("."):
                return "CORRECT"
            else:
                return "INCORRECT"
        except:
            return "INCORRECT"
    elif len(numbers) == 3:
        if str(numbers[2]) == str(target):
            return "CORRECT"
        else:
            return "INCORRECT"
    return "AVOIDANCE"


def label_anagram_response(response, target, confidence=None):
    """
    Task: anagram
    Once the response has been cleaned, it can tag CORRECT, INCORRECT and AVOIDANCE for every prompt type
    """
    AVOIDANCE_INDICATORS = ["i don't know", "i am not sure", "i cannot", "i give up", "no answer", "unable to", "cannot determine", 
                            "not able to", "no idea", "don't know", "do not know", "not sure", "not able", "not possible", "unsure", "unknown",
                            "no valid", "none", "no word"]

    if response == "None":
        return "AVOIDANCE"
    if response is None:
        return "AVOIDANCE"
    response = remove_punctuation(response.strip().lower())
    target = remove_punctuation(target.strip().lower())

    if target in response:
        return "CORRECT"
    elif any(indicator in response for indicator in AVOIDANCE_INDICATORS):
        return "AVOIDANCE"
    elif len(response) > len(target) * 20:  # Allowing a small margin for extra words
        return "AVOIDANCE"
    # If there are only numbers in the response, it is avoidance
    elif all(char.isdigit() or char.isspace() for char in response):
        return "AVOIDANCE"
    else:
        return "INCORRECT"


def label_locality_response(response, target, confidence=None):
    if response == "None":
        return "AVOIDANCE"
    if response is None:
        return "AVOIDANCE"
    
    AVOIDANCE_INDICATORS = ["i don't know", "i am not sure", "i cannot", "i give up", "no answer", "unable to", "cannot determine", "can’t determine",
                            "not able to", "no idea", "don't know", "do not know", "not sure", "not able", "not possible", "unsure", "unknown",
                            "no valid", "no word", "ambiguous"]
    
    response = remove_punctuation(response.strip().lower())
    target = remove_punctuation(target.strip().lower())
    confidence = str(confidence).strip().lower()

    if target in response:
        return "CORRECT"
    elif str(response)== str(confidence):
        return "AVOIDANCE"
    elif any(indicator in response for indicator in AVOIDANCE_INDICATORS):
        return "AVOIDANCE"
    elif all(char.isdigit() or char.isspace() or char == ',' for char in response):
        return "AVOIDANCE"
    else:
        return "INCORRECT"


def label_science_response(response, target, correct_answer):

    answer_options = ['A', 'B', 'C', 'D', 'E']

    if response is None:
        return "AVOIDANCE"

    target = target.strip()
    response = str(response).strip()
    response = re.sub(r'[, ]*\b\d{1,3}\s*%\s*$', '', response).strip()

    # Extract the FIRST option letter in common MC formats:
    #   "A", "(A)", "A.", "A)", "A -", "A: ...", "A An animal", "A. An animal"
    m = re.search(r'(?:^|[\s\(\[\{])([A-Z])(?:\s*[\.\,\)\]\}\:\-]|\s+|$)', response)
    if not m:
        if response.lower() == correct_answer.strip().lower():
            return "CORRECT"
        elif not any(char.isalpha() for char in response):
            return "AVOIDANCE"
        else:
            return "INCORRECT"

    pred = m.group(1).upper()

    return "CORRECT" if pred == target else "INCORRECT"


def label_omega_response(response, target, confidence=None, prompt_type=None):
    if response is None:
        return "AVOIDANCE"
    

    if len(response) > 100:
        return "AVOIDANCE"
    
    response = response.strip().lower()

    if any(value in response for value in ("infinity", "infinite", r"\infty", "∞")):
        return "INCORRECT"

    if "$" in response or "boxed" in response:
        boxed_answer = extract_dollar(response)
        try:
            if str(boxed_answer) == str(target):
                return "CORRECT"
            elif boxed_answer is None:
                return "AVOIDANCE"
            elif float(boxed_answer) == float(target):
                return "CORRECT"
            else:
                return "INCORRECT"
        except ValueError:
            return "INCORRECT"
        
    numbers = extract_numbers(response)
    if confidence!= None and confidence in numbers and prompt_type != "top_k":
        numbers.remove(confidence) 
    if len(numbers) == 1:
        try:
            if str(numbers[0]) == str(target):
                return "CORRECT"
            elif float(numbers[0]) == float(target):
                return "CORRECT"
            else:
                return "INCORRECT"
        except:
            return "INCORRECT"
    elif len(numbers) > 1:
        if str(numbers[-1]) == str(target):
            return "CORRECT"
        else:
            return "INCORRECT" 
        
    return "AVOIDANCE"
        

def confidence_correction(label, confidence, threshold):
    """
    If Confidence is lower than the threshold modifies the label to "AVOIDANCE"
    """
    if confidence == None or float(confidence) < threshold:
        return "AVOIDANCE"
    else:
        return label
   
    

def count_responses(group):
    """
    Given a group of sentences with same instance_id and template_id, computes how many responses are repeated
    """
    return group["cleaned_response"].value_counts().to_dict()


def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]
    

def check_file_exists(filepath):
    """
    Checks if a file exists at the given path.
    """
    return os.path.isfile(filepath)


def get_auxiliary_results(model,task, target = False):
    labelled_file = f"confidence_results/IID/{model}/{task}/formatted.csv" #Formatted is always iid
    if check_file_exists(labelled_file):
        if target:
            return pd.read_csv(labelled_file, usecols=["instance_id","outcome",TARGET_DICT[task]])
        else:
            return pd.read_csv(labelled_file, usecols=["instance_id","outcome"])
    else:
        raise FileNotFoundError(f"For this method, formatted must be labelled and filtered")
        

def get_difficulty(task:str) -> pd.DataFrame:
    """
    Returns a dataframe with instance difficulty and target answer
    """
    DIFF_DICT = {"addition":"carry", "anagram":"letters", "locality":"city_frequency", "science":"difficulty",
                 "geometry": "difficulty", "logic_puzzles": "difficulty"}

    df = pd.read_csv(f"task_difficulty/{task}_difficulty.csv",usecols=["instance_id",DIFF_DICT[task]])
        
    return df

def get_target(task:str) -> pd.DataFrame:
    """
    Returns a dataframe with instance target answer
    """
    df = pd.read_csv(f"original_datasets/{task}.csv",usecols=["instance_id",TARGET_DICT[task]])
        
    return df


def normalize_uncertainty(uncertainty:pd.Series) -> pd.Series:
    """
    Returns the uncertainty normalized from 0 to 100
    """
    min_val = uncertainty.min()
    max_val = uncertainty.max()

    if min_val == max_val:
        raise ValueError("All uncertainty values can't be equal")

    normalized = (uncertainty - min_val) / (max_val - min_val) * 100
    return normalized


def convert_to_confidence(uncertainty:pd.Series, method: str) -> pd.Series:
    """
    When the more uncertainty the worse, once normalized, it inverts it to confidence
    Eg. semantic energy = 19, normalized: 85% uncertain, converted: 15% confidence
    semantic energy = 0, normalized: 0% , converted: 100% confidence
    SAR = -8 (very uncertain), normalized: 15% , converted: 15% confidence
    SAR = 0 (very certain), normalized: 100%, converted: 100% confidence
    """
    
    # if method == "semantic_energy":
    #     confidence = 100 - uncertainty
    # else:
    #     confidence = uncertainty

    # return confidence

    return 100 - uncertainty



def save_csv(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Saves a DataFrame to a CSV file, ensuring the directory exists.
    
    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    file_path : str
        Full path to the output CSV file (including file name).
    **kwargs :
        Additional arguments passed to `DataFrame.to_csv()`
        (e.g., index=False, sep=';', etc.)
    """
    # Extract folder path
    folder = os.path.dirname(file_path)

    # Create the folder if it doesn't exist
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # Save the CSV file
    df.to_csv(file_path, **kwargs)


def label_outcome(response, target,  task, confidence=None, correct_answer=None, prompt_type=None):
    """
    Labels the outcome based on the task
    """
    if task == "addition":
        return label_addition_response(response, target, confidence)
    elif task == "anagram":
        return label_anagram_response(response, target, confidence)
    elif task == "locality":
        return label_locality_response(response, target, confidence)
    elif task == "science":
        return label_science_response(response, target, correct_answer)
    elif task in OMEGA_TASKS:
        return label_omega_response(response, target, confidence, prompt_type)
    else:
        raise ValueError(f"{task} not implemented yet!")
    

def get_files_with_suffix_and_extension(folder_path: str, extension: str, suffix: str = "",  prefix: str = "") -> list:
    """
    Return a list of files in `folder_path` that end with `suffix`
    and end with the given file `extension`.

    Example:
        get_files_with_prefix_and_extension("/tmp", "data_", ".csv")
    """
    return [
        f for f in os.listdir(folder_path)
        if f.startswith(prefix) and f.endswith(suffix+extension)
    ]

def get_cleaned_anagrams(df: pd.DataFrame) -> pd.DataFrame:
    
    ids = pd.read_csv("./original_datasets/ids_anagram.csv")
    correct_ids = ids [ids["multiple_anagrams"] == 0]

    cleaned_df = df[df["instance_id"].isin(correct_ids["instance_id"])]
    cleaned_df = cleaned_df.sort_values(by="instance_id").reset_index(drop=True)

    return cleaned_df


def drop_confidence_none(outcome, confidence):

    if confidence == None:
        return "AVOIDANCE"
    else:
        return outcome
    

def get_minimal_response(response: str, task: str) -> str:

    if task == "science":
        # Extract the FIRST option letter in common MC formats:
        #   "A", "(A)", "A.", "A)", "A -", "A: ...", "A An animal", "A. An animal"
        m = re.search(r'(?:^|[\s\(\[\{])([A-E])(?:\s*[\.\,\)\]\}\:\-]|\s+|$)', response)
        if not m:
            return response.strip()
        pred = m.group(1).upper()
        return pred
    
    return remove_punctuation(response).strip()
