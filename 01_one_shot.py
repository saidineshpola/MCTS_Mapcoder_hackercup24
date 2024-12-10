from dataclasses import dataclass
from pathlib import Path
import logging
import time

import openai
import weave
import simple_parsing
from tqdm import tqdm

from mini_lib.problem import Problem
from mini_lib.utils import maybe_remove_backticks, check_solution, setup_logger, run

client = openai.AzureOpenAI(
    api_key='',
    api_version= '2024-02-15-preview',
    azure_endpoint="https://openai-ppcaz166.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview",
)

@weave.op
def call_model(messages, **kwargs):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        **kwargs
    ).choices[0].message.content

@weave.op
def generate_code(
    problem: Problem, 
    system_prompt: str, 
    prompt_template: str, 
    extract_prompt: str,
    use_images: bool = False,
    attempt: int = 1,
    previous_attempt_info: dict = None
) -> str:
    logging.info(f"Generating code solution for: {problem.name} (Attempt {attempt})")
    formatted_prompt = prompt_template.format(
        problem_description=problem.problem_description,
        sample_input=problem.sample_input,
        sample_output=problem.sample_output,
        attempt=attempt,
        previous_attempt_info=previous_attempt_info
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": formatted_prompt}
        ] + ([{"type": "image_url", "image_url": {"url": img}} for img in problem.images] if use_images else [])}
    ]

    out = call_model(messages=messages)
    logging.info("Generating initial analysis and solution")

    messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": [
        {"type": "text", "text": extract_prompt}
    ]})

    solution = call_model(messages=messages)
    logging.info("Extracting the solution from the previous generation...")

    solution = maybe_remove_backticks(solution)
    return solution


system_prompt = """You are an expert Python programmer with a top rank on Codeforces (Grandmaster or International Grandmaster level).\
    Your task is to create optimal Python code to solve the given problem, ensuring it can handle large inputs within the time constraints."""
prompt_template = """
Problem: 
{problem_description}

Input: 
{sample_input}

Output: 
{sample_output}

Create a python program that returns the correct output for the given input. 
The file should have a single `solve` method that has the following signature:
input: [str]: The same Input provided above
output [str]: The same Output provided above

{previous_attempt_info}

```python
from tqdm import tqdm
def solve(input: str) -> str: 
```
"""

extract_prompt = """
Extract the code from the response. reply with the code only. Omit any additional example or explanation.
- If the solution involves a for loop, please use `for sample in tqdm(range(samples))` to show progress.
- The code should be a valid python program.
- Get the `solve` function with the corresponding imports"""

@weave.op
def solve_problem(problem: Problem, use_images=False, timeout=60, max_attempts=3) -> dict:
    previous_attempt = None
    for attempt in range(1, max_attempts + 1):
        previous_attempt_info = ""
        if previous_attempt:
            previous_attempt_info = f"""
            Previous attempt code:
            {previous_attempt['code']}
            
            Previous attempt output:
            {previous_attempt['generated_output']}
            
            Expected output:
            {previous_attempt['expected_output']}
            
            Please analyze the previous attempt, identify any issues, and provide an improved solution.
            """
        
        code = generate_code(
            problem, 
            system_prompt=system_prompt, 
            prompt_template=prompt_template, 
            extract_prompt=extract_prompt, 
            use_images=use_images,
            attempt=attempt,
            previous_attempt_info=previous_attempt_info
        )

        
        input, output = problem.sample_input, problem.sample_output
        generated_output = run(code, input=input, timeout=timeout) 
        
        matches = check_solution(output, generated_output)
        logging.info(f"Attempt {attempt} - Sample Matches: {matches}")
        
        if matches['matches']:
            logging.info(f"Solution found on attempt {attempt}")
            return {"code": code, "generated_output": generated_output, "expected_output": output, "attempts": attempt}
        
        previous_attempt = {"code": code, "generated_output": generated_output, "expected_output": output}
        logging.info(f"Attempt {attempt} failed. Trying again...")
        time.sleep(2)  # Add a small delay between attempts
    
    logging.warning(f"Failed to find a solution after {max_attempts} attempts")
    return {"code": code, "generated_output": generated_output, "expected_output": output, "attempts": max_attempts}

@dataclass
class Args(simple_parsing.Serializable):
    problem_name: str = "cheeseburger_corollary_ch1" # name of the problem to solve
    folder_path: Path = Path("./dataset/2023/practice/") # path to the folder containing the problems
    weave_log: bool = False # set to True to log to weave
    use_images: bool = False # set to True to use images in the prompt
    save_output: bool = True # set to True to save the output to a file
    debug: bool = False # set to True to see the debug logs
    timeout: int = 60 # timeout for the code execution
    max_attempts: int = 3 # maximum number of attempts to solve the problem

if __name__=="__main__":
    args = simple_parsing.parse(Args)

    setup_logger(args.debug)

    problem = Problem.from_name(args.problem_name, args.folder_path)

    if args.weave_log: 
        weave.init("hack-starter")
    
    logging.info("> Solving on sample input...")
    problem_solution = solve_problem(problem, use_images=args.use_images, timeout=args.timeout, max_attempts=args.max_attempts)
    # Debug
    if args.save_output:
        logging.info("> Saving output to files")
        problem.save_output(problem_solution["generated_output"])
        problem.save_code(problem_solution["code"])
    logging.info(f"> Using final Solution after {problem_solution['attempts']} attempts")
    
    logging.info("> Solving on full input...")
    expected_output = problem.get_output()
    generated_output = run(problem_solution["code"], input=problem.get_input(), timeout=args.timeout) 
    matches = check_solution(expected_output, generated_output)
    logging.info("Final Matches:")
    logging.info(matches)
