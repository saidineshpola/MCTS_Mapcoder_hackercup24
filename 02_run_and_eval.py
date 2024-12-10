import asyncio
import json
import logging
from dataclasses import dataclass
import os
from pathlib import Path
import re
import time
from typing import Any, Dict

import openai
import weave
import simple_parsing
from tqdm.asyncio import tqdm

from mini_lib.problem import Problem
from mini_lib.utils import (
    maybe_remove_backticks,
    setup_logger,
    check_solution,
    arun,
    run,
)

from dotenv import load_dotenv

load_dotenv()
client = openai.AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPEN_AI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint="https://openai-ppcaz166.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview",
)


@weave.op
async def call_model(messages, **kwargs):
    out = await client.chat.completions.create(
        model="gpt-4o", messages=messages, **kwargs
    )
    return out.choices[0].message.content
    # response = (
    #     client.chat.completions.create(model="gpt-4o", messages=messages, **kwargs)
    #     .choices[0]
    #     .message.content
    # )

    # return response


@weave.op
async def generate_code(
    problem: Problem,
    system_prompt: str,
    prompt_template: str,
    extract_prompt: str,
    use_images: bool = False,
) -> str:
    logging.info(f"Generating code solution for: {problem.name}")

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_template.format(
                        problem_description=problem.problem_description,
                        sample_input=problem.sample_input,
                        sample_output=problem.sample_output,
                    ),
                }
            ]
            + (
                [
                    {"type": "image_url", "image_url": {"url": img}}
                    for img in problem.images
                ]
                if use_images
                else []
            ),
        },
    ]

    # call model one first time to get the code
    out = await call_model(messages=messages)
    logging.info("Generating initial analysis and solution")

    # Let's make a second call to the model to extract the code from the response
    messages.append({"role": "assistant", "content": out})
    messages.append(
        {"role": "user", "content": [{"type": "text", "text": extract_prompt}]}
    )

    # call model second time to extract the code
    solution = await call_model(messages=messages)
    logging.info("Extracting the solution from the previous generation...")

    # in case we have ```python stuff...`
    solution = maybe_remove_backticks(solution)
    return solution


# Prompts
SYSTEM_PROMPT = """You are an expert Python programmer with a top rank on Codeforces (Grandmaster or International Grandmaster level).
Your task is to create optimal Python code to solve the given problem, ensuring it can handle large inputs within the time constraints."""

TIME_COMPLEXITY_SYSTEM_PROMPT = """
You are an expert Python programmer with a top rank on Codeforces (Grandmaster or International Grandmaster level).
Your task is to create optimal Python code to solve the given problem.
Guide to Planning based on given problem statment variables.
Let n be the main variable in the problem.
Maximum allowed Time Complexity Guidelines:
n ≤ 12: O(n!)
n ≤ 25: O(2^n)
n ≤ 100: O(n^4)
n ≤ 500: O(n^3)
n ≤ 10^4: O(n^2)
n ≤ 10^6: O(n log n)
n ≤ 10^8: O(n)
n > 10^8: O(log n) or O(1)

Examples:
O(n!): Permutations
O(2^n): Subset exhaustion
O(n^3): Triangle enumeration
O(n^2): Slow sorting (Bubble, Insertion, Selection)
O(n log n): Fast sorting (Merge Sort)
O(n): Linear Search, Counting Sort
O(log n): Binary Search, Euclidean GCD
O(1): Simple calculations
"""

EXEMPLARS_PROMPT = """Given a problem, provide relevant problems then identify the algorithm behind it and also explain the tutorial of the algorithm.
# Problem:
{problem_description}

# Exemplars:
Recall {k} relevant and distinct problems (different from problem mentioned above). For each problem,
1. describe it
2. generate Python code step by step to solve that problem
3. finally generate a planning to solve that problem
4. provide a confidence score (0-100) for how well this problem relates to the original problem

# Algorithm:

----------------
Important:
Your response must follow the JSON format specified in the response_format parameter.
{{"problems": [
    {{
        "description": "Problem Description",
        "planning": "Step by Step Planning along with code",
    }}
],
"algorithm": "Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem. Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code."
}}
"""

PLANNING_PROMPT = """Given a competitive programming problem, generate a concrete planning and confidence score to solve the problem and evaluate its correctness.
# Example Problem:
{example_description}
# Example Problem planning:
{example_planning}
# Original Problem:
{problem_description}
## Sample Test cases:
Input:
{sample_input}
Output:
{sample_output}
## Algorithm:
{algorithm}

----------------
Important: Your response must strictly adhere to the following JSON format:

{{
    "planning": "<A detailed string containing the step-by-step plan to solve the problem>",
    "confidence_score": <An integer between 0 and 100, where 0 means no confidence and 100 means absolute confidence>
}}

Ensure that the "planning" field contains a comprehensive, step-by-step approach to solving the problem, and the "confidence_score" is an integer value between 0 and 100.
"""
# TODO : Remove '#' code After TACO evaluation

CODE_PROMPT = """Given a competitive programming problem generate Python code to solve the problem.
## Algorithm:
{algorithm}
## Problem to be solved:
{problem_description}
## Planning:
{planning}
## Sample Test cases:
Input:
{sample_input}
Output:
{sample_output}
## Let's think step by step.

----------------
Important:
## Your response must contain only the Python code to solve this problem. Do not add extra explanation or words.
Create a python program that returns the correct output for the given input. 
Make the code efficient and fast, so we can solve large inputs.
The file should have a single `solve` method that has the following signature:
input: [str]: The same Input provided above 
output [str]: The same Output provided above

```python
from tqdm import tqdm
def solve(input: str) -> str: 
    data = input.strip().split('\n')
```
"""


ERROR_PROMPT = """The code execution failed due to an error: {error}. Please fix the code to handle this error.
Current code:
{code}
Problem description:
{problem_description}
Sample input:
{sample_input}
Expected output:
{sample_output}

Please provide a corrected version of the code."""

FULL_INPUT_ERROR_PROMPT = """The code execution failed on the full input. Please fix the code to handle this error.
Problem description:
{problem_description}
Current code:
{code}
Error message:
{error}

Please provide a corrected version of the code that can handle the full input without errors."""


class MapCoder:
    def __init__(self, k=3, t=5):
        self.k = k
        self.t = t

    def parse_json(self, response: str) -> Dict[str, Any]:
        response = response.replace("```json", "").replace("```", "")
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                return json.loads('{"root": ' + response + "}")
            except json.JSONDecodeError:
                return {}

    def parse_code(self, response: str) -> str:
        if "```" not in response:
            return response
        code_pattern = r"```(?:python)?((.|\n)*?)```"
        code_blocks = re.findall(code_pattern, response, re.DOTALL)
        if code_blocks:
            return code_blocks[-1][0].strip()
        return response

    @staticmethod
    def trim_text(text: str, trimmed_text: str):
        return text.replace(trimmed_text, "").strip()

    async def generate_code(
        self, problem: Problem, system_prompt: str, use_images: bool = False
    ) -> str:
        # Step 1: Generate exemplars and algorithm
        exemplars_prompt = EXEMPLARS_PROMPT.format(
            problem_description=problem.problem_description, k=self.k
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": exemplars_prompt},
        ]
        exemplars_response = await call_model(
            messages=messages, response_format={"type": "json_object"}
        )
        exemplars_data = json.loads(exemplars_response)

        # Step 2: Generate planning and confidence score for each exemplar
        plannings = []
        for i, example in enumerate(exemplars_data.get("problems", [])):
            planning_prompt = PLANNING_PROMPT.format(
                example_description=example["description"],
                example_planning=example["planning"],
                problem_description=problem.problem_description,
                sample_input=problem.sample_input,
                sample_output=problem.sample_output,
                algorithm=exemplars_data["algorithm"],
            )
            messages = [
                {"role": "system", "content": TIME_COMPLEXITY_SYSTEM_PROMPT},
                {"role": "user", "content": planning_prompt},
            ]
            planning_response = await call_model(
                messages=messages, response_format={"type": "json_object"}
            )
            planning_data = json.loads(planning_response)
            if planning_data["planning"]:
                plannings.append(
                    (
                        planning_data["planning"],
                        planning_data["confidence_score"],
                        example,
                    )
                )

        if not plannings:
            logging.warning("No plannings found")
            return ""

        plannings.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Generate and evaluate code based on the plannings
        best_code = ""
        min_offending_cases = float("inf")
        temperature = 0.0
        for planning, confidence, example in plannings:
            code_prompt = CODE_PROMPT.format(
                algorithm=exemplars_data["algorithm"],
                problem_description=problem.problem_description,
                planning=planning,
                sample_input=problem.sample_input,
                sample_output=problem.sample_output,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": code_prompt},
            ]
            code_response = await call_model(messages=messages, temperature=temperature)
            code = self.parse_code(code_response)

            # Evaluate the code
            try:
                generated_output = await arun(
                    code, input=problem.sample_input, timeout=30
                )
                matches = check_solution(problem.sample_output, generated_output)

                if matches["matches"]:
                    return code  # Perfect match, return immediately

                if len(matches["offending_cases"]) < min_offending_cases:
                    min_offending_cases = len(matches["offending_cases"])
                    best_code = code

                # Try to correct the code if it's not perfect
                error_prompt = ERROR_PROMPT.format(
                    error="Failed to produce expected output",
                    code=code,
                    problem_description=problem.problem_description,
                    sample_input=problem.sample_input,
                    sample_output=problem.sample_output,
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": error_prompt},
                ]
                corrected_code_response = await call_model(messages=messages)
                corrected_code = self.parse_code(corrected_code_response)

                # Evaluate the corrected code
                generated_output = await arun(
                    corrected_code, input=problem.sample_input, timeout=30
                )
                matches = check_solution(problem.sample_output, generated_output)

                if matches["matches"]:
                    return corrected_code  # Perfect match, return immediately

                if (
                    matches["offending_cases"]
                    and len(matches["offending_cases"]) < min_offending_cases
                ):
                    min_offending_cases = len(matches["offending_cases"])
                    best_code = corrected_code

            except Exception as e:
                logging.error(f"Code execution failed: {str(e)}")

        return best_code  # Return the code with the least number of offending cases


@dataclass
class Args(simple_parsing.Serializable):
    folder_path: Path = Path(
        "./dataset/2023/practice"
    )  # path to the folder containing the problems
    weave_log: bool = False  # set to True to log to weave
    weave_eval: bool = False  # set to True to evaluate the code
    max_num_problems: int = 5  # set maximum number of problems to evaluate
    on_sample: bool = True  # run evaluation on sample inputs/outputs
    use_images: bool = False  # set to True to use images in the prompt
    save_output: bool = True  # set to True to save the output to a file
    debug: bool = False  # set to True to see the debug logs
    timeout: int = 120  # timeout for the code execution


async def solve_problem(problem: Problem, on_sample=False) -> dict:
    mapcoder = MapCoder()
    code = await mapcoder.generate_code(
        problem, system_prompt=SYSTEM_PROMPT, use_images=False
    )
    if on_sample:
        input, output = problem.sample_input, problem.sample_output
    else:
        input, output = problem.get_input(), problem.get_output()
    generated_output = await arun(code, input=input, timeout=60)
    problem.save_output(generated_output)
    problem.save_code(code)
    return {
        "code": code,
        "generated_output": generated_output,
        "expected_output": output,
    }


def match(model_output: str):
    matches = check_solution(
        model_output["expected_output"], model_output["generated_output"]
    )
    return matches


async def solve_one(problem):
    try:
        model_output = await solve_problem(problem)
        matches = match(model_output)
        logging.info(f"Problem {problem.name} results: {matches}")
        return {"runs": "✅", "error": None, **matches}
    except Exception as e:
        logging.error(f"Problem {problem.name} failed with error: {e}")
        return {
            "runs": "❌",
            "error": str(e),
            "matches": -1,
            "total": -1,
            "offending_cases": [],
        }


async def evaluate(problems):
    tasks = [solve_one(problem) for problem in problems]
    eval_results = await tqdm.gather(*tasks, desc="Solving problems...")
    return eval_results


async def main():
    args = simple_parsing.parse(Args)
    setup_logger(args.debug)
    t0 = time.perf_counter()

    problems = Problem.find_all(args.folder_path)[: args.max_num_problems]
    if args.weave_log:
        weave.init("hack-starter")

    if args.weave_eval:
        dataset = [{"problem": problem} for problem in problems]
        evaluation = weave.Evaluation(dataset=dataset, scorers=[match])
        await evaluation.evaluate(solve_problem)
    else:
        eval_results = await evaluate(problems)

        # Format the results in a pandas dataframe
        import pandas as pd
        from tabulate import tabulate

        df = pd.DataFrame(
            [
                {
                    "problem": problem.name,
                    "runs": result["runs"],
                    "error": result["error"],
                    "matches": result["total"] - len(result["offending_cases"]),
                    "offending_cases": len(result["offending_cases"]),
                    "total": result["total"],
                    "valid": (
                        "✅"
                        if (result["matches"] and result["matches"] != -1)
                        else "❌"
                    ),  # == result["total"]
                }
                for problem, result in zip(problems, eval_results)
            ]
        )
        logging.info("Evaluation results:")
        table = tabulate(df, headers="keys", tablefmt="pretty", showindex=False)
        print(table)
        logging.info(f"Evaluation took {time.perf_counter() - t0:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
