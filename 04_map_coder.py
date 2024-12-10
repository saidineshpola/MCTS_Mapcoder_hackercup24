import asyncio
import json
import logging
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Tuple

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


def load_unique_tags(filename="unique_tags.json"):
    with open(filename, "r") as f:
        loaded_tags = json.load(f)
    return loaded_tags


# Load and verify
loaded_tags = load_unique_tags()

load_dotenv()
MODEL = "gpt-4o"
client = openai.AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPEN_AI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint="https://openai-ppcaz166.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview",
)

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
# Consider incorporating or drawing inspiration from the following algorithmic approaches:
{tags}
# Exemplars:
Recall {k} relevant and distinct problems (different from problem mentioned above). For each problem,
1. describe it
2. generate key algorithm from the problem and explain the solution following those algorithm.
3. finally generate a planning to solve that problem using the example problem
----------------
Important:
Your response must follow the JSON format specified in the response_format parameter.
{{"problems": [
    {{
        "description": "Problem Description",
        "algoritm": "algorithm",
        "planning": "Step by Step Planning along with code",
    }}
],
}}
"""

PLANNING_PROMPT = """Given a competitive programming problem, generate a concrete planning wthout any code and confidence score to solve the problem and evaluate its correctness.
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

----------------
Important: Your response must strictly adhere to the following JSON format:

{{
    "planning": "<A detailed string containing the step-by-step plan to solve the problem>",
    "algorithm_complexity":"the expected time complexity of the solution based on the given Input constraints",
    "confidence_score": <An integer between 0 and 100, where 0 means no confidence and 100 means absolute confidence>
}}

Ensure that the "planning" field contains a comprehensive, step-by-step approach to solving the problem, and the "confidence_score" is an integer value between 0 and 100.
"""

CODE_PROMPT = """Given a competitive programming problem generate most efficient Python code to solve the problem witout giving TLE for larger inputs.
## Problem to be solved:
{problem_description}
## Planning:
{planning}
## Expected Time Algorithm:
{algorithm_complexity}
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


TLE_PROMPT = """
Problem Statement:
{statement}
Code:
{code}
Your Task:
1. You are given a problem statement, together with a code that someone has written to solve the problem, above.
2. Next, you are given a piece of code.
3. Determine the time complexity of the code.
4. With the time complexity in (3), is the code fast enough to execute given the problem constraints?
5. If your answer to (4) is NO, think of an alternative approach to solve the problem, such that it is fast enough.
6. Ask yourself, does your new approach has equal or lower time complexity than the existing solution?
7. If your answer to (4) is YES or (6) is NO, code the exact same program that was given.
   Else, implement the code with better time complexity.
python
<insert necessary imports>
def solve(input: str) -> str:
"""

TIME_OUT = 10  # in seconds


@weave.op
async def call_model(messages, **kwargs):
    out = await client.chat.completions.create(model=MODEL, messages=messages, **kwargs)
    return out.choices[0].message.content


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


@dataclass
class Args(simple_parsing.Serializable):
    folder_path: Path = Path("./round3Data")
    weave_log: bool = False
    weave_eval: bool = False
    max_num_problems: int = 5
    on_sample: bool = True
    use_images: bool = False
    save_output: bool = True
    debug: bool = False
    timeout: int = 60


class BaseAgent:
    async def process(self, *args, **kwargs):
        raise NotImplementedError


class ExemplarAgent(BaseAgent):
    def __init__(self, k: int):
        self.k = k

    async def process(self, problem: Problem, system_prompt: str) -> Dict[str, Any]:
        exemplars_prompt = EXEMPLARS_PROMPT.format(
            problem_description=problem.problem_description,
            k=self.k,
            tags=loaded_tags["tags"],
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": exemplars_prompt},
        ]
        exemplars_response = await call_model(
            messages=messages, response_format={"type": "json_object"}, temperature=0.3
        )
        return json.loads(exemplars_response)


class PlanningAgent(BaseAgent):
    async def process(
        self, problem: Problem, exemplar_data: Dict[str, Any], system_prompt: str
    ) -> List[tuple]:
        plannings = []
        for example in exemplar_data.get("problems", []):
            planning_prompt = PLANNING_PROMPT.format(
                example_description=example["description"],
                example_planning=example["planning"],
                problem_description=problem.problem_description,
                sample_input=problem.sample_input,
                sample_output=problem.sample_output,
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
                        planning_data["algorithm_complexity"],
                    )
                )
        return sorted(plannings, key=lambda x: x[1], reverse=True)


class CodeGenerationAgent(BaseAgent):
    def __init__(self):
        self.code_pattern = r"```(?:python)?((.|\n)*?)```"

    def parse_code(self, response: str) -> str:
        if "```" not in response:
            return response
        code_blocks = re.findall(self.code_pattern, response, re.DOTALL)
        if code_blocks:
            return code_blocks[-1][0].strip()
        return response

    async def process(
        self,
        problem: Problem,
        planning: str,
        algorithm_complexity: str,
        system_prompt: str,
    ) -> str:
        code_prompt = CODE_PROMPT.format(
            algorithm_complexity=algorithm_complexity,
            problem_description=problem.problem_description,
            planning=planning,
            sample_input=problem.sample_input,
            sample_output=problem.sample_output,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": code_prompt},
        ]
        code_response = await call_model(messages=messages)
        return self.parse_code(code_response)


class DebuggerAgent(BaseAgent):
    async def process(
        self, problem: Problem, code: str, error: str, system_prompt: str
    ) -> str:
        error_prompt = ERROR_PROMPT.format(
            error=error,
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
        return CodeGenerationAgent().parse_code(corrected_code_response)


class OptimizationAgent(BaseAgent):
    async def process(self, problem: Problem, code: str, system_prompt: str) -> str:
        tle_prompt = TLE_PROMPT.format(
            statement=problem.problem_description,
            code=code,
        )
        messages = [
            {"role": "system", "content": TIME_COMPLEXITY_SYSTEM_PROMPT},
            {"role": "user", "content": tle_prompt},
        ]
        optimized_code_response = await call_model(messages=messages)
        return CodeGenerationAgent().parse_code(optimized_code_response)


class ManagerAgent:
    def __init__(self, system_prompt: str, k: int = 3, max_iterations: int = 1):
        self.system_prompt = system_prompt
        self.exemplar_agent = ExemplarAgent(k)
        self.planning_agent = PlanningAgent()
        self.code_generation_agent = CodeGenerationAgent()
        self.debugger_agent = DebuggerAgent()
        self.optimization_agent = OptimizationAgent()
        self.max_iterations = max_iterations

    async def run(self, problem: Problem) -> str:
        exemplar_data = await self.exemplar_agent.process(problem, self.system_prompt)
        plannings = await self.planning_agent.process(
            problem, exemplar_data, self.system_prompt
        )

        best_code = ""
        min_offending_cases = float("inf")

        for planning, confidence, algorithm_complexity in plannings:
            for _ in range(self.max_iterations):
                try:
                    code = await self.code_generation_agent.process(
                        problem,
                        planning,
                        algorithm_complexity,
                        self.system_prompt,
                    )
                    generated_output = await arun(
                        code, input=problem.sample_input, timeout=TIME_OUT
                    )
                    matches = check_solution(
                        problem.sample_output,
                        generated_output,
                        problem_name=problem.name,
                    )

                    if matches["matches"]:
                        return code

                    if len(matches["offending_cases"]) < min_offending_cases:
                        min_offending_cases = len(matches["offending_cases"])
                        best_code = code

                    # Try to debug the code
                    code = await self.debugger_agent.process(
                        problem,
                        code,
                        "Failed to produce expected output",
                        self.system_prompt,
                    )

                except asyncio.TimeoutError:
                    logging.warning(f"TLE for sample input")
                    code = await self.optimization_agent.process(
                        problem, code, self.system_prompt
                    )

                except Exception as e:
                    logging.error(f"Code execution failed: {str(e)}")
                    code = await self.debugger_agent.process(
                        problem, code, str(e), self.system_prompt
                    )

        return best_code


async def solve_problem(problem: Problem, on_sample=False) -> dict:
    manager = ManagerAgent(SYSTEM_PROMPT)
    code = await manager.run(problem)
    if on_sample:
        input, output = problem.sample_input, problem.sample_output
        full_input = problem.get_input()
        try:
            generated_full_output = await arun(code, input=full_input, timeout=60)
            problem.save_output(generated_full_output)
        except asyncio.TimeoutError:
            logging.warning(
                f"Full input execution timed out for problem: {problem.name}"
            )
        generated_output = await arun(code, input=input, timeout=20)
    else:
        logging.info(f"Solving on FULL INPUT")
        input, output = problem.get_input(), problem.get_output()
        generated_output = await arun(code, input=input, timeout=60)
        problem.save_output(generated_output)
    problem.save_code(code)
    return {
        "code": code,
        "generated_output": generated_output,
        "expected_output": output,
    }


def match(model_output: str, problem_name: str):
    matches = check_solution(
        model_output["expected_output"],
        model_output["generated_output"],
        problem_name,
    )
    return matches


async def solve_one(problem):
    try:
        model_output = await solve_problem(problem, Args.on_sample)
        matches = match(model_output, problem.name)
        logging.info(f"Problem {problem.name} results: {matches}")
        return {"runs": "✅", "error": None, **matches}
    except asyncio.TimeoutError:
        logging.error(f"Problem {problem.name} timed out")
        return {
            "runs": "❌",
            "error": "Time Limit Exceeded",
            "matches": -1,
            "total": -1,
            "offending_cases": [],
        }
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
        await evaluation.evaluate(
            solve_problem,
        )
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
                    ),
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
