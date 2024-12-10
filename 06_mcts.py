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
import instructor

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

CODE_PROMPT = """Given a competitive programming problem and planning generate Python code to solve the problem.

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

TIME_OUT = 6  # in seconds


TLE_PROMPT = """
Problem Statement:
{statement}
Code:
{code}
Your Task:
1. You are given a problem statement, together with a code that someone has written to solve the problem, above.
2. Next, you are given a piece of code.
3. The code is not fast enough to execute given the problem constraints.
4. Think of an alternative approach to solve the problem, such that it is fast enough.
5. Ask yourself, does your new approach has lower time complexity than the existing solution?
6. Implement the code with better time complexity.
```python
from tqdm import tqdm
def solve(input: str) -> str: 
    data = input.strip().split('\n')
```
"""

TIME_OUT = 6  # in seconds


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


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_actions = None  # Will be set in expand()


class MCTS:
    def __init__(self, k=3, t=5, c=4, cbase=10, num_simulations=4):
        self.k = k
        self.t = t
        self.c = c  # exploration weight
        self.cbase = cbase
        self.num_simulations = num_simulations

    def parse_code(self, response: str) -> str:
        if "```" not in response:
            return response
        code_pattern = r"```(?:python)?((.|\n)*?)```"
        code_blocks = re.findall(code_pattern, response, re.DOTALL)
        if code_blocks:
            return code_blocks[-1][0].strip()
        return response

    async def generate_code(
        self, problem: Problem, system_prompt: str, use_images: bool = False
    ) -> str:
        root = MCTSNode(state=problem.problem_description)
        for _ in range(self.num_simulations):
            node = await self.select(root)
            if not node.untried_actions and not node.children:
                node = await self.expand(node)
            reward, feedback = await self.evaluate(node, problem, system_prompt)
            if reward >= 1.0:
                return feedback
            self.backpropagate(node, reward, feedback)
            if feedback:
                await self.rethink(node, feedback)
        best_child = max(root.children, key=lambda c: c.value / c.visits)
        return await self.generate_final_code(best_child, problem, system_prompt)

    async def select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            if not all(child.visits > 0 for child in node.children):
                return await self.expand(node)
            node = max(node.children, key=lambda c: self.puct_score(c))
        return node

    def puct_score(self, node: MCTSNode) -> float:
        parent_visits = sum(sibling.visits for sibling in node.parent.children)
        exploration_factor = math.sqrt(math.log(parent_visits) / node.visits)
        beta = math.log((parent_visits + self.cbase + 1) / self.cbase) + self.c
        return node.value / node.visits + beta * node.prior * exploration_factor

    async def expand(self, node: MCTSNode) -> MCTSNode:
        if not node.untried_actions:
            thoughts_and_scores = await self.generate_thoughts(
                node.state, node.feedback if hasattr(node, "feedback") else None
            )
            node.untried_actions = thoughts_and_scores

        if not node.untried_actions:  # Add safety check
            return node

        thought_data = node.untried_actions.pop(0)
        thought = thought_data["thought"]
        prior = thought_data["score"]
        new_state = node.state + "\n" + thought
        child = MCTSNode(state=new_state, parent=node)
        child.prior = prior
        node.children.append(child)
        return child

    async def evaluate(
        self, node: MCTSNode, problem: Problem, system_prompt: str
    ) -> tuple[float, str]:  # Added type hint
        code = await self.generate_code_from_thoughts(
            node.state, problem, system_prompt
        )
        try:
            generated_output = await arun(
                code, input=problem.sample_input, timeout=TIME_OUT
            )
            matches = check_solution(problem.sample_output, generated_output)

            if matches["matches"]:
                return 1.0, code

            v_test = 1 - (len(matches["offending_cases"]) / len(problem.sample_output))

            if v_test < 1:
                feedback = await self.get_block_level_feedback(code, problem)
                return v_test, feedback

            v_llm = await self.get_llm_evaluation(code, problem)
            return 0.5 * v_test + 0.5 * v_llm, None
        except asyncio.TimeoutError:
            return -0.5, "Time Limit Exceeded, Improve the code's time complexity"
        except Exception as e:
            return 0, str(e)

    def backpropagate(self, node: MCTSNode, reward: float, feedback: str):
        while node:
            node.visits += 1
            node.value += reward
            if feedback:
                node.feedback = feedback
            node = node.parent

    async def rethink(self, node: MCTSNode, feedback: str):
        improved_thoughts = await self.generate_improved_thoughts(node.state, feedback)
        if improved_thoughts and node.untried_actions:  # Add safety check
            node.untried_actions = improved_thoughts + node.untried_actions
        elif improved_thoughts:  # If only improved_thoughts exists
            node.untried_actions = improved_thoughts
        # If neither exists, do nothing

    async def generate_thoughts(self, state: str, feedback: str = None) -> list:
        try:
            prompt = f"""
            Given the current state of thoughts and problem description:
            {state}
            
            {"And considering the following feedback:" if feedback else ""}
            {feedback if feedback else ""}
            
            Generate {self.k} distinct thoughts or strategies for writing code to solve this problem.
            For each thought, provide a reasonableness score between 0 and 1.
            
            Format your response as a JSON object with a 'data' key containing an array of objects: 
            {{'data': [{{"thought": "thought1", "score": score1}}, {{"thought": "thought2", "score": score2}}, ...]}}
            """

            response = await call_model(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            thoughts_and_scores = json.loads(response)["data"]
            return thoughts_and_scores
        except Exception as e:
            logging.error(f"Error generating thoughts: {e}")
            return []

    async def generate_improved_thoughts(self, state: str, feedback: str) -> list:
        try:
            prompt = f"""
            Given the current state of thoughts:
            {state}
            
            And considering the following feedback:
            {feedback}
            
            Generate {self.k} improved thoughts or strategies for writing code to solve this problem.
            For each thought, provide a reasonableness score between 0 and 1.
            
            Format your response as a JSON object with a 'data' key containing an array of objects: 
            {{'data': [{{"thought": "thought1", "score": score1}}, {{"thought": "thought2", "score": score2}}, ...]}}
            """

            response = await call_model(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            # Parse the response to extract thoughts and scores
            thoughts_and_scores = json.loads(response).get(
                "data", []
            )  # Use .get() with default
            return thoughts_and_scores
        except Exception as e:
            logging.error(f"Error generating improved thoughts: {e}")
            return []

    async def generate_code_from_thoughts(
        self, thoughts: str, problem: Problem, system_prompt: str
    ) -> str:
        prompt = CODE_PROMPT.format(
            problem_description=problem.problem_description,
            planning=thoughts,
            sample_input=problem.sample_input,
            sample_output=problem.sample_output,
        )

        response = await call_model(messages=[{"role": "user", "content": prompt}])
        return self.parse_code(response)

    async def get_block_level_feedback(self, code: str, problem: Problem) -> str:
        try:
            generated_output = await arun(
                code, input=problem.sample_input, timeout=TIME_OUT
            )
            matches = check_solution(problem.sample_output, generated_output)

            if not matches["matches"]:
                return f"The code failed on the following test cases: {matches['offending_cases']}"
            return "The code passed all test cases, but may need optimization."
        except Exception as e:
            return f"Error during execution: {str(e)}"

    async def get_llm_evaluation(self, code: str, problem: Problem) -> float:
        prompt = f"""
        Given the following problem description and code solution, evaluate the correctness and efficiency of the code.
        Provide a score between 0 and 1, where 1 is perfectly correct and efficient.
        
        Problem:
        {problem.problem_description}
        
        Code:
        {code}
        
        Evaluate the code and provide ONLY single float value between 0 and 1 as the score.
        """

        response = await call_model(messages=[{"role": "user", "content": prompt}])
        return float(response.strip())

    async def generate_final_code(
        self, best_node: MCTSNode, problem: Problem, system_prompt: str
    ) -> str:
        thoughts = best_node.state
        return await self.generate_code_from_thoughts(thoughts, problem, system_prompt)


@dataclass
class Args(simple_parsing.Serializable):
    folder_path: Path = Path("./round3Data")
    weave_log: bool = False
    weave_eval: bool = False
    max_num_problems: int = 7
    on_sample: bool = True
    use_images: bool = False
    save_output: bool = True
    debug: bool = False
    timeout: int = 80


async def solve_problem(problem: Problem, on_sample=False) -> dict:
    mctscoder = MCTS()
    code = await mctscoder.generate_code(
        problem, system_prompt=SYSTEM_PROMPT, use_images=False
    )
    problem.save_code(code)
    if on_sample:
        input, output = problem.sample_input, problem.sample_output
        full_input = problem.get_input()
        try:
            generated_full_output = await arun(
                code, input=full_input, timeout=TIME_OUT * 15
            )
            problem.save_output(generated_full_output)
        except asyncio.TimeoutError:
            logging.warning(
                f"Full input execution timed out for problem: {problem.name}"
            )
        generated_output = await arun(code, input=input, timeout=10)
    else:
        logging.info(f"Solving on FULL INPUT")
        input, output = problem.get_input(), problem.get_output()
        generated_output = await arun(code, input=input, timeout=60)
        problem.save_output(generated_output)

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


# async def tle_fix(problem: Problem):
#     # step1 get the code
#     code = generated_codes[problem.name]
#     # step2 run the tle with code for fixing it
#     input, output = problem.sample_input, problem.sample_output
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": TLE_PROMPT.format(
#                         problem_statement=problem.problem_description, code=code
#                     ),
#                 }
#             ],
#         },
#     ]

#     code = await call_model(messages=messages)
#     full_input = problem.get_input()
#     try:
#         generated_full_output = await arun(code, input=full_input, timeout=60)
#         problem.save_output(generated_full_output)
#         {
#             "code": code,
#             "expected_output": output,
#         }
#     except asyncio.TimeoutError:
#         logging.warning(f"TLE feedback also gave TLE: {problem.name}")
#         return {
#             "runs": "❌",
#             "error": "Time Limit Exceeded",
#             "matches": -1,
#             "total": -1,
#             "offending_cases": [],
#         }


async def solve_one(problem):
    try:
        model_output = await solve_problem(problem, Args.on_sample)
        matches = match(model_output)
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


async def evaluate(problems: List[Problem]):
    tasks = [solve_one(problem) for problem in problems if "Coin" in problem.name]
    eval_results = await tqdm.gather(*tasks, desc="Solving problems...")
    return eval_results


async def main():
    args = simple_parsing.parse(Args)
    setup_logger(args.debug)
    t0 = time.perf_counter()

    problems = Problem.find_all(args.folder_path)[: args.max_num_problems]
    problems = [problem for problem in problems]
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
