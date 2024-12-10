import asyncio
from dataclasses import dataclass
from pathlib import Path
import logging
import time
import re
import xml.etree.ElementTree as ET
import json
import yaml

import openai
import weave
import simple_parsing
from tqdm.asyncio import tqdm

from mini_lib.problem import Problem
from mini_lib.utils import (
    arun,
    maybe_remove_backticks,
    check_solution,
    setup_logger,
    run,
)
from pydantic import BaseModel
from typing import Any, Dict, List
import sys
import subprocess

import os
from dotenv import load_dotenv

load_dotenv()

client = openai.AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPEN_AI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint="https://openai-ppcaz166.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview",
)


class ThinkerResponse:
    conditions: List[str]
    objectives: List[str]


class NewThinkerResponse:
    conditions: List[str]


class JudgeResponse:
    is_correct: List[bool]


class ExecutorResponse:
    code: str


system_prompt = """You are an expert Python programmer with a top rank on Codeforces (Grandmaster or International Grandmaster level).
    Your task is to create optimal Python code to solve the given problem, ensuring it can handle large inputs within the time constraints."""


@weave.op
async def call_model(messages, **kwargs):
    out = await client.chat.completions.create(
        model="gpt-4o", messages=messages, **kwargs
    )
    return out.choices[0].message.content


class MACM:
    def __init__(self, problem, k=3):
        self.k = k
        self.conditions = []
        self.objectives = []
        self.num_original_conditions = 0
        self.problem = problem

        prompt_template = """
        Problem: 
        {problem_description}

        Input: 
        {sample_input}

        Output: 
        {sample_output}
        """

        self.problem_statement = prompt_template.format(
            problem_description=problem.problem_description,
            sample_input=problem.sample_input,
            sample_output=problem.sample_output,
        )

    def parse_json(self, response: str) -> Dict[str, Any]:
        response = response.replace("```json", "").replace("```", "")
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # If parsing fails, try to wrap the content in a root object
            try:
                return json.loads('{"root": ' + response + "}")
            except json.JSONDecodeError:
                # If it still fails, return an empty dict
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

    @weave.op()
    async def think_initial(self):

        Analysis_conditions_objective = """
                                        Help me to analyze the conditions and the objective of a problem.
                                        You should only provide one objective.
                                        The conditions must be derived directly from the problem; deductions or calculations to establish these conditions are not allowed.
                                        You don't need to provide me with a solution for the time being.

                                        Example 1:
                                        Question:
                                        Problem:

                                        Dora has a set s containing integers. In the beginning, she will put all integers in [L, R] into the set s. That is, an integer x is initially contained in the set if and only if \(L \leq x \leq R\). Then she allows you to perform the following operations:

                                        1. Select three distinct integers a, b, c from the set s, such that gcd(a, b) = gcd(b, c) = gcd(a, c) = 1.
                                        2. Remove these three integers from the set s.

                                        What is the maximum number of operations you can perform?

                                        # Constraints

                                        \(1 \leq T \leq 500\)
                                        \(1 \leq L, R \leq 1000\)

                                        # Input Format

                                        Input begins with an integer \(T\), the number of test cases. Each case will contain one line with two space-separated integers, \(L\) and \(R\).


                                        # Output Format

                                        For the \(i\)th test case, print `"Case #i: "` followed by a single integer — the maximum number of operations you can perform.

                                        Conditions and Objectives:
                                        {{"conditions": ['Dora has a set s containing integers. In the beginning, she will put all integers in [L, R] into the set s. That is, an integer x is initially contained in the set if and only if \(L \leq x \leq R\). Then she allows you to perform the following operations:', 'Select three distinct integers a, b, c from the set s, such that gcd(a, b) = gcd(b, c) = gcd(a, c) = 1.', 'Remove these three integers from the set s.'], "objective": ['What is the maximum number of operations you can perform?']}}
                                        

                                        Real question:
                                        {Question}

                                        ----------------
                                        Important: Your response must strictly adhere to the following JSON format:

                                        {{
                                            "conditions": "<A list consisting of all the conditions>",
                                            "objective": "<A list consisting of all the objective>"
                                        }}
                                        """

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": Analysis_conditions_objective.format(
                    Question=self.problem_statement
                ),
            },
        ]

        response = await call_model(
            messages=messages, response_format={"type": "json_object"}
        )
        response = response.replace("```json", "").replace("```", "")
        try:
            response = json.loads(response)
        except:
            response = {}

        return response

    @weave.op()
    async def think_new(self):

        Discover_new_conditions = """
                                    I have some known conditions:
                                    {Known_conditions}
                                    And my objective is:
                                    {Objective}
                                    Please derive one direct condition with logical relationships based on the known conditions.
                                    NOTE:
                                    1. You are only allowed to use the known conditions to derive new conclusions.
                                    2. Feel free to use mathematical reasoning or coding to derive the new condition.

                                    ----------------
                                    Important:
                                    Your response must follow the JSON format specified in the response_format parameter.
                                    {{"conditions": ["Identify the new condition"], "reasoning": "Your reasoning to arrive at the new condition"}}
                                    """

        numbered_conditions = "\n".join(
            f"{i + 1}. {condition}" for i, condition in enumerate(self.conditions)
        )
        numbered_objective = "\n".join(
            f"{i + 1}. {objective}" for i, objective in enumerate(self.objectives)
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": Discover_new_conditions.format(
                    Known_conditions=numbered_conditions, Objective=numbered_objective
                ),
            },
        ]

        response = await (
            call_model(messages=messages, response_format={"type": "json_object"})
            .replace("```json", "")
            .replace("```", "")
        )
        logging.info(response)
        try:
            response = json.loads(response)
        except:
            response = {}

        return response

    @weave.op()
    async def judge(self):

        Judge_condition = """
                        I need you to determine if the statement is a condition included inside the problem.
                        You are only allowed to use the 'True' or 'False' as the final answer.
                        If it is correct, answer 'True'. 
                        If it is not correct, answer 'False'. 

                        Example 1:
                        problem:
                        Avery earns 30$ each day, how much will he earn for 30 days?
                        Statement:
                        Avery earns 20$ each day
                        Judgement:
                        False

                        Example 2:
                        problem:
                        The sum of two numbers is 25 and their difference is 9. What is their product?
                        Statement:
                        ['The sum of two numbers is 25']
                        Judgement:
                        True

                        Real question:
                        {question}
                        Statement:
                        {Initial_conditions}
                        Judgement:
                        """

        new_conditions = []
        for condition in self.conditions:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": Judge_condition.format(
                        question=self.problem_statement,
                        Initial_conditions=self.conditions,
                    ),
                },
            ]
            response = await call_model(messages=messages)
            response = bool(response)
            assert response in [True, False]
            if response == True:
                new_conditions.append(condition)

        self.conditions = new_conditions

    @weave.op()
    async def judge_new(self):

        Judge_T_F = """
                    I have some known conditions:
                    {Known_condtions}

                    and my problem objective:
                    {objective}

                    From the above conditions, I derived at this new condition:
                    {condition_from_thinker}

                    How confident are you that my new derived condition is correct? Think out loud, providing your mathematical reasoning.
                    ----------------
                    Important:
                    Your response must follow the JSON format specified in the response_format parameter.
                    {{"confidence": "Confidence score (from 0-100)", "reasoning": "Your reasoning"}}
                    """

        new_conditions = []

        for index, condition in enumerate(self.conditions):
            if index < len(self.conditions) - 1:
                new_conditions.append(condition)
                continue

            numbered_conditions = "\n".join(
                f"{i + 1}. {condition}" for i, condition in enumerate(new_conditions)
            )
            numbered_objective = "\n".join(
                f"{i + 1}. {objective}" for i, objective in enumerate(self.objectives)
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": Judge_T_F.format(
                        objective=numbered_objective,
                        Known_condtions=numbered_conditions,
                        condition_from_thinker=condition,
                    ),
                },
            ]
            response = await call_model(
                messages=messages, response_format={"type": "json_object"}
            )
            response = response.replace("```json", "").replace("```", "")

            try:
                response = json.loads(response)
                if int(response["confidence"]) >= 70:
                    new_conditions.append(condition)
            except:
                response = {}

        self.conditions = new_conditions

    @weave.op()
    async def determine_steps(self):

        Determine_Steps = """
                            Help me to come up with one solution targeting at our objective. 

                            Example 1:
                            Known conditions: 
                            1. f(3) = 5
                            2. for all x > 0, f(3x) = f(x) + 2
                            3. Because f(3x) = f(x) + 2, so f(9) = f(3) + 2
                            4. Because f(3x) = f(x) + 2, and f(3) = 5, so f(9) = 7
                            5. Because f(3x) = f(x) + 2, so f(27) = f(9) + 2
                            Objective:
                            find $f^{{-1}}(11)$
                            Solutions:
                            Step 1:
                            Use f(9) to find f(27)
                            Step 2:
                            Use f(27) to find f(81)
                            Step 3:
                            Repeat until find x that f(x) = 11

                            Example 2:
                            Known conditions: 
                            1. a + 1 / b = 22 / 7
                            2. b + 1 / c = 8
                            3. abc = 21
                            4. c = 1 / (8 - b)
                            5. a = (22 / 7) - (1 / b) 
                            Objective:
                            Calaculate c + 1 / a
                            Solutions:
                            Step 1:
                            Substitude a in abc = 21 by (22 / 7) - (1 / b) 
                            Step 2:
                            Substitude c in abc = 21 by 1 / (8 - b) 
                            Step 3:
                            Calculate b 
                            Step 4:
                            Calculate a and c
                            Step 5:
                            Calculate c + 1 / a

                            Real question:
                            Known conditions:
                            {Known_conditions}
                            Objective:
                            {Objective}
                            Solutions:
                            """

        numbered_conditions = "\n".join(
            f"{i + 1}. {condition}" for i, condition in enumerate(self.conditions)
        )
        numbered_objective = "\n".join(
            f"{i + 1}. {objective}" for i, objective in enumerate(self.objectives)
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": Determine_Steps.format(
                    Known_conditions=numbered_conditions, Objective=numbered_objective
                ),
            },
        ]

        self.solution = await call_model(messages=messages)

    @weave.op()
    async def solve_problem(self):

        find_target = """
                    Our objective is:
                    {Objective}
                    We have:
                    {Conditions}

                    ## Sample Test cases for reference:
                    Input:
                    {sample_input}
                    Output:
                    {sample_output}
                    
                    Now that you have the objectives, conditions and sample test cases:
                    1. Create a python program that returns the correct output for the given input. 
                    2. Make the code efficient and fast, so we can solve large inputs.

                    3. The file should have a single `solve` method that has the following signature:
                    input: [str]: The same Input provided above
                    output: [str]: The same Output provided above

                    ```python
                    from tqdm import tqdm
                    def solve(input: str) -> str: 
                    ```
                    """

        numbered_conditions = "\n".join(
            f"{i + 1}. {condition}" for i, condition in enumerate(self.conditions)
        )
        numbered_objective = "\n".join(
            f"{i + 1}. {objective}" for i, objective in enumerate(self.objectives)
        )

        final_prompt = find_target.format(
            Objective=numbered_objective,
            Conditions=numbered_conditions,
            sample_input=self.problem.sample_input,
            sample_output=self.problem.sample_output,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ]

        response = await call_model(messages=messages)
        code = self.parse_code(response)

        last_run_code = code

        # Check if the code passes sample test cases
        input_data, expected_output = (
            self.problem.sample_input,
            self.problem.sample_output,
        )
        try:
            generated_output = run(code, input=input_data, timeout=60)
            matches = check_solution(expected_output, generated_output)
            if matches["matches"]:
                return code
            else:
                least_offending_cases = len(matches["offending_cases"])
                last_run_code = code
                for _ in range(6):
                    # If matches fail, pass the failed input-output cases to the model
                    error_prompt = f"""
                                        Problem description:
                                        {self.problem.problem_description}

                                        Our objective is:
                                        {numbered_objective}
                                        We have:
                                        {numbered_conditions}
                                        
                                        Current code:
                                        {code}

                                        Your current code failed some of the sample test cases.
                                        
                                        Failed cases:
                                        {[f'Expected Output {x[0]}, Your Output {x[1]}' for x in matches['offending_cases']]}

                                        Please think out loud on what went wrong, and provide a corrected version of the code."""

                    logging.info(error_prompt)
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": error_prompt},
                    ]

                    corrected_code_response = call_model(messages=messages)
                    corrected_code = self.parse_code(corrected_code_response)

                    # Try running the corrected code
                    try:
                        generated_output = run(
                            corrected_code, input=input_data, timeout=60
                        )
                        matches = check_solution(expected_output, generated_output)
                        if matches["matches"]:
                            return corrected_code
                    except Exception as e:
                        # If corrected code still fails, continue to the next planning
                        logging.error(f"Corrected code execution failed: {str(e)}")

                    current_offending_cases = len(matches["offending_cases"])
                    if current_offending_cases < least_offending_cases:
                        last_run_code = corrected_code
                        least_offending_cases = current_offending_cases

        except Exception as e:
            # Handle other exceptions
            error_prompt = f"""The code execution failed due to an error: {str(e)}. Please fix the code to handle this error.
                                Current code:
                                {code}
                                Problem description:
                                {self.problem.problem_description}
                                Sample input:
                                {self.problem.sample_input}
                                Expected output:
                                {self.problem.sample_output}

                                Please think out loud on what went wrong, and provide a corrected version of the code."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": error_prompt},
            ]

            corrected_code_response = await call_model(messages=messages)
            corrected_code = self.parse_code(corrected_code_response)

            # Try running the corrected code
            try:
                generated_output = await asyncio.to_thread(
                    run, code, input=input_data, timeout=60
                )
                matches = check_solution(expected_output, generated_output)
                if matches["matches"]:
                    return corrected_code
            except Exception as e:
                # If corrected code still fails, continue to the next planning
                logging.error(f"Corrected code execution failed: {str(e)}")

        logging.info("LAST RUN CODE")
        logging.info(last_run_code)

        return last_run_code

    @weave.op()
    async def start_execution(self, timeout: int = 60):
        response = await self.think_initial()
        self.conditions.extend(response["conditions"])
        self.objectives.extend(response["objective"])
        await self.judge()

        for _ in range(self.k):
            response = await self.think_new()
            self.conditions.extend(response["conditions"])
            self.judge_new()
            logging.info(self.conditions)

        # self.determine_steps()
        code = await self.solve_problem()
        return code
        # with open("code_2.py", "w", encoding="utf-8") as fout:
        #     fout.write(code)

        # logging.info("> Solving on full input...")
        # expected_output = self.problem.get_output()
        # generated_output = run(code, input=self.problem.get_input(), timeout=timeout)
        # matches = check_solution(expected_output, generated_output)


@dataclass
class Args(simple_parsing.Serializable):
    folder_path: Path = Path(
        "./dataset/2023/practice"
    )  # path to the folder containing the problems
    weave_log: bool = False  # set to True to log to weave
    weave_eval: bool = False  # set to True to evaluate the code
    max_num_problems: int = 5  # set maximum number of problems to evaluate
    on_sample: bool = False  # run evaluation on sample inputs/outputs
    use_images: bool = False  # set to True to use images in the prompt
    save_output: bool = True  # set to True to save the output to a file
    debug: bool = False  # set to True to see the debug logs
    timeout: int = 120  # timeout for the code execution


async def solve_problem(problem: Problem, on_sample=False) -> dict:
    macm_agent = MACM(problem, k=0)
    code = await macm_agent.start_execution()
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
