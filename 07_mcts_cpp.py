# OBSERVATION: Not working good with gpt4o so many TLEs

import asyncio
import json
import logging
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Literal, Tuple

import openai
import weave
import simple_parsing
from tqdm.asyncio import tqdm

from mini_lib.problem import Problem
from mini_lib.utils import (
    maybe_remove_backticks,
    setup_logger,
    check_solution,
    arun_cpp,
    run,
)

from dotenv import load_dotenv
import instructor
from anthropic import AsyncAnthropic


def load_unique_tags(filename="unique_tags.json"):
    with open(filename, "r") as f:
        loaded_tags = json.load(f)
    return loaded_tags


# Load and verify
loaded_tags = load_unique_tags()
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_AI_KEY",""))
load_dotenv()
MODEL = "gpt-4o"
client = None

# Prompts
SYSTEM_PROMPT = """You are an expert C++ programmer with a top rank on Codeforces (Grandmaster or International Grandmaster level).
Your task is to create optimal C++ code to solve the given problem, ensuring it can handle large inputs within the time constraints."""

TIME_COMPLEXITY_SYSTEM_PROMPT = """
You are an expert CPP programmer with a top rank on Codeforces (Grandmaster or International Grandmaster level).
Your task is to create optimal C++ code to solve the given problem.
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

TIME_OUT = 20  # in seconds


CODE_PROMPT = """Given a competitive programming problem and planning generate C++ code to solve the problem.
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
--------------------------------------------------------
Important:
## Your response must contain only the C++ code to solve this problem. Do not add extra explanation or words.
Create a C++ program that returns the correct output for the given input. 
Make the code efficient and fast, so we can solve large inputs.
Include both the solve function and a main function along with all the imports that are necessary.
CODE TEMPLATE:

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef vector<int> vi;
typedef vector<ll> vll;
typedef pair<int,int> pii;

template<typename T>
T safe_convert(const string& str) {{
    T result;
    stringstream ss(str);
    ss >> result;
    return result;
}}

template<typename T>
string safe_to_string(const T& value) {{
    stringstream ss;
    ss << value;
    return ss.str();
}}

class Solution {{
public:
    string solve(const string& input) {{
        stringstream ss(input);
        string result;
        
        // Solution implementation here
        
        return result;
    }}
}};

int main() {{
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    string input;
    string line;
    while (getline(cin, line)) {{
        input += line + '\\n';
    }}
    
    Solution solution;
    try {{
        string result = solution.solve(input);
        cout << result;
    }} catch (const exception& e) {{
        cerr << "Error: " << e.what() << endl;
        return 1;
    }}
    
    return 0;
}}
```
"""


@weave.op
async def call_model(messages, **kwargs):
    out = await client.chat.completions.create(model=MODEL, messages=messages, **kwargs)
    return out.choices[0].message.content


@weave.op
async def call_model_claude(messages, **kwargs):
    out = await anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=8192,
        messages=messages,
        **kwargs,
    )
    return out.content[0].text


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

    # in case we have ```cpp stuff...`
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
    def __init__(self, k=5, t=5, c=4, cbase=10, num_simulations=8):
        self.k = k  # No. of thoughts
        self.t = t  # Not used
        self.c = c  # exploration weight
        self.cbase = cbase
        self.num_simulations = num_simulations

    def parse_code(self, response: str) -> str:
        if "```" not in response:
            return response
        code_pattern = r"```(?:cpp)?((.|\n)*?)```"
        code_blocks = re.findall(code_pattern, response, re.DOTALL)
        if code_blocks:
            return code_blocks[-1][0].strip()
        return response

    async def generate_code(
        self, problem: Problem, system_prompt: str, use_images: bool = False
    ) -> str:
        root = MCTSNode(state=problem.problem_description)
        best_solutions = []  # Store solutions with best performance metrics

        for _ in range(self.num_simulations):
            node = await self.select(root)
            if not node.untried_actions and not node.children:
                node = await self.expand(node)

            # Enhanced evaluation with multiple performance metrics
            reward, performance_metrics, all_pass_test_cases = (
                await self.evaluate_comprehensive(node, problem, system_prompt)
            )

            # Store solutions based on comprehensive performance
            if (
                all_pass_test_cases
            ):  # or performance_metrics["test_case_coverage"] >= 0.5:
                logging.info(f"DEBUG:::::::{reward}, {all_pass_test_cases}")
                best_solutions.append(
                    {
                        "code": performance_metrics["code"],
                        "metrics": performance_metrics,
                    }
                )

            self.backpropagate(node, reward, performance_metrics.get("feedback", ""))

            if performance_metrics.get("feedback"):
                await self.rethink(node, performance_metrics["feedback"])

        # If no solutions were found, use the best explored solution
        if not best_solutions:
            logging.warning(f"No solution found in best_solution")
            best_child = max(root.children, key=lambda c: c.value / c.visits)
            return await self.generate_final_code(best_child, problem, system_prompt)

        # Select best solution based on comprehensive performance metrics
        return await self.select_best_solution(best_solutions, problem)

    async def select_best_solution(
        self, solutions: List[dict], problem: Problem
    ) -> str:
        """
        Select the best solution based on multiple performance metrics
        """

        def score_solution(solution):
            metrics = solution["metrics"]
            # if metrics["test_case_coverage"] >= 1.0:
            #     return metrics["time_complexity_score"]
            # return 0
            return (
                metrics["test_case_coverage"] * 0.4
                + metrics["time_complexity_score"] * 0.3
                + metrics["memory_efficiency_score"] * 0.2
                + metrics["code_quality_score"] * 0.1
            )

        # Sort solutions by comprehensive performance score
        ranked_solutions = sorted(solutions, key=score_solution, reverse=True)
        return ranked_solutions[0]["code"]

    async def evaluate_comprehensive(
        self, node: MCTSNode, problem: Problem, system_prompt: str
    ) -> tuple[float, dict]:
        """
        Comprehensive code evaluation with multiple performance metrics
        """
        code = await self.generate_code_from_thoughts(
            node.state, problem, system_prompt, retry_from_error=False
        )

        async def execute_comprehensive_evaluation(code, problem, retry=False):
            try:
                # Execute code and collect multiple metrics
                performance_metrics = await self.analyze_code_performance(code, problem)

                # Calculate overall reward
                test_case_reward = performance_metrics["test_case_coverage"]
                complexity_reward = performance_metrics["time_complexity_score"]
                logging.info(
                    f"Overall testcase coverage for {problem.name}: {test_case_reward},Time Complexity score:{complexity_reward}"
                )
                # Combined reward calculation
                overall_reward = 0.6 * test_case_reward + 0.4 * complexity_reward

                # If performance is poor, attempt to regenerate code
                if overall_reward < 0.5 and not retry:
                    logging.info(
                        "Attempting to improve code based on performance metrics"
                    )
                    new_code = await self.generate_code_from_thoughts(
                        json.dumps(performance_metrics),
                        problem,
                        system_prompt,
                        retry_from_error=True,
                        code=code,
                    )
                    return await execute_comprehensive_evaluation(
                        new_code, problem, retry=True
                    )

                performance_metrics["code"] = code
                return (
                    overall_reward,
                    performance_metrics,
                    True if test_case_reward == 1.0 else False,
                )

            except Exception as e:
                logging.error(f"Comprehensive evaluation error: {e}")
                return (
                    0,
                    {
                        "test_case_coverage": 0,
                        "time_complexity_score": 0,
                        "memory_efficiency_score": 0,
                        "code_quality_score": 0,
                        "feedback": str(e),
                        "code": code,
                    },
                    False,
                )

        return await execute_comprehensive_evaluation(code, problem)

    async def analyze_code_performance(self, code: str, problem: Problem) -> dict:
        """
        Analyze code performance with multiple metrics
        """
        try:
            # Execute code with sample input
            generated_output = await arun_cpp(
                code, input=problem.sample_input, timeout=TIME_OUT
            )

            # Check solution correctness
            matches = check_solution(
                problem.sample_output, generated_output, problem.name
            )

            # Test case coverage
            test_case_coverage = 1 - (
                len(matches["offending_cases"]) / matches["total"]
            )

            # Time complexity analysis (basic estimation)
            time_complexity_score = await self.estimate_time_complexity(code)

            # Memory efficiency estimation
            memory_efficiency_score = 0  # TODO: self.estimate_memory_efficiency(code)

            # Code quality score (basic heuristics)
            code_quality_score = 0  # TODO: self.estimate_code_quality(code)

            # Generate specific feedback if test cases fail
            feedback = (
                f"Failed test cases: {matches['offending_cases']}"
                if test_case_coverage < 1.0
                else ""
            )

            return {
                "test_case_coverage": test_case_coverage,
                "time_complexity_score": time_complexity_score,
                "memory_efficiency_score": memory_efficiency_score,
                "code_quality_score": code_quality_score,
                "feedback": feedback,
            }

        except Exception as e:
            logging.error(f"Performance analysis error: {e}")
            return {
                "test_case_coverage": 0,
                "time_complexity_score": 0,
                "memory_efficiency_score": 0,
                "code_quality_score": 0,
                "feedback": str(e),
            }

    async def estimate_time_complexity(self, code: str) -> float:
        """
        Estimate time complexity and return integer scores.

        Args:
            code (str): The code snippet to analyze

        Returns:
            Dict with 'scores' key containing integer scores
        """
        try:
            # Comprehensive prompt for time complexity scoring
            prompt = f"""Analyze the time complexity of the following code and provide integer scores:

    Code:
    ```python
    {code}
    ```

    Score Guidelines:
    - O(1) Constant Time: 100 points
    - O(log n) Logarithmic Time: 90 points
    - O(n) Linear Time: 80 points
    - O(n log n) Linearithmic Time: 70 points
    - O(n^2) Quadratic Time: 50 points
    - O(n^3) Cubic Time: 30 points
    - O(2^n) Exponential Time: 10 points

    Scoring Criteria:
    1. Evaluate algorithmic structure
    2. Consider nested loops, recursive calls
    3. Assess data structure operations
    4. Focus on worst-case computational complexity

    Response must be a valid JSON object with a 'scores' key:
    {{
        "score": <integer score>
    }}

    Provide a single integer score from 0 to 100 based on the most dominant time complexity."""

            # Actual LLM call would replace this mock implementation
            response = await call_model(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            # Ensure we return a valid dictionary with scores
            return json.loads(response).get("score", 0) / 100

        except Exception as e:
            logging.error(f"Error in time complexity estimation: {e}")
            return {"scores": 0}

    def estimate_memory_efficiency(self, code: str) -> float:
        """
        Estimate memory efficiency based on data structures and allocation
        """
        memory_indicators = {
            "vector<vector<": 0.2,  # High memory usage
            "new ": 0.3,  # Dynamic allocation
            "malloc(": 0.3,  # C-style memory allocation
        }

        score = 1.0  # Start with perfect score
        for indicator, penalty in memory_indicators.items():
            if indicator in code:
                score -= penalty

        return max(0, score)

    def estimate_code_quality(self, code: str) -> float:
        """
        Basic code quality estimation
        """
        quality_indicators = {
            "magic numbers": -0.2,
            "long functions": -0.1,
            "global variables": -0.1,
            "complex conditionals": -0.1,
        }

        score = 1.0  # Start with perfect score
        for indicator, penalty in quality_indicators.items():
            # Add simple heuristics for each indicator
            if indicator == "magic numbers" and re.search(r"\b\d+\b", code):
                score += penalty
            # Add more complex checks as needed

        return max(0, score)

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
    ) -> tuple[float, str]:
        code = await self.generate_code_from_thoughts(
            node.state, problem, system_prompt, retry_from_error=False
        )

        async def execute_code(code, problem, retry=False):
            try:
                generated_output = await arun_cpp(
                    code, input=problem.sample_input, timeout=TIME_OUT
                )
                matches = check_solution(
                    problem.sample_output, generated_output, problem.name
                )

                if matches["matches"]:
                    return 1.0, code

                v_test = 1 - (
                    len(matches["offending_cases"]) / len(problem.sample_output)
                )

                if v_test < 1:
                    feedback = await self.get_block_level_feedback(code, problem)
                    # return v_test, feedback

                v_llm = await self.get_llm_evaluation(code, problem)
                return 0.5 * v_test + 0.5 * v_llm, feedback

            except asyncio.TimeoutError:
                return -0.5, "Time Limit Exceeded, Improve the code's time complexity"

            except Exception as e:
                logging.error(f"Error executing code: {e}")

                # Add retry logic if not already retried
                if not retry:
                    logging.info("Attempting to regenerate and retry code")
                    # Optionally, you could modify the code generation here
                    new_code = await self.generate_code_from_thoughts(
                        str(e), problem, system_prompt, retry_from_error=True, code=code
                    )
                    return await execute_code(new_code, problem, retry=True)

                return 0, str(e)

        return await execute_code(code, problem)

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
    
    Generate {self.k} distinct thoughts or strategies for solving this problem. Consider incorporating or drawing inspiration from the following algorithmic approaches:
    {', '.join(loaded_tags['tags'])}
    
    For each thought, do the following:
    1. Describe a potential approach to solving the problem
    2. If applicable, mention how a specific algorithm from the suggested list might be relevant
    3. Provide a reasonableness score between 0 and 1, considering:
       - Feasibility of implementation
       - Potential effectiveness for the given problem
       - Alignment with the suggested algorithmic approaches

    Format your response as a JSON object with a 'data' key containing an array of objects: 
    {{'data': [
        {{"thought": "detailed thought description", 
          "algorithm_inspiration": "relevant algorithm name (optional)", 
          "score": score}}, 
        ...
    ]}}
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

    async def generate_thoughts_claude(self, state: str, feedback: str = None) -> list:
        try:
            claude_prompt = f"""
            Given the current state of thoughts and problem description:
            {state}
            
            {"And considering the following feedback:" if feedback else ""}
            {feedback if feedback else ""}
            
            Generate {self.k} distinct thoughts or strategies or useful mathmatical algorithms for writing code to solve this problem.
            For each thought, provide a reasonableness score between 0 and 1.
            NOTE: Provided thoughts and strategies should consider TLE with high priority.
            
            """

            claude_response = await call_model_claude(
                messages=[{"role": "user", "content": claude_prompt}],
            )
            prompt = f"""
            Given the response of thoughts create JSON response:
            # Problem
            {claude_response}
            Format the response as a JSON object with a 'data' key containing an array of objects: 
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
        self,
        thoughts_or_error: str,
        problem: Problem,
        system_prompt: str,
        retry_from_error=False,
        code="",
    ) -> str:
        if retry_from_error:
            prompt = ERROR_PROMPT.format(
                problem_description=problem.problem_description,
                code=code,
                error=thoughts_or_error,
                sample_input=problem.sample_input,
                sample_output=problem.sample_output,
            )
        else:
            prompt = CODE_PROMPT.format(
                problem_description=problem.problem_description,
                planning=thoughts_or_error,
                sample_input=problem.sample_input,
                sample_output=problem.sample_output,
            )

        response = await call_model(messages=[{"role": "user", "content": prompt}])
        return self.parse_code(response)

    async def get_block_level_feedback(self, code: str, problem: Problem) -> str:
        try:
            generated_output = await arun_cpp(
                code, input=problem.sample_input, timeout=TIME_OUT
            )
            matches = check_solution(
                problem.sample_output, generated_output, problem.name
            )

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




async def solve_problem(problem: Problem, on_sample=False) -> dict:
    mctscoder = MCTS(k=4)
    code = await mctscoder.generate_code(
        problem, system_prompt=SYSTEM_PROMPT, use_images=False
    )
    problem.save_code_cpp(code)

    input, output = problem.sample_input, problem.sample_output

    try:
        generated_output = await arun_cpp(code, input=input, timeout=TIME_OUT)
        matches = match(
            {
                "code": code,
                "generated_output": generated_output,
                "expected_output": output,
            },
            problem.name,
        )

        # If sample test cases fail, return the failure details without further execution
        if not matches["matches"]:
            return {
                "code": code,
                "generated_output": generated_output,
                "expected_output": output,
                "matches": matches,
            }

        # If sample tests pass and on_sample is True, run on full input
        if on_sample:
            logging.info(f"**Running on full input from on sample for {problem.name}")
            full_input = problem.get_input()
            generated_full_output = await arun_cpp(
                code, input=full_input, timeout=TIME_OUT * 6
            )
            problem.save_output(generated_full_output)
            return {
                "code": code,
                "generated_output": generated_full_output,
                "expected_output": problem.get_output(),
            }

        # If not on_sample, run on full input directly
        else:
            logging.info(f"Solving on FULL INPUT")
            input, output = problem.get_input(), problem.get_output()
            generated_full_output = await arun_cpp(
                code, input=input, timeout=TIME_OUT * 8
            )
            problem.save_output(generated_full_output)
            return {
                "code": code,
                "generated_output": generated_full_output,
                "expected_output": output,
            }

    except Exception as e:
        logging.error(f"Error solving problem {problem.name}: {e}")
        raise


def match(model_output: str, problem_name: Any):
    matches = check_solution(
        model_output["expected_output"], model_output["generated_output"], problem_name
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


async def evaluate(problems: List[Problem]):
    tasks = [solve_one(problem) for problem in problems]
    eval_results = await tqdm.gather(*tasks, desc="Solving problems...")
    return eval_results


@dataclass
class Args(simple_parsing.Serializable):
    folder_path: Path = Path("./round3Data")
    weave_log: bool = False
    weave_eval: bool = False
    max_num_problems: int = 7
    on_sample: bool = False
    use_images: bool = False
    save_output: bool = True
    debug: bool = False
    timeout: int = 80
    client_type: Literal['anthropic', 'azure_openai', 'openai'] = 'azure_openai'
    model: str = None  # Optional specific model override

class AIClientFactory:
    @staticmethod
    def get_client(
        client_type: Literal['anthropic', 'azure_openai', 'openai'] = 'azure_openai', 
        model: str = None
    ):
        """
        Factory method to create different AI clients
        
        :param client_type: Type of AI client to create
        :param model: Specific model to use
        :return: Configured AI client
        """
        # Default models for each client
        default_models = {
            'anthropic': 'claude-3-sonnet-20240229',
            'azure_openai': 'gpt-4o',
            'openai': 'gpt-3.5-turbo'
        }
        
        # Use provided model or default
        model = model or default_models.get(client_type, 'gpt-4o')
        
        if client_type == 'anthropic':
            return AsyncAnthropic(
                api_key=os.getenv("ANTHROPIC_AI_KEY", "")
            )
        elif client_type == 'azure_openai':
            return openai.AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_OPEN_AI_KEY"),
                api_version=os.getenv("AZURE_OPEN_AI_KEY"),
                azure_endpoint=os.getenv("AZURE_OPEN_AI_KEY"),
            )
        elif client_type == 'openai':
            return openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "")
            )
        else:
            raise ValueError(f"Unsupported client type: {client_type}")


async def main():
    args = simple_parsing.parse(Args)
    setup_logger(args.debug)
    global client
    global MODEL
    MODEL= args.model
    client = AIClientFactory.get_client(
        client_type=args.client_type, 
        model=args.model
    )
    t0 = time.perf_counter()

    problems = Problem.find_all(args.folder_path)[: args.max_num_problems]
    problems = [problem for problem in problems ] #if problem.name in ["Set, Cover"]]
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
