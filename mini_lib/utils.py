import asyncio

# from asyncio import subprocess
import subprocess
import concurrent.futures
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Optional, List
import concurrent.futures
import signal
from functools import partial
from rich.logging import RichHandler
import re
import weave


def load_jsonl(file: Path) -> List[dict]:
    """Load a JSONL file"""
    with open(file, "r") as f:
        return [json.loads(line) for line in f]


class TimeoutException(Exception):
    pass


def setup_logger(debug=False, silence_openai=True):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    # silence openai logger
    if silence_openai:
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)


def maybe_remove_backticks(solution: str) -> str:
    "Remove backticks from the solution"
    solution = solution.strip()
    solution = re.sub(r"^```python\s*", "", solution)
    solution = re.sub(r"\s*```$", "", solution)
    return solution


@weave.op()
def check_solution(expected: str, actual: str, problem_name: str) -> dict:
    "Check the solution against the expected output"
    matches = 0
    expected_lines = expected.strip().split("\n")
    # logging.debug(f"Expected lines: {expected_lines}")
    actual_lines = actual.strip().split("\n")
    # logging.debug(f"Actual lines: {actual_lines}")
    offending_cases = []
    for expected_line, actual_line in zip(expected_lines, actual_lines):
        expected_line = expected_line.strip()
        actual_line = actual_line.strip()

        if expected_line == actual_line:
            matches += 1  # +1 for the whole line match
        else:
            offending_cases.append((expected_line, actual_line))
    logging.info(
        f"Passed testcases for {problem_name}: {matches}/{len(expected_lines)}"
    )

    return {
        "matches": matches == len(expected_lines),
        "total": len(expected_lines),
        "offending_cases": offending_cases,
    }


def run_with_timeout(code: str, input: Optional[str], timeout: int):
    def signal_handler(signum, frame):
        raise TimeoutException("Function call timed out")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(timeout)

    try:
        vars = {}
        exec(code, vars)
        fn = vars.get("solve", lambda x: x)
        result = fn(input)
        signal.alarm(0)  # Cancel the alarm
        return result
    except TimeoutException:
        logging.error("Function call timed out")
        raise
    except Exception as e:
        # logging.error(f"Error executing code: {e}")
        raise
    finally:
        signal.alarm(0)  # Ensure the alarm is canceled


async def arun(
    code: Optional[str] = None, input: Optional[str] = None, timeout: int = 60
):
    logging.info("Running solution asynchronously...")
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()

    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future = loop.run_in_executor(
                executor, run_with_timeout, code, input, timeout
            )

            result = await asyncio.wait_for(
                future, timeout=timeout + 5
            )  # Add a small buffer

        return result
    except asyncio.TimeoutError:
        logging.error("Function call timed out (outer timeout)")
        raise TimeoutException("Function call timed out (outer timeout)")
    except Exception as e:
        logging.error(f"Error executing code: {e}")
        raise e
    finally:
        t1 = time.perf_counter()
        logging.info(f"Code solution runtime: {t1 - t0:.2f} seconds")


def run(code: Optional[str] = None, input: Optional[str] = None, timeout: int = 60):
    logging.info("Running solution synchronously...")
    return asyncio.run(arun(code, input, timeout))


def compile_cpp(code: str) -> str:
    """Compiles C++ code and returns the path to the executable."""
    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
        f.write(code)
        cpp_file = f.name

    exe_file = cpp_file[:-4]  # Remove .cpp extension
    if os.name == "nt":  # Windows
        exe_file += ".exe"

    # Compile with optimization flags
    compile_command = ["g++", "-O2", "-std=c++17", cpp_file, "-o", exe_file]
    try:
        subprocess.run(compile_command, check=True, capture_output=True)
        return exe_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Compilation error: {e.stderr.decode()}")
    finally:
        os.unlink(cpp_file)  # Clean up source file


def run_with_timeout_cpp(
    executable: str, input_str: Optional[str], timeout: int
) -> str:
    """Runs the compiled C++ executable with timeout."""
    try:
        process = subprocess.Popen(
            [executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate(input=input_str, timeout=timeout)

        if process.returncode != 0:
            raise RuntimeError(f"Runtime error: {stderr}")

        return stdout.strip()

    except subprocess.TimeoutExpired:
        process.kill()
        raise TimeoutException("Function call timed out")
    finally:
        if os.path.exists(executable):
            os.unlink(executable)  # Clean up executable


async def arun_cpp(
    code: Optional[str] = None, input: Optional[str] = None, timeout: int = 60
) -> str:
    """Asynchronously compiles and runs C++ code."""
    logging.info("Running C++ solution asynchronously...")
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()

    try:
        # Compile C++ code in a separate process
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executable = await loop.run_in_executor(executor, compile_cpp, code)

        # Run the compiled executable
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, run_with_timeout_cpp, executable, input, timeout
            )

        return result

    except asyncio.TimeoutError:
        logging.error("Function call timed out (outer timeout)")
        raise TimeoutException("Function call timed out (outer timeout)")
    except Exception as e:
        # logging.error(f"Error executing code: {e}")
        raise
    finally:
        t1 = time.perf_counter()
        logging.info(f"Code solution runtime: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    # Test check_solution
    expected = "Case #1: YES\nCase #2: NO\nCase #3: YES"
    actual = "Case #1: YES\nCase #2: Yes\nCase #3: YES"
    result = check_solution(expected, actual)
    assert result["matches"] == 2, "Expected 2 matches"
    assert result["total"] == 3, "Expected 3 total lines"
    assert len(result["offending_cases"]) == 1, "Expected 1 offending case"
    assert result["offending_cases"][0] == (
        "Case #2: NO",
        "Case #2: Yes",
    ), "Unexpected offending case"

    # Test maybe_remove_backticks
    assert maybe_remove_backticks("print('hello')\n```") == "print('hello')"
    assert maybe_remove_backticks("print('hello')\n```  ") == "print('hello')"
    assert maybe_remove_backticks("```python\nprint('hello')") == "print('hello')"
    assert maybe_remove_backticks("```python\nprint('hello')\n```") == "print('hello')"

    # test exec
    code = "def solve(x: int):\n    return x + 1"
    input = 2
    result = run(code, input)
    assert result == 3, "Expected 3"

    # async test
    result = asyncio.run(arun(code, input))
    assert result == 3, "Expected 3"
    print("All tests passed!")
