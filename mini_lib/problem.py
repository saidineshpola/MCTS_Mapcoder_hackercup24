import re
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Optional, List


def _find_used_images(description_text: str, folder_path: Path) -> list[str]:
    # Find all text files that might contain image links
    image_link_files = list(folder_path.glob("image_*.txt"))

    # Dictionary to store image links
    image_links = {}

    # Read image links from text files
    for file in image_link_files:
        with file.open("r") as f:
            image_links[file.stem] = f.read().strip()

    photo_ids = set(re.findall(r"{{PHOTO_ID:(\d+)", description_text))
    markdown_images = set(re.findall(r"!\[.*?\]\((.*?)\)", description_text))

    used_images = [
        image_links[f"image_{photo_id}"]
        for photo_id in photo_ids
        if f"image_{photo_id}" in image_links
    ]

    used_images.extend(
        [
            link
            for link in markdown_images
            if any(link == value for value in image_links.values())
        ]
    )

    return used_images


def _replace_img_links(description_text: str, image_urls: list[str]) -> str:
    for image_url in image_urls:
        image_id = image_url.split("/")[-1].split(".")[0]  # Extract ID from URL
        old_ref = f"{{{{PHOTO_ID:{image_id}|WIDTH:600}}}}"
        new_ref = f"![{image_id}]({image_url})"
        description_text = description_text.replace(old_ref, new_ref)

    return description_text


@dataclass
class Problem:
    name: str
    problem_description: str
    sample_input: str
    sample_output: str
    input_path: Path  # this is sometimes a big file
    output_path: Path  # this is sometimes a big file
    folder_path: Path
    code: Optional[str] = None
    images: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._process_description_and_images()

    def _process_description_and_images(self):
        used_images = _find_used_images(self.problem_description, self.folder_path)
        self.problem_description = _replace_img_links(
            self.problem_description, used_images
        )
        self.images = used_images

    def get_input(self) -> str:
        return self.input_path.read_text()

    def get_output(self) -> str:
        return self.output_path.read_text()

    def save_code(
        self,
        code: str,
        code_path: Optional[str] = None,
        outfile_name: Optional[str] = None,
    ):
        final_code = f"from pathlib import Path\ninput = Path('./{self.input_path.name}').read_text()\n\n"
        code_name = f"{self.name}_generated.txt"
        code_path = (
            Path(self.folder_path) / code_name if code_path is None else code_path
        )
        final_code += code
        outfile_name = (
            f"./{self.name}_generated.out.txt" if outfile_name is None else outfile_name
        )
        final_code += (
            f"\n\noutput = solve(input)\nPath('{outfile_name}').write_text(output)\n"
        )
        code_path.write_text(final_code)
        return Path(code_path)

    def save_code_cpp(
        self,
        code: str,
        code_path: Optional[str] = None,
        outfile_name: Optional[str] = None,
    ):
        """
        Saves C++ code with boilerplate for file I/O operations.

        Args:
            code: The C++ solution code
            code_path: Optional custom path to save the code
            outfile_name: Optional custom output file name

        Returns:
            Path object pointing to the saved code file
        """
        # Define the C++ includes and helper functions
        cpp_header = """#include <iostream>
    #include <fstream>
    #include <string>
    #include <sstream>

    std::string read_file(const std::string& filename) {
        std::ifstream file(filename);
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    void write_file(const std::string& filename, const std::string& content) {
        std::ofstream file(filename);
        file << content;
    }
    """

        # Create the main function with file I/O
        code_name = f"{self.name}_generated.cpp"
        code_path = (
            Path(self.folder_path) / code_name if code_path is None else Path(code_path)
        )
        outfile_name = (
            f"./{self.name}_generated.out.txt" if outfile_name is None else outfile_name
        )

        main_function = f"""
    int main() {{
        std::string input = read_file("./{self.input_path.name}");
        std::string output = solve(input);
        write_file("{outfile_name}", output);
        return 0;
    }}
    """

        # Combine all parts of the code
        final_code = cpp_header + "\n" + code + "\n" + main_function

        # Write the code to file
        code_path.write_text(final_code)
        return code_path

    def save_output(self, output: str, outfile: Optional[str] = None):
        outfile_name = f"{self.name}_generated.out.txt"
        outfile = Path(self.folder_path) / outfile_name if outfile is None else outfile
        outfile.write_text(output)
        return Path(outfile)

    @classmethod
    def from_name(cls, name: str, folder_path: Path):
        description_path = folder_path / f"{name}.md"
        input_path = folder_path / f"{name}.in"
        output_path = folder_path / f"{name}.out"
        sample_input_path = folder_path / f"{name}_sample_input.txt"
        sample_output_path = folder_path / f"{name}_sample_output.txt"

        return cls.from_files(
            name=name,
            description_path=description_path,
            sample_input_path=sample_input_path,
            sample_output_path=sample_output_path,
            input_path=input_path,
        )

    @classmethod
    def from_files(
        cls,
        name: str,
        description_path: Path,
        sample_input_path: Path,
        sample_output_path: Path,
        input_path: Path,
        output_path=None,
    ):
        return cls(
            name=name,
            problem_description=description_path.read_text(),
            sample_input=sample_input_path.read_text(),
            sample_output=sample_output_path.read_text(),
            input_path=input_path,
            output_path=(
                input_path.with_suffix(".out") if not output_path else output_path
            ),
            folder_path=input_path.parent,
        )

    @classmethod
    def from_name_2024(cls, name: str, folder_path: Path):
        description_path = folder_path / f"statement.txt"
        input_path = folder_path / f"full_in.txt"
        output_path = folder_path / f"full_out.txt"
        sample_input_path = folder_path / f"sample_in.txt"
        sample_output_path = folder_path / f"sample_out.txt"

        return cls.from_files(
            name=name,
            description_path=description_path,
            sample_input_path=sample_input_path,
            sample_output_path=sample_output_path,
            input_path=input_path,
            output_path=output_path,
        )

    @classmethod
    def find_all(cls, folder_path: Path, is_2024=True) -> List["Problem"]:
        problems = []

        if is_2024:
            # Find all folders in the given path
            problem_folders = [f for f in folder_path.iterdir() if f.is_dir()]
            for problem_folder in problem_folders:
                # Check if the folder contains the required files
                required_files = [
                    "full_in.txt",
                    "sample_in.txt",
                    "sample_out.txt",
                    "statement.txt",
                    # "full_out.txt",
                ]

                if all(
                    problem_folder.joinpath(file).exists() for file in required_files
                ):
                    problem_name = problem_folder.name
                    try:
                        problem = cls.from_name_2024(problem_name, problem_folder)
                        problems.append(problem)
                    except FileNotFoundError as e:
                        print(
                            f"Warning: Couldn't create problem from {problem_name}. Error: {e}"
                        )
                else:
                    print(
                        f"Warning: Folder {problem_folder} is missing required files."
                    )

            logging.info(
                f"Found {len(problems)} problems in 2024's folder: {folder_path}"
            )
            return problems

        # Find all markdown files in the folder
        md_files = folder_path.rglob("*.md")

        for md_file in md_files:
            # Skip files that end with '_sol.md' as they might be solution files
            if md_file.stem.endswith("_sol"):
                continue

            problem_name = md_file.stem
            try:
                problem = cls.from_name(problem_name, md_file.parent)
                problems.append(problem)
            except FileNotFoundError as e:
                print(
                    f"Warning: Couldn't create problem from {problem_name}. Error: {e}"
                )
        logging.info(f"Found {len(problems)} problems in folder: {folder_path}")
        return problems

    def __repr__(self):
        return f"""Problem: {self.name}
    Description: {self.problem_description[:50]}...
    Sample Input: {self.sample_input[:50]}...
    Sample Output: {self.sample_output[:50]}...
    Input Path: {self.input_path}
    Output Path: {self.output_path}
    Images: {len(self.images)} image(s)
"""

if __name__ == "__main__":
    problem_name = "cheeseburger_corollary_ch1"
    folder_path = Path("../dataset/2023/practice/")

    # load 1 problem by name
    problem = Problem.from_name(problem_name, folder_path)
    print(problem)

    # load all problems in folder
    folder_path = Path("../dataset/2023/")
    problems = Problem.find_all(folder_path)
    print(f"Found {len(problems)} problems in folder: {folder_path}")
    assert len(problems) == 29
