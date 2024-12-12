# AI Problem Solving Evaluation Script

## Overview
This script provides a flexible framework for solving and evaluating problems using AI models using MCTS mapcoder across different clients and providers.

## Prerequisites

### Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Environment Setup
1. Create a `.env` file with your API keys and deployments(for AzureOpenAI):
   ```
   ANTHROPIC_AI_KEY=your_anthropic_api_key
   AZURE_OPEN_AI_KEY=your_azure_openai_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Configuration Options

### Command-line Arguments
The script supports various configuration options:

- `--folder_path`: Path to the problem data folder (default: `./round3Data`)
- `--weave_log`: Enable Weave logging (default: `False`)
- `--weave_eval`: Enable Weave evaluation (default: `False`)
- `--max_num_problems`: Maximum number of problems to process (default: `7`)
- `--client_type`: AI client type (options: 'anthropic', 'azure_openai', 'openai')
- `--model`: Specific model to use (optional)
- `--debug`: Enable debug logging (default: `False`)

## Supported AI Clients and Models

### Configurable Clients
The script supports multiple AI clients:
- Anthropic
- Azure OpenAI
- OpenAI

### Model Flexibility
You can specify different models for each client type.

## Running the Script

### Basic Usage
```bash
python 07_mcts_cpp.py
```

### Example Configurations
```bash
# Use Azure OpenAI with default gpt-4o model
python 07_mcts_cpp.py

# Use Anthropic
python 07_mcts_cpp.py --client_type 'anthropic' --model 'claude-3-sonnet-20240229'

# Use OpenAI with debug mode
python 07_mcts_cpp.py --client_type 'openai' --debug True
```

## Evaluation Output
The script generates a tabulated evaluation result showing:
- Problem name
- Number of runs
- Error cases
- Matched solutions
- Total test cases
- Validity status

## Weave Logging Support
Enable detailed logging for debugging:
```bash
python 07_mcts_cpp.py --weave_log True
```

## Troubleshooting
- Ensure all API keys are correctly set in the `.env` file
- Verify the problem data folder exists
- Check internet connectivity and proxy for gateway problems
  
## Research Contributions

We extend our sincere gratitude to these research ideas for our Hackercup24 solution
1. [RethinkMCTS](https://www.arxiv.org/pdf/2409.09584)
2. [MapCoder](https://arxiv.org/pdf/2405.11403)


