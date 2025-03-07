# DREST: PPO Implementation for Shutdownable Agents

This repository contains a PyTorch implementation of Proximal Policy Optimisation (PPO) for training agents with a Discounted REward for Same-Length Trajectories (DREST) reward function. This is part of ongoing research into training shutdownable agents via stochastic choice.

## Overview

This implementation explores training agents to be:
- USEFUL (pursuing goals effectively conditional on each trajectory length)
- NEUTRAL (choosing stochastically between different trajectory lengths)

The project uses gridworld environments to test and validate the DREST reward function approach.

## Project Structure

```
PPO_torch/
├── config.py              # Main configuration parameters
├── train.py               # Training script
├── requirements.txt       # Project dependencies
├── Generalist/            # Core agent implementation
│   ├── grid_env.py        # Gridworld environment
│   └── PPO2.py            # PPO agent implementation
├── gridworld_construction/  # Gridworld utilities
│   ├── calculateM.py
│   ├── convertToTensors.py
│   ├── directM.py
│   ├── draw_gridworld.py
│   └── gridworlds_*.pkl   # Gridworld datasets
├── utilities/             # Helper functions and utilities
│   ├── classes.py
│   ├── evals_utils.py
│   ├── IMPALA_CNN.py
│   ├── parse_utils.py
│   ├── utils.py
│   ├── config_device.py
│   ├── config_env.py
│   ├── config_train_loop.py
│   └── config_wrapup.py
└── models/                # Directory for saved models
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/haz-g/ipp.git
cd ipp
```

2. Ensure you have Python 3.11 installed (this project is not compatible with Python 3.12+):
```bash
# On macOS with Homebrew:
brew install python@3.11
brew install libomp # Required for PyTorch on macOS

# On Ubuntu/Debian:
sudo apt-get install python3.11

# On Windows:
# Download Python 3.11 from python.org
```

3. Set up a virtual environment with Python 3.11 (recommended):
```bash
# On macOS/Linux:
python3.11 -m venv venv
source venv/bin/activate

# On Windows:
python3.11 -m venv venv
venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Hardware Requirements and Configuration

The implementation supports both local and cloud-based execution:

### Local Setup
- Can run on CPU with sufficient cores
- Recommended configuration: 4 parallel environments
- Expect slower training times compared to GPU execution
- Modify `num_envs` in config.py accordingly:
```python
CONFIG = {
    'num_envs': 4,  # For local CPU execution
    ...
}
```

### Cloud Setup (Recommended)
- Tested on Lambda Labs' 1xGH200 instance
- Adjust parallel environments count to suit CPU core count
- Significantly faster training times

## Cloud Deployment

### Syncing Files to Cloud

To sync your local files to a cloud instance (e.g., Lambda Labs), use a form of the below command (feel free to adjust the 'exclude' prompts as necessary):

```bash
rsync -avz --exclude=".git" --exclude=".gitattributes" --exclude="venv/" \
  --exclude="wandb/" --exclude="__pycache__/" --exclude="models/*" \
  --exclude=".gitignore" /path/to/your/files lambda:/home/ubuntu/Lambda_IPP/
```

Note: If you're going to be continuing training from an existing model ensure you remove the ```--exclude="models/*"``` prompt.

### Cloud Setup

Setting up on cloud instances is straightforward:

1. Sync your files to the cloud instance
2. Navigate to your project directory
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Run the training script:
   ```bash
   python train.py
   ```
6. When prompted, select option 2 to log in to Weights & Biases
7. Paste your API key from https://wandb.ai/settings

Note: The default configuration in `config.py` is optimised for cloud environments. You can run it as-is for cloud deployment or adjust parameters for your specific setup.

## Training

The training script supports various configurations defined in `config.py`. 

Basic training command:
```bash
python train.py
```

Key configuration parameters can be modified in `config.py` or passed as command-line arguments:
```bash
python train.py --num_envs 4  # For local execution
```

## Implementation Details

The core PPO implementation is adapted from [CleanRL](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy), modified to support the DREST reward function and specific environment requirements for training shutdownable agents.

Key features:
- IMPALA CNN preprocessing for observations
- Vectorised environment support
- Configurable DREST reward function
- Comprehensive evaluation metrics for USEFULNESS and NEUTRALITY

## Research Context

This implementation is part of ongoing research into training shutdownable agents. The approach uses a novel DREST reward function to train agents that are both effective at pursuing goals and neutral about trajectory lengths.

For more details about the theoretical framework and research context, please refer to the associated paper (forthcoming).

## Contributing

This is an active research project. For questions or contributions, please open an issue to discuss proposed changes.