# Randomness, Not Representation: Evaluating Cultural Alignment in LLMs

This repository contains the code for the paper ["Randomness, Not Representation: The Unreliability of Evaluating Cultural Alignment in LLMs"](https://arxiv.org/abs/2503.08688).

## Abstract

Research on the 'cultural alignment' of Large Language Models (LLMs) has emerged in response to growing interest in understanding representation across diverse stakeholders. Current approaches to evaluating cultural alignment through survey-based assessments that borrow from social science methodologies often overlook systematic robustness checks. Here, we identify and test three assumptions behind current survey-based evaluation methods: (1) _Stability_: that cultural alignment is a property of LLMs rather than an artifact of evaluation design, (2) _Extrapolability_: that alignment with one culture on a narrow set of issues predicts alignment with that culture on others, and (3) _Steerability_: that LLMs can be reliably prompted to represent specific cultural perspectives. Through experiments examining both explicit and implicit preferences of leading LLMs, we find a high level of instability across presentation formats, incoherence between evaluated versus held-out cultural dimensions, and erratic behavior under prompt steering. We show that these inconsistencies can cause the results of an evaluation to be very sensitive to minor variations in methodology. Finally, we demonstrate in a case study on evaluation design that narrow experiments and a selective assessment of evidence can be used to paint an incomplete picture of LLMs' cultural alignment properties. Overall, these results highlight significant limitations of current survey-based approaches to evaluating the cultural alignment of LLMs and highlight a need for systematic robustness checks and red-teaming for evaluation results.
## Repository Overview

The repository is organized around the three main experiments and a case study:

- **Stability**: Tests whether cultural alignment is a stable property of LLMs or an artifact of evaluation design
- **Extrapolability**: Tests whether alignment on certain cultural dimensions predicts alignment on others
- **Steerability**: Tests if LLMs can be reliably prompted to represent specific cultural perspectives
- **Case Study**: Examines how forced binary choices affect LLM evaluation results. Inspired by Mazeika et al.'s work (arXiv:2502.08640), it demonstrates how using Likert scales with versus without a 'neutral' option can suggest different conclusions about an LLM's cultural preferences.

## Requirements

To install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiments

### Core Scripts

Each experiment has a similar pattern of scripts:

- `data_*.py`: Generates necessary data for experiments
- `run_*.py`: Runs the experiments and outputs results to the `results/` folder
- `analyze_*.py`: Analyzes results and outputs visualizations to the `analysis/` folder

All scripts are designed to run without command-line arguments. Configuration parameters are defined within the scripts themselves.

### Utility Files

Files at the repository root provide shared functionality:

- `environment.py`: API configuration including credentials - edit before running experiments
- `models.py`: Interfaces for LLM providers (OpenAI, Anthropic, etc.)
- `style.py`: Formatting utilities for experiment outputs
- `constants.py`: Shared constants used across experiments
- `requirements.txt`: Python dependencies

### 1. Stability Experiments

#### Explicit Stability Experiments

```bash
# Generate data for explicit stability experiments
python stability/explicit/data_explicit_stability.py

# Run the explicit stability experiments
python stability/explicit/run_explicit_stability.py

# Analyze the results and generate visualizations
python stability/explicit/analyze_explicit_stability.py
```

#### Implicit Stability Experiments

```bash
# Generate culturally distinct cover letters
python stability/implicit/data_implicit_stability.py

# Run the implicit stability experiments
python stability/implicit/run_implicit_stability.py
```

#### Implicit Stability Experiments

The implicit stability experiments include several subexperiments:

- **Main Experiment**
  - `data_implicit_stability.py`: Generates culturally distinct cover letters
  - `run_implicit_stability.py`: Runs the main implicit stability experiment
  - `constants_implicit_stability.py`: Contains adjustable experiment parameters (trials, dimensions, etc.) - edit before running experiment
  - `models_implicit_stability.py`: Model-specific implementations
  - `utils_implicit_stability.py`: Utility functions

- **Comparative vs. Absolute Experiment**
  - `run_implicit_stability_comparative.py`: Runs both comparative and absolute experiment types
  - `analyze_implicit_stability_comparative.py`: Analyzes differences between comparative and absolute judgment approaches

- **Context Experiment (Hiring Manager vs. Career Coach vs. Job Applicant)**
  - `analyze_implicit_stability_context.py`: Analyzes how different evaluation contexts affect results

- **Reasoning vs. No Reasoning Experiment**
  - `analyze_implicit_stability_reason.py`: Analyzes effects of enabling vs. disabling model reasoning

- **Scale Experiment (4-point vs. 5-point vs. 6-point Likert scales)**
  - `analyze_implicit_stability_scale.py`: Analyzes how different Likert scale sizes affect results

### 2. Extrapolability Experiments

```bash
# Run extrapolability experiments
python extrapolability/run_extrapolability.py

# Analyze the results and generate visualizations
python extrapolability/analyze_extrapolability.py
```

### 3. Steerability Experiments

```bash
# Run steerability experiments
python steerability/run_steerability.py

# Analyze the results and generate visualizations
python steerability/analyze_steerability.py
```

### 4. Case Study: Forced Binary Choices

```bash
# Run the case study
python case_study/run_case_study.py

# Analyze the results and generate visualizations
python case_study/analyze_case_study.py
```

## Data

Our experiments use a combination of external datasets and our own generated data:

### External Datasets

- [Cover Letter Dataset](https://huggingface.co/datasets/ShashiVish/cover-letter-dataset): Used for implicit stability experiments
- [LLM Global Opinions](https://huggingface.co/datasets/Anthropic/llm_global_opinions): Used for stability and steerability experiments
- [Values Survey Module (VSM-2013)](https://geerthofstede.com/research-and-vsm/vsm-2013/): Used for extrapolability experiments

### Our Generated Dataset

We've made our culturally distinct cover letters dataset available on Hugging Face:
- **Dataset**: [cultural-dimension-cover-letters](https://huggingface.co/datasets/akhan02/cultural-dimension-cover-letters)

#### Dataset Structure

The Hugging Face version consolidates all cultural dimensions into a single CSV with 13 columns: one for the original cover letter and 12 for the six cultural dimensions (both poles).

Our `data_implicit_stability.py` script generates separate files for each dimension:
1. `idv.xlsx`: Individualism vs. Collectivism
2. `lto.xlsx`: Long-Term vs. Short-Term Orientation
3. `mas.xlsx`: Masculinity vs. Femininity
4. `pdi.xlsx`: High vs. Low Power Distance
5. `uai.xlsx`: High vs. Low Uncertainty Avoidance
6. `ivr.xlsx`: Indulgence vs. Restraint

The Hugging Face version is simply an alternative consolidated format for ease of external use.



## Citation

If you use this code or our findings in your research, please cite:

```bibtex
@article{khan2025randomness,
  title={Randomness, Not Representation: The Unreliability of Evaluating Cultural Alignment in LLMs},
  author={Khan, Ariba and Casper, Stephen and Hadfield-Menell, Dylan},
  journal={arXiv preprint arXiv:2503.08688},
  year={2025},
  url={https://arxiv.org/abs/2503.08688}
}
```

## Contact

- Ariba Khan: akhan02@mit.edu
- Stephen Casper: scasper@mit.edu