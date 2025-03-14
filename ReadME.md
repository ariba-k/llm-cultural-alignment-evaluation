# Randomness, Not Representation: Evaluating Cultural Alignment in LLMs

This repository contains the code for the paper ["Randomness, Not Representation: The Unreliability of Evaluating Cultural Alignment in LLMs"](https://arxiv.org/abs/2503.08688).

## Abstract

Research on the 'cultural alignment' of Large Language Models (LLMs) has emerged in response to growing interest in understanding representation across diverse stakeholders. Current approaches to evaluating cultural alignment borrow social science methodologies but often overlook systematic robustness checks. We identify and test three assumptions behind current evaluation methods: (1) Stability: that cultural alignment is a property of LLMs rather than an artifact of evaluation design, (2) Extrapolability: that alignment with one culture on a narrow set of issues predicts alignment with that culture on others, and (3) Steerability: that LLMs can be reliably prompted to represent specific cultural perspectives. Through experiments examining both explicit and implicit preferences of leading LLMs, we find a high level of instability across presentation formats, incoherence between evaluated versus held-out cultural dimensions, and erratic behavior under prompt steering. We show that these inconsistencies can cause the results of an evaluation to be very sensitive to minor variations in methodology.

## Data

Our experiments use a combination of external datasets and internally generated data:
### External Datasets

- [Cover Letter Dataset](https://huggingface.co/datasets/ShashiVish/cover-letter-dataset): Used for implicit stability experiments to test how LLMs evaluate culturally distinct content without explicitly discussing values.

- [LLM Global Opinions](https://huggingface.co/datasets/Anthropic/llm_global_opinions): Used for stability experiments to examine format sensitivity and for steerability experiments to test if LLMs can be prompted to align with specific cultural perspectives.

- [Values Survey Module (VSM-2013)](https://geerthofstede.com/research-and-vsm/vsm-2013/): Used for extrapolability experiments to test whether alignment on certain cultural dimensions predicts alignment on others.

### Internal Data Generation

- `data_explicit_stability.py`: Generates survey question variations with different formats to test the impact of presentation on LLM responses.

- `data_implicit_stability.py`: Creates culturally distinct cover letter versions across Hofstede's dimensions to evaluate implicit cultural biases.

## Requirements

To install dependencies:

```bash
pip install -r requirements.txt
```

## Case Study: Forced Binary Choices

Our case study examining the impact of forced binary choices on LLM evaluations was inspired by the methodology in Mazeika et al.'s "Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs" (2025), available at [arXiv:2502.08640](https://arxiv.org/abs/2502.08640).

In this case study, we demonstrate how using Likert scales with versus without a 'neutral' option can suggest very different conclusions about an LLM's cultural preferences.

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