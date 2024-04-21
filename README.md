# About

This is the repository for report on the paper

***Is Your Code Generated by ChatGPT Really Correct?* Rigorous Evaluation of Large Language Models for Code Generation**

by Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, Lingming Zhang

Jiawei Liu et al. *Is Your Code Generated by ChatGPT Really Correct?* Rigorous Evaluation
of Large Language Models for Code Generation. 2023. arXiv: 2305.01210 [cs.SE].

for the Lecture **Data Mining und maschinelles Lernen** in Wintersemester 2023/2024

from Eric Jäkel, Eric Bühler & Sümeyye Özel


## Requirements & Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the packages.

```bash
git clone https://github.com/evalplus/evalplus.git
cd evalplus
pip install -r tools/requirements.txt
pip install evalplus --upgrade
```

# Code Generation
## Description
This repository provides code for generating code solutions using the PolyCoder-
2.7B large language model (LLM) and evaluating them with the EvalPlus
framework, specifically the HumanEval+ dataset.

## Usage
1. Download the PolyCoder-2.7B model weights from [huggingface](
https://huggingface.co/NinedayWang/PolyCoder-2.7B)
2. Run the script
```bash
python gen_sol.py
```

This script will:
* Download the HumanEval+ dataset from EvalPlus if not already present.
* Generate code solutions for each problem in the HumanEval+ dataset using
PolyCoder-2.7B.
* Save the results as `samples.jsonl`
## Customization
* You can modify the hyperparameters in `gen_sol.py` to experiment with different
beam search settings, maximum length of generated code, etc.
* Consider exploring PolyCoder's other features, such as fine-tuning on specific
coding domains or tasks.

# Evaluation of Pregenerated Samples
1. Extract the contents of the file pregenerated_samples/tested_models.zip
2. Install the evalplus CLI-Tool (python package) using pip
3. Open a terminal and navigate to the location of the extracted files
4. Type the command
```bash
    evalplus.evaluate --dataset humaneval --samples <model name>
```   
and replace <model name> with the name of the directory containing the pregenerated samples. 


# References
* Paper available at [arxiv](https://arxiv.org/abs/2305.01210)
* Code of the authors and further information available at their [GitHub](https://github.com/evalplus/evalplus)
