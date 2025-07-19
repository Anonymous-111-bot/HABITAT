# This is the official code of "A Small Leak Sinks All: Exploring the Transferable Vulnerability of Source Code Models"

## Paper Overview

Source Code Model (SCM) aims to learn the proper embeddings from source codes, demonstrating significant success in various software engineering or security tasks. The recent explosive development of Large Language Models (LLMs) extends the family of SCMs, bringing LLMs for code (LLM4Code) that revolutionize development workflows. Investigating different kinds of SCM vulnerability is the cornerstone for software security; however, the fundamental one, transferable vulnerability, remains critically underexplored.

Existing studies neither offer practical ways (i.e., require access to the downstream classifier of SCMs) to produce effective adversarial samples for adversarial defense, nor give heed to the widely used LLM4Code. Therefore, this work systematically studies the intrinsic vulnerability transferability of both traditional SCMs and LLM4Code, and proposes a victim-agnostic approach to generate practical adversarial samples. We design a Hierarchical Adaptive Bandit-based Intelligent method for Transferable Attack (HABITAT), consisting of a tailored perturbation-inserting mechanism and a hierarchical Reinforcement Learning (RL) framework that adaptively selects optimal perturbations without requiring any access to the downstream classifier of SCMs.

Furthermore, an intrinsic transferability analysis of SCM vulnerabilities is conducted, revealing the potential vulnerability correlation between traditional SCMs and LLM4Code, together with fundamental factors that govern the success rate of victim-agnostic transfer attacks. These findings of SCM vulnerabilities underscore the critical focal points for developing robust defenses in the future. Experimental evaluation demonstrates that our constructed adversarial examples crafted based on traditional SCMs achieve up to 64% success rates against LLM4Code, representing over 15% improvement over the existing state-of-the-art method.

## Requirements

- Python >= 3.7
- torch
- numpy
- transformers
- astor
- tqdm
- scikit-learn
- tree_sitter

You can install the dependencies with:

```bash
pip install torch numpy transformers astor tqdm scikit-learn tree_sitter
```

## Attack Methods

This repository implements three main attack methods based on the HABITAT framework:

### 1. PATD (Position-Aware Transformation Discovery) - MAB Attack
- **Method**: `MAB_attack`
- **Description**: Position-aware transformation discovery that identifies safe insertion positions and important positions using Multi-Armed Bandit (MAB) to learn optimal perturbation strategies through exploration and exploitation
- **Usage**: Run the standard attack pipeline without transfer memory paths

### 2. PGSA (Preference-Guided Strategy Adaptation) - Adaptive Transfer Attack  
- **Method**: `adaptive_transfer_attack`
- **Description**: Preference-guided strategy adaptation that leverages learned preferences from previous attacks stored in memory to improve transferability
- **Usage**: Use `--transfer_memory_path` with a single memory file

### 3. MMMT (Multi-Model Memory Transfer) - Adaptive Stacked Attack
- **Method**: `adaptive_stacked_attack`
- **Description**: Multi-model memory transfer that combines preferences from multiple models using MAB model selection for enhanced transferability
- **Usage**: Use `--transfer_memory_path` with two memory files separated by comma

## Usage

### 1. Training and Evaluation

See `Authorship_Attribution/code/scripts.sh` and `Defeat_Detection/code/script.sh` for example commands to train and evaluate models.

### 2. Attack Execution

#### PATD (Position-Aware Transformation Discovery)
```bash
# For Authorship Attribution task
cd Authorship_Attribution/code/CodeBERT
python attack_ablation_here.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --eval_data_file=../data/valid.txt \
    --block_size 512 \
    --eval_batch_size 32

# For Defeat Detection task  
cd Defeat_Detection/code/CodeBERT
python attack_ablation_here.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base-mlm \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=../dataset/test_subs_0_400.jsonl \
    --block_size 512 \
    --eval_batch_size 64
```

#### PGSA (Preference-Guided Strategy Adaptation)
```bash
# Use single memory file for preference-guided transfer attack
cd Defeat_Detection/code/CodeBERT
python attack_ablation_here.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base-mlm \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=../dataset/test_subs_0_400.jsonl \
    --block_size 512 \
    --eval_batch_size 64 \
    --transfer_memory_path mab_preferences_unix.json
```

#### MMMT (Multi-Model Memory Transfer)
```bash
# Use two memory files for multi-model memory transfer attack
cd Defeat_Detection/code/CodeBERT
python attack_ablation_here.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base-mlm \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=../dataset/test_subs_0_400.jsonl \
    --block_size 512 \
    --eval_batch_size 64 \
    --transfer_memory_path mab_preferences_unix.json,mab_preferences_codet5.json
```

### 3. LLM4Code Transferability Testing

To evaluate the transferability of adversarial samples on LLM4Code models, use the code pair generation feature:

```bash
# Generate code pairs for LLM4Code evaluation
cd Authorship_Attribution/code/CodeBERT
python attack_ablation_here.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --eval_data_file=../data/valid.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --generate_code_pairs \
    --max_pairs 100 \
    --output_code_pairs_file mab_code_pairs.json
```

The generated JSON file contains original and adversarial code pairs in the format:
```json
{
    "id": 1,
    "original": "// Source code",
    "modified": "// Adversarial code", 
    "expected": "yes"
}
```

These code pairs can be used to test functional equivalence on various LLM4Code models by prompting: "Are these code snippets functionally equivalent? Ignore unexecuted code, unused functions/variables, and comments. Answer only 'yes' or 'no'."

### 4. Data Preparation

Place your training, validation, and test data in the appropriate `data/` or `dataset/` directories. Adjust paths in scripts as needed.

## Directory Structure

```
HABITAT/
├── Authorship_Attribution/     # Authorship attribution task
│   └── code/
│       ├── CodeBERT/          # CodeBERT implementation
│       ├── CodeT5/            # CodeT5 implementation  
│       └── UniXcoder/         # UniXcoder implementation
├── Defeat_Detection/          # Defeat detection task
│   └── code/
│       ├── CodeBERT/          # CodeBERT implementation
│       ├── CodeT5/            # CodeT5 implementation
│       └── UniXcoder/         # UniXcoder implementation
└── README.md
```

## Citation

If you use this code, please cite our paper:

```
@inproceedings{tobeupdated,
  title={A Small Leak Sinks All: Exploring the Transferable Vulnerability of Source Code Models},
  author={...},
  booktitle={...},
  year={2024}
}
``` 