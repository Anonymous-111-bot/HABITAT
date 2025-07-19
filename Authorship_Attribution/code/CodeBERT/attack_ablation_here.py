from run import InputFeatures  # 修改为实际定义 InputFeatures 类的模块
import os
import argparse  # 确保正确导入 argparse 模块
import warnings
import logging
import torch
import random
import json
import time
import numpy as np
from run import TextDataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForMaskedLM
import sys
from Crossdomainattacker import  MABMemoryManager
sys.path.append('../codebert_downstream_model/run')
# 设置环境变量和日志配置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 运行代码时使用离线模式
warnings.simplefilter(action='ignore', category=FutureWarning)  # 仅报告警告

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import numpy as np
from tqdm import tqdm
import torch
import numpy as np


import os
import logging

logger = logging.getLogger(__name__)

class CodePairsGenerator:
    def __init__(self, attacker, target_model, tokenizer, args):
        self.attacker = attacker
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.args = args

    def _get_prediction(self, code, label):
        inputs = self.tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=self.args.block_size,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(self.args.device)
        attention_mask = inputs['attention_mask'].to(self.args.device)
        with torch.no_grad():
            outputs = self.target_model(input_ids=input_ids, labels=label)
            loss, logits = outputs
            pred = torch.argmax(logits, dim=-1).item()

        return pred

    def generate_code_pairs_from_dataset(self, dataset, output_file, max_samples=None, start_idx=0):
        code_pairs = []
        successful_attacks = 0
        total_attempts = 0
        skipped_samples = 0

        # Determine how many samples to process
        if max_samples is None:
            samples_to_process = len(dataset)
        else:
            samples_to_process = min(max_samples, len(dataset) - start_idx)

        logger.info(f"Starting to generate {samples_to_process} code pairs using MAB attack...")
        for idx in tqdm(range(start_idx, start_idx + samples_to_process), desc="Generating code pairs"):
            try:
                example = dataset[idx]
                input_ids = example[0].unsqueeze(0).to(self.args.device)
                label = example[1].unsqueeze(0).to(self.args.device)
                original_code = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                original_pred = self._get_prediction(original_code, label)
                if original_pred != label.item():
                    logger.info(f"Sample {idx}: Original prediction already wrong, skipping...")
                    skipped_samples += 1
                    continue
                total_attempts += 1
                logger.info(f"Sample {idx}: Performing MAB attack...")
                adversarial_code = self.attacker.MAB_attack(original_code, label)
                adversarial_pred = self._get_prediction(adversarial_code, label)

                code_pair = {
                    "id": len(code_pairs) + 1,
                    "original": original_code,
                    "modified": adversarial_code,
                    "expected":"yes",
                    "description": f"MAB Attack - Sample {idx}"
                }

                code_pairs.append(code_pair)

                if len(code_pairs) % 10 == 0:
                    success_rate = (successful_attacks / total_attempts * 100) if total_attempts > 0 else 0
                    logger.info(f"Progress: {len(code_pairs)} pairs generated, "
                                f"Attack success rate: {success_rate:.2f}%")

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue

        self._save_code_pairs(code_pairs, output_file)

        if total_attempts > 0:
            success_rate = (successful_attacks / total_attempts) * 100
            logger.info(f"\nFinal Statistics:")
            logger.info(f"Total samples processed: {start_idx + samples_to_process}")
            logger.info(f"Samples with correct original prediction: {total_attempts}")
            logger.info(f"Samples with wrong original prediction (skipped): {skipped_samples}")
            logger.info(f"Successful attacks: {successful_attacks}")
            logger.info(f"Attack success rate: {success_rate:.2f}%")
            logger.info(f"Total code pairs generated: {len(code_pairs)}")
            logger.info(f"Results saved to: {output_file}")

        return code_pairs

    def _save_code_pairs(self, code_pairs, output_file):
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(code_pairs, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully saved {len(code_pairs)} code pairs to {output_file}")

        except Exception as e:
            logger.error(f"Error saving code pairs to {output_file}: {str(e)}")
            raise

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from datetime import datetime
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--eval_target_file", default=None, type=str,
                        help="Target label for targeted attack on valid dataset.")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--csv_store_path", type=str,
                        help="Path to store the CSV file")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--language_type", type=str,
                        help="The programming language type of dataset")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--number_labels", type=int, default=2,
                        help="The number of labels for classification.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--use_encoder_features", action='store_true',
                        help="Whether to use encoder features for attack guidance.")
    parser.add_argument("--max_transformations", type=int, default=3,
                        help="Maximum number of transformations to apply per sample.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    # Hyper-parameters
    parser.add_argument("--num_of_changes", default=1, type=int,
                        help="Number of inserted obfuscations.")
    parser.add_argument("--do_position_selection", action='store_true')

    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument("--generate_code_pairs", action='store_true',
                        help="Generate code pairs in JSON format")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Maximum number of code pairs to generate")
    parser.add_argument("--attack_strategy", type=str, default="adaptive",
                        choices=["sequential", "ensemble", "competitive", "adaptive"],
                        help="Attack strategy to use")
    parser.add_argument("--output_code_pairs_file", type=str, default=None,
                        help="Output file for code pairs (JSON format)")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    set_seed(args.seed)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.model_name_or_path is None:
        args.model_name_or_path = 'microsoft/codebert-base'

    config = config_class.from_pretrained(args.model_name_or_path)
    config.num_labels = args.number_labels

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)
    else:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
    args.block_size = min(args.block_size, tokenizer.model_max_length)

    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
    from model import Model
    model_tgt = Model(encoder, config, tokenizer, args).to(args.device)
    model_sub = RobertaModel.from_pretrained("microsoft/codebert-base").to(args.device)

    try:
        codebert_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    except:
        codebert_mlm = RobertaForMaskedLM.from_pretrained("../cache/microsoft/codebert-base-mlm/")
    try:
        tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    except:
        tokenizer_mlm = RobertaTokenizer.from_pretrained("../cache/microsoft/codebert-base-mlm/")
    codebert_mlm.to('cuda')

    checkpoint_path = os.path.join(args.output_dir, 'model.bin')

    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)
    from Crossdomainattacker import CrossDomainAttacker
    attacker = CrossDomainAttacker(
        args=args,
        model_sub=model_sub,
        model_tgt=model_tgt,
        tokenizer=tokenizer,
        model_mlm=codebert_mlm,
        tokenizer_mlm=tokenizer_mlm,
        use_bpe=False,
        threshold_pred_score=0.5,
        targeted=False,
        epsilon=0.1
    )
    if args.generate_code_pairs:
        if args.output_code_pairs_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.output_code_pairs_file = f'mab_code_pairs_{timestamp}.json'

        generator = CodePairsGenerator(
            attacker=attacker,
            target_model=model_tgt,
            tokenizer=tokenizer,
            args=args
        )

        # Generate code pairs
        code_pairs = generator.generate_code_pairs_from_dataset(
            dataset=eval_dataset,
            output_file=args.output_code_pairs_file,
            max_samples=args.max_pairs,
            start_idx=0
        )

        return

    if args.do_position_selection:
        best_position = position_selection_experiment(
            attacker=attacker,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_tgt=model_tgt,
            device=args.device,
            args=args
        )

    model_tgt.eval()
    start_time = time.time()

    total_cnt = 0
    success_attack = 0
    adv_examples = []
    wrong = 0

    results = {
        'original_codes': [],
        'adversarial_codes': [],
        'original_predictions': [],
        'adversarial_predictions': [],
        'attack_success': [],
        'feature_distances': []
    }


    for index, example in enumerate(eval_dataset):
        print(f"Processing example {index + 1}/{len(eval_dataset)}")
        input_ids = example[0].unsqueeze(0).to(args.device)
        label = example[1].unsqueeze(0).to(args.device)
        with torch.no_grad():
            inputs = tokenizer.encode_plus(
                source_code,
                add_special_tokens=True,
                max_length=args.block_size,
                truncation=True,
                return_tensors='pt'
            )
            input_ids = inputs['input_ids'].to(args.device)

            attention_mask = inputs['attention_mask'].to(args.device)

            outputs = model_tgt(input_ids=input_ids, labels=label)
            loss, logits = outputs

            prob_orig = torch.softmax(logits, dim=-1)
            pred_orig = torch.argmax(prob_orig, dim=-1).item()

        results['original_codes'].append(source_code)
        results['original_predictions'].append(pred_orig)
        adv_code = attacker.MAB_attack(source_code,label)

        if adv_code:
            adv_inputs = tokenizer.encode_plus(
                adv_code,
                add_special_tokens=True,
                max_length=args.block_size,
                truncation=True,
                return_tensors='pt'
            )
            adv_input_ids = adv_inputs['input_ids'].to(args.device)
            adv_attention_mask = adv_inputs['attention_mask'].to(args.device)

            with torch.no_grad():
                outputs_adv = model_tgt(input_ids=adv_input_ids, labels=label)
                loss_adv, logits_adv = outputs_adv
                prob_adv = torch.softmax(logits_adv, dim=-1)
                pred_adv = torch.argmax(prob_adv, dim=-1).item()

            if pred_orig == label.item():
                if pred_adv != pred_orig:
                    adv_examples.append(adv_code)
                    success_attack += 1
                    print(f"Attack successful! Original prediction: {pred_orig}, Adversarial prediction: {pred_adv}")
                    results['attack_success'].append(True)
                else:
                    print(f"Attack failed! Original prediction: {pred_orig}, Adversarial prediction: {pred_adv}")
                    results['attack_success'].append(False)
            else:
                print(f"Original prediction was wrong")
                wrong += 1
                results['attack_success'].append(None)  # 原始预测已错误

        else:
            print("Attack did not generate adversarial example.")

        total_cnt += 1
        print(f"Wrong predictions in original model: {wrong}")
        print(f"Total processed: {total_cnt}, Successful attacks: {success_attack}")

    success_rate = success_attack / (total_cnt - wrong) * 100 if (total_cnt - wrong) > 0 else 0
    print(f"Attack success rate: {success_rate:.2f}%")
    print(f"Total samples: {total_cnt}, Correct predictions: {total_cnt - wrong}, Successful attacks: {success_attack}")
    with open(os.path.join(args.output_dir, 'adv_examples.txt'), 'w', encoding='utf-8') as f:
        for code in adv_examples:
            f.write(code + '\n')

    print("Adversarial examples saved.")


def position_selection_experiment(attacker, eval_dataset, tokenizer, model_tgt, device, args):
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    os.makedirs("results", exist_ok=True)

    position_counts = [1, 2, 3, 5, 8, 10, 15]

    all_results = {}

    sample_size = len(eval_dataset)
    sample_indices = list(range(sample_size))
    for idx in tqdm(sample_indices, desc="分析样本"):
        example = eval_dataset[idx]
        input_ids = example[0].unsqueeze(0).to(device)
        label = example[1].unsqueeze(0).to(device)

        source_code = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        results = attacker.analyze_position_selection(
            source_code=source_code,
            label=label,
            position_counts=position_counts,
            iterations=5
        )

        if results:
            all_results[f"sample_{idx}"] = results

    summary = {pos: {
        'feature_distances': [],
        'code_modifications': [],
        'success_count': 0,
        'total_valid_samples': 0,
        'success_distances': []
    } for pos in position_counts}

    for sample_id, sample_results in all_results.items():
        for pos_str, pos_data in sample_results.items():
            pos = int(pos_str)
            if pos_data is not None:
                summary[pos]['feature_distances'].append(pos_data['feature_distance'])
                summary[pos]['code_modifications'].append(pos_data['code_modification'])
                summary[pos]['total_valid_samples'] += 1
                if pos_data['is_success']:
                    summary[pos]['success_count'] += 1
                    summary[pos]['success_distances'].append(pos_data['feature_distance'])

    for pos in position_counts:
        if summary[pos]['total_valid_samples'] > 0:
            summary[pos]['avg_feature_distance'] = np.mean(summary[pos]['feature_distances'])
            summary[pos]['avg_code_modification'] = np.mean(summary[pos]['code_modifications'])
            summary[pos]['success_rate'] = summary[pos]['success_count'] / summary[pos]['total_valid_samples']
            summary[pos]['avg_success_distance'] = np.mean(summary[pos]['success_distances']) if summary[pos][
                'success_distances'] else 0
        else:
            summary[pos]['avg_feature_distance'] = 0
            summary[pos]['avg_code_modification'] = 0
            summary[pos]['success_rate'] = 0
            summary[pos]['avg_success_distance'] = 0
    with open(os.path.join("results", "position_selection.json"), 'w') as f:
        json.dump({
            'summary': {str(k): v for k, v in summary.items()},
            'samples': all_results
        }, f, indent=2)

    best_position_balanced = max(position_counts,
                                 key=lambda p: summary[p]['success_rate'] - 0.1 * summary[p]['avg_feature_distance'])

    distances = [summary[p]['avg_feature_distance'] for p in position_counts]
    growth_rates = [0]
    for i in range(1, len(distances)):
        rate = (distances[i] - distances[i - 1]) / distances[i - 1] if distances[i - 1] > 0 else 0
        growth_rates.append(rate)

    critical_idx = np.argmax(growth_rates[1:]) + 1 if len(growth_rates) > 1 else 0
    critical_position = position_counts[critical_idx]
    return critical_position


if __name__ == '__main__':
    main()