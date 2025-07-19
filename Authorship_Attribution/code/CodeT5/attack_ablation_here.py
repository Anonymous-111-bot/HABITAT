'''For attacking CodeBERT models'''
import sys
import os

sys.path.append('../../../')
sys.path.append('../../../python_parser')
retval = os.getcwd()
import csv
import copy
import json
import logging
import argparse
import warnings
import torch
import numpy as np
import pickle
import time
from run import convert_examples_to_features
from run import set_seed
from run import TextDataset
from run import InputFeatures
from utils_main import Recorder
from utils_main import python_keywords, is_valid_substitute, _tokenize
from utils_main import get_identifier_posistions_from_code
from T5_model import CodeT5
from parser.run_parser import get_code_tokens
from tqdm import  tqdm
from datetime import datetime

from torch.utils.data.dataset import Dataset
from torch.utils.data import SequentialSampler, DataLoader
from transformers import RobertaForMaskedLM
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # run the code in offline mode
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

MODEL_CLASSES = {
    't5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
}

logger = logging.getLogger(__name__)
class CodePairsGenerator:
    def __init__(self, attacker, target_model, tokenizer, args):
        """
        Initialize code pairs generator

        Args:
            attacker: CrossDomainAttacker instance
            target_model: Target model for prediction
            tokenizer: Tokenizer for the model
            args: Arguments containing device and other settings
        """
        self.attacker = attacker
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.args = args

    def _get_prediction(self, code, label):
        """Get model prediction for code"""
        # 编码输入
        inputs = self.tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=self.args.block_size,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(self.args.device)
        attention_mask = inputs['attention_mask'].to(self.args.device)

        # 预测
        with torch.no_grad():
            outputs = self.target_model(input_ids=input_ids, labels=label)
            loss, logits = outputs
            pred = torch.argmax(logits, dim=-1).item()

        return pred

    def generate_code_pairs_from_dataset(self, dataset, output_file, max_samples=None, start_idx=0):
        """
        Generate adversarial code pairs from dataset using MAB attack and save to JSON

        Args:
            dataset: TextDataset instance
            output_file: Output JSON file path
            max_samples: Maximum number of samples to process (None for all)
            start_idx: Starting index in dataset

        Returns:
            List of code pairs
        """
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

        # Process each sample
        for idx in tqdm(range(start_idx, start_idx + samples_to_process), desc="Generating code pairs"):
            try:
                # Get sample from dataset
                example = dataset[idx]
                input_ids = example[0].unsqueeze(0).to(self.args.device)
                label = example[1].unsqueeze(0).to(self.args.device)

                # Decode source code
                original_code = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

                # Get original prediction
                original_pred = self._get_prediction(original_code, label)

                # Only attack if original prediction matches label
                if original_pred != label.item():
                    logger.info(f"Sample {idx}: Original prediction already wrong, skipping...")
                    skipped_samples += 1
                    continue

                # Perform MAB attack
                total_attempts += 1
                logger.info(f"Sample {idx}: Performing MAB attack...")

                # Use MAB attack
                adversarial_code = self.attacker.MAB_attack(original_code, label)

                # Check if attack was successful
                adversarial_pred = self._get_prediction(adversarial_code, label)
                #
                # if adversarial_pred != label.item():
                #     successful_attacks += 1
                #     expected = "yes"  # They are different (attack succeeded)
                #     logger.info(f"Sample {idx}: Attack successful!")
                # else:
                #     expected = "yes"  # They are the same (attack failed)
                #     logger.info(f"Sample {idx}: Attack failed, codes are still equivalent")

                # Create code pair entry
                code_pair = {
                    "id": len(code_pairs) + 1,
                    "original": original_code,
                    "modified": adversarial_code,
                    "expected":"yes",
                    "description": f"MAB Attack - Sample {idx}"
                }

                code_pairs.append(code_pair)

                # Log progress every 10 samples
                if len(code_pairs) % 10 == 0:
                    success_rate = (successful_attacks / total_attempts * 100) if total_attempts > 0 else 0
                    logger.info(f"Progress: {len(code_pairs)} pairs generated, "
                                f"Attack success rate: {success_rate:.2f}%")

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue

        # Save results to JSON file
        self._save_code_pairs(code_pairs, output_file)

        # Print final statistics
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
        """Save code pairs to JSON file"""
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(code_pairs, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully saved {len(code_pairs)} code pairs to {output_file}")

        except Exception as e:
            logger.error(f"Error saving code pairs to {output_file}: {str(e)}")
            raise



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

    parser.add_argument("--model_type", default="bert", type=str,
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
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
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
    parser.add_argument("--number_labels", type=int,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--use_ga", action='store_true',
                        help="Whether to GA-Attack.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
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
    parser.add_argument("--num_of_changes", required=True, type=int,
                        help="number of inserted obfuscations.")
    parser.add_argument("--do_position_selection", action='store_true',
                        help="是否进行重要性位置选择分析")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # New arguments for code pairs generation
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

    args.device = torch.device("cuda")
    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0

    ## Load Target Model
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')  # 读取model的路径
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained('Salesforce/codet5-small')
    config.num_labels = args.number_labels
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model_tgt = model_class.from_pretrained('Salesforce/codet5-small')
    else:
        model_tgt = model_class(config)

    model_tgt = CodeT5(model_tgt, config, tokenizer, args).to(args.device)

    # 加载模型检查点
    # 加载模型检查点
    checkpoint_prefix = 'model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    ckpt = torch.load(output_dir)
    model_tgt.load_state_dict(torch.load(output_dir))
    model_tgt.to(args.device)
    from Authorship_Attribution.code.UniXcoder.Crossdomainattacker import MABMemoryManager
    # 加载memory managers
    # memory_manager2 = MABMemoryManager("mab_preferences_unixcoder.json")
    # memory_manager1 = MABMemoryManager("mab_preferences_codebert.json")
    # memory_manager1.load_preferences()
    # memory_manager2.load_preferences()
    # 调试：打印模型权重是否加载成功
    print("Model weights loaded successfully:", all([param.requires_grad for param in model_tgt.parameters()]))

    try:
        model_sub = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small").to(args.device)
    except:
        model_sub = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small').to(args.device)

    codebert_mlm = model_sub
    tokenizer_mlm = tokenizer
    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)

    from Authorship_Attribution.code.UniXcoder.Crossdomainattacker import CrossDomainAttacker
    attacker = CrossDomainAttacker(args, model_sub, model_tgt, tokenizer, codebert_mlm, tokenizer_mlm, use_bpe=1,
                                   threshold_pred_score=0)
    # memory_manager1 = MABMemoryManager("mab_preferences_unixcoder.json")
    # memory_manager2 = MABMemoryManager("mab_preferences_codet5.json")
    # memory_manager1.load_preferences()
    # memory_manager2.load_preferences()

    if args.generate_code_pairs:
            print("\n==== 开始生成MAB攻击代码对 ====")

            # Create output filename if not provided
            if args.output_code_pairs_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                args.output_code_pairs_file = f'mab_code_pairs_{timestamp}.json'

            # Create generator (简化版本)
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

            print(f"\n成功生成 {len(code_pairs)} 个MAB攻击代码对，保存在 {args.output_code_pairs_file}")

            # 返回而不执行后续的攻击流程
            return

    if args.do_position_selection:
        print("\n==== 开始进行位置选择实验 ====")
        best_position = position_selection_experiment(
            attacker=attacker,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_tgt=model_tgt,
            device=args.device,
            args=args
        )

    # 开始攻击
    model_tgt.eval()
    total_cnt = 0
    success_attack = 0
    adv_examples = []
    start_time = time.time()
    wrong = 0
    # 初始化结果存储
    results = {
        'original_codes': [],
        'adversarial_codes': [],
        'original_predictions': [],
        'adversarial_predictions': [],
        'attack_success': [],
        'feature_distances': []
    }



    # success = memory_manager_codebert.load_preferences()
    # print(f"加载CodeBERT偏好文件: {'成功' if success else '失败'}")

    # if success:
    #     print(f"偏好文件中包含 {len(memory_manager_codebert.preferences)} 个条目")
    #     if memory_manager_codebert.preferences:
    #         print(f"前5个键: {list(memory_manager_codebert.preferences.keys())[:5]}")

    for index, example in enumerate(eval_dataset):
        print(f"Processing example {index + 1}/{len(eval_dataset)}")

        # 获取源代码和标签
        input_ids = example[0].unsqueeze(0).to(args.device)
        label = example[1].unsqueeze(0).to(args.device)

        # Convert input IDs to tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # Print tokens to check for None or invalid tokens
        # print(tokens)

        # Filter out None values from the tokens list
        filtered_tokens = [token for token in tokens if token is not None]

        source_code = tokenizer.decode([tokenizer.convert_tokens_to_ids(token) for token in filtered_tokens],
                                       skip_special_tokens=True)

        # 对原始样本进行预测
        # Predict on the original sample
        with torch.no_grad():
            inputs = tokenizer.encode_plus(
                source_code,
                add_special_tokens=True,
                max_length=args.block_size,
                truncation=True,
                return_tensors='pt'
            )
            input_ids = inputs['input_ids'].to(args.device)  # Shape: [1, seq_length]
            attention_mask = inputs['attention_mask'].to(args.device)  # Shape: [1, seq_length]

            # Check shapes
            # print(f"input_ids shape: {input_ids.shape}")
            # print(f"labels shape: {label.shape}")

            outputs = model_tgt(input_ids=input_ids, labels=label)
            loss, logits = outputs  # Unpack outputs

            prob_orig = torch.softmax(logits, dim=-1)  # Shape: [1, num_classes]
            pred_orig = torch.argmax(prob_orig, dim=-1).item()  # Predicted label

        # source_code_1 = attacker.blankattack(source_code)
        # source_code_2 = attacker.deleteblankattack(source_code_1)

        #
        # adv_code = attacker.stacked_transfer_attack(
        #     source_code,
        #     label,
        #     memory_manager1,
        #     memory_manager2,
        #     example_index=index,
        #     strategy="adaptive"  # Puedes elegir: "sequential", "ensemble", "competitive", "adaptive"
        # )
        adv_code = attacker.MAB_attack(source_code,label)

        # from baseline import BaselineAttacker
        # attacker = BaselineAttacker(args, model_sub, model_tgt, tokenizer)
        # adv_code = attacker.baseline_attack(source_code, label)
        if adv_code:
            # 编码对抗样本
            adv_inputs = tokenizer.encode_plus(
                adv_code,
                add_special_tokens=True,
                max_length=args.block_size,
                truncation=True,
                return_tensors='pt'
            )
            adv_input_ids = adv_inputs['input_ids'].to(args.device)
            adv_attention_mask = adv_inputs['attention_mask'].to(args.device)

            # # 调试：打印对抗样本的 input_ids 和 attention_mask
            # print(f"adv_input_ids shape: {adv_input_ids.shape}, content: {adv_input_ids}")
            # print(f"adv_attention_mask shape: {adv_attention_mask.shape}, content: {adv_attention_mask}")

            # 预测对抗样本
            with torch.no_grad():
                outputs_adv = model_tgt(input_ids=adv_input_ids)
                logits_adv = outputs_adv[0]

                # 调试：打印对抗样本的 logits 和概率分布
                # print(f"logits_adv shape: {logits_adv.shape}, content: {logits_adv}")
                prob_adv = torch.softmax(logits_adv, dim=-1)
                # print(f"prob_adv shape: {prob_adv.shape}, content: {prob_adv}")

                pred_adv = torch.argmax(prob_adv, dim=-1).item()
                # print(f"pred_adv: {pred_adv}, pred_orig: {pred_orig}, label: {label.item()}")
            # 比较预测结果，判断攻击是否成功
            if pred_orig == label.item():
                # If the original prediction was correct, attack success if adversarial prediction is wrong
                if pred_adv != pred_orig:
                    adv_examples.append(adv_code)
                    success_attack += 1

                    print(f"Attack successful! Original prediction: {pred_orig}, Adversarial prediction: {pred_adv}")
                else:
                    print(f"Attack failed! Original prediction: {pred_orig}, Adversarial prediction: {pred_adv}")
            else:
                wrong += 1;
                print(f"Original prediction was wrong")

        else:
            print("Attack did not generate adversarial example.")

        total_cnt += 1
        print(f"Total processed: {total_cnt}, Successful attacks: {success_attack}")
        print(wrong)
    # memory_manager.save_preferences()


    # 打印攻击成功率
    success_rate = success_attack / (total_cnt - wrong) * 100 if total_cnt > 0 else 0
    print(f"Attack success rate: {success_rate:.2f}%")

    # 保存对抗样本
    with open(os.path.join(args.output_dir, 'adv_examples.txt'), 'w', encoding='utf-8') as f:
        for code in adv_examples:
            f.write(code + '\n')

    print("Adversarial examples saved.")

def position_selection_experiment(attacker, eval_dataset, tokenizer, model_tgt, device, args):
        """
        进行位置选择实验，分析不同位置数量的影响
        """
        import os
        import json
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        # 创建结果目录
        os.makedirs("results", exist_ok=True)

        # 要测试的位置数量
        position_counts = [1, 2, 3, 5, 8, 10, 15]

        # 所有样本的结果
        all_results = {}

        # 选择样本数量
        sample_size = len(eval_dataset)
        sample_indices = list(range(sample_size))

        # 样本级别的数据
        for idx in tqdm(sample_indices, desc="分析样本"):
            # 获取样本
            example = eval_dataset[idx]
            input_ids = example[0].unsqueeze(0).to(device)
            label = example[1].unsqueeze(0).to(device)

            # 解码源代码
            source_code = tokenizer.decode(input_ids[0], skip_special_tokens=True)

            # 分析不同位置数量的影响
            results = attacker.analyze_position_selection(
                source_code=source_code,
                label=label,
                position_counts=position_counts,
                iterations=5
            )

            if results:
                all_results[f"sample_{idx}"] = results

        # 汇总结果
        # 汇总结果
        summary = {pos: {
            'feature_distances': [],
            'code_modifications': [],
            'success_count': 0,
            'total_valid_samples': 0,  # 只计算原始预测正确的样本数
            'success_distances': []
        } for pos in position_counts}

        # 计算平均值
        for sample_id, sample_results in all_results.items():
            for pos_str, pos_data in sample_results.items():
                pos = int(pos_str)
                # 只有在分析了该样本在这个位置数量下的结果时才处理
                if pos_data is not None:
                    summary[pos]['feature_distances'].append(pos_data['feature_distance'])
                    summary[pos]['code_modifications'].append(pos_data['code_modification'])
                    summary[pos]['total_valid_samples'] += 1  # 每个分析的样本都是原始预测正确的
                    if pos_data['is_success']:
                        summary[pos]['success_count'] += 1
                        summary[pos]['success_distances'].append(pos_data['feature_distance'])

        # 计算平均值和成功率
        for pos in position_counts:
            if summary[pos]['total_valid_samples'] > 0:
                summary[pos]['avg_feature_distance'] = np.mean(summary[pos]['feature_distances'])
                summary[pos]['avg_code_modification'] = np.mean(summary[pos]['code_modifications'])
                # 正确计算成功率：成功样本数 / 原始预测正确的样本数
                summary[pos]['success_rate'] = summary[pos]['success_count'] / summary[pos]['total_valid_samples']
                summary[pos]['avg_success_distance'] = np.mean(summary[pos]['success_distances']) if summary[pos][
                    'success_distances'] else 0
            else:
                summary[pos]['avg_feature_distance'] = 0
                summary[pos]['avg_code_modification'] = 0
                summary[pos]['success_rate'] = 0
                summary[pos]['avg_success_distance'] = 0
        # 保存结果
        with open(os.path.join("results", "position_selection.json"), 'w') as f:
            json.dump({
                'summary': {str(k): v for k, v in summary.items()},
                'samples': all_results
            }, f, indent=2)

        # 可视化结果
        # visualize_position_selection(summary, position_counts)

        # 找出最优位置数量
        # 1. 基于成功率与特征距离的平衡
        best_position_balanced = max(position_counts,
                                     key=lambda p: summary[p]['success_rate'] - 0.1 * summary[p][
                                         'avg_feature_distance'])

        # 2. 基于特征距离增长率的临界点
        distances = [summary[p]['avg_feature_distance'] for p in position_counts]
        growth_rates = [0]
        for i in range(1, len(distances)):
            rate = (distances[i] - distances[i - 1]) / distances[i - 1] if distances[i - 1] > 0 else 0
            growth_rates.append(rate)

        critical_idx = np.argmax(growth_rates[1:]) + 1 if len(growth_rates) > 1 else 0
        critical_position = position_counts[critical_idx]

        print(f"\n基于分析结果:")
        print(f"平衡型最优位置数量: {best_position_balanced}")
        print(f"临界点位置数量: {critical_position}")

        # 返回建议的最优位置数量
        return critical_position


if __name__ == '__main__':
    main()