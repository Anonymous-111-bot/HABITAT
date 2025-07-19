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
from Crossdomainattacker import MABMemoryManager
from tqdm import tqdm
from datetime import datetime

sys.path.append('../codebert_downstream_model/run')
# 设置环境变量和日志配置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 运行代码时使用离线模式
warnings.simplefilter(action='ignore', category=FutureWarning)  # 仅报告警告

MODEL_CLASSES = {
    'unixcoder': (RobertaConfig, RobertaModel, RobertaTokenizer),
}

logging.basicConfig(level=logging.INFO)
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    parser.add_argument("--do_position_selection", action='store_true')
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
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints.')
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
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="编码序列的最大长度")

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

    # 设置设备
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # 设置随机种子
    set_seed(args.seed)

    # 加载模型配置、分词器和模型
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.model_name_or_path is None:
        args.model_name_or_path = 'microsoft/unixcoder-base'

    # 加载配置
    config = config_class.from_pretrained(args.model_name_or_path)
    config.num_labels = args.number_labels

    # 加载分词器
    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)
    else:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    # 设置 block_size
    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
    args.block_size = min(args.block_size, tokenizer.model_max_length)

    # 初始化模型编码器
    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)

    # 从 model.py 导入 Model 类
    from model import Model
    model_tgt = Model(encoder, config, tokenizer, args).to(args.device)
    model_sub = RobertaModel.from_pretrained("microsoft/unixcoder-base").to(args.device)

    # 加载 CodeBERT (MLM) 模型
    try:
        codebert_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    except:
        codebert_mlm = RobertaForMaskedLM.from_pretrained("../cache/microsoft/codebert-base-mlm/")
    try:
        tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    except:
        tokenizer_mlm = RobertaTokenizer.from_pretrained("../cache/microsoft/codebert-base-mlm/")
    codebert_mlm.to('cuda')

    # 加载模型检查点
    checkpoint_path = os.path.join(args.output_dir, 'model_backup.bin')
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=args.device)
        model_tgt.load_state_dict(state_dict)
        print(f"成功加载模型参数：{checkpoint_path}")
    else:
        print(f"模型参数文件不存在：{checkpoint_path}")
        exit(1)  # 无法继续，退出程序

    # 加载数据集
    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)

    # 实例化攻击类
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

    # # 加载memory managers
    # memory_manager1 = MABMemoryManager("mab_preferences_codet5.json")
    # memory_manager2 = MABMemoryManager("mab_preferences_codebert.json")
    # memory_manager1.load_preferences()
    # memory_manager2.load_preferences()

    # Generate code pairs if requested
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
    start_time = time.time()

    # 初始化计数器和结果列表
    total_cnt = 0
    success_attack = 0
    adv_examples = []
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

    for index, example in enumerate(eval_dataset):
        print(f"Processing example {index + 1}/{len(eval_dataset)}")

        # 获取源代码和标签
        input_ids = example[0].unsqueeze(0).to(args.device)
        label = example[1].unsqueeze(0).to(args.device)

        # 解码源代码
        source_code = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # 对原始样本进行预测
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

        # 将原始代码信息添加到结果中

        # # 生成对抗样本
        # source_code_1 = attacker.blankattack(source_code)
        # source_code_2 = attacker.deleteblankattack(source_code_1)
        # #
        # adv_code = attacker.stacked_transfer_attack(
        #     source_code,
        #     label,
        #     memory_manager1,
        #     memory_manager2,
        #     example_index=index,
        #     strategy="adaptive"  # Puedes elegir: "sequential", "ensemble", "competitive", "adaptive"
        # )
        adv_code = attacker.MAB_attack(source_code,label)
        # adv_code = attacker.adapt_transfer_attack(source_code,label,memory_manager=memory_manager2,example_index=index)

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

            # 对对抗样本进行预测
            with torch.no_grad():
                outputs_adv = model_tgt(input_ids=adv_input_ids, labels=label)
                loss_adv, logits_adv = outputs_adv
                prob_adv = torch.softmax(logits_adv, dim=-1)
                pred_adv = torch.argmax(prob_adv, dim=-1).item()

            # 比较预测结果，判断攻击是否成功
            if pred_orig == label.item():
                # 原始预测正确，攻击成功如果对抗样本预测错误
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
    # memory_manager.save_preferences()

    # 计算并打印攻击成功率
    success_rate = success_attack / (total_cnt - wrong) * 100 if (total_cnt - wrong) > 0 else 0
    print(f"Attack success rate: {success_rate:.2f}%")
    print(f"Total samples: {total_cnt}, Correct predictions: {total_cnt - wrong}, Successful attacks: {success_attack}")
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