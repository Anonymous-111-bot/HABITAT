'''For attacking CodeBERT models'''
import sys
import os

sys.path.append('../../../')
sys.path.append('../../../python_parser')

import json
import logging
import argparse
import warnings
import torch
import time
from model import Model
from run import TextDataset
from utils import set_seed
# from python_parser.parser_folder import remove_comments_and_docstrings
from utils import Recorder
from Crossdomainattacker import CrossDomainAttacker
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          RobertaModel)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

logger = logging.getLogger(__name__)


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
    parser.add_argument("--eval_data_file_2", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--csv_store_path", default=None, type=str,
                        help="Path to store the CSV file")

    # Hyper-parameters
    parser.add_argument("--num_of_changes", required=True, type=int,
                        help="number of changes")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--use_ga", action='store_true',
                        help="Whether to GA-Attack.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--do_position_selection", action='store_true',
                        help="是否进行重要性位置选择分析")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--transfer_memory_path", default=None, type=str,
                        help="迁移攻击时使用的memory json文件路径（如codet5.json）")

    parser.add_argument("--use_mab_memory", action='store_true',
                        help="是否使用MAB带记忆攻击（MAB_attack_with_memory）")

    parser.add_argument("--mab_memory_path", default="mab_preferences_codetbe.json", type=str,
                        help="MAB记忆文件保存路径")

    args = parser.parse_args()

    args.device = torch.device("cuda")
    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0

    ## Load Target Model
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')  # 读取model的路径
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        # 如果路径存在且有内容，则从checkpoint load模型
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
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 1  # 只有一个label?
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    
    # 严格按照训练时的逻辑创建模型
    # 检查是否有训练好的模型路径
    checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    
    if os.path.exists(output_dir):
        print(f"Found trained model at: {output_dir}")
        # 如果有训练好的模型，直接创建空模型然后加载权重
        model = model_class(config)
    else:
        print(f"No trained model found at: {output_dir}")
        # 如果没有训练好的模型，使用预训练模型
        if args.model_name_or_path:
            model = model_class.from_pretrained(args.model_name_or_path,
                                                from_tf=bool('.ckpt' in args.model_name_or_path),
                                                config=config,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
        else:
            model = model_class(config)

    # 创建Model实例，与训练时完全一致
    from model import Model
    model_tgt = Model(model, config, tokenizer, args)

    # 加载训练好的模型权重
    print(f"Loading model from: {output_dir}")
    print(f"Model structure:")
    print(f"  - encoder type: {type(model_tgt.encoder)}")
    print(f"  - config: {type(model_tgt.config)}")
    print(f"  - num_labels: {model_tgt.config.num_labels}")

    
    # 详细检查模型参数
    try:
        state_dict = torch.load(output_dir, map_location=args.device)
        print(f"Loaded state dict keys: {list(state_dict.keys())}")
        
        model_dict = model_tgt.state_dict()
        print(f"Current model keys: {list(model_dict.keys())}")
        
        # 检查参数匹配情况
        missing_keys = []
        unexpected_keys = []
        size_mismatch_keys = []
        
        for key in state_dict.keys():
            if key not in model_dict:
                unexpected_keys.append(key)
            elif state_dict[key].shape != model_dict[key].shape:
                size_mismatch_keys.append(f"{key}: {state_dict[key].shape} vs {model_dict[key].shape}")
        
        for key in model_dict.keys():
            if key not in state_dict:
                missing_keys.append(key)
        
        print(f"Parameter analysis:")
        if missing_keys:
            print(f"  Missing keys in loaded state dict: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys in loaded state dict: {unexpected_keys}")
        if size_mismatch_keys:
            print(f"  Size mismatches: {size_mismatch_keys}")
        
        # 尝试严格加载
        try:
            model_tgt.load_state_dict(state_dict, strict=True)
            print("✓ Model loaded successfully with strict=True")
        except Exception as strict_error:
            print(f"✗ Strict loading failed: {strict_error}")
            print("Attempting to load with parameter mapping...")
            
            # 尝试参数映射
            mapped_state_dict = {}
            for key in state_dict.keys():
                if key in model_dict:
                    if state_dict[key].shape == model_dict[key].shape:
                        mapped_state_dict[key] = state_dict[key]
                    else:
                        print(f"  Skipping {key} due to shape mismatch: {state_dict[key].shape} vs {model_dict[key].shape}")
                else:
                    print(f"  Skipping {key} - not found in current model")
            
            # 检查映射后的覆盖率
            mapped_keys = set(mapped_state_dict.keys())
            total_keys = set(model_dict.keys())
            coverage = len(mapped_keys) / len(total_keys) * 100
            print(f"Parameter coverage: {coverage:.1f}% ({len(mapped_keys)}/{len(total_keys)})")
            
            if coverage < 50:
                raise ValueError(f"Parameter coverage too low ({coverage:.1f}%). Model structure may be incompatible.")
            
            model_tgt.load_state_dict(mapped_state_dict, strict=False)
            print("✓ Model loaded with parameter mapping")
            
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("Model loading failed. Please check:")
        print("1. Model file exists and is not corrupted")
        print("2. Model architecture matches training configuration")
        print("3. All required parameters are present")
        raise e
    
    model_tgt.to(args.device)
    print(f"✓ Model moved to device: {args.device}")

    # 创建替代模型用于特征提取
    print("Creating substitute model for feature extraction...")
    model_sub = RobertaModel.from_pretrained('microsoft/codebert-base').to(args.device)
    print(f"✓ Substitute model created: {type(model_sub)}")
    print(f"  - Model device: {next(model_sub.parameters()).device}")
    
    # 测试替代模型的前向传播
    try:
        test_input = torch.randint(0, 1000, (1, 10)).to(args.device)
        with torch.no_grad():
            test_output = model_sub(test_input)
        print(f"✓ Substitute model forward pass successful")
        print(f"  - Output shape: {test_output[0].shape}")
    except Exception as e:
        print(f"✗ Substitute model test failed: {e}")
        raise e


    ## Load CodeBERT (MLM) model
    print("Loading CodeBERT MLM model...")
    if not args.base_model:
        print("✗ Error: --base_model argument is required for MLM model")
        raise ValueError("--base_model argument is required")
    
    try:
        codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
        tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
        codebert_mlm.to('cuda')
        print(f"✓ CodeBERT MLM model loaded: {type(codebert_mlm)}")
        print(f"  - Model device: {next(codebert_mlm.parameters()).device}")
        
        # 测试MLM模型
        test_input = torch.randint(0, 1000, (1, 10)).to('cuda')
        with torch.no_grad():
            test_output = codebert_mlm(test_input)
        print(f"✓ MLM model forward pass successful")
        print(f"  - Output shape: {test_output[0].shape}")
        
    except Exception as e:
        print(f"✗ Failed to load MLM model: {e}")
        raise e

    ## Load Dataset
    print("Loading evaluation dataset...")
    try:
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
        print(f"✓ Dataset loaded: {len(eval_dataset)} examples")
        
        # 测试数据集
        if len(eval_dataset) > 0:
            sample = eval_dataset[0]
            print(f"  - Sample input shape: {sample[0].shape}")
            print(f"  - Sample label: {sample[1]}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        raise e

    from Crossdomainattacker import CrossDomainAttacker
    print("Creating CrossDomainAttacker...")
    try:
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
        print(f"✓ CrossDomainAttacker created successfully")
        print(f"  - Number of transformations: {len(attacker.code_transformations)}")
        print(f"  - Top positions: {attacker.top_positions}")
        print(f"  - Max iterations: {attacker.max_iterations}")
    except Exception as e:
        print(f"✗ Failed to create CrossDomainAttacker: {e}")
        raise e
    model_tgt.eval()
    start_time = time.time()

    # 初始化计数器和结果列表
    total_cnt = 0
    success_attack = 0
    adv_examples = []
    wrong = 0
    if args.do_position_selection:
            best_position = position_selection_experiment
            attacker=attacker,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_tgt=model_tgt,
            device=args.device,
            args=args


    # 初始化结果存储
    results = {
        'original_codes': [],
        'adversarial_codes': [],
        'original_predictions': [],
        'adversarial_predictions': [],
        'attack_success': [],
        'feature_distances': []
    }

    # 加载两个模型的memory（如codet5和codebert）
    memory_manager1 = None
    memory_manager2 = None
    if args.transfer_memory_path and args.transfer_memory_path.endswith('.json'):
        from Crossdomainattacker import MABMemoryManager
        paths = args.transfer_memory_path.split(',')
        memory_manager1 = MABMemoryManager(save_path=paths[0].strip())
        memory_manager1.load_preferences()
        if len(paths) > 1:
            memory_manager2 = MABMemoryManager(save_path=paths[1].strip())
            memory_manager2.load_preferences()
        else:
            memory_manager2 = None

    all_position_mabs = {}

    # 新增：用于保存前后代码对
    code_pairs = []
    pair_id = 1

    # from baseline import BaselineAttacker
    # attacker = BaselineAttacker(args, model_sub, model_tgt, tokenizer)

    for index, example in enumerate(eval_dataset):
        print(f"Processing example {index + 1}/{len(eval_dataset)}")

        # 获取源代码和标签
        input_ids = example[0].unsqueeze(0).to(args.device)
        label = example[1].unsqueeze(0).to(args.device)

        # 解码源代码
        # 直接用 example[0] 的原始内容（假设是 input_ids）
        source_code = tokenizer.decode(example[0], skip_special_tokens=True)
        if memory_manager1 is not None and memory_manager2 is not None:
                    adv_code = attacker.stacked_transfer_attack(source_code, label, memory_manager1, memory_manager2, example_index=index, strategy="adaptive")
        if memory_manager1 is not None:
                    adv_code = attacker.adapt_transfer_attack(source_code, label, memory_manager1, example_index=index)
        else:
                    adv_code = attacker.MAB_attack(source_code, label)

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
            loss, prob_orig = outputs

            pred_orig = (prob_orig[:, 0] > 0.5).long().item()


        if adv_code is None:
            print(f"Attack skipped - original prediction was wrong")
            wrong += 1
            results['attack_success'].append(None)  # 原始预测已错误
        elif adv_code:
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
                loss_adv, prob_adv = outputs_adv
                # Model已经返回概率，不需要再次应用softmax
                pred_adv = (prob_adv[:, 0] > 0.5).long().item()

            # 比较预测结果，判断攻击是否成功
            if pred_orig == label.item():
                # 新增：只保存预测正确的样本的前后代码对
                code_pairs.append({
                    "id": pair_id,
                    "original": source_code,
                    "modified": adv_code,
                    "expected": "yes",
                    "description": f"Attack adaptive - Sample {index}"
                })
                pair_id += 1
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
            results['attack_success'].append(False)

        total_cnt += 1
        print(f"Wrong predictions in original model: {wrong}")
        print(f"Total processed: {total_cnt}, Successful attacks: {success_attack}")

    success_rate = success_attack / (total_cnt - wrong) * 100 if (total_cnt - wrong) > 0 else 0
    print(f"Attack success rate: {success_rate:.2f}%")
    print(f"Total samples: {total_cnt}, Correct predictions: {total_cnt - wrong}, Successful attacks: {success_attack}")
    with open(os.path.join(args.output_dir, 'adv_examples.txt'), 'w', encoding='utf-8') as f:
        for code in adv_examples:
            f.write(code + '\n')

    print("codebertAdversarial examples saved.")
    # --------------------------------------------------------------------------------------------
    with open('./adv_codebert_ensemble_train.txt', 'w') as f:
        for example in adv_examples:
            f.write(example)

    num = 0

    with open(os.path.join(args.output_dir, 'code_pairs.json'), 'w', encoding='utf-8') as f:
        json.dump(code_pairs, f, indent=2, ensure_ascii=False)

def position_selection_experiment(attacker, eval_dataset, tokenizer, model_tgt, device, args):
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

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