import hashlib
import json
import os
import logging

logger = logging.getLogger(__name__)
import copy
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
import re
import math
import torch
import torch.nn as nn
import numpy as np
import sys
import ast
import astor
import io
import tokenize
import builtins
import keyword
from transformers import AutoTokenizer, T5ForSequenceClassification, T5Config, T5Tokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label=None,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label


class UCB1MAB:
    """UCB1算法实现的多臂老虎机"""

    def __init__(self, num_arms: int, alpha: float = 1.0):
        """
        初始化UCB1多臂老虎机

        Args:
            num_arms: 臂的数量
            alpha: 探索参数，控制探索的激进程度
        """
        self.num_arms = num_arms
        self.alpha = alpha
        self.values = np.zeros(num_arms)  # 每个臂的估计价值
        self.counts = np.zeros(num_arms)  # 每个臂被选择的次数
        self.total_pulls = 0  # 总共选择臂的次数

    def select_arm(self) -> int:
        """
        使用UCB1策略选择一个臂

        Returns:
            选中的臂的索引
        """
        # 如果有臂未被选择过，优先选择它
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        # 计算UCB值
        ucb_values = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            exploitation = self.values[arm]  # 利用项
            exploration = self.alpha * np.sqrt(2 * np.log(self.total_pulls) / self.counts[arm])  # 探索项
            ucb_values[arm] = exploitation + exploration

        # 选择UCB值最高的臂
        return np.argmax(ucb_values)

    def update(self, arm: int, reward: float) -> None:
        """
        更新指定臂的估计价值

        Args:
            arm: 选中的臂的索引
            reward: 获得的奖励
        """
        self.counts[arm] += 1
        self.total_pulls += 1

        # 增量更新公式
        step_size = 1.0 / self.counts[arm]
        self.values[arm] = (1 - step_size) * self.values[arm] + step_size * reward

    def log_arm_parameters(self):
        """记录并打印每个臂的参数（估计价值和选择次数）"""
        for arm in range(self.num_arms):
            logger.info(f"臂 {arm} -> 估计价值: {self.values[arm]:.4f}, 被选择次数: {self.counts[arm]}")


class PositionMAB:
    """
    位置级别的多臂老虎机，管理单个位置的插入策略
    """

    def __init__(self, position_id: int, num_transformations: int):
        self.position_id = position_id
        self.mab = UCB1MAB(num_transformations)  # 创建UCB1MAB对象
        self.best_transformation = None
        self.best_reward = -float('inf')

    def select_transformation(self) -> int:
        """选择一个转换操作"""
        return self.mab.select_arm()  # 委托给UCB1MAB的select_arm

    def update(self, transformation_id: int, reward: float) -> None:
        """更新转换操作的奖励"""
        self.mab.update(transformation_id, reward)

        # 跟踪最佳转换
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_transformation = transformation_id

    def select_arm(self):
        """
        代理UCB1MAB中的select_arm方法
        """
        return self.mab.select_arm()  # 返回MAB中选择的臂（转换操作）

    def log_arm_parameters(self):
        """记录并打印每个臂的参数（估计价值和选择次数）"""
        logger.info(f"位置 {self.position_id} - MAB状态:")
        self.mab.log_arm_parameters()  # 委托给UCB1MAB的log_arm_parameters方法


class MABMemoryManager:
    def __init__(self, save_path='mab_preferences_codebert.json'):

        self.save_path = save_path
        self.preferences = {}

    def store_preference(self, code, position_id, best_transformation, all_values, example_index=None):
        """
        存储特定代码样本的 MAB 偏好
        简化版 - 直接使用 example_index 作为键

        Args:
            code: 原始代码字符串或哈希
            position_id: 位置ID
            best_transformation: 该位置最佳的转换索引
            all_values: 该位置所有转换的估计值
            example_index: 可选的示例索引
        """
        # 使用 example_index 或哈希值作为键
        key = str(example_index) if example_index is not None else self.compute_code_hash(code)

        if key not in self.preferences:
            self.preferences[key] = {}

        position_count = len(self.preferences[key])
        position_number = str(position_count + 1)

        self.preferences[key][position_number] = {
            'best_transformation': best_transformation,
            'values': all_values
        }

    def save_preferences(self):
        """
        保存偏好到JSON文件，处理NumPy类型
        """

        # 创建一个自定义的JSON编码器来处理NumPy类型
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        # 使用自定义编码器保存JSON
        with open(self.save_path, 'w') as f:
            json.dump(self.preferences, f, indent=2, cls=NumpyEncoder)

        logger.info(f"MAB 偏好已保存到 {self.save_path}")

    def load_preferences(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                self.preferences = json.load(f)
            logger.info(f"已从 {self.save_path} 加载 MAB 偏好")
            return True
        else:
            logger.warning(f"在 {self.save_path} 中未找到偏好文件")
            return False

    def get_preferences(self, code_hash):
        return self.preferences.get(code_hash)

    def compute_code_hash(self, code):

        return hashlib.md5(code.encode()).hexdigest()

    def get_all_preferences(self):
        """获取所有存储的偏好"""
        return self.preferences

    def get_index_by_hash(self, code_hash):
        """通过代码哈希获取索引"""
        return self.code_to_index.get(code_hash)



# 添加一个必要的辅助函数，用于迁移学习中的位置映射
def normalize_positions(old_positions, new_positions):
    normalized = {}

    # 根据新代码的位置数量限制使用的旧位置数量
    available_positions = min(len(old_positions), len(new_positions))

    # 只使用前available_positions个位置
    for i in range(1, available_positions + 1):
        pos_id = str(i)
        if pos_id in old_positions:
            normalized[str(i)] = old_positions[pos_id]

    return normalized
class CrossDomainAttacker:

    def __init__(self, args, model_sub, model_tgt, tokenizer, model_mlm, tokenizer_mlm,
                 use_bpe, threshold_pred_score, targeted=False, epsilon=0.1):
        """
        初始化跨域攻击器

        Args:
            args: 参数配置
            model_sub: 替代模型
            model_tgt: 目标模型
            tokenizer: 分词器
            model_mlm: 掩码语言模型
            tokenizer_mlm: 掩码语言模型分词器
            use_bpe: 是否使用BPE
            threshold_pred_score: 预测阈值
            targeted: 是否进行定向攻击
            epsilon: 攻击扰动大小
        """
        self.args = args
        self.model_sub = model_sub  # 替代模型
        self.model_tgt = model_tgt  # 目标模型
        self.tokenizer = tokenizer  # 分词器
        self.model_mlm = model_mlm  # 掩码语言模型
        self.tokenizer_mlm = tokenizer_mlm  # 掩码语言模型分词器
        self.use_bpe = use_bpe  # 是否使用BPE
        self.threshold_pred_score = threshold_pred_score  # 预测阈值
        self.targeted = targeted  # 是否为定向攻击


        # 攻击参数
        self.top_positions = 3 # 选择影响最大的前k个位置
        self.max_iterations = 1# 最大攻击迭代次数\
        self.exi =1.0
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # C语言代码转换操作
        self.code_transformations = [
            # 1. 添加无用的if语句
            lambda indent, _: "{}if (0) {{\n{}    int unused_var_{} = 42;\n{}}}".format(
                indent, indent, random.randint(1000, 9999), indent),
            
            # 2. 添加无用的while循环
            lambda indent, _: "{}while (0) {{\n{}    int unused_var_{} = 0;\n{}    break;\n{}}}".format(
                indent, indent, random.randint(1000, 9999), indent, indent),
            
            # 3. 添加无用的变量声明
            lambda indent, _: "{}int unused_var_{} = {};".format(
                indent, random.randint(1000, 9999), random.choice([0, 1, -1, 42, 100])),
            
            # 4. 添加无用的函数声明
            lambda indent, _: "{}void unused_func_{}() {{\n{}    return;\n{}}}".format(
                indent, random.randint(1000, 9999), indent, indent),
            
            # 5. 添加注释
            lambda indent, _: "{}/* NOTE: This is a comment */".format(indent),
            
            # 6. 添加无用的printf语句
            lambda indent, _: '{}printf("");'.format(indent),
            
            # 7. 添加无用的宏定义
            lambda indent, _: "{}#define UNUSED_MACRO_{} 0".format(
                indent, random.randint(1000, 9999)),
            
            # 8. 添加无用的结构体声明
            lambda indent, _: "{}struct unused_struct_{} {{\n{}    int dummy;\n{}}};".format(
                indent, random.randint(1000, 9999), indent, indent),
            
            # 9. 添加无用的枚举
            lambda indent, _: "{}enum unused_enum_{} {{\n{}    DUMMY = 0\n{}}};".format(
                indent, random.randint(1000, 9999), indent, indent),
            
            # 10. 添加无用的类型定义
            lambda indent, _: "{}typedef int unused_type_{};".format(
                indent, random.randint(1000, 9999)),
        ]

    def _get_prediction(self, code, label):
        """
        获取模型对代码的预测

        Args:
            code: 代码字符串
            label: 真实标签

        Returns:
            预测标签
        """
        features = self._extract_code_features(code)
        with torch.no_grad():
            # Model.forward() 返回概率，不是logits
            probs = self.model_tgt(torch.tensor([features.input_ids]).to(self.args.device))
            # 对于二分类，概率大于0.5为1，否则为0
            preds = (probs[:, 0] > 0.5).long()
            return preds[0].item()

    def _calculate_loss(self, orig_features, new_features):
        """
        计算两个特征表示之间的损失/距离

        Args:
            orig_features: 原始特征
            new_features: 新特征

        Returns:
            损失值
        """
        # 提取特征向量
        with torch.no_grad():
            # 使用RobertaModel
            orig_outputs = self.model_sub(torch.tensor([orig_features.input_ids]).to(self.args.device))
            new_outputs = self.model_sub(torch.tensor([new_features.input_ids]).to(self.args.device))

            orig_embeddings = orig_outputs[0]  # RobertaModel返回(last_hidden_state, ...)
            new_embeddings = new_outputs[0]

        # 计算距离 - 使用第一个token的表示
        orig_cls = orig_embeddings[:, 0, :]  # 第一个token
        new_cls = new_embeddings[:, 0, :]  # 第一个token
        distance = torch.norm(orig_cls - new_cls)
        return distance.item()

    def _extract_code_features(self, code):
        """
        提取代码特征

        Args:
            code: 代码字符串

        Returns:
            InputFeatures对象
        """
        # 使用unixcoder的分词方式
        code_tokens = self.tokenizer.tokenize(code)[:self.args.block_size - 2]
        source_tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = self.args.block_size - len(source_ids)
        source_ids += [self.tokenizer.pad_token_id] * padding_length
        return InputFeatures(source_tokens, source_ids, 0)

    def _check_syntax_validity(self, code):
        """
        检查C语言代码语法是否有效（简化版本）

        Args:
            code: 代码字符串

        Returns:
            布尔值,表示代码语法是否有效
        """
        # 简化的C语言语法检查
        try:
            # 检查基本的C语言语法特征
            lines = code.split('\n')
            
            # 检查括号匹配
            open_braces = code.count('{')
            close_braces = code.count('}')
            if open_braces != close_braces:
                return False
            
            # 检查括号匹配
            open_parens = code.count('(')
            close_parens = code.count(')')
            if open_parens != close_parens:
                return False
            
            # 检查方括号匹配
            open_brackets = code.count('[')
            close_brackets = code.count(']')
            if open_brackets != close_brackets:
                return False
            
            # 检查分号（C语言语句结束符）
            # 但要注意字符串中的分号
            semicolons = code.count(';')
            if semicolons == 0 and len(code.strip()) > 0:
                # 如果没有分号，可能是预处理指令或函数定义
                if not any(code.strip().startswith(prefix) for prefix in ['#', 'typedef', 'enum', 'struct']):
                    return False
            
            return True
        except Exception:
            return False

    def _get_insert_masked_code(self, code, pos):
        """
        在指定位置插入掩码

        Args:
            code: 代码字符串
            pos: 插入位置 (可能是整数或元组)

        Returns:
            插入掩码后的代码
        """
        # 处理新的位置格式 (line_idx, char_pos) 或旧格式 (line_idx)
        if isinstance(pos, tuple):
            line_idx, char_pos = pos
            lines = code.split('\n')
            
            # 如果是单行代码，在字符位置插入
            if len(lines) == 1:
                line = lines[0]
                new_line = line[:char_pos] + '<mask>' + line[char_pos:]
                return new_line
            else:
                # 多行代码，在指定行插入
                if line_idx < len(lines):
                    line = lines[line_idx]
                    new_line = line[:char_pos] + '<mask>' + line[char_pos:]
                    lines[line_idx] = new_line
                    return '\n'.join(lines)
                else:
                    return code
        else:
            # 旧格式处理
            def count_space(line):
                count = 0
                for char in line:
                    if char == ' ':
                        count += 1
                    if char != ' ':
                        break
                return count

            splited_code = code.split('\n')
            if pos == 0:
                space_num = count_space(splited_code[pos])
            elif pos == len(splited_code) - 1:
                space_num = count_space(splited_code[pos])
            else:
                space_num = max(count_space(splited_code[pos]), count_space(splited_code[pos + 1]))

            splited_code.insert(pos, ' ' * space_num + '<mask>')
            inserted_code_str = ''
            for line in splited_code:
                inserted_code_str += (line + '\n')
            return inserted_code_str

    def _apply_transformation(self, code, position, transformation_index):
        """
        在指定位置应用代码转换，确保插入时的缩进保持不变。

        Args:
            code: 代码字符串
            position: 位置索引
            transformation_index: 转换操作索引

        Returns:
            转换后的代码
        """
        # 获取指定转换函数
        transformation = self.code_transformations[transformation_index]

        # 获取插入位置的缩进信息
        lines = code.split('\n')
        if position < 0 or position >= len(lines):
            return code

        # 获取当前行的缩进
        current_line = lines[position]
        indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
        indentation = ' ' * indent

        # 应用转换
        try:
            # 调用转换函数，插入转换后的内容，保持缩进
            insert_content = transformation(indentation, "")

            # 在当前行后插入转换内容，确保缩进不丢失
            new_lines = lines.copy()
            new_lines.insert(position + 1, insert_content)  # 在当前行后插入
            return '\n'.join(new_lines)
        except Exception as e:
            logger.error(f"转换应用错误: {e}")
            return code

    def _find_safe_insertion_lines(self, code_lines):
        """
        找出可以安全插入C语言代码的行号
        注意：由于数据集处理，代码可能被合并为一行

        Args:
            code_lines: 代码行列表

        Returns:
            可以安全插入代码的行号列表
        """
        safe_lines = []
        
        # 如果代码只有一行（被合并了），我们需要在合适的位置插入
        if len(code_lines) == 1:
            # 在单行代码中寻找合适的插入点
            code = code_lines[0]
            
            # 寻找合适的分号位置（语句结束）
            semicolon_positions = []
            brace_positions = []
            
            # 找到所有分号位置
            for i, char in enumerate(code):
                if char == ';':
                    semicolon_positions.append(i)
                elif char == '{':
                    brace_positions.append(i)
                elif char == '}':
                    brace_positions.append(i)
            
            # 在分号后插入（语句结束）
            for pos in semicolon_positions:
                if pos < len(code) - 1:  # 不是最后一个字符
                    safe_lines.append((0, pos + 1))  # (行号, 字符位置)
            
            # 如果没有找到分号，尝试在大括号后插入
            if not safe_lines and brace_positions:
                for pos in brace_positions:
                    if pos < len(code) - 1:
                        safe_lines.append((0, pos + 1))
            
            # 如果还是没有找到，在代码中间插入
            if not safe_lines:
                mid_pos = len(code) // 2
                safe_lines.append((0, mid_pos))
                
        else:
            # 多行代码的处理（原有逻辑）
            in_multiline_comment = False
            in_string = False
            brace_level = 0
            in_function_def = False

            for i, line in enumerate(code_lines):
                line_stripped = line.strip()

                # 跳过空行
                if not line_stripped:
                    continue

                # 处理多行注释 /* ... */
                if '/*' in line and '*/' not in line:
                    in_multiline_comment = True
                    continue
                elif '*/' in line:
                    in_multiline_comment = False
                    continue
                elif in_multiline_comment:
                    continue

                # 跳过单行注释
                if line_stripped.startswith('//') or line_stripped.startswith('/*'):
                    continue

                # 跳过预处理指令
                if line_stripped.startswith('#'):
                    continue

                # 检测函数定义开始
                if (line_stripped.startswith('int ') or line_stripped.startswith('void ') or 
                    line_stripped.startswith('char ') or line_stripped.startswith('float ') or
                    line_stripped.startswith('double ') or line_stripped.startswith('long ') or
                    line_stripped.startswith('short ') or line_stripped.startswith('unsigned ') or
                    line_stripped.startswith('signed ') or line_stripped.startswith('struct ') or
                    line_stripped.startswith('enum ') or line_stripped.startswith('union ') or
                    line_stripped.startswith('typedef ')) and '(' in line:
                    in_function_def = True
                    continue

                # 检测函数定义结束（找到分号或大括号）
                if in_function_def and (';' in line or '{' in line):
                    in_function_def = False
                    if '{' in line:
                        brace_level += 1
                    continue

                # 跳过结构体/枚举/联合体定义
                if (line_stripped.startswith('struct ') or line_stripped.startswith('enum ') or 
                    line_stripped.startswith('union ')) and '{' in line:
                    continue

                # 跳过宏定义
                if line_stripped.startswith('#define '):
                    continue

                # 跳过包含文件
                if line_stripped.startswith('#include '):
                    continue

                # 检查括号层级
                for char in line:
                    if char == '{':
                        brace_level += 1
                    elif char == '}':
                        brace_level -= 1

                # 处理字符串（但不跳过整行）
                if '"' in line:
                    # 简单的字符串检测（不处理转义字符）
                    quote_count = line.count('"')
                    if quote_count % 2 == 1:  # 奇数个引号
                        in_string = not in_string

                # 如果在字符串中，跳过
                if in_string:
                    continue

                # 如果在函数定义中，跳过
                if in_function_def:
                    continue

                # 添加到安全行列表
                safe_lines.append((i, 0))  # (行号, 字符位置)

        return safe_lines

    def _find_important_positions(self, code, top_k):
        """
        找出代码中的重要位置，通过掩码前后的特征差异来确定。

        Args:
            code: 代码字符串
            top_k: 返回的重要位置数量

        Returns:
            重要且安全的位置列表
        """
        lines = code.split('\n')

        # 找出安全的插入位置
        safe_lines = self._find_safe_insertion_lines(lines)

        if len(safe_lines) == 0:
            logger.warning("没有找到安全的插入位置，尝试使用简单启发式方法")
            # 简单启发式：找到非空行，不在字符串或注释中的行
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('#'):
                    safe_lines.append(i)

        if len(safe_lines) == 0:
            logger.warning("仍然没有找到安全的插入位置，返回前几行作为默认位置")
            # 如果仍然没有找到安全位置，使用前几行作为默认位置
            safe_lines = list(range(min(5, len(lines))))

        logger.info(f"找到 {len(safe_lines)} 个安全插入位置")

        # 如果安全位置不足top_k个，全部返回
        if len(safe_lines) <= top_k:
            return safe_lines

        # 提取原始代码特征
        orig_features = self._extract_code_features(code)

        # 计算每个位置的重要性分数
        position_scores = []
        for pos in safe_lines:
            try:
                # 在该位置插入掩码
                masked_code = self._get_insert_masked_code(code, pos)
                # 提取掩码后的特征
                masked_features = self._extract_code_features(masked_code)
                # 计算特征差异（使用某种距离度量）
                feature_diff = self._calculate_loss(orig_features, masked_features)
                position_scores.append((pos, feature_diff))
            except Exception as e:
                logger.error(f"计算位置 {pos} 的重要性时出错: {str(e)}")
                position_scores.append((pos, 0.0))  # 出错时分配最低分数

        # 按重要性分数降序排序
        position_scores.sort(key=lambda x: x[1], reverse=True)

        # 选择前top_k个重要位置
        important_positions = [pos for pos, score in position_scores[:top_k]]

        logger.info(f"选择了 {len(important_positions)} 个重要位置: {important_positions}")
        return important_positions

    def _apply_transformation_with_indentation(self, code, position, transform_idx):
        """
        应用转换并保持正确的缩进

        Args:
            code: 源代码字符串
            position: 要应用转换的位置
            transform_idx: 转换方法的索引

        Returns:
            转换后的代码字符串
        """
        # 将代码分割成行
        lines = code.split('\n')

        # 计算每行的起始位置
        line_starts = [0]
        current_pos = 0
        for i, line in enumerate(lines[:-1]):  # 不包括最后一行
            current_pos += len(line) + 1  # +1 是换行符
            line_starts.append(current_pos)

        # 找出position所在的行
        line_idx = 0
        for i, start in enumerate(line_starts):
            if start <= position and (i == len(line_starts) - 1 or line_starts[i + 1] > position):
                line_idx = i
                break

        # 计算在行内的位置
        line_pos = position - line_starts[line_idx]

        # 获取当前行的缩进
        current_line = lines[line_idx]
        indent = ''
        for char in current_line:
            if char in [' ', '\t']:
                indent += char
            else:
                break

        # 获取转换函数并执行
        transformation_func = self.code_transformations[transform_idx]

        # 所有转换函数都需要indent和code两个参数
        insert_content = transformation_func(indent, code)

        # 处理插入内容的缩进
        content_lines = insert_content.split('\n')
        if len(content_lines) > 1:
            # 第一行与插入点同级
            indented_content = content_lines[0]
            # 后续行需要添加与当前行相同的缩进
            for i in range(1, len(content_lines)):
                indented_content += '\n' + indent + content_lines[i]
        else:
            indented_content = insert_content

        # 在代码中插入转换内容
        before = current_line[:line_pos]
        after = current_line[line_pos:]
        lines[line_idx] = before + indented_content + after

        # 重新组合代码
        return '\n'.join(lines)

    def MAB_attack(self, source_code, label, return_position_mabs=False):
        """
        Using Multi-Armed Bandit strategy for code attack with proper PositionMAB utilization.
        Args:
            source_code: Source code string
            label: Target label
            return_position_mabs: 是否返回 position_mabs 以便全量 memory 存储
        Returns:
            Attacked code string 或 (Attacked code string, position_mabs)
        """
        # Get original prediction
        pred_orig = self._get_prediction(source_code, label)
        if pred_orig != label.item():
            logger.info("Original prediction is already incorrect. Skipping attack.")
            if return_position_mabs:
                return None, None
            else:
                return None

        # Extract original code features
        orig_features = self._extract_code_features(source_code)

        # Split code into lines
        code_lines = source_code.split('\n')

        # Find important positions in the code
        insertion_points = self._find_important_positions(source_code, self.top_positions)

        if len(insertion_points) == 0:
            logger.warning("No usable insertion points found, unable to perform attack")
            if return_position_mabs:
                return source_code, {}
            else:
                return source_code

        logger.info(f"Selected {len(insertion_points)} insertion points: {insertion_points}")

        # Number of transformation methods
        num_transforms = len(self.code_transformations)

        # Initialize PositionMAB for each insertion point
        position_mabs = {pos: PositionMAB(pos, num_transforms) for pos in insertion_points}

        # Track best results
        best_code = source_code
        best_reward = -float('inf')
        attack_success = False

        # Total iterations
        total_iterations = self.max_iterations
        # Random exploration phase iterations
        exploration_ratio = self.exi
        exploration_iterations = round(total_iterations * exploration_ratio)

        logger.info(f"Starting random exploration phase ({exploration_iterations} iterations)...")

        # PHASE 1: Random exploration
        for iteration in range(exploration_iterations):
            # Randomly select transformations
            selected_transforms = {}
            for pos in insertion_points:
                transform_idx = random.randint(0, num_transforms - 1)
                selected_transforms[pos] = transform_idx

            # Apply transformations
            modified_lines = code_lines.copy()

            # Apply transformations in reverse order to avoid line number changes
            for pos in sorted(selected_transforms.keys(), reverse=True):
                transform_idx = selected_transforms[pos]
                try:
                    # Handle new position format (line, char_pos) or old format (line)
                    if isinstance(pos, tuple):
                        line_idx, char_pos = pos
                        # For single-line code, insert at character position
                        if len(modified_lines) == 1:
                            current_line = modified_lines[0]
                            # Insert transformation code at character position
                            transform_code = self.code_transformations[transform_idx]
                            if isinstance(transform_code, str):
                                insert_code = transform_code
                            else:
                                insert_code = transform_code("", "")  # No indentation for inline
                            
                            # Insert at character position
                            new_line = current_line[:char_pos] + " " + insert_code + " " + current_line[char_pos:]
                            modified_lines[0] = new_line
                        else:
                            # Multi-line code, insert at line position
                            current_line = modified_lines[line_idx]
                            indent = len(current_line) - len(current_line.lstrip())
                            indent_str = ' ' * indent

                            transform_code = self.code_transformations[transform_idx]
                            if isinstance(transform_code, str):
                                transform_code = indent_str + transform_code
                            else:
                                transform_code = transform_code(indent_str, "")

                            modified_lines.insert(line_idx + 1, transform_code)
                    else:
                        # Old format - line number only
                        current_line = modified_lines[pos]
                        indent = len(current_line) - len(current_line.lstrip())
                        indent_str = ' ' * indent

                        transform_code = self.code_transformations[transform_idx]
                        if isinstance(transform_code, str):
                            transform_code = indent_str + transform_code
                        else:
                            transform_code = transform_code(indent_str, "")

                        modified_lines.insert(pos + 1, transform_code)
                except Exception as e:
                    logger.error(f"Error applying transformation {transform_idx} at position {pos}: {str(e)}")
                    continue

            # Recombine code
            transformed_code = '\n'.join(modified_lines)

            # Evaluate transformation effect
            transformed_features = self._extract_code_features(transformed_code)
            feature_distance = self._calculate_loss(orig_features, transformed_features)
            pred_new = self._get_prediction(transformed_code, label)

            # Calculate reward
            current_success = (pred_new != label.item())
            reward = feature_distance + (1.0 if current_success else 0.0)

            # Update PositionMAB instances with rewards
            for pos, transform_idx in selected_transforms.items():
                position_mabs[pos].update(transform_idx, reward)

            # Update best result
            if reward > best_reward:
                best_reward = reward
                best_code = transformed_code
                attack_success = current_success

            # If attack succeeded, finish early
            if current_success:
                logger.info(f"Attack succeeded during random exploration! Iteration {iteration + 1}")
                if return_position_mabs:
                    return transformed_code, position_mabs
                else:
                    return transformed_code

        # PHASE 2: MAB-guided search
        logger.info("Starting MAB-guided search phase...")

        for iteration in range(exploration_iterations, total_iterations):
            # Select transformations using PositionMAB strategies
            selected_transforms = {}
            for pos in insertion_points:
                # Use PositionMAB to select the transformation
                transform_idx = position_mabs[pos].select_transformation()
                selected_transforms[pos] = transform_idx

            logger.info(
                f"MAB search iteration {iteration - exploration_iterations + 1}/{total_iterations - exploration_iterations}, selected transformations: {selected_transforms}")

            # Apply transformations
            modified_lines = code_lines.copy()

            # Apply transformations in reverse order
            for pos in sorted(selected_transforms.keys(), reverse=True):
                transform_idx = selected_transforms[pos]
                try:
                    # Handle new position format (line, char_pos) or old format (line)
                    if isinstance(pos, tuple):
                        line_idx, char_pos = pos
                        # For single-line code, insert at character position
                        if len(modified_lines) == 1:
                            current_line = modified_lines[0]
                            # Insert transformation code at character position
                            transform_code = self.code_transformations[transform_idx]
                            if isinstance(transform_code, str):
                                insert_code = transform_code
                            else:
                                insert_code = transform_code("", "")  # No indentation for inline
                            
                            # Insert at character position
                            new_line = current_line[:char_pos] + " " + insert_code + " " + current_line[char_pos:]
                            modified_lines[0] = new_line
                        else:
                            # Multi-line code, insert at line position
                            current_line = modified_lines[line_idx]
                            indent = len(current_line) - len(current_line.lstrip())
                            indent_str = ' ' * indent

                            transform_code = self.code_transformations[transform_idx]
                            if isinstance(transform_code, str):
                                transform_code = indent_str + transform_code
                            else:
                                transform_code = transform_code(indent_str, "")

                            modified_lines.insert(line_idx + 1, transform_code)
                    else:
                        # Old format - line number only
                        current_line = modified_lines[pos]
                        indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                        indent_str = ' ' * indent

                        transform_code = self.code_transformations[transform_idx]
                        if isinstance(transform_code, str):
                            transform_code = indent_str + transform_code
                        else:
                            transform_code = transform_code(indent_str, "")

                        modified_lines.insert(pos + 1, transform_code)
                except Exception as e:
                    logger.error(f"Error applying transformation {transform_idx} at position {pos}: {str(e)}")
                    continue

            # Recombine code
            transformed_code = '\n'.join(modified_lines)

            # Evaluate transformation effect
            transformed_features = self._extract_code_features(transformed_code)
            feature_distance = self._calculate_loss(orig_features, transformed_features)
            pred_new = self._get_prediction(transformed_code, label)

            # Calculate reward
            current_success = (pred_new != label.item())
            reward = feature_distance + (1.0 if current_success else 0.0)

            # Update PositionMAB instances with rewards
            for pos, transform_idx in selected_transforms.items():
                position_mabs[pos].update(transform_idx, reward)

            # Update best result
            if reward > best_reward:
                best_reward = reward
                best_code = transformed_code
                attack_success = current_success

            # If attack succeeded, finish early
            if current_success:
                logger.info(
                    f"Attack succeeded during MAB-guided search! Iteration {iteration - exploration_iterations + 1}")
                if return_position_mabs:
                    return transformed_code, position_mabs
                else:
                    return transformed_code

        # If all iterations failed, return best result
        logger.info(
            f"Reached maximum iterations. Attack {'succeeded' if attack_success else 'failed'}. Returning best code.")
        if return_position_mabs:
            return best_code, position_mabs
        else:
            return best_code
    def compute_loss(self, embedding_a, embedding_b):
        '''
        Compute the squared distance between embedding_a and embedding_b
        '''
        return self.loss_fn(embedding_a.to(self.args.device), embedding_b.to(self.args.device)).item()

    def blankattack(self, code):
        # Add spaces to disrupt code structure
        code = re.sub(r'(\b\w+)(?=\()', r'\1   ', code)
        code = re.sub(r'(\b\w+)(?=\[)', r'\1  ', code)
        code = re.sub(r'(\b\w+)(?=\:)', r'\1  ', code)
        code = re.sub(r'(\b\w+)(?=\.)', r'\1  ', code)

        code = re.sub(r'(?<!")\==', '=   =', code)
        code = re.sub(r'(?<!")\+=', '+   =', code)
        code = re.sub(r'(?<!")\-=', '-   =', code)
        code = re.sub(r'(?<!")\!=', '!   =', code)

        return code

    def deleteblankattack(self, code):
        """
        Removes unnecessary spaces around specific operators in the code.

        Args:
            code (str): The original source code as a string.

        Returns:
            str: The modified code with spaces around operators removed.
        """
        # Split the code into lines for individual processing
        lines = code.split('\n')
        new_lines = []

        for line in lines:
            # Remove spaces around '='
            stripped_line = re.sub(r'\s*=\s*', '=', line)
            # Remove spaces around '=='
            stripped_line = re.sub(r'\s*==\s*', '==', stripped_line)
            # Remove spaces around '-='
            stripped_line = re.sub(r'\s*-=\s*', '-=', stripped_line)
            # Remove spaces around '!='
            stripped_line = re.sub(r'\s*!=\s*', '!=', stripped_line)

            new_lines.append(stripped_line)

        # Reassemble the processed lines into a single string
        new_code = '\n'.join(new_lines)

        return new_code


    def imitateattack(self, code):
        '''
        Use the most common lowercase and uppercase letters for mapping attack, only changing letters in variable names, ignoring keywords.

        Args:
            code (str): Input code snippet.

        Returns:
            str: Attacked code snippet.
        '''

        # Extract variable names (words made of letters) and filter out Python keywords
        variable_names = re.findall(r'\b[a-zA-Z]+\b', code)
        python_keywords = set(keyword.kwlist)  # Get all Python reserved keywords

        # Count lowercase and uppercase letter occurrences
        lowercase_letter_count = {}
        uppercase_letter_count = {}
        import builtins

        # Get all builtin function names
        builtin_functions = dir(builtins)
        for var in variable_names:
            # If variable is neither a keyword nor a builtin function name
            if var not in python_keywords and var not in builtin_functions:

                for char in var:
                    if char.islower():
                        if char in lowercase_letter_count:
                            lowercase_letter_count[char] += 1
                        else:
                            lowercase_letter_count[char] = 1
                    elif char.isupper():
                        if char in uppercase_letter_count:
                            uppercase_letter_count[char] += 1
                        else:
                            uppercase_letter_count[char] = 1

        # Find most common lowercase and uppercase letters
        most_common_lowercase = max(lowercase_letter_count, key=lowercase_letter_count.get, default=None)
        most_common_uppercase = max(uppercase_letter_count, key=uppercase_letter_count.get, default=None)

        print(
            f"Most common lowercase letter: {most_common_lowercase} with count: {lowercase_letter_count.get(most_common_lowercase, 0)}")
        print(
            f"Most common uppercase letter: {most_common_uppercase} with count: {uppercase_letter_count.get(most_common_uppercase, 0)}")

        # Only replace letters in variable names
        mapped_code = code
        for var in variable_names:
            if var not in python_keywords:  # If variable is not a keyword
                if most_common_lowercase and most_common_lowercase in var:  # Only change variables containing most common lowercase letter
                    mapped_var = ''.join(
                        [self.char_small_farest.get(char.lower(),
                                                    char) if char.lower() == most_common_lowercase else char for char in
                         var])
                    mapped_code = mapped_code.replace(var, mapped_var)

                if most_common_uppercase and most_common_uppercase in var:  # Only change variables containing most common uppercase letter
                    mapped_var = ''.join(
                        [self.char_big_farest.get(char.lower(), char) if char.lower() == most_common_uppercase else char
                         for char in var])
                    mapped_code = mapped_code.replace(var, mapped_var)

        return mapped_code

    def initialize_position_mabs_from_preferences(self, positions, preferences):
        """
        根据存储的偏好初始化 PositionMAB 实例

        Args:
            positions: 位置 ID 列表
            preferences: 加载的偏好字典

        Returns:
            dict: 初始化的 PositionMAB 实例字典
        """
        position_mabs = {}

        for pos in positions:
            # 创建新的 PositionMAB
            mab = PositionMAB(pos, len(self.code_transformations))

            # 如果这个位置有存储的偏好，使用它们
            if str(pos) in preferences:
                pos_prefs = preferences[str(pos)]

                # 设置最佳转换
                mab.best_transformation = pos_prefs['best_transformation']

                # 设置 UCB1MAB 的值
                for i, value in enumerate(pos_prefs['values']):
                    # 使用较高的 counts 值确保偏好被重视
                    mab.mab.values[i] = value
                    mab.mab.counts[i] = 10  # 给予偏好一定的初始权重
                    mab.mab.total_pulls += 10

            position_mabs[pos] = mab

        return position_mabs

    def MAB_attack_with_memory(self, source_code, label, memory_manager=None, example_index=None):
        """
        使用多臂老虎机策略进行代码攻击，并将 MAB 偏好存储在内存管理器中
        修复版本：处理NumPy类型

        Args:
            source_code: 源代码字符串
            label: 目标标签
            memory_manager: MABMemoryManager 实例
            example_index: 示例索引用于存储偏好

        Returns:
            攻击后的代码字符串
        """
        # 调用原始的 MAB_attack 方法获取对抗样本
        adv_code = self.MAB_attack(source_code, label)
        if adv_code is None:
            return None

        # 如果攻击成功且提供了内存管理器，存储 MAB 偏好
        if memory_manager is not None:
            # 检查攻击是否成功
            pred_orig = self._get_prediction(source_code, label)
            pred_adv = self._get_prediction(adv_code, label)

            attack_success = (pred_orig == label.item() and pred_adv != label.item())

            if attack_success:
                # 重新运行 MAB_attack 的步骤，但只为了获取 position_mabs
                # 提取原始代码特征
                orig_features = self._extract_code_features(source_code)

                # 找到重要位置
                insertion_points = self._find_important_positions(source_code, self.top_positions)

                if len(insertion_points) > 0:
                    # 为每个插入点初始化 PositionMAB
                    num_transforms = len(self.code_transformations)
                    position_mabs = {pos: PositionMAB(pos, num_transforms) for pos in insertion_points}

                    # 进行简单训练以获取偏好
                    # 这里只进行简单的训练，因为我们主要是为了记录偏好
                    code_lines = source_code.split('\n')

                    for iteration in range(3):  # 只做少量迭代
                        for pos in insertion_points:
                            for transform_idx in range(num_transforms):
                                try:
                                    # 应用转换
                                    modified_lines = code_lines.copy()

                                    # 获取当前行缩进
                                    if pos < len(modified_lines):
                                        current_line = modified_lines[pos]
                                        indent = len(current_line) - len(
                                            current_line.lstrip()) if current_line.strip() else 0
                                        indent_str = ' ' * indent
                                    else:
                                        indent_str = '    '  # 默认缩进

                                    # 获取转换代码
                                    transform_code = self.code_transformations[transform_idx]
                                    if isinstance(transform_code, str):
                                        transform_code = indent_str + transform_code
                                    else:
                                        transform_code = transform_code(indent_str, "")

                                    # 插入转换
                                    if pos < len(modified_lines):
                                        modified_lines.insert(pos + 1, transform_code)
                                    else:
                                        modified_lines.append(transform_code)

                                    # 评估转换
                                    transformed_code = '\n'.join(modified_lines)
                                    transformed_features = self._extract_code_features(transformed_code)
                                    feature_distance = self._calculate_loss(orig_features, transformed_features)
                                    pred_new = self._get_prediction(transformed_code, label)

                                    # 计算奖励
                                    current_success = (pred_new != label.item())
                                    reward = feature_distance + (1.0 if current_success else 0.0)

                                    # 更新 MAB
                                    position_mabs[pos].update(transform_idx, reward)
                                except Exception as e:
                                    continue

                    # 存储 MAB 偏好 - 处理NumPy类型
                    import numpy as np

                    for i, pos in enumerate(insertion_points):
                        position_number = str(i + 1)  # 位置从1开始编号
                        mab = position_mabs[pos]

                        # 转换NumPy类型
                        best_transform = int(mab.best_transformation) if mab.best_transformation is not None else 0

                        # 转换值数组
                        values_list = []
                        for val in mab.mab.values:
                            values_list.append(float(val) if isinstance(val, np.number) else val)

                        # 存储转换后的值
                        memory_manager.store_preference(
                            source_code,
                            pos,  # 原始位置ID
                            best_transform,
                            values_list,
                            example_index  # 传递示例索引
                        )

                    # 保存所有偏好
                    memory_manager.save_preferences()

        # 返回对抗样本
        return adv_code

    def transfer_based_attack(self, source_code, label, memory_manager, example_index=None):
        """
        使用基于迁移的 MAB 攻击，利用之前学到的偏好 (修复NumPy类型)

        Args:
            source_code: 源代码字符串
            label: 目标标签
            memory_manager: MABMemoryManager 实例
            example_index: 示例索引，用于存储偏好

        Returns:
            攻击后的代码字符串
        """
        # 获取原始预测
        pred_orig = self._get_prediction(source_code, label)
        if pred_orig != label.item():
            logger.info("原始预测已经不正确。跳过攻击。")
            return source_code

        # 查找偏好 - 直接使用example_index
        key = str(example_index) if example_index is not None else memory_manager.compute_code_hash(source_code)
        preferences = memory_manager.preferences.get(key)

        # 打印调试信息
        print(f"查找偏好: key={key}")
        if preferences:
            print(f"找到偏好: {list(preferences.keys())}")
        else:
            print(
                f"未找到偏好。可用键: {list(memory_manager.preferences.keys())[:5] if memory_manager.preferences else '无'}")

        # 如果未找到存储的偏好，回退到标准 MAB 攻击
        if preferences is None:
            logger.info("未找到存储的 MAB 偏好。使用标准 MAB 攻击。")
            return self._fresh_mab_attack(source_code, label)

        logger.info(f"找到存储的 MAB 偏好。应用基于迁移的攻击。")

        # 提取原始代码特征
        orig_features = self._extract_code_features(source_code)

        # 找到代码中的重要位置
        insertion_points = self._find_important_positions(source_code, self.top_positions)

        if len(insertion_points) == 0:
            logger.warning("未找到可用的插入点，无法执行攻击")
            return source_code

        # 使用存储的偏好初始化 PositionMAB
        position_mabs = {}
        for i, pos in enumerate(insertion_points):
            position_number = str(i + 1)  # 从1开始编号
            mab = PositionMAB(pos, len(self.code_transformations))

            # 如果这个位置有存储的偏好，使用它们
            if position_number in preferences:
                pos_prefs = preferences[position_number]
                print(f"应用位置{pos}的偏好: {pos_prefs}")

                # 设置最佳转换
                if 'best_transformation' in pos_prefs:
                    mab.best_transformation = pos_prefs['best_transformation']

                # 设置 UCB1MAB 的值
                if 'values' in pos_prefs and isinstance(pos_prefs['values'], list):
                    values = pos_prefs['values']
                    for j, value in enumerate(values):
                        if j < len(mab.mab.values):
                            mab.mab.values[j] = value
                            mab.mab.counts[j] = 10  # 给予偏好一定权重
                            mab.mab.total_pulls += 10

            position_mabs[pos] = mab

        # 将代码分割成行
        code_lines = source_code.split('\n')

        # 跟踪最佳结果
        best_code = source_code
        best_reward = -float('inf')
        attack_success = False

        total_iterations = self.max_iterations

        logger.info(f"开始基于迁移的 MAB 攻击 ({total_iterations} 次迭代)...")

        # 迭代攻击
        for iteration in range(total_iterations):
            # 使用存储的偏好选择转换
            selected_transforms = {}
            for pos in insertion_points:
                # 使用 PositionMAB 选择转换
                transform_idx = position_mabs[pos].select_transformation()
                selected_transforms[pos] = transform_idx

            # 应用转换
            modified_lines = code_lines.copy()

            # 按逆序应用转换
            for pos in sorted(selected_transforms.keys(), reverse=True):
                transform_idx = selected_transforms[pos]
                try:
                    # 安全地获取当前行及其缩进
                    if pos < len(modified_lines):
                        current_line = modified_lines[pos]
                        indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                    else:
                        # 超出范围时使用默认缩进
                        current_line = ""
                        indent = 4

                    indent_str = ' ' * indent

                    # 获取转换代码
                    transform_code = self.code_transformations[transform_idx]
                    # 确保正确缩进
                    if isinstance(transform_code, str):
                        transform_code = indent_str + transform_code
                    else:
                        transform_code = transform_code(indent_str, "")

                    # 插入转换
                    if pos < len(modified_lines):
                        modified_lines.insert(pos + 1, transform_code)
                    else:
                        # 超出范围时追加到末尾
                        modified_lines.append(transform_code)
                except Exception as e:
                    logger.error(f"在行 {pos} 应用转换 {transform_idx} 时出错: {str(e)}")
                    continue

            # 重新组合代码
            transformed_code = '\n'.join(modified_lines)

            # 评估转换效果
            transformed_features = self._extract_code_features(transformed_code)
            feature_distance = self._calculate_loss(orig_features, transformed_features)
            pred_new = self._get_prediction(transformed_code, label)

            # 计算奖励
            current_success = (pred_new != label.item())
            reward = feature_distance + (1.0 if current_success else 0.0)

            # 用奖励更新 PositionMAB 实例
            for pos, transform_idx in selected_transforms.items():
                position_mabs[pos].update(transform_idx, reward)

            # 更新最佳结果
            if reward > best_reward:
                best_reward = reward
                best_code = transformed_code
                attack_success = current_success

            # 如果攻击成功，提前结束
            if current_success:
                logger.info(f"在迁移 MAB 搜索期间攻击成功！迭代 {iteration + 1}")

                # # 更新当前代码的偏好 - 将NumPy值转换为Python标准类型
                # for i, pos in enumerate(insertion_points):
                #     position_number = str(i + 1)  # 位置序号从1开始
                #     mab = position_mabs[pos]
                #     best_transform = int(mab.best_transformation) if mab.best_transformation is not None else 0
                #
                #     # 将NumPy数组转换为纯Python列表
                #     import numpy as np
                #     values_list = []
                #     for val in mab.mab.values:
                #         values_list.append(float(val) if isinstance(val, np.number) else val)
                    #
                    # # 存储转换后的值
                    # memory_manager.store_preference(
                    #     source_code,
                    #     pos,  # 原始位置ID
                    #     best_transform,
                    #     values_list,
                    #     example_index  # 传递示例索引
                    # )


                return transformed_code

        # 如果迁移攻击失败，回退到标准 MAB 攻击
        if not attack_success:
            logger.info("基于迁移的攻击失败。回退到标准 MAB 攻击。")
            return self._fresh_mab_attack(source_code, label)
        # 返回最佳代码
        return best_code

    def _fresh_mab_attack(self, source_code, label):
        """
        封装标准MAB攻击，确保状态完全干净
        """
        # 创建一个新的CrossDomainAttacker实例，确保状态完全干净
        fresh_attacker = CrossDomainAttacker(
            args=self.args,
            model_sub=self.model_sub,
            model_tgt=self.model_tgt,
            tokenizer=self.tokenizer,
            model_mlm=self.model_mlm,
            tokenizer_mlm=self.tokenizer_mlm,
            use_bpe=self.use_bpe,
            threshold_pred_score=self.threshold_pred_score,
            targeted=self.targeted
        )

        # 复制其他可能需要的参数
        fresh_attacker.top_positions = self.top_positions
        fresh_attacker.max_iterations = self.max_iterations
        fresh_attacker.code_transformations = self.code_transformations.copy()

        # 使用全新的实例执行标准MAB攻击
        logger.info("使用全新状态执行标准MAB攻击")
        return fresh_attacker.MAB_attack(source_code, label)

    def adapt_transfer_attack(self, source_code, label, memory_manager, example_index=None):
        """
        改进的迁移攻击方法:
        - 如果检测到偏好，先进行随机迭代初始化，再执行MAB攻击
        - 如果没有偏好，直接使用标准MAB攻击

        Args:
            source_code: 源代码字符串
            label: 目标标签
            memory_manager: MABMemoryManager实例
            example_index: 示例索引，用于存储偏好

        Returns:
            攻击后的代码字符串
        """
        # 获取原始预测
        pred_orig = self._get_prediction(source_code, label)
        if pred_orig != label.item():
            logger.info("原始预测已经不正确。跳过攻击。")
            return source_code

        # 查找偏好 - 使用example_index或代码哈希
        key = str(example_index) if example_index is not None else memory_manager.compute_code_hash(source_code)
        preferences = memory_manager.preferences.get(key)

        # 打印调试信息
        logger.info(f"查找偏好: key={key}")
        if preferences:
            logger.info(f"找到偏好: {list(preferences.keys())}")
        else:
            logger.info(
                f"未找到偏好。可用键: {list(memory_manager.preferences.keys())[:5] if memory_manager.preferences else '无'}")

        # 如果未找到偏好，直接使用标准MAB攻击
        if preferences is None:
            logger.info("未找到偏好。使用标准MAB攻击。")
            return self.MAB_attack(source_code, label)

        logger.info("找到偏好。使用改进的迁移攻击流程。")

        # 提取原始代码特征
        orig_features = self._extract_code_features(source_code)

        # 找到重要位置
        insertion_points = self._find_important_positions(source_code, self.top_positions)
        if len(insertion_points) == 0:
            logger.warning("未找到可用的插入点，无法执行攻击")
            return source_code

        # 转换数量
        num_transforms = len(self.code_transformations)

        # 创建新的PositionMAB实例
        position_mabs = {pos: PositionMAB(pos, num_transforms) for pos in insertion_points}

        # 将代码分割成行
        code_lines = source_code.split('\n')

        # 跟踪最佳结果
        best_code = source_code
        best_reward = -float('inf')
        attack_success = False

        # 设置迭代次数
        total_iterations = self.max_iterations

        # ======== 关键改动：从随机迭代开始 ========
        # 随机迭代阶段 - 根据偏好引导

        exploration_ratio = self.exi
        random_iterations = round(total_iterations * exploration_ratio)  # 使用1/3的迭代用于随机探索
        logger.info(f"开始随机探索阶段 ({random_iterations} 次迭代)...")

        for iteration in range(random_iterations):
            # 随机选择转换，但权重受到偏好的影响
            selected_transforms = {}

            for i, pos in enumerate(insertion_points):
                position_number = str(i + 1)  # 位置编号从1开始

                # 如果这个位置有偏好数据，以一定概率使用最佳转换
                if position_number in preferences and random.random() < 0.7:  # 70%概率使用最佳转换
                    best_transform = preferences[position_number].get('best_transformation', 0)
                    transform_idx = best_transform
                else:
                    # 否则随机选择
                    transform_idx = random.randint(0, num_transforms - 1)

                selected_transforms[pos] = transform_idx

            # 应用转换
            modified_lines = code_lines.copy()

            # 按逆序应用转换
            for pos in sorted(selected_transforms.keys(), reverse=True):
                transform_idx = selected_transforms[pos]
                actual_pos = pos[0] if isinstance(pos, tuple) else pos
                try:
                    if actual_pos < len(modified_lines):
                        current_line = modified_lines[actual_pos]
                        indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                    else:
                        current_line = ""
                        indent = 4
                    indent_str = ' ' * indent
                    transform_code = self.code_transformations[transform_idx]
                    if isinstance(transform_code, str):
                        transform_code = indent_str + transform_code
                    else:
                        transform_code = transform_code(indent_str, "")
                    if actual_pos < len(modified_lines):
                        modified_lines.insert(actual_pos + 1, transform_code)
                    else:
                        modified_lines.append(transform_code)
                except Exception as e:
                    logger.error(f"Error applying transformation {transform_idx} at line {actual_pos}: {str(e)}")
                    continue

            # 重新组合代码
            transformed_code = '\n'.join(modified_lines)

            # 评估转换效果
            transformed_features = self._extract_code_features(transformed_code)
            feature_distance = self._calculate_loss(orig_features, transformed_features)
            pred_new = self._get_prediction(transformed_code, label)

            # 计算奖励
            current_success = (pred_new != label.item())
            reward = feature_distance + (1.0 if current_success else 0.0)

            # 更新PositionMAB
            for pos, transform_idx in selected_transforms.items():
                position_key = pos[0] if isinstance(pos, tuple) else pos
                if position_key in position_mabs:
                    position_mabs[position_key].update(transform_idx, reward)

            # 更新最佳结果
            if reward > best_reward:
                best_reward = reward
                best_code = transformed_code
                attack_success = current_success

            # 如果攻击成功，提前结束
            if current_success:
                logger.info(f"在随机探索阶段攻击成功！迭代 {iteration + 1}")
                return transformed_code

        # ======== MAB引导搜索阶段 ========
        logger.info(f"开始MAB引导搜索阶段 ({total_iterations - random_iterations} 次迭代)...")

        for iteration in range(random_iterations, total_iterations):
            # 使用MAB策略选择转换
            selected_transforms = {}
            for pos in insertion_points:
                # 直接用pos作为键查找position_mabs，保持类型一致
                transform_idx = position_mabs[pos].select_transformation()
                selected_transforms[pos] = transform_idx

            # 应用转换
            modified_lines = code_lines.copy()

            # 按逆序应用转换
            for pos in sorted(selected_transforms.keys(), reverse=True):
                transform_idx = selected_transforms[pos]
                actual_pos = pos[0] if isinstance(pos, tuple) else pos
                try:
                    if actual_pos < len(modified_lines):
                        current_line = modified_lines[actual_pos]
                        indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                    else:
                        current_line = ""
                        indent = 4
                    indent_str = ' ' * indent
                    transform_code = self.code_transformations[transform_idx]
                    if isinstance(transform_code, str):
                        transform_code = indent_str + transform_code
                    else:
                        transform_code = transform_code(indent_str, "")
                    if actual_pos < len(modified_lines):
                        modified_lines.insert(actual_pos + 1, transform_code)
                    else:
                        modified_lines.append(transform_code)
                except Exception as e:
                    logger.error(f"Error applying transformation {transform_idx} at line {actual_pos}: {str(e)}")
                    continue

            # 重新组合代码
            transformed_code = '\n'.join(modified_lines)

            # 评估转换效果
            transformed_features = self._extract_code_features(transformed_code)
            feature_distance = self._calculate_loss(orig_features, transformed_features)
            pred_new = self._get_prediction(transformed_code, label)

            # 计算奖励
            current_success = (pred_new != label.item())
            reward = feature_distance + (1.0 if current_success else 0.0)

            # 更新PositionMAB
            for pos, transform_idx in selected_transforms.items():
                # 直接用pos作为键，保持类型一致
                if pos in position_mabs:
                    position_mabs[pos].update(transform_idx, reward)

            # 更新最佳结果
            if reward > best_reward:
                best_reward = reward
                best_code = transformed_code
                attack_success = current_success

            # 如果攻击成功，提前结束
            if current_success:
                logger.info(f"在MAB引导搜索阶段攻击成功！迭代 {iteration - random_iterations + 1}")
                return transformed_code

        # 返回最佳结果
        logger.info(f"达到最大迭代次数。攻击{'成功' if attack_success else '失败'}。")
        return best_code

    def analyze_position_selection(self, source_code, label, position_counts=[1, 2, 3, 5, 8, 10, 15], iterations=5):
        """
        对单个样本分析不同重要性位置数量的影响，返回嵌入表示差异

        Args:
            source_code: 源代码字符串
            label: 目标标签
            position_counts: 要测试的不同重要性位置数量列表
            iterations: 每次攻击的迭代次数

        Returns:
            结果字典，包含每个位置数量的特征距离和代码修改信息
        """
        # 保存原始配置
        original_top_positions = self.top_positions
        original_max_iterations = self.max_iterations

        # 设置迭代次数
        self.max_iterations = iterations

        # 获取原始特征
        orig_features = self._extract_code_features(source_code)

        # 保存原始预测
        pred_orig = self._get_prediction(source_code, label)

        # 存储结果
        results = {}

        # 检查原始预测是否正确
        if pred_orig != label.item():
            logger.info("原始预测已经不正确，跳过分析")
            # 恢复原始设置
            self.top_positions = original_top_positions
            self.max_iterations = original_max_iterations
            return None

        # 分析不同位置数量
        for pos_count in position_counts:
            logger.info(f"分析重要性位置数量: {pos_count}")

            # 设置位置数量
            self.top_positions = pos_count

            # 执行攻击
            adv_code = self.MAB_attack(source_code, label)

            # 计算特征距离
            adv_features = self._extract_code_features(adv_code)
            feature_distance = self._calculate_loss(orig_features, adv_features)

            # 计算代码修改比例
            orig_lines = source_code.split('\n')
            adv_lines = adv_code.split('\n')
            modification_ratio = (len(adv_lines) - len(orig_lines)) / len(orig_lines)

            # 检查攻击是否成功
            pred_adv = self._get_prediction(adv_code, label)
            is_success = (pred_adv != label.item())

            # 记录结果
            results[pos_count] = {
                'feature_distance': feature_distance,
                'code_modification': modification_ratio,
                'is_success': is_success,
                'adv_code': adv_code
            }

            logger.info(
                f"位置数量: {pos_count}, 特征距离: {feature_distance:.4f}, 代码修改比例: {modification_ratio:.4f}, 攻击成功: {is_success}")

        # 恢复原始设置
        self.top_positions = original_top_positions
        self.max_iterations = original_max_iterations

        return results


    def stacked_transfer_attack(self, source_code, label, memory_manager1, memory_manager2, example_index=None,
                                strategy="ensemble"):
        """
        叠加攻击方法：利用两个模型的记忆进行攻击

        参数:
            source_code: 源代码字符串
            label: 目标标签
            memory_manager1: 第一个记忆管理器
            memory_manager2: 第二个记忆管理器
            example_index: 样本索引，用于存储偏好
            strategy: 攻击策略 ("sequential"顺序, "ensemble"集成, "competitive"竞争, "adaptive"自适应)

        返回:
            攻击后的代码
        """
        # 检查原始预测
        pred_orig = self._get_prediction(source_code, label)
        if pred_orig != label.item():
            logger.info("原始预测已经不正确，跳过攻击")
            return source_code

        # 获取两个模型的偏好
        key1 = str(example_index) if example_index is not None else memory_manager1.compute_code_hash(source_code)
        key2 = str(example_index) if example_index is not None else memory_manager2.compute_code_hash(source_code)

        preferences1 = memory_manager1.preferences.get(key1)
        preferences2 = memory_manager2.preferences.get(key2)

        logger.info(f"查找偏好: key1={key1}, key2={key2}")
        logger.info(f"模型1偏好: {list(preferences1.keys()) if preferences1 else '未找到'}")
        logger.info(f"模型2偏好: {list(preferences2.keys()) if preferences2 else '未找到'}")

        # 如果两个模型都没有偏好，使用标准MAB攻击
        if preferences1 is None and preferences2 is None:
            logger.info("未找到任何偏好，使用标准MAB攻击")
            return self.MAB_attack(source_code, label)

        # 根据选择的策略执行攻击
        if strategy == "sequential":
            return self._sequential_stacked_attack(source_code, label, preferences1, preferences2)
        elif strategy == "ensemble":
            return self._ensemble_stacked_attack(source_code, label, preferences1, preferences2)
        elif strategy == "competitive":
            return self._competitive_stacked_attack(source_code, label, preferences1, preferences2)
        elif strategy == "adaptive":
            return self._adaptive_stacked_attack(source_code, label, preferences1, preferences2)
        else:
            logger.warning(f"未知策略: {strategy}，默认使用集成策略")
            return self._ensemble_stacked_attack(source_code, label, preferences1, preferences2)

    def _sequential_stacked_attack(self, source_code, label, preferences1, preferences2):
        """
        Sequential attack strategy: tries first model's preferences, then second model's if first fails.
        Uses MAB-guided exploration rather than purely random exploration.

        Args:
            source_code: Source code string
            label: Target label
            preferences1: First model's preferences
            preferences2: Second model's preferences

        Returns:
            Attacked code string
        """
        logger.info("Executing sequential stacked attack with MAB-guided approach")

        # Try first model if preferences exist
        if preferences1:
            logger.info("Trying attack with model 1 preferences")

            # Extract original code features
            orig_features = self._extract_code_features(source_code)

            # Find important positions
            insertion_points = self._find_important_positions(source_code, self.top_positions)

            if len(insertion_points) == 0:
                logger.warning("No usable insertion points found")
                return source_code

            # Number of transformations
            num_transforms = len(self.code_transformations)

            # Initialize PositionMABs with model 1 preferences
            position_mabs = {}
            for i, pos in enumerate(insertion_points):
                position_number = str(i + 1)
                mab = PositionMAB(pos, num_transforms)

                # Apply model 1 preferences
                if position_number in preferences1:
                    pos_prefs = preferences1[position_number]
                    if 'best_transformation' in pos_prefs:
                        mab.best_transformation = pos_prefs['best_transformation']

                    if 'values' in pos_prefs and isinstance(pos_prefs['values'], list):
                        values = pos_prefs['values']
                        for j, value in enumerate(values):
                            if j < len(mab.mab.values):
                                mab.mab.values[j] = value
                                mab.mab.counts[j] = 10  # Give initial weight
                                mab.mab.total_pulls += 10

                position_mabs[pos] = mab

            # Split code into lines
            code_lines = source_code.split('\n')

            # Track best results
            best_code = source_code
            best_reward = -float('inf')
            attack_success = False

            # Set iterations
            total_iterations = self.max_iterations

            # MAB-guided exploration ratio
            exploration_ratio = self.exi
            exploration_iterations = round(total_iterations * exploration_ratio)

            logger.info(
                f"Starting MAB-guided approach with model 1: {exploration_iterations} exploration iterations, {total_iterations - exploration_iterations} exploitation iterations")

            # Phase 1: MAB-guided exploration with model 1 preferences
            for iteration in range(exploration_iterations):
                # Select transformations with MAB guidance
                selected_transforms = {}
                for pos in insertion_points:
                    # Use MAB with 70% probability, random selection with 30%
                    if random.random() < 0.7:
                        transform_idx = position_mabs[pos].select_transformation()
                    else:
                        transform_idx = random.randint(0, num_transforms - 1)
                    selected_transforms[pos] = transform_idx

                # Apply transformations
                modified_lines = code_lines.copy()

                # Apply transformations in reverse order
                for pos in sorted(selected_transforms.keys(), reverse=True):
                    transform_idx = selected_transforms[pos]
                    try:
                        # Get current line indentation
                        if pos < len(modified_lines):
                            current_line = modified_lines[pos]
                            indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                        else:
                            current_line = ""
                            indent = 4

                        indent_str = ' ' * indent

                        # Get transformation code
                        transform_code = self.code_transformations[transform_idx]

                        # Ensure proper indentation
                        if isinstance(transform_code, str):
                            transform_code = indent_str + transform_code
                        else:
                            transform_code = transform_code(indent_str, "")

                        # Insert transformation
                        if pos < len(modified_lines):
                            modified_lines.insert(pos + 1, transform_code)
                        else:
                            modified_lines.append(transform_code)
                    except Exception as e:
                        logger.error(f"Error applying transformation {transform_idx} at line {pos}: {str(e)}")
                        continue

                # Recombine code
                transformed_code = '\n'.join(modified_lines)

                # Evaluate transformation effect
                transformed_features = self._extract_code_features(transformed_code)
                feature_distance = self._calculate_loss(orig_features, transformed_features)
                pred_new = self._get_prediction(transformed_code, label)

                # Calculate reward
                current_success = (pred_new != label.item())
                reward = feature_distance + (1.0 if current_success else 0.0)

                # Update MABs with rewards
                for pos, transform_idx in selected_transforms.items():
                    position_mabs[pos].update(transform_idx, reward)

                # Update best result
                if reward > best_reward:
                    best_reward = reward
                    best_code = transformed_code
                    attack_success = current_success

                # If attack succeeded, finish early
                if current_success:
                    logger.info(f"Attack succeeded during model 1 exploration! Iteration {iteration + 1}")
                    return transformed_code

            # Phase 2: MAB-guided exploitation with model 1 preferences
            for iteration in range(exploration_iterations, total_iterations):
                # Select transformations using MAB
                selected_transforms = {}
                for pos in insertion_points:
                    transform_idx = position_mabs[pos].select_transformation()
                    selected_transforms[pos] = transform_idx

                # Apply transformations
                modified_lines = code_lines.copy()

                # Apply transformations in reverse order
                for pos in sorted(selected_transforms.keys(), reverse=True):
                    transform_idx = selected_transforms[pos]
                    try:
                        # Get current line indentation
                        if pos < len(modified_lines):
                            current_line = modified_lines[pos]
                            indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                        else:
                            current_line = ""
                            indent = 4

                        indent_str = ' ' * indent

                        # Get transformation code
                        transform_code = self.code_transformations[transform_idx]

                        # Ensure proper indentation
                        if isinstance(transform_code, str):
                            transform_code = indent_str + transform_code
                        else:
                            transform_code = transform_code(indent_str, "")

                        # Insert transformation
                        if pos < len(modified_lines):
                            modified_lines.insert(pos + 1, transform_code)
                        else:
                            modified_lines.append(transform_code)
                    except Exception as e:
                        logger.error(f"Error applying transformation {transform_idx} at line {pos}: {str(e)}")
                        continue

                # Recombine code
                transformed_code = '\n'.join(modified_lines)

                # Evaluate transformation effect
                transformed_features = self._extract_code_features(transformed_code)
                feature_distance = self._calculate_loss(orig_features, transformed_features)
                pred_new = self._get_prediction(transformed_code, label)

                # Calculate reward
                current_success = (pred_new != label.item())
                reward = feature_distance + (1.0 if current_success else 0.0)

                # Update MABs with rewards
                for pos, transform_idx in selected_transforms.items():
                    position_mabs[pos].update(transform_idx, reward)

                # Update best result
                if reward > best_reward:
                    best_reward = reward
                    best_code = transformed_code
                    attack_success = current_success

                # If attack succeeded, finish early
                if current_success:
                    logger.info(
                        f"Attack succeeded during model 1 exploitation! Iteration {iteration - exploration_iterations + 1}")
                    return transformed_code

            # If model 1 failed, but we found a somewhat promising result, return it
            if best_reward > 0:
                logger.info(f"Model 1 did not achieve full success, but found code with reward {best_reward}")
                return best_code

        # If model 1 failed or had no preferences, try model 2
        if preferences2:
            logger.info("Trying attack with model 2 preferences")

            # Extract original code features
            orig_features = self._extract_code_features(source_code)

            # Find important positions
            insertion_points = self._find_important_positions(source_code, self.top_positions)

            if len(insertion_points) == 0:
                logger.warning("No usable insertion points found")
                return source_code

            # Number of transformations
            num_transforms = len(self.code_transformations)

            # Initialize PositionMABs with model 2 preferences
            position_mabs = {}
            for i, pos in enumerate(insertion_points):
                position_number = str(i + 1)
                mab = PositionMAB(pos, num_transforms)

                # Apply model 2 preferences
                if position_number in preferences2:
                    pos_prefs = preferences2[position_number]
                    if 'best_transformation' in pos_prefs:
                        mab.best_transformation = pos_prefs['best_transformation']

                    if 'values' in pos_prefs and isinstance(pos_prefs['values'], list):
                        values = pos_prefs['values']
                        for j, value in enumerate(values):
                            if j < len(mab.mab.values):
                                mab.mab.values[j] = value
                                mab.mab.counts[j] = 10  # Give initial weight
                                mab.mab.total_pulls += 10

                position_mabs[pos] = mab

            # Split code into lines
            code_lines = source_code.split('\n')

            # Track best results
            best_code = source_code
            best_reward = -float('inf')
            attack_success = False

            # Set iterations
            total_iterations = self.max_iterations

            # MAB-guided exploration ratio
            exploration_ratio = self.exi
            exploration_iterations = round(total_iterations * exploration_ratio)

            logger.info(
                f"Starting MAB-guided approach with model 2: {exploration_iterations} exploration iterations, {total_iterations - exploration_iterations} exploitation iterations")

            # Similar structure to model 1 attack - Phase 1: MAB-guided exploration with model 2 preferences
            for iteration in range(exploration_iterations):
                # Select transformations with MAB guidance
                selected_transforms = {}
                for pos in insertion_points:
                    # Use MAB with 70% probability, random selection with 30%
                    if random.random() < 0.7:
                        transform_idx = position_mabs[pos].select_transformation()
                    else:
                        transform_idx = random.randint(0, num_transforms - 1)
                    selected_transforms[pos] = transform_idx

                # Apply transformations
                modified_lines = code_lines.copy()

                # Apply transformations in reverse order
                for pos in sorted(selected_transforms.keys(), reverse=True):
                    transform_idx = selected_transforms[pos]
                    try:
                        # Get current line indentation
                        if pos < len(modified_lines):
                            current_line = modified_lines[pos]
                            indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                        else:
                            current_line = ""
                            indent = 4

                        indent_str = ' ' * indent

                        # Get transformation code
                        transform_code = self.code_transformations[transform_idx]

                        # Ensure proper indentation
                        if isinstance(transform_code, str):
                            transform_code = indent_str + transform_code
                        else:
                            transform_code = transform_code(indent_str, "")

                        # Insert transformation
                        if pos < len(modified_lines):
                            modified_lines.insert(pos + 1, transform_code)
                        else:
                            modified_lines.append(transform_code)
                    except Exception as e:
                        logger.error(f"Error applying transformation {transform_idx} at line {pos}: {str(e)}")
                        continue

                # Recombine code
                transformed_code = '\n'.join(modified_lines)

                # Evaluate transformation effect
                transformed_features = self._extract_code_features(transformed_code)
                feature_distance = self._calculate_loss(orig_features, transformed_features)
                pred_new = self._get_prediction(transformed_code, label)

                # Calculate reward
                current_success = (pred_new != label.item())
                reward = feature_distance + (1.0 if current_success else 0.0)

                # Update MABs with rewards
                for pos, transform_idx in selected_transforms.items():
                    position_mabs[pos].update(transform_idx, reward)

                # Update best result
                if reward > best_reward:
                    best_reward = reward
                    best_code = transformed_code
                    attack_success = current_success

                # If attack succeeded, finish early
                if current_success:
                    logger.info(f"Attack succeeded during model 2 exploration! Iteration {iteration + 1}")
                    return transformed_code

            # Phase 2: MAB-guided exploitation with model 2 preferences
            for iteration in range(exploration_iterations, total_iterations):
                # Select transformations using MAB
                selected_transforms = {}
                for pos in insertion_points:
                    transform_idx = position_mabs[pos].select_transformation()
                    selected_transforms[pos] = transform_idx

                # Apply transformations
                modified_lines = code_lines.copy()

                # Apply transformations in reverse order
                for pos in sorted(selected_transforms.keys(), reverse=True):
                    transform_idx = selected_transforms[pos]
                    try:
                        # Get current line indentation
                        if pos < len(modified_lines):
                            current_line = modified_lines[pos]
                            indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                        else:
                            current_line = ""
                            indent = 4

                        indent_str = ' ' * indent

                        # Get transformation code
                        transform_code = self.code_transformations[transform_idx]

                        # Ensure proper indentation
                        if isinstance(transform_code, str):
                            transform_code = indent_str + transform_code
                        else:
                            transform_code = transform_code(indent_str, "")

                        # Insert transformation
                        if pos < len(modified_lines):
                            modified_lines.insert(pos + 1, transform_code)
                        else:
                            modified_lines.append(transform_code)
                    except Exception as e:
                        logger.error(f"Error applying transformation {transform_idx} at line {pos}: {str(e)}")
                        continue

                # Recombine code
                transformed_code = '\n'.join(modified_lines)

                # Evaluate transformation effect
                transformed_features = self._extract_code_features(transformed_code)
                feature_distance = self._calculate_loss(orig_features, transformed_features)
                pred_new = self._get_prediction(transformed_code, label)

                # Calculate reward
                current_success = (pred_new != label.item())
                reward = feature_distance + (1.0 if current_success else 0.0)

                # Update MABs with rewards
                for pos, transform_idx in selected_transforms.items():
                    position_mabs[pos].update(transform_idx, reward)

                # Update best result
                if reward > best_reward:
                    best_reward = reward
                    best_code = transformed_code
                    attack_success = current_success

                # If attack succeeded, finish early
                if current_success:
                    logger.info(
                        f"Attack succeeded during model 2 exploitation! Iteration {iteration - exploration_iterations + 1}")
                    return transformed_code

        # If both models failed, use standard MAB attack
        logger.info("Both models failed or had no preferences. Using standard MAB attack.")
        return self.MAB_attack(source_code, label)
    def _ensemble_stacked_attack(self, source_code, label, preferences1, preferences2):
        """
        Ensemble attack strategy: Combines preferences from both models and uses MAB-guided approach
        instead of random exploration.

        Args:
            source_code: Source code string
            label: Target label
            preferences1: First model's preferences
            preferences2: Second model's preferences

        Returns:
            Attacked code string
        """
        logger.info("Executing ensemble stacked attack with MAB-guided approach")

        # Extract original code features
        orig_features = self._extract_code_features(source_code)

        # Find important insertion points
        insertion_points = self._find_important_positions(source_code, self.top_positions)

        if len(insertion_points) == 0:
            logger.warning("No usable insertion points found")
            return source_code

        # Number of transformations
        num_transforms = len(self.code_transformations)

        # Initialize PositionMAB for each insertion point with combined preferences
        position_mabs = {}

        # Combine preferences from both models
        for i, pos in enumerate(insertion_points):
            position_number = str(i + 1)
            mab = PositionMAB(pos, num_transforms)

            # Arrays for combining preferences
            combined_values = np.zeros(num_transforms)
            value_counts = np.zeros(num_transforms)

            # Process model 1 preferences
            if preferences1 and position_number in preferences1:
                pos_prefs1 = preferences1[position_number]
                if 'values' in pos_prefs1 and isinstance(pos_prefs1['values'], list):
                    for j, value in enumerate(pos_prefs1['values']):
                        if j < num_transforms:
                            combined_values[j] += value
                            value_counts[j] += 1

            # Process model 2 preferences
            if preferences2 and position_number in preferences2:
                pos_prefs2 = preferences2[position_number]
                if 'values' in pos_prefs2 and isinstance(pos_prefs2['values'], list):
                    for j, value in enumerate(pos_prefs2['values']):
                        if j < num_transforms:
                            combined_values[j] += value
                            value_counts[j] += 1

            # Calculate average values (avoid division by zero)
            for j in range(num_transforms):
                if value_counts[j] > 0:
                    mab.mab.values[j] = combined_values[j] / value_counts[j]
                    mab.mab.counts[j] = 10  # Give initial weight
                    mab.mab.total_pulls += 10

            # Determine best transformation
            best_transform_idx = np.argmax(mab.mab.values)
            mab.best_transformation = best_transform_idx

            position_mabs[pos] = mab

        # Split code into lines
        code_lines = source_code.split('\n')

        # Track best results
        best_code = source_code
        best_reward = -float('inf')
        attack_success = False

        # Set iterations
        total_iterations = self.max_iterations

        # MAB-guided exploration ratio
        exploration_ratio = self.exi
        exploration_iterations = round(total_iterations * exploration_ratio)

        logger.info(
            f"Starting ensemble MAB-guided attack: {exploration_iterations} MAB exploration iterations, {total_iterations - exploration_iterations} MAB exploitation iterations")

        # Phase 1: MAB-guided exploration - Using MAB with some randomness
        for iteration in range(exploration_iterations):
            # Select transformations guided by MAB with some randomness
            selected_transforms = {}

            for pos in insertion_points:
                # Use MAB values to create weighted probabilities with 70% probability
                # Pure random selection with 30% probability for exploration
                if random.random() < 0.7:
                    # Use MAB values to create weighted probability distribution
                    values = position_mabs[pos].mab.values
                    min_val = min(values)
                    if min_val < 0:
                        # Shift values to ensure positive weights
                        shifted_values = [v - min_val + 0.1 for v in values]
                    else:
                        shifted_values = [v + 0.1 for v in values]  # Add small constant to avoid zeros

                    # Create probability distribution
                    total = sum(shifted_values)
                    probs = [v / total for v in shifted_values]

                    # Select based on probabilities
                    transform_idx = np.random.choice(range(num_transforms), p=probs)
                else:
                    # Pure random for exploration
                    transform_idx = random.randint(0, num_transforms - 1)

                selected_transforms[pos] = transform_idx

            # Apply transformations
            modified_lines = code_lines.copy()

            # Apply transformations in reverse order
            for pos in sorted(selected_transforms.keys(), reverse=True):
                transform_idx = selected_transforms[pos]
                try:
                    # Get current line indentation
                    if pos < len(modified_lines):
                        current_line = modified_lines[pos]
                        indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                    else:
                        current_line = ""
                        indent = 4

                    indent_str = ' ' * indent

                    # Get transformation code
                    transform_code = self.code_transformations[transform_idx]

                    # Ensure proper indentation
                    if isinstance(transform_code, str):
                        transform_code = indent_str + transform_code
                    else:
                        transform_code = transform_code(indent_str, "")

                    # Insert transformation
                    if pos < len(modified_lines):
                        modified_lines.insert(pos + 1, transform_code)
                    else:
                        modified_lines.append(transform_code)
                except Exception as e:
                    logger.error(f"Error applying transformation {transform_idx} at line {pos}: {str(e)}")
                    continue

            # Recombine code
            transformed_code = '\n'.join(modified_lines)

            # Evaluate transformation effect
            transformed_features = self._extract_code_features(transformed_code)
            feature_distance = self._calculate_loss(orig_features, transformed_features)
            pred_new = self._get_prediction(transformed_code, label)

            # Calculate reward
            current_success = (pred_new != label.item())
            reward = feature_distance + (1.0 if current_success else 0.0)

            # Update PositionMAB
            for pos, transform_idx in selected_transforms.items():
                position_mabs[pos].update(transform_idx, reward)

            # Update best result
            if reward > best_reward:
                best_reward = reward
                best_code = transformed_code
                attack_success = current_success

            # If attack succeeded, finish early
            if current_success:
                logger.info(f"Attack succeeded during ensemble exploration! Iteration {iteration + 1}")
                return transformed_code

        # Phase 2: MAB exploitation - Use pure MAB selection
        logger.info("Starting MAB exploitation phase...")
        for iteration in range(exploration_iterations, total_iterations):
            # Use MAB to select transformations
            selected_transforms = {}
            for pos in insertion_points:
                transform_idx = position_mabs[pos].select_transformation()
                selected_transforms[pos] = transform_idx

            # Apply transformations
            modified_lines = code_lines.copy()

            # Apply transformations in reverse order
            for pos in sorted(selected_transforms.keys(), reverse=True):
                transform_idx = selected_transforms[pos]
                try:
                    # Get current line indentation
                    if pos < len(modified_lines):
                        current_line = modified_lines[pos]
                        indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                    else:
                        current_line = ""
                        indent = 4

                    indent_str = ' ' * indent

                    # Get transformation code
                    transform_code = self.code_transformations[transform_idx]

                    # Ensure proper indentation
                    if isinstance(transform_code, str):
                        transform_code = indent_str + transform_code
                    else:
                        transform_code = transform_code(indent_str, "")

                    # Insert transformation
                    if pos < len(modified_lines):
                        modified_lines.insert(pos + 1, transform_code)
                    else:
                        modified_lines.append(transform_code)
                except Exception as e:
                    logger.error(f"Error applying transformation {transform_idx} at line {pos}: {str(e)}")
                    continue

            # Recombine code
            transformed_code = '\n'.join(modified_lines)

            # Evaluate transformation effect
            transformed_features = self._extract_code_features(transformed_code)
            feature_distance = self._calculate_loss(orig_features, transformed_features)
            pred_new = self._get_prediction(transformed_code, label)

            # Calculate reward
            current_success = (pred_new != label.item())
            reward = feature_distance + (1.0 if current_success else 0.0)

            # Update PositionMAB
            for pos, transform_idx in selected_transforms.items():
                position_mabs[pos].update(transform_idx, reward)

            # Update best result
            if reward > best_reward:
                best_reward = reward
                best_code = transformed_code
                attack_success = current_success

            # If attack succeeded, finish early
            if current_success:
                logger.info(
                    f"Attack succeeded during ensemble exploitation! Iteration {iteration - exploration_iterations + 1}")
                return transformed_code

        # Return best result
        logger.info(f"Reached maximum iterations. Attack {'succeeded' if attack_success else 'failed'}.")
        return best_code

    def _competitive_stacked_attack(self, source_code, label, preferences1, preferences2):
        """
        Competitive attack strategy: Divides iterations between two models with MAB-guided approach
        for each model instead of pure exploration.

        Args:
            source_code: Source code string
            label: Target label
            preferences1: First model's preferences
            preferences2: Second model's preferences

        Returns:
            Attacked code string
        """
        logger.info("Executing competitive stacked attack with MAB-guided approach")

        # Extract original code features
        orig_features = self._extract_code_features(source_code)

        # Find important insertion points
        insertion_points = self._find_important_positions(source_code, self.top_positions)

        if len(insertion_points) == 0:
            logger.warning("No usable insertion points found")
            return source_code

        # Number of transformations
        num_transforms = len(self.code_transformations)

        # Initialize separate PositionMAB sets for each model
        position_mabs1 = {}
        position_mabs2 = {}

        # Initialize from model 1 preferences
        for i, pos in enumerate(insertion_points):
            position_number = str(i + 1)
            mab1 = PositionMAB(pos, num_transforms)

            if preferences1 and position_number in preferences1:
                pos_prefs = preferences1[position_number]
                if 'best_transformation' in pos_prefs:
                    mab1.best_transformation = pos_prefs['best_transformation']

                if 'values' in pos_prefs and isinstance(pos_prefs['values'], list):
                    values = pos_prefs['values']
                    for j, value in enumerate(values):
                        if j < len(mab1.mab.values):
                            mab1.mab.values[j] = value
                            mab1.mab.counts[j] = 10
                            mab1.mab.total_pulls += 10

            position_mabs1[pos] = mab1

        # Initialize from model 2 preferences
        for i, pos in enumerate(insertion_points):
            position_number = str(i + 1)
            mab2 = PositionMAB(pos, num_transforms)

            if preferences2 and position_number in preferences2:
                pos_prefs = preferences2[position_number]
                if 'best_transformation' in pos_prefs:
                    mab2.best_transformation = pos_prefs['best_transformation']

                if 'values' in pos_prefs and isinstance(pos_prefs['values'], list):
                    values = pos_prefs['values']
                    for j, value in enumerate(values):
                        if j < len(mab2.mab.values):
                            mab2.mab.values[j] = value
                            mab2.mab.counts[j] = 10
                            mab2.mab.total_pulls += 10

            position_mabs2[pos] = mab2

        # Split code into lines
        code_lines = source_code.split('\n')

        # Track best results
        best_code = source_code
        best_reward = -float('inf')
        attack_success = False

        # Set iterations
        total_iterations = self.max_iterations

        # Iterations for each model
        iterations_per_model = total_iterations // 2

        # MAB-guided exploration ratio for each model's portion
        exploration_ratio = self.exi
        model1_exploration = round(iterations_per_model * exploration_ratio)
        model2_exploration = round(iterations_per_model * exploration_ratio)

        logger.info(
            f"Starting competitive attack with MAB-guided approach: each model gets {iterations_per_model} iterations")
        logger.info(
            f"Model 1: {model1_exploration} exploration, {iterations_per_model - model1_exploration} exploitation")
        logger.info(
            f"Model 2: {model2_exploration} exploration, {iterations_per_model - model2_exploration} exploitation")

        # Phase 1: Model 1 with MAB-guided exploration and exploitation
        logger.info("Starting model 1 phase...")

        # Model 1 exploration
        for iteration in range(model1_exploration):
            # Select transformations guided by MAB with some randomness
            selected_transforms = {}
            for pos in insertion_points:
                if random.random() < 0.7:  # 70% MAB-guided, 30% random
                    # Use MAB values to guide selection with some probability weighting
                    values = position_mabs1[pos].mab.values
                    min_val = min(values)
                    if min_val < 0:
                        shifted_values = [v - min_val + 0.1 for v in values]
                    else:
                        shifted_values = [v + 0.1 for v in values]

                    # Create probability distribution
                    total = sum(shifted_values)
                    probs = [v / total for v in shifted_values]

                    # Select based on probabilities
                    transform_idx = np.random.choice(range(num_transforms), p=probs)
                else:
                    # Random selection
                    transform_idx = random.randint(0, num_transforms - 1)

                selected_transforms[pos] = transform_idx

            # Apply transformations
            modified_lines = code_lines.copy()

            # Apply transformations in reverse order
            for pos in sorted(selected_transforms.keys(), reverse=True):
                transform_idx = selected_transforms[pos]
                try:
                    # Get current line indentation
                    if pos < len(modified_lines):
                        current_line = modified_lines[pos]
                        indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                    else:
                        current_line = ""
                        indent = 4

                    indent_str = ' ' * indent

                    # Get transformation code
                    transform_code = self.code_transformations[transform_idx]

                    # Ensure proper indentation
                    if isinstance(transform_code, str):
                        transform_code = indent_str + transform_code
                    else:
                        transform_code = transform_code(indent_str, "")

                    # Insert transformation
                    if pos < len(modified_lines):
                        modified_lines.insert(pos + 1, transform_code)
                    else:
                        modified_lines.append(transform_code)
                except Exception as e:
                    logger.error(f"Error applying transformation {transform_idx} at line {pos}: {str(e)}")
                    continue

            # Recombine code
            transformed_code = '\n'.join(modified_lines)

            # Evaluate transformation effect
            transformed_features = self._extract_code_features(transformed_code)
            feature_distance = self._calculate_loss(orig_features, transformed_features)
            pred_new = self._get_prediction(transformed_code, label)

            # Calculate reward
            current_success = (pred_new != label.item())
            reward = feature_distance + (1.0 if current_success else 0.0)

            # Update model 1 PositionMAB
            for pos, transform_idx in selected_transforms.items():
                position_mabs1[pos].update(transform_idx, reward)

            # Update best result
            if reward > best_reward:
                best_reward = reward
                best_code = transformed_code
                attack_success = current_success

            # If attack succeeded, finish early
            if current_success:
                logger.info(f"Attack succeeded during model 1 exploration! Iteration {iteration + 1}")
                return transformed_code

        # Model 1 exploitation
        for iteration in range(model1_exploration, iterations_per_model):
            # Use MAB to select transformations
            selected_transforms = {}
            for pos in insertion_points:
                transform_idx = position_mabs1[pos].select_transformation()
                selected_transforms[pos] = transform_idx

            # Apply transformations
            modified_lines = code_lines.copy()

            # Apply transformations in reverse order
            for pos in sorted(selected_transforms.keys(), reverse=True):
                transform_idx = selected_transforms[pos]
                try:
                    # Get current line indentation
                    if pos < len(modified_lines):
                        current_line = modified_lines[pos]
                        indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                    else:
                        current_line = ""
                        indent = 4

                    indent_str = ' ' * indent

                    # Get transformation code
                    transform_code = self.code_transformations[transform_idx]

                    # Ensure proper indentation
                    if isinstance(transform_code, str):
                        transform_code = indent_str + transform_code
                    else:
                        transform_code = transform_code(indent_str, "")

                    # Insert transformation
                    if pos < len(modified_lines):
                        modified_lines.insert(pos + 1, transform_code)
                    else:
                        modified_lines.append(transform_code)
                except Exception as e:
                    logger.error(f"Error applying transformation {transform_idx} at line {pos}: {str(e)}")
                    continue

            # Recombine code
            transformed_code = '\n'.join(modified_lines)

            # Evaluate transformation effect
            transformed_features = self._extract_code_features(transformed_code)
            feature_distance = self._calculate_loss(orig_features, transformed_features)
            pred_new = self._get_prediction(transformed_code, label)

            # Calculate reward
            current_success = (pred_new != label.item())
            reward = feature_distance + (1.0 if current_success else 0.0)

            # Update model 1 PositionMAB
            for pos, transform_idx in selected_transforms.items():
                position_mabs1[pos].update(transform_idx, reward)

            # Update best result
            if reward > best_reward:
                best_reward = reward
                best_code = transformed_code
                attack_success = current_success

            # If attack succeeded, finish early
            if current_success:
                logger.info(
                    f"Attack succeeded during model 1 exploitation! Iteration {iteration - model1_exploration + 1}")
                return transformed_code

        # Phase 2: Model 2 with MAB-guided exploration and exploitation
        logger.info("Starting model 2 phase...")

        # Model 2 exploration
        for iteration in range(model2_exploration):
            # Select transformations guided by MAB with some randomness
            selected_transforms = {}
            for pos in insertion_points:
                if random.random() < 0.7:  # 70% MAB-guided, 30% random
                    # Use MAB values to guide selection with some probability weighting
                    values = position_mabs2[pos].mab.values
                    min_val = min(values)
                    if min_val < 0:
                        shifted_values = [v - min_val + 0.1 for v in values]
                    else:
                        shifted_values = [v + 0.1 for v in values]

                    # Create probability distribution
                    total = sum(shifted_values)
                    probs = [v / total for v in shifted_values]

                    # Select based on probabilities
                    transform_idx = np.random.choice(range(num_transforms), p=probs)
                else:
                    # Random selection
                    transform_idx = random.randint(0, num_transforms - 1)

                selected_transforms[pos] = transform_idx

            # Apply transformations
            modified_lines = code_lines.copy()

            # Apply transformations in reverse order
            for pos in sorted(selected_transforms.keys(), reverse=True):
                transform_idx = selected_transforms[pos]
                try:
                    # Get current line indentation
                    if pos < len(modified_lines):
                        current_line = modified_lines[pos]
                        indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                    else:
                        current_line = ""
                        indent = 4

                    indent_str = ' ' * indent

                    # Get transformation code
                    transform_code = self.code_transformations[transform_idx]

                    # Ensure proper indentation
                    if isinstance(transform_code, str):
                        transform_code = indent_str + transform_code
                    else:
                        transform_code = transform_code(indent_str, "")

                    # Insert transformation
                    if pos < len(modified_lines):
                        modified_lines.insert(pos + 1, transform_code)
                    else:
                        modified_lines.append(transform_code)
                except Exception as e:
                    logger.error(f"Error applying transformation {transform_idx} at line {pos}: {str(e)}")
                    continue

            # Recombine code
            transformed_code = '\n'.join(modified_lines)

            # Evaluate transformation effect
            transformed_features = self._extract_code_features(transformed_code)
            feature_distance = self._calculate_loss(orig_features, transformed_features)
            pred_new = self._get_prediction(transformed_code, label)

            # Calculate reward
            current_success = (pred_new != label.item())
            reward = feature_distance + (1.0 if current_success else 0.0)

            # Update model 2 PositionMAB
            for pos, transform_idx in selected_transforms.items():
                position_mabs2[pos].update(transform_idx, reward)

            # Update best result
            if reward > best_reward:
                best_reward = reward
                best_code = transformed_code
                attack_success = current_success

            # If attack succeeded, finish early
            if current_success:
                logger.info(f"Attack succeeded during model 2 exploration! Iteration {iteration + 1}")
                return transformed_code

        # Model 2 exploitation
        for iteration in range(model2_exploration, iterations_per_model):
            # Use MAB to select transformations
            selected_transforms = {}
            for pos in insertion_points:
                transform_idx = position_mabs2[pos].select_transformation()
                selected_transforms[pos] = transform_idx

            # Apply transformations
            modified_lines = code_lines.copy()

            # Apply transformations in reverse order
            for pos in sorted(selected_transforms.keys(), reverse=True):
                transform_idx = selected_transforms[pos]
                try:
                    # Get current line indentation
                    if pos < len(modified_lines):
                        current_line = modified_lines[pos]
                        indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                    else:
                        current_line = ""
                        indent = 4

                    indent_str = ' ' * indent

                    # Get transformation code
                    transform_code = self.code_transformations[transform_idx]

                    # Ensure proper indentation
                    if isinstance(transform_code, str):
                        transform_code = indent_str + transform_code
                    else:
                        transform_code = transform_code(indent_str, "")

                    # Insert transformation
                    if pos < len(modified_lines):
                        modified_lines.insert(pos + 1, transform_code)
                    else:
                        modified_lines.append(transform_code)
                except Exception as e:
                    logger.error(f"Error applying transformation {transform_idx} at line {pos}: {str(e)}")
                    continue

            # Recombine code
            transformed_code = '\n'.join(modified_lines)

            # Evaluate transformation effect
            transformed_features = self._extract_code_features(transformed_code)
            feature_distance = self._calculate_loss(orig_features, transformed_features)
            pred_new = self._get_prediction(transformed_code, label)

            # Calculate reward
            current_success = (pred_new != label.item())
            reward = feature_distance + (1.0 if current_success else 0.0)

            # Update model 2 PositionMAB
            for pos, transform_idx in selected_transforms.items():
                position_mabs2[pos].update(transform_idx, reward)

            # Update best result
            if reward > best_reward:
                best_reward = reward
                best_code = transformed_code
                attack_success = current_success

            # If attack succeeded, finish early
            if current_success:
                logger.info(
                    f"Attack succeeded during model 2 exploitation! Iteration {iteration - model2_exploration + 1}")
                return transformed_code

        # Return best result
        logger.info(f"Reached maximum iterations. Attack {'succeeded' if attack_success else 'failed'}.")
        return best_code

    def _adaptive_stacked_attack(self, source_code, label, preferences1, preferences2):
        """
        Improved adaptive attack strategy: Uses a separate Model-level MAB to decide which
        model to use for each position, instead of fixed weights.

        Args:
            source_code: Source code string
            label: Target label
            preferences1: First model's preferences
            preferences2: Second model's preferences

        Returns:
            Attacked code string
        """
        logger.info("Executing improved adaptive stacked attack with model-level MAB")

        # Extract original code features
        orig_features = self._extract_code_features(source_code)

        # Find important insertion points
        insertion_points = self._find_important_positions(source_code, self.top_positions)

        if len(insertion_points) == 0:
            logger.warning("No usable insertion points found")
            return source_code

        # Number of transformations
        num_transforms = len(self.code_transformations)

        # Initialize MABs for transformations from each model
        position_mabs1 = {}  # Model 1 MAB for transformations
        position_mabs2 = {}  # Model 2 MAB for transformations

        # Initialize model-selection MAB for each position
        # This is a 2-armed bandit where arms are: [use model1, use model2]
        model_selection_mabs = {}

        # Initialize all MABs
        for i, pos in enumerate(insertion_points):
            position_number = str(i + 1)

            # Initialize model 1 transformation MAB
            mab1 = PositionMAB(pos, num_transforms)
            if preferences1 and position_number in preferences1:
                pos_prefs1 = preferences1[position_number]
                if 'best_transformation' in pos_prefs1:
                    mab1.best_transformation = pos_prefs1['best_transformation']
                if 'values' in pos_prefs1 and isinstance(pos_prefs1['values'], list):
                    values = pos_prefs1['values']
                    for j, value in enumerate(values):
                        if j < len(mab1.mab.values):
                            mab1.mab.values[j] = value
                            mab1.mab.counts[j] = 10
                            mab1.mab.total_pulls += 10
            position_mabs1[pos] = mab1

            # Initialize model 2 transformation MAB
            mab2 = PositionMAB(pos, num_transforms)
            if preferences2 and position_number in preferences2:
                pos_prefs2 = preferences2[position_number]
                if 'best_transformation' in pos_prefs2:
                    mab2.best_transformation = pos_prefs2['best_transformation']
                if 'values' in pos_prefs2 and isinstance(pos_prefs2['values'], list):
                    values = pos_prefs2['values']
                    for j, value in enumerate(values):
                        if j < len(mab2.mab.values):
                            mab2.mab.values[j] = value
                            mab2.mab.counts[j] = 10
                            mab2.mab.total_pulls += 10
            position_mabs2[pos] = mab2

            # Initialize model selection MAB (2 arms: model1, model2)
            model_selection_mabs[pos] = UCB1MAB(num_arms=2)

        # Split code into lines
        code_lines = source_code.split('\n')

        # Track best results
        best_code = source_code
        best_reward = -float('inf')
        attack_success = False

        # Set iterations
        total_iterations = self.max_iterations

        # MAB-guided exploration ratio
        exploration_ratio = self.exi
        exploration_iterations = round(total_iterations * exploration_ratio)

        logger.info(f"Starting improved adaptive attack: {exploration_iterations} exploration iterations, "
                    f"{total_iterations - exploration_iterations} exploitation iterations")

        # Phase 1: Exploration phase with model-level MAB
        logger.info("Starting exploration phase...")

        for iteration in range(exploration_iterations):
            logger.info(f"Exploration iteration {iteration + 1}")

            # Select models and transformations for each position
            selected_transforms = {}
            transform_models = {}  # Track which model was used for each position

            for pos in insertion_points:
                # In exploration phase, we alternate between models with some randomness
                if iteration < exploration_iterations // 3:
                    # First third: random exploration with slight bias
                    if random.random() < 0.55:  # Slight bias to model 1 initially
                        model_idx = 0  # model 1
                    else:
                        model_idx = 1  # model 2
                elif iteration < 2 * exploration_iterations // 3:
                    # Second third: start using UCB1 model selection but with randomness
                    if random.random() < 0.7:
                        model_idx = model_selection_mabs[pos].select_arm()
                    else:
                        model_idx = random.randint(0, 1)
                else:
                    # Last third: rely more on UCB1 model selection
                    model_idx = model_selection_mabs[pos].select_arm()

                # Select transformation from the chosen model's MAB
                current_mab = position_mabs1[pos] if model_idx == 0 else position_mabs2[pos]
                transform_idx = current_mab.select_transformation()

                # Store selected transform and model
                selected_transforms[pos] = transform_idx
                transform_models[pos] = model_idx

            # Apply transformations
            modified_lines = code_lines.copy()

            # Apply transformations in reverse order
            for pos in sorted(selected_transforms.keys(), reverse=True):
                transform_idx = selected_transforms[pos]
                actual_pos = pos[0] if isinstance(pos, tuple) else pos
                try:
                    if actual_pos < len(modified_lines):
                        current_line = modified_lines[actual_pos]
                        indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                    else:
                        current_line = ""
                        indent = 4
                    indent_str = ' ' * indent
                    transform_code = self.code_transformations[transform_idx]
                    if isinstance(transform_code, str):
                        transform_code = indent_str + transform_code
                    else:
                        transform_code = transform_code(indent_str, "")
                    if actual_pos < len(modified_lines):
                        modified_lines.insert(actual_pos + 1, transform_code)
                    else:
                        modified_lines.append(transform_code)
                except Exception as e:
                    logger.error(f"Error applying transformation {transform_idx} at line {actual_pos}: {str(e)}")
                    continue

            # Recombine code
            transformed_code = '\n'.join(modified_lines)

            # Evaluate transformation effect
            transformed_features = self._extract_code_features(transformed_code)
            feature_distance = self._calculate_loss(orig_features, transformed_features)
            pred_new = self._get_prediction(transformed_code, label)

            # Calculate reward
            current_success = (pred_new != label.item())
            reward = feature_distance + (1.0 if current_success else 0.0)

            # Update MABs with rewards
            for pos, transform_idx in selected_transforms.items():
                model_idx = transform_models[pos]

                # Update the transformation MAB for the model that was used
                if model_idx == 0:
                    position_mabs1[pos].update(transform_idx, reward)
                else:
                    position_mabs2[pos].update(transform_idx, reward)

                # Update the model selection MAB
                model_selection_mabs[pos].update(model_idx, reward)

            # Update best result
            if reward > best_reward:
                best_reward = reward
                best_code = transformed_code
                attack_success = current_success

            # If attack succeeded, finish early
            if current_success:
                logger.info(f"Attack succeeded during exploration! Iteration {iteration + 1}")
                return transformed_code

        # Phase 2: Exploitation phase - Use model selection MAB to choose model, then use that model's MAB
        logger.info("Starting exploitation phase...")

        for iteration in range(exploration_iterations, total_iterations):
            # Use MAB for both model selection and transformation selection
            selected_transforms = {}
            transform_models = {}

            for pos in insertion_points:
                # Select model using model selection MAB
                model_idx = model_selection_mabs[pos].select_arm()

                # Select transformation using the chosen model's MAB
                current_mab = position_mabs1[pos] if model_idx == 0 else position_mabs2[pos]
                transform_idx = current_mab.select_transformation()

                # Store selections
                selected_transforms[pos] = transform_idx
                transform_models[pos] = model_idx

            # Apply transformations
            modified_lines = code_lines.copy()

            # Apply transformations in reverse order
            for pos in sorted(selected_transforms.keys(), reverse=True):
                transform_idx = selected_transforms[pos]
                actual_pos = pos[0] if isinstance(pos, tuple) else pos
                try:
                    if actual_pos < len(modified_lines):
                        current_line = modified_lines[actual_pos]
                        indent = len(current_line) - len(current_line.lstrip()) if current_line.strip() else 0
                    else:
                        current_line = ""
                        indent = 4
                    indent_str = ' ' * indent
                    transform_code = self.code_transformations[transform_idx]
                    if isinstance(transform_code, str):
                        transform_code = indent_str + transform_code
                    else:
                        transform_code = transform_code(indent_str, "")
                    if actual_pos < len(modified_lines):
                        modified_lines.insert(actual_pos + 1, transform_code)
                    else:
                        modified_lines.append(transform_code)
                except Exception as e:
                    logger.error(f"Error applying transformation {transform_idx} at line {actual_pos}: {str(e)}")
                    continue

            # Recombine code
            transformed_code = '\n'.join(modified_lines)

            # Evaluate transformation effect
            transformed_features = self._extract_code_features(transformed_code)
            feature_distance = self._calculate_loss(orig_features, transformed_features)
            pred_new = self._get_prediction(transformed_code, label)

            # Calculate reward
            current_success = (pred_new != label.item())
            reward = feature_distance + (1.0 if current_success else 0.0)

            # Update MABs with rewards
            for pos, transform_idx in selected_transforms.items():
                model_idx = transform_models[pos]

                # Update the transformation MAB for the model that was used
                if model_idx == 0:
                    position_mabs1[pos].update(transform_idx, reward)
                else:
                    position_mabs2[pos].update(transform_idx, reward)

                # Update the model selection MAB
                model_selection_mabs[pos].update(model_idx, reward)

            # Update best result
            if reward > best_reward:
                best_reward = reward
                best_code = transformed_code
                attack_success = current_success

            # If attack succeeded, finish early
            if current_success:
                # Log model usage statistics
                model1_count = sum(1 for pos in transform_models if transform_models[pos] == 0)
                model2_count = sum(1 for pos in transform_models if transform_models[pos] == 1)
                total = len(transform_models)

                logger.info(f"Attack succeeded during exploitation! Iteration {iteration - exploration_iterations + 1}")
                logger.info(f"Model usage: Model 1: {model1_count}/{total} ({model1_count / total:.2f}), "
                            f"Model 2: {model2_count}/{total} ({model2_count / total:.2f})")

                # Log MAB state for model selection
                for pos in insertion_points:
                    logger.info(f"Position {pos} model selection values: "
                                f"Model 1: {model_selection_mabs[pos].values[0]:.4f}, "
                                f"Model 2: {model_selection_mabs[pos].values[1]:.4f}")

                return transformed_code

        # Return best result if we reach maximum iterations
        if attack_success:
            logger.info(f"Attack succeeded but reached maximum iterations.")
        else:
            logger.info(f"Attack failed after maximum iterations.")

        # Log final model selection state
        for pos in insertion_points:
            model1_val = model_selection_mabs[pos].values[0]
            model2_val = model_selection_mabs[pos].values[1]
            preferred = "Model 1" if model1_val > model2_val else "Model 2"
            logger.info(f"Position {pos} final model preferences: "
                        f"Model 1: {model1_val:.4f}, Model 2: {model2_val:.4f}, Preferred: {preferred}")

        return best_code

def mab_memory_all(save_path, all_position_mabs):
    """
    全量保存每个样本每个位置的最佳转换和 value，无论攻击是否成功。
    Args:
        save_path: 保存路径
        all_position_mabs: {sample_index: {position: PositionMAB实例或dict}}
    """
    import json
    import numpy as np
    result = {}
    for sample_idx, pos_mabs in all_position_mabs.items():
        result[str(sample_idx)] = {}
        for pos, mab in pos_mabs.items():
            # 兼容 PositionMAB 实例或 dict
            if hasattr(mab, 'best_transformation') and hasattr(mab, 'mab'):
                best_trans = int(mab.best_transformation) if mab.best_transformation is not None else 0
                values = [float(v) if isinstance(v, (float, int, np.number)) else v for v in getattr(mab.mab, 'values', [])]
            elif isinstance(mab, dict):
                best_trans = int(mab.get('best_transformation', 0))
                values = [float(v) if isinstance(v, (float, int, np.number)) else v for v in mab.get('values', [])]
            else:
                best_trans = 0
                values = []
            result[str(sample_idx)][str(pos)] = {
                'best_transformation': best_trans,
                'values': values
            }
    # 保存为json
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    logger.info(f"全量MAB memory已保存到 {save_path}")


