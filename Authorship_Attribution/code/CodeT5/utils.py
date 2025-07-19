import collections
import re
from io import StringIO
import  tokenize
import torch
import torch.nn as nn
import copy
from tree_sitter import Language, Parser
import random
import sys
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import os
import sys
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
import numpy as np
import csv
from parser.run_parser import  get_example_batch
class CodeDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)



def select_parents(population):
        length = range(len(population))
        index_1 = random.choice(length)
        index_2 = random.choice(length)
        chromesome_1 = population[index_1]
        chromesome_2 = population[index_2]
        return chromesome_1, index_1, chromesome_2, index_2

def mutate(chromesome, variable_substitute_dict):
        tgt_index = random.choice(range(len(chromesome)))
        tgt_word = list(chromesome.keys())[tgt_index]
        chromesome[tgt_word] = random.choice(variable_substitute_dict[tgt_word])

        return chromesome

def crossover(csome_1, csome_2, r=None):
        if r is None:
            r = random.choice(range(len(csome_1)))  # 随机选择一个位置.
            # 但是不能选到0

        child_1 = {}
        child_2 = {}
        for index, variable_name in enumerate(csome_1.keys()):
            if index < r:  # 前半段
                child_2[variable_name] = csome_1[variable_name]
                child_1[variable_name] = csome_2[variable_name]
            else:
                child_1[variable_name] = csome_1[variable_name]
                child_2[variable_name] = csome_2[variable_name]
        return child_1, child_2

def map_chromesome(chromesome: dict, code: str, lang: str) -> str:

        temp_replace = get_example_batch(code, chromesome, lang)

        return temp_replace
