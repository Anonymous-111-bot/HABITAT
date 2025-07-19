import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def adapt_unixcoder_features(self, input_ids=None, labels=None):
        """
        Adapter method to make CodeBERT compatible with UniXcoder features
        """
        # Reshaping input to match block_size
        original_shape = input_ids.shape

        # Determine correct padding token (important!)
        pad_token = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 1

        # Ensure input shape is correct
        if len(original_shape) == 2 and original_shape[1] != self.args.block_size:
            if original_shape[1] < self.args.block_size:
                # Pad with the CORRECT pad token
                pad_length = self.args.block_size - original_shape[1]
                padding = torch.full((original_shape[0], pad_length),
                                     pad_token,
                                     dtype=input_ids.dtype,
                                     device=input_ids.device)
                input_ids = torch.cat([input_ids, padding], dim=1)
            else:
                # Truncate to correct size
                input_ids = input_ids[:, :self.args.block_size]

        # Reshape properly
        input_ids = input_ids.view(-1, self.args.block_size)

        # Create proper attention mask using the correct pad token
        attention_mask = input_ids.ne(pad_token)

        # Process through the encoder with the correct format
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Extract hidden states properly - account for different output formats
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]  # CodeBERT style
        else:
            hidden_states = outputs.last_hidden_state  # UniXcoder style

        logits = self.classifier(hidden_states)
        prob = F.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

    def get_encoder_feature(self, input_ids=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        outputs = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
        return outputs

    def forward(self, input_ids=None, labels=None, use_adapter=True):
        if use_adapter:
            return self.adapt_unixcoder_features(input_ids, labels)

        if input_ids is not None:
            current_size = input_ids.size(1)  # 假设input_ids是[batch_size, sequence_length]
            if current_size < self.args.block_size:
                pad_size = self.args.block_size - current_size
                padding = torch.zeros(input_ids.size(0), pad_size, dtype=input_ids.dtype, device=input_ids.device)
                input_ids = torch.cat((input_ids, padding), dim=1)

            input_ids = input_ids.view(-1, self.args.block_size)

        outputs = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        return self

    def get_results(self, dataset, batch_size):
        '''
        给定example和tgt model，返回预测的label和probability
        '''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        ## Evaluate Model

        eval_loss = 0.0
        nb_eval_steps = 0
        self.eval()
        logits = []
        labels = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            label = batch[1].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs, label)
                # print(lm_loss)
                # 调用这个模型. 重写了反前向传播模型.
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

            nb_eval_steps += 1
        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)

        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))

        return probs, pred_labels


class Model_sub(nn.Module):
    def __init__(self, encoder, args):
        super(Model_sub, self).__init__()
        self.encoder = encoder
        self.args = args

    def forward(self, input_ids=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        outputs = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
        return outputs


class Model_for_sim(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model_for_sim, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, input_ids=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        outputs = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, outputs
        else:
            return prob, outputs


class Model_for_loss(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model_for_loss, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, input_ids=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        outputs = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

    def get_results(self, dataset, batch_size):
        '''
        给定example和tgt model，返回预测的label和probability
        '''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        ## Evaluate Model

        eval_loss = 0.0
        nb_eval_steps = 0
        self.eval()
        logits = []
        labels = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            label = batch[1].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs, label)
                # print(lm_loss)
                # 调用这个模型. 重写了反前向传播模型.
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

            nb_eval_steps += 1
        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)

        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))

        return probs, pred_labels, eval_loss