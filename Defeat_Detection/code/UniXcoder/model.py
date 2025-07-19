import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
from transformers import RobertaModel, RobertaConfig


class UniXcoderClassificationHead(nn.Module):
    """Head for sentence-level classification tasks on UniXcoder."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels if hasattr(config, 'num_labels') else 2)

    def forward(self, features, **kwargs):
        # For UniXcoder, typically use the first token representation ([CLS])
        x = features[:, 0, :]  # take CLS token
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
        self.classifier = UniXcoderClassificationHead(config)
        self.args = args
        self.query = 0

        # Ensure padding token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if hasattr(encoder, 'config'):
                encoder.config.pad_token_id = self.tokenizer.pad_token_id

    def get_encoder_feature(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        attention_mask = self._create_attention_mask(input_ids) if attention_mask is None else attention_mask

        # Get UniXcoder hidden states
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Use last layer hidden states for classification
        hidden_states = outputs.last_hidden_state
        return hidden_states

    def _create_attention_mask(self, input_ids):
        """Safely create attention mask handling None pad_token_id"""
        if self.tokenizer.pad_token_id is None:
            # Create a mask of 1s with the same shape as input_ids
            return torch.ones_like(input_ids)
        else:
            # Create a mask based on padding tokens with the same shape as input_ids
            return input_ids.ne(self.tokenizer.pad_token_id)

    def forward(self, input_ids=None, labels=None):
        batch_size = input_ids.size(0)

        # Calculate how many complete blocks we can fit
        total_elements = input_ids.numel()

        # If needed, pad the input to make it divisible by block_size
        if total_elements % self.args.block_size != 0:
            # Reshape to 1D first
            input_ids_flat = input_ids.reshape(-1)
            # Calculate padding needed
            pad_length = self.args.block_size - (total_elements % self.args.block_size)
            # Pad the flattened tensor
            input_ids_padded = F.pad(input_ids_flat, (0, pad_length), "constant", self.tokenizer.pad_token_id)
            # Use the padded tensor
            input_ids = input_ids_padded

        # Now reshape safely
        input_ids = input_ids.view(-1, self.args.block_size)

        # Create proper attention mask with the same dimensions as input_ids
        attention_mask = self._create_attention_mask(input_ids)

        # Get UniXcoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Use last layer hidden states for classification
        hidden_states = outputs.last_hidden_state
        logits = self.classifier(hidden_states)
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
        Given examples and target model, return predicted labels and probabilities
        '''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        # Evaluate model
        eval_loss = 0.0
        nb_eval_steps = 0
        self.eval()
        logits = []
        labels = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            label = batch[1].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs, labels=label)
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

    def forward(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.view(-1, self.args.block_size)

        # Create attention mask if not provided
        if attention_mask is None:
            if hasattr(self, 'tokenizer') and self.tokenizer.pad_token_id is not None:
                attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            else:
                attention_mask = torch.ones_like(input_ids)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs.last_hidden_state  # Return last layer hidden states


class Model_for_sim(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model_for_sim, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = UniXcoderClassificationHead(config)
        self.args = args
        self.query = 0

        # Ensure padding token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if hasattr(encoder, 'config'):
                encoder.config.pad_token_id = self.tokenizer.pad_token_id

    def _create_attention_mask(self, input_ids):
        """Safely create attention mask handling None pad_token_id"""
        if self.tokenizer.pad_token_id is None:
            return torch.ones_like(input_ids)
        else:
            return input_ids.ne(self.tokenizer.pad_token_id)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        attention_mask = self._create_attention_mask(input_ids) if attention_mask is None else attention_mask

        # Get UniXcoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Use last layer hidden states for classification
        hidden_states = outputs.last_hidden_state
        logits = self.classifier(hidden_states)
        prob = F.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, hidden_states
        else:
            return prob, hidden_states


class ModelForLoss(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelForLoss, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = UniXcoderClassificationHead(config)
        self.args = args
        self.query = 0

        # Ensure padding token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if hasattr(encoder, 'config'):
                encoder.config.pad_token_id = self.tokenizer.pad_token_id

    def _create_attention_mask(self, input_ids):
        """Safely create attention mask handling None pad_token_id"""
        if self.tokenizer.pad_token_id is None:
            return torch.ones_like(input_ids)
        else:
            return input_ids.ne(self.tokenizer.pad_token_id)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        attention_mask = self._create_attention_mask(input_ids) if attention_mask is None else attention_mask

        # Get UniXcoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Use last layer hidden states for classification
        hidden_states = outputs.last_hidden_state
        logits = self.classifier(hidden_states)
        prob = F.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

    def get_results(self, dataset, batch_size):
        '''
        Given examples and target model, return predicted labels, probabilities, and loss
        '''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        # Evaluate model
        eval_loss = 0.0
        nb_eval_steps = 0
        self.eval()
        logits = []
        labels = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            label = batch[1].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs, labels=label)
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


# Helper function to load and initialize UniXcoder model
def load_unixcoder_model(model_path, tokenizer, args, num_labels=66):
    # Load UniXcoder model configuration
    config = RobertaConfig.from_pretrained(model_path)

    # Modify configuration for classification task
    config.num_labels = num_labels

    # Load pretrained UniXcoder model
    encoder = RobertaModel.from_pretrained(model_path)

    # Ensure padding token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        encoder.config.pad_token_id = tokenizer.pad_token_id

    # Create classification model
    model = Model(encoder=encoder, config=config, tokenizer=tokenizer, args=args)

    return model