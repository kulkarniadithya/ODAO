from transformers import BertModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.module = BertModel.from_pretrained('bert-base-uncased')
        self.config = self.module.config
        self.module_output = nn.Linear(self.config.hidden_size, 2)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,
                module_start_positions=None, module_end_positions=None):
        module_bert_output = self.module(input_ids, token_type_ids, attention_mask)
        module_logits = self.module_output(module_bert_output[0])
        module_start_logits, module_end_logits = module_logits.split(1, dim=-1)
        module_start_logits = module_start_logits.squeeze(-1)
        module_end_logits = module_end_logits.squeeze(-1)

        if module_start_positions is not None and module_end_positions is not None:
            if len(module_start_positions.size()) > 1:
                module_start_positions = module_start_positions.squeeze(-1)
            if len(module_end_positions.size()) > 1:
                module_end_positions = module_end_positions.squeeze(-1)

        loss_func = CrossEntropyLoss()
        start_index = torch.nonzero(module_start_positions)
        end_index = torch.nonzero(module_end_positions)
        start_index = start_index.tolist()
        end_index = end_index.tolist()
        module_start_loss = 0
        module_end_loss = 0
        for i in range(0, len(start_index)):
            for j in range(0, len(start_index[i])):
                module_start_loss = module_start_loss + loss_func(module_start_logits, torch.tensor([start_index[i][j]]))
        for i in range(0, len(end_index)):
            for j in range(0, len(end_index[i])):
                module_end_loss = module_end_loss + loss_func(module_end_logits, torch.tensor([end_index[i][j]]))
        module_loss = (module_start_loss + module_end_loss) / 2
        return module_loss, module_bert_output
