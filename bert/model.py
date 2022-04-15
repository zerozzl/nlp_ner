import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from torchcrf import CRF


class NER(nn.Module):
    def __init__(self, config_path, model_path, num_tags, bert_freeze, use_crf):
        super(NER, self).__init__()
        self.use_crf = use_crf

        config = BertConfig.from_json_file(config_path)
        self.embedding = BertModel.from_pretrained(model_path, config=config)
        self.linear = nn.Linear(config.hidden_size, num_tags)

        if bert_freeze:
            for param in self.embedding.parameters():
                param.requires_grad = False

        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, tokens, segments, masks, decode=True, tags=None):
        out = self.embedding(input_ids=tokens, token_type_ids=segments, attention_mask=masks)
        out = out.last_hidden_state
        out = self.linear(out)

        if decode:
            if self.use_crf:
                pred = self.crf.decode(out, masks)
            else:
                out = F.softmax(out, dim=2)
                out = torch.argmax(out, dim=2)
                pred = out.cpu().numpy()
            return pred
        else:
            if self.use_crf:
                loss = -self.crf(out, tags, masks)
            else:
                out_shape = out.size()
                loss = self.ce_loss(out.reshape(out_shape[0] * out_shape[1], out_shape[2]),
                                    tags.reshape(out_shape[0] * out_shape[1]))
            return loss
