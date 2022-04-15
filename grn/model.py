import torch
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF


class ContextLayer(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(ContextLayer, self).__init__()
        self.conv_1 = nn.Conv1d(embed_size, hidden_size, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv1d(embed_size, hidden_size, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(embed_size, hidden_size, kernel_size=5, padding=2)

    def forward(self, inp):
        inp = inp.permute(0, 2, 1)

        conv_1_out = torch.tanh(self.conv_1(inp)).permute(0, 2, 1).unsqueeze(2)
        conv_3_out = torch.tanh(self.conv_3(inp)).permute(0, 2, 1).unsqueeze(2)
        conv_5_out = torch.tanh(self.conv_5(inp)).permute(0, 2, 1).unsqueeze(2)

        out = torch.cat((conv_1_out, conv_3_out, conv_5_out), dim=2)
        out = F.max_pool2d(out, kernel_size=(out.size(2), 1)).squeeze(2)

        return out


class RelationLayer(nn.Module):
    def __init__(self, hidden_size):
        super(RelationLayer, self).__init__()
        self.hidden_size = hidden_size

        self.linear_0 = nn.Linear(hidden_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, inp, masks):
        batch_size, max_seq_size, embed_size = inp.size()
        sentence_lengths = torch.sum(masks, 1)

        # attention-based context information
        # batch x max_seq x embed --> batch x max_seq x max_seq x embed
        gate_0 = self.linear_0(inp)
        gate_1 = self.linear_1(inp)

        sigmoid_input = gate_0.view(batch_size, max_seq_size, 1, embed_size) \
                            .expand(batch_size, max_seq_size, max_seq_size, embed_size) + gate_1 \
                            .view(batch_size, 1, max_seq_size, embed_size) \
                            .expand(batch_size, max_seq_size, max_seq_size, embed_size)

        # batch x max_seq x max_seq x embed
        forget_gate = torch.sigmoid(sigmoid_input)

        input_row_expanded = inp.view(batch_size, 1, max_seq_size, embed_size) \
            .expand(batch_size, max_seq_size, max_seq_size, embed_size)

        forget_result = torch.mul(input_row_expanded, forget_gate)

        # start_t0 = time.time()
        selection_mask = masks.view(batch_size, max_seq_size, 1) \
            .mul(masks.view(batch_size, 1, max_seq_size))

        selection_mask = selection_mask.view(batch_size, max_seq_size, max_seq_size, 1) \
            .expand(batch_size, max_seq_size, max_seq_size, self.hidden_size)

        # batch x max_seq x max_seq x embed
        forget_result_masked = torch.mul(forget_result, selection_mask.float())

        # batch x max_seq x embed
        context_sumup = torch.sum(forget_result_masked, 2)

        # average
        context_vector = torch.div(context_sumup, sentence_lengths.view(batch_size, 1, 1)
                                   .expand(batch_size, max_seq_size, self.hidden_size).float())

        output_result = F.tanh(context_vector)

        return output_result


class NER(nn.Module):
    def __init__(self, num_tags, char_vocab_size, char_embed_size, input_dropout_rate, hidden_dim, hidden_dropout_rate,
                 use_seg_embed, seg_vocab_size, seg_embed_size, use_bigram_embed, bigram_vocab_size, bigram_embed_size,
                 use_crf, embed_freeze):
        super(NER, self).__init__()
        self.use_seg_embed = use_seg_embed
        self.use_bigram_embed = use_bigram_embed
        self.use_crf = use_crf

        embed_size = char_embed_size
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_size)

        if use_seg_embed:
            embed_size += seg_embed_size
            self.seg_embedding = nn.Embedding(seg_vocab_size, seg_embed_size)

        if use_bigram_embed:
            embed_size += bigram_embed_size * 2
            self.bigram_embedding = nn.Embedding(bigram_vocab_size, bigram_embed_size)

        self.context_layer = ContextLayer(embed_size, hidden_dim)
        self.relation_layer = RelationLayer(hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_tags)

        self.in_dropout = nn.Dropout(input_dropout_rate)
        self.hid_dropout = nn.Dropout(hidden_dropout_rate)

        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        if embed_freeze:
            for param in self.char_embedding.parameters():
                param.requires_grad = False
            if use_bigram_embed:
                for param in self.bigram_embedding.parameters():
                    param.requires_grad = False

    def init_char_embedding(self, pretrained_embeddings):
        self.char_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_bigram_embedding(self, pretrained_embeddings):
        self.bigram_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, tokens, masks, seg_tags=None, decode=True, tags=None):
        out = self.char_embedding(tokens[:, :, 0])
        if self.use_bigram_embed:
            w_emb = torch.cat([self.bigram_embedding(tokens[:, :, i]) for i in range(1, tokens.size()[2])], dim=2)
            out = torch.cat((out, w_emb), dim=2)
        if self.use_seg_embed:
            s_emb = self.seg_embedding(seg_tags)
            out = torch.cat((out, s_emb), dim=2)
        out = self.in_dropout(out)

        out = self.context_layer(out)
        out = self.relation_layer(out, masks)
        out = self.hid_dropout(out)

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
