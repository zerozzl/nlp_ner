"""PyTorch  StarTransformerLayer  """
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from torchcrf import CRF


class MultiAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(MultiAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v):
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(k)
        mixed_value_layer = self.value(v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return self.out(context_layer)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class StarTransformerLayer(nn.Module):

    def __init__(self, cycle_num, hidden_size, num_attention_heads, attention_dropout_prob):
        super().__init__()
        self.cycle_num = cycle_num
        self.multi_att_satellite = MultiAttention(hidden_size, num_attention_heads, attention_dropout_prob)
        self.multi_att_relay = copy.deepcopy(self.multi_att_satellite)
        self.ln_satellite = LayerNorm(hidden_size)
        self.ln_relay = copy.deepcopy(self.ln_satellite)

    def cycle_shift(self, e: torch.Tensor, forward=True):
        b, l, d = e.size()

        if forward:
            temp = e[:, -1, :]
            for i in range(l - 1):
                e[:, i + 1, :] = e[:, i, :]
            e[:, 0, :] = temp
        else:
            temp = e[:, 0, :]
            for i in range(1, l):
                e[:, i - 1, :] = e[:, i, :]
            e[:, -1, :] = temp

        return e

    def forward(self, e: torch.Tensor):
        # Initialization
        h = e.clone()
        b, l, d = h.size()
        s = F.avg_pool2d(h, (h.shape[1], 1)).squeeze(1)
        for _ in range(self.cycle_num):
            # update the satellite nodes
            h_last, h_next = self.cycle_shift(h.clone(), False), self.cycle_shift(h.clone(), True)
            s_m = s.unsqueeze(1).expand_as(h)
            c = torch.cat(
                [h_last.unsqueeze(-2), h.unsqueeze(-2), h_next.unsqueeze(-2), e.unsqueeze(-2), s_m.unsqueeze(-2)],
                dim=-2)
            c = c.view(b * l, -1, d)
            h = h.unsqueeze(-2).view(b * l, -1, d)
            h = self.ln_satellite(F.relu(self.multi_att_satellite(h, c, c))).squeeze(-2).view(b, l, -1)
            # update the relay node
            s = s.unsqueeze(1)
            m_c = torch.cat([s, h], dim=1)
            s = self.ln_relay(F.relu(self.multi_att_relay(s, m_c, m_c))).squeeze(1)

        return h, s


class StarTransformerClassifier(nn.Module):

    def __init__(self, v_size, cycle_num, hidden_size, num_attention_heads, attention_dropout_prob, label_num=2):
        super().__init__()
        self.star_trm = StarTransformerLayer(cycle_num, hidden_size, num_attention_heads, attention_dropout_prob)
        self.emb = nn.Embedding(v_size, hidden_size)
        self.fc = nn.Linear(hidden_size, label_num)

    def forward(self, x):
        x = self.emb(x)
        h, s = self.star_trm(x)
        h_max = F.avg_pool2d(h, (h.shape[1], 1)).squeeze(1)
        return self.fc(h_max + s)


class StarTransformerTokenClassifier(nn.Module):

    def __init__(self, v_size, cycle_num, hidden_size, num_attention_heads, attention_dropout_prob, label_num):
        super().__init__()
        self.cycle_num = cycle_num
        self.label_num = label_num
        self.emb = nn.Embedding(v_size, hidden_size)
        self.star_trm = StarTransformerLayer(cycle_num, hidden_size, num_attention_heads, attention_dropout_prob)
        self.fc = nn.Linear(hidden_size, label_num)

    def forward(self, x):
        x = self.emb(x)
        h, s = self.star_trm(x)
        return self.fc(h)


class NER(nn.Module):

    def __init__(self, num_tags, char_vocab_size, char_embed_size, input_dropout_rate,
                 cycle_num, num_attention_heads, attention_dropout_prob, fc_dropout_rate,
                 use_seg_embed, seg_vocab_size, seg_embed_size,
                 use_bigram_embed, bigram_vocab_size, bigram_embed_size,
                 use_crf, add_boundary, embed_freeze):
        super(NER, self).__init__()
        self.use_seg_embed = use_seg_embed
        self.use_bigram_embed = use_bigram_embed
        self.use_crf = use_crf
        self.add_boundary = add_boundary

        embed_size = char_embed_size
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_size)

        if use_seg_embed:
            embed_size += seg_embed_size
            self.seg_embedding = nn.Embedding(seg_vocab_size, seg_embed_size)

        if use_bigram_embed:
            embed_size += bigram_embed_size * 2
            self.bigram_embedding = nn.Embedding(bigram_vocab_size, bigram_embed_size)

        self.star_encoder = StarTransformerLayer(cycle_num, embed_size, num_attention_heads, attention_dropout_prob)
        self.star_w = nn.Linear(embed_size, embed_size)

        if add_boundary:
            self.head_gru = nn.GRU(embed_size, embed_size, num_layers=1, bidirectional=True, batch_first=True)
            self.tail_gru = nn.GRU(embed_size, embed_size, num_layers=1, bidirectional=True, batch_first=True)

            self.head_w = nn.Linear(embed_size * 2, embed_size)
            self.tail_w = nn.Linear(embed_size * 2, embed_size)

            self.head_fc = nn.Linear(embed_size * 2, 2)
            self.tail_fc = nn.Linear(embed_size * 2, 2)

        self.o_fc = nn.Linear(embed_size, num_tags)

        self.in_dropout = nn.Dropout(input_dropout_rate)
        self.fc_dropout = nn.Dropout(fc_dropout_rate)

        self.ce_loss = nn.CrossEntropyLoss()

        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)

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

    def forward(self, tokens, masks, heads, tails, seg_tags=None, decode=True, tags=None):
        embed = self.char_embedding(tokens[:, :, 0])
        if self.use_bigram_embed:
            w_emb = torch.cat([self.bigram_embedding(tokens[:, :, i]) for i in range(1, tokens.size()[2])], dim=2)
            embed = torch.cat((embed, w_emb), dim=2)
        if self.use_seg_embed:
            s_emb = self.seg_embedding(seg_tags)
            embed = torch.cat((embed, s_emb), dim=2)
        embed = self.in_dropout(embed)

        star_out, _ = self.star_encoder(embed)

        if self.add_boundary:
            head_out, _ = self.head_gru(embed)
            tail_out, _ = self.tail_gru(embed)
            out = self.head_w(head_out) + self.tail_w(tail_out) + self.star_w(star_out)
        else:
            out = star_out

        out = self.fc_dropout(out)
        out = self.o_fc(out)

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
                o_loss = -self.crf(out, tags, masks)
            else:
                out_shape = out.size()
                o_loss = self.ce_loss(out.reshape(out_shape[0] * out_shape[1], out_shape[2]),
                                      tags.reshape(out_shape[0] * out_shape[1]))

            loss = o_loss
            if self.add_boundary:
                head_o = self.head_fc(head_out)
                tail_o = self.tail_fc(tail_out)

                head_shape = head_o.size()
                tail_shape = tail_o.size()
                head_loss = self.ce_loss(head_o.reshape(head_shape[0] * head_shape[1], head_shape[2]),
                                         heads.reshape(head_shape[0] * head_shape[1]))
                tail_loss = self.ce_loss(tail_o.reshape(tail_shape[0] * tail_shape[1], tail_shape[2]),
                                         tails.reshape(tail_shape[0] * tail_shape[1]))

                loss = o_loss + head_loss + tail_loss

            return loss
