import math
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF


class RelativeSinusoidalPositionalEmbedding(nn.Module):
    """
    This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        """
        :param embedding_dim: Ã¿¸öÎ»ÖÃµÄdimension
        :param padding_idx:
        :param init_size:
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        weights = self.get_embedding(init_size + 1, embedding_dim, padding_idx)
        self.origin_shift = (init_size + 1) // 2 + 1
        self.register_buffer('weights', weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings // 2, num_embeddings // 2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        # self.origin_shift = num_embeddings // 2 + 1
        return emb

    def forward(self, input):
        """
        Input is expected to be of size [bsz x seqlen].
        """
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            # recompute/expand embeddings if needed
            weights = self.get_embedding(
                max_pos * 2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0) // 2
            self.register_buffer('weights', weights)

        positions = torch.arange(-seq_len, seq_len).to(input.device).long() + self.origin_shift  # 2*seq_len
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed


class RelativeMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout, scale=False):
        """
        :param int d_model:
        :param int n_head:
        :param dropout: ¶Ôattention mapµÄdropout
        :param r_w_bias: n_head x head_dim or None, Èç¹ûÎªdim
        :param r_r_bias: n_head x head_dim or None,
        :param scale:
        :param rel_pos_embed:
        """
        super().__init__()
        self.qkv_linear = nn.Linear(d_model, d_model * 3, bias=False)
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model // n_head, 0, 1200)

        if scale:
            self.scale = math.sqrt(d_model // n_head)
        else:
            self.scale = 1

        self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
        self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))

    def forward(self, x, mask):
        """
        :param x: batch_size x max_len x d_model
        :param mask: batch_size x max_len
        :return:
        """
        batch_size, max_len, d_model = x.size()
        pos_embed = self.pos_embed(mask)  # l x head_dim

        qkv = self.qkv_linear(x)  # batch_size x max_len x d_model3
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)  # b x n x l x d

        rw_head_q = q + self.r_r_bias[:, None]
        AC = torch.einsum('bnqd,bnkd->bnqk', [rw_head_q, k])  # b x n x l x d, nÊÇhead

        D_ = torch.einsum('nd,ld->nl', self.r_w_bias, pos_embed)[None, :, None]  # head x 2max_len, Ã¿¸öhead¶ÔÎ»ÖÃµÄbias
        B_ = torch.einsum('bnqd,ld->bnql', q, pos_embed)  # bsz x head  x max_len x 2max_len£¬Ã¿¸öquery¶ÔÃ¿¸öshiftµÄÆ«ÒÆ
        E_ = torch.einsum('bnqd,ld->bnql', k, pos_embed)  # bsz x head x max_len x 2max_len, key¶ÔrelativeµÄbias
        BD = B_ + D_  # bsz x head x max_len x 2max_len, Òª×ª»»Îªbsz x head x max_len x max_len
        BDE = self._shift(BD) + self._transpose_shift(E_)
        attn = AC + BDE

        attn = attn / self.scale

        attn = attn.masked_fill(mask[:, None, None, :].eq(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, max_len, d_model)  # b x n x l x d

        return v

    def _shift(self, BD):
        """
        ÀàËÆ
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2

        ×ª»»Îª
        0   1  2
        -1  0  1
        -2 -1  0

        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)  # bsz x n_head x (2max_len+1) x max_len
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)  # bsz x n_head x 2max_len x max_len
        BD = BD[:, :, :, max_len:]
        return BD

    def _transpose_shift(self, E):
        """
        ÀàËÆ
          -3   -2   -1   0   1   2
         -30  -20  -10  00  10  20
        -300 -200 -100 000 100 200

        ×ª»»Îª
          0  -10   -200
          1   00   -100
          2   10    000

        :param E: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = E.size()
        zero_pad = E.new_zeros(bsz, n_head, max_len, 1)
        # bsz x n_head x -1 x (max_len+1)
        E = torch.cat([E, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)
        indice = (torch.arange(max_len) * 2 + 1).to(E.device)
        E = E.index_select(index=indice, dim=-2).transpose(-1, -2)  # bsz x n_head x max_len x max_len

        return E


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feedforward_dim, after_norm, dropout):
        """
        :param int d_model: Ò»°ã512Ö®ÀàµÄ
        :param self_attn: self attentionÄ£¿é£¬ÊäÈëÎªx:batch_size x max_len x d_model, mask:batch_size x max_len, Êä³öÎª
            batch_size x max_len x d_model
        :param int feedforward_dim: FFNÖÐ¼ä²ãµÄdimensionµÄ´óÐ¡
        :param bool after_norm: normµÄÎ»ÖÃ²»Ò»Ñù£¬Èç¹ûÎªFalse£¬Ôòembedding¿ÉÒÔÖ±½ÓÁ¬µ½Êä³ö
        :param float dropout: Ò»¹²Èý¸öÎ»ÖÃµÄdropoutµÄ´óÐ¡
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = self_attn

        self.after_norm = after_norm

        self.ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(feedforward_dim, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x, mask):
        """
        :param x: batch_size x max_len x hidden_size
        :param mask: batch_size x max_len, Îª0µÄµØ·½Îªpad
        :return: batch_size x max_len x hidden_size
        """
        residual = x
        if not self.after_norm:
            x = self.norm1(x)

        x = self.self_attn(x, mask)
        x = x + residual
        if self.after_norm:
            x = self.norm1(x)
        residual = x
        if not self.after_norm:
            x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        if self.after_norm:
            x = self.norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, feedforward_dim, dropout, after_norm=True, scale=False):
        super().__init__()
        self.d_model = d_model

        self_attn = RelativeMultiHeadAttn(d_model, n_head, dropout, scale=scale)
        self.layers = nn.ModuleList([EncoderLayer(d_model, deepcopy(self_attn), feedforward_dim, after_norm, dropout)
                                     for _ in range(num_layers)])

    def forward(self, x, mask):
        """
        :param x: batch_size x max_len
        :param mask: batch_size x max_len. ÓÐvalueµÄµØ·½Îª1
        :return:
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class NER(nn.Module):

    def __init__(self, num_tags, char_vocab_size, char_embed_size, input_dropout_rate,
                 num_layers, n_head, head_dims, fc_dropout_rate, attn_dropout_rate, after_norm, attn_scale,
                 use_seg_embed, seg_vocab_size, seg_embed_size,
                 use_bigram_embed, bigram_vocab_size, bigram_embed_size,
                 use_crf, embed_freeze):
        super(NER, self).__init__()
        self.use_seg_embed = use_seg_embed
        self.use_bigram_embed = use_bigram_embed
        self.use_crf = use_crf

        d_model = n_head * head_dims
        ff_dim = int(2 * d_model)

        embed_size = char_embed_size
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_size)

        if use_seg_embed:
            embed_size += seg_embed_size
            self.seg_embedding = nn.Embedding(seg_vocab_size, seg_embed_size)

        if use_bigram_embed:
            embed_size += bigram_embed_size * 2
            self.bigram_embedding = nn.Embedding(bigram_vocab_size, bigram_embed_size)

        self.in_fc = nn.Linear(embed_size, d_model)

        self.encoders = Encoder(num_layers, d_model, n_head, ff_dim, attn_dropout_rate,
                                after_norm=after_norm, scale=attn_scale)

        self.out_fc = nn.Linear(d_model, num_tags)

        self.in_dropout = nn.Dropout(input_dropout_rate)
        self.fc_dropout = nn.Dropout(fc_dropout_rate)

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

        out = self.in_fc(out)
        out = self.encoders(out, masks)
        out = self.fc_dropout(out)
        out = self.out_fc(out)

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
