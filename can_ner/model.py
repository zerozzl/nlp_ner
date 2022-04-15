import numpy as np
import torch
from torch import nn
from torch.nn.utils import rnn
import torch.nn.functional as F
from torchcrf import CRF


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def score(self, ht, hs):
        re = self.w1(ht)
        rc = self.w2(hs)
        re = re.unsqueeze(1)
        out = torch.tanh(re + rc)
        out = self.v(out)
        return out

    def forward(self, entities, contexts):
        scores = self.score(entities, contexts)
        weights = F.softmax(scores, dim=1)
        out = torch.mul(weights, contexts)
        return out


class LocalConvolutionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, window_size, use_pos_embed, use_cpu):
        super(LocalConvolutionLayer, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.use_pos_embed = use_pos_embed
        self.use_cpu = use_cpu

        feature_dim = input_dim
        if use_pos_embed:
            feature_dim += window_size
        self.attn = AttentionLayer(feature_dim, hidden_dim)
        self.cnn = nn.Linear(feature_dim * window_size, hidden_dim)

    def forward(self, embeddings, masks):
        batch_n, inp_len, feature_dim = embeddings.size()

        cnn_out = []
        for i in range(inp_len):
            entities, contexts, _ = self.get_attn_inp(embeddings, masks, i, batch_n, inp_len)
            out = self.attn(entities, contexts)
            out = out.reshape(batch_n, -1)
            out = self.cnn(out)
            cnn_out.append(out)
        cnn_out = torch.stack(cnn_out, 1)
        return cnn_out

    def get_attn_inp(self, embeddings, masks, idx, batch_n, inp_len):
        begin = idx - self.window_size // 2
        end = idx + self.window_size // 2 + 1

        emb_pad_left = None
        emb_pad_right = None
        mask_pad_left = None
        mask_pad_right = None
        if begin < 0:
            emb_pad_left = torch.zeros(batch_n, -begin, self.input_dim)
            mask_pad_left = torch.zeros(batch_n, -begin)

            emb_pad_left = emb_pad_left.cpu() if self.use_cpu else emb_pad_left.cuda()
            mask_pad_left = mask_pad_left.cpu() if self.use_cpu else mask_pad_left.cuda()
        if end > inp_len:
            emb_pad_right = torch.zeros(batch_n, end - inp_len, self.input_dim)
            mask_pad_right = torch.zeros(batch_n, end - inp_len)

            emb_pad_right = emb_pad_right.cpu() if self.use_cpu else emb_pad_right.cuda()
            mask_pad_right = mask_pad_right.cpu() if self.use_cpu else mask_pad_right.cuda()

        idx = idx - begin
        begin = 0 if begin < 0 else begin
        end = inp_len if end > inp_len else end
        contexts = embeddings[:, begin: end, :]
        mask = masks[:, begin: end]

        if emb_pad_left is not None:
            contexts = torch.cat((emb_pad_left, contexts), dim=1)
            mask = torch.cat((mask_pad_left, mask), dim=1)
        if emb_pad_right is not None:
            contexts = torch.cat((contexts, emb_pad_right), dim=1)
            mask = torch.cat((mask, mask_pad_right), dim=1)

        if self.use_pos_embed:
            pos_emb = torch.eye(self.window_size).repeat(batch_n, 1).reshape(batch_n, self.window_size,
                                                                             self.window_size)
            pos_emb = pos_emb.cpu() if self.use_cpu else pos_emb.cuda()
            contexts = torch.cat((contexts, pos_emb), dim=2)

        entities = contexts[:, idx, :]

        return entities, contexts, mask


class NER(nn.Module):
    def __init__(self, num_tags, char_vocab_size, char_embed_size, input_dropout_rate,
                 hidden_dim, window_size, hidden_dropout_rate,
                 use_seg_embed, seg_vocab_size, seg_embed_size,
                 use_bigram_embed, bigram_vocab_size, bigram_embed_size,
                 use_pos_embed, use_rnn, use_crf, embed_freeze, use_cpu):
        super(NER, self).__init__()
        self.use_seg_embed = use_seg_embed
        self.use_bigram_embed = use_bigram_embed
        self.use_rnn = use_rnn
        self.use_crf = use_crf

        embed_size = char_embed_size
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_size)

        if use_seg_embed:
            embed_size += seg_embed_size
            self.seg_embedding = nn.Embedding(seg_vocab_size, seg_embed_size)

        if use_bigram_embed:
            embed_size += bigram_embed_size * 2
            self.bigram_embedding = nn.Embedding(bigram_vocab_size, bigram_embed_size)

        self.cnn = LocalConvolutionLayer(embed_size, hidden_dim, window_size, use_pos_embed, use_cpu)

        if use_rnn:
            self.rnn = torch.nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1,
                                    bidirectional=True, batch_first=True)
        self.attn = AttentionLayer(hidden_dim, hidden_dim)
        self.liner = nn.Linear(hidden_dim, num_tags)

        self.cnn_active = torch.nn.LeakyReLU(0.01)
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

    def forward(self, tokens, masks, sents_len, seg_tags=None, decode=True, tags=None):
        inp_len = np.max(sents_len)

        tokens = tokens[:, :inp_len, :]
        masks = masks[:, :inp_len]
        if seg_tags is not None:
            seg_tags = seg_tags[:, :inp_len]
        if tags is not None:
            tags = tags[:, :inp_len]

        embeddings = self.char_embedding(tokens[:, :, 0])
        if self.use_bigram_embed:
            w_emb = torch.cat([self.bigram_embedding(tokens[:, :, i]) for i in range(1, tokens.size()[2])], dim=2)
            embeddings = torch.cat((embeddings, w_emb), dim=2)
        if self.use_seg_embed:
            s_emb = self.seg_embedding(seg_tags)
            embeddings = torch.cat((embeddings, s_emb), dim=2)
        embeddings = self.in_dropout(embeddings)

        contexts = self.cnn(embeddings, masks)
        contexts = self.cnn_active(contexts)
        contexts = self.hid_dropout(contexts)

        if self.use_rnn:
            # contexts = rnn.pack_padded_sequence(contexts, sents_len, batch_first=True)
            contexts, _ = self.rnn(contexts)
            # contexts, _ = rnn.pad_packed_sequence(contexts, batch_first=True)

        attn_out = []
        for i in range(inp_len):
            entities = contexts[:, i, :]
            out = self.attn(entities, contexts)
            out = torch.sum(out, dim=1)
            attn_out.append(out)
        attn_out = torch.stack(attn_out, 1)
        attn_out = attn_out * masks.unsqueeze(2).int()
        attn_out = self.hid_dropout(attn_out)

        out = self.liner(attn_out)

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
