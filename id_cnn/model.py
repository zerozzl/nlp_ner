import torch
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_block_layer, dropout_rate):
        super(DilatedConvBlock, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                       padding=1, dilation=1)] +
            [nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                       padding=2 ** (l + 1), dilation=2 ** (l + 1)) for l in range(num_block_layer - 2)] +
            [nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                       padding=1, dilation=1)]
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = x
        for conv in self.convs:
            out = conv(out)
        out = self.dropout(out)
        return out


class NER(nn.Module):
    def __init__(self, num_tags, char_vocab_size, char_embed_size, channel_size, kernel_size, num_block_layer,
                 num_block, input_dropout_rate, hidden_dropout_rate,
                 use_seg_embed, seg_vocab_size, seg_embed_size,
                 use_bigram_embed, bigram_vocab_size, bigram_embed_size,
                 use_crf, embed_freeze):
        super(NER, self).__init__()
        self.num_block = num_block
        self.use_seg_embed = use_seg_embed
        self.use_bigram_embed = use_bigram_embed
        self.use_crf = use_crf

        input_size = char_embed_size
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_size)

        if use_bigram_embed:
            input_size += bigram_embed_size * 2
            self.bigram_embedding = nn.Embedding(bigram_vocab_size, bigram_embed_size)

        if use_seg_embed:
            input_size += seg_embed_size
            self.seg_embedding = nn.Embedding(seg_vocab_size, seg_embed_size)

        self.conv0 = nn.Conv1d(in_channels=input_size, out_channels=channel_size, kernel_size=kernel_size,
                               padding=1, dilation=1)
        self.block = DilatedConvBlock(channel_size, channel_size, kernel_size, num_block_layer, hidden_dropout_rate)
        self.linear = nn.Linear(channel_size, num_tags)

        self.dropout = nn.Dropout(input_dropout_rate)

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

    def forward(self, x, masks, seg_tags=None, decode=True, tags=None):
        out = self.char_embedding(x[:, :, 0])
        if self.use_bigram_embed:
            w_emb = torch.cat([self.bigram_embedding(x[:, :, i]) for i in range(1, x.size()[2])], dim=2)
            out = torch.cat((out, w_emb), dim=2)
        if self.use_seg_embed:
            s_emb = self.seg_embedding(seg_tags)
            out = torch.cat((out, s_emb), dim=2)
        out = self.dropout(out)

        out = out.permute(0, 2, 1)
        out = self.conv0(out)
        for _ in range(self.num_block):
            out = self.block(out)
        out = out.permute(0, 2, 1)

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
