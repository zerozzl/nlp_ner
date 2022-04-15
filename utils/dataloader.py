import codecs
import numpy as np
import jieba
from torch.utils.data import Dataset

TOKEN_PAD = '[PAD]'
TOKEN_UNK = '[UNK]'
TOKEN_CLS = '[CLS]'
TOKEN_SEP = '[SEP]'
TOKEN_EDGES_START = '<s>'
TOKEN_EDGES_END = '</s>'
SEG_TAG_TO_ID = {'B': 0, 'M': 1, 'E': 2, 'S': 3}


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def read_from_file(self, data_path, max_len=0,
                       do_pad=False, pad_token=TOKEN_PAD,
                       do_to_id=False, tag_to_id=None, char_tokenizer=None,
                       for_bert=False, do_sort=False,
                       add_bigram_feature=False, bigram_tokenizer=None,
                       debug=False):
        self.data = []
        sent = []
        ner_tag = []
        seg_tag = []
        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    sent, ner_tag, seg_tag, mask, sent_len, head, tail = self.process_sent(sent, ner_tag, seg_tag,
                                                                                           max_len,
                                                                                           do_pad, pad_token,
                                                                                           do_to_id, tag_to_id,
                                                                                           char_tokenizer,
                                                                                           for_bert,
                                                                                           add_bigram_feature,
                                                                                           bigram_tokenizer)

                    self.data.append([sent, ner_tag, seg_tag, mask, sent_len, head, tail])
                    sent = []
                    ner_tag = []
                    seg_tag = []

                    if debug:
                        if len(self.data) >= 10:
                            break
                else:
                    word, ne, seg = line.split()
                    sent.append(word)
                    ner_tag.append(ne)
                    seg_tag.append(seg)

            if len(sent) > 0:
                sent, ner_tag, seg_tag, mask, sent_len, head, tail = self.process_sent(sent, ner_tag, seg_tag, max_len,
                                                                                       do_pad, pad_token,
                                                                                       do_to_id, tag_to_id,
                                                                                       char_tokenizer,
                                                                                       for_bert,
                                                                                       add_bigram_feature,
                                                                                       bigram_tokenizer)

                self.data.append([sent, ner_tag, seg_tag, mask, sent_len, head, tail])

        if do_sort:
            self.data = sorted(self.data, key=lambda x: x[4], reverse=True)

    def process_sent(self, sent, ner_tag, seg_tag, max_len, do_pad, pad_token,
                     do_to_id, tag_to_id, char_tokenizer, for_bert,
                     add_bigram_feature, bigram_tokenizer):
        sent_len = len(sent)
        sent_len = max_len if sent_len > max_len else sent_len

        if for_bert:
            sent = [TOKEN_CLS] + sent
            ner_tag = ['O'] + ner_tag
            seg_tag = ['S'] + seg_tag

        if do_pad:
            mask = [1] * len(sent[:max_len]) + [0] * (max_len - len(sent))
            sent = sent[:max_len]

            if add_bigram_feature and (len(sent) < max_len):
                sent += [TOKEN_EDGES_END] + [pad_token] * (max_len - len(sent) - 1)
            else:
                sent += [pad_token] * (max_len - len(sent))

            ner_tag = ner_tag[:max_len] + ['O'] * (max_len - len(ner_tag))
            seg_tag = seg_tag[:max_len] + ['S'] * (max_len - len(seg_tag))
        else:
            mask = [1] * len(sent)

        if add_bigram_feature:
            sent = [TOKEN_EDGES_START] + sent + [TOKEN_PAD]
            sent = [[sent[i]] + [sent[i - 1] + sent[i]] + [sent[i] + sent[i + 1]] for i in
                    range(1, len(sent) - 1)]

        head = [1 if (tag[0] == 'B' or tag[0] == 'S') else 0 for tag in ner_tag]
        tail = [1 if (tag[0] == 'E' or tag[0] == 'S') else 0 for tag in ner_tag]

        if do_to_id:
            if add_bigram_feature:
                sent = np.array(sent)
                uni_id = char_tokenizer.convert_tokens_to_ids(sent[:, 0])
                bi_id = bigram_tokenizer.convert_tokens_to_ids(sent[:, 1:])
                sent = np.concatenate((uni_id, bi_id), axis=1).tolist()
            else:
                sent = char_tokenizer.convert_tokens_to_ids(sent)
            ner_tag = [tag_to_id.get(tag) for tag in ner_tag]
            seg_tag = [SEG_TAG_TO_ID.get(tag) for tag in seg_tag]

        return sent, ner_tag, seg_tag, mask, sent_len, head, tail


class WeiboDataset(BaseDataset):
    def __init__(self, data_path, max_len=0,
                 do_pad=False, pad_token=TOKEN_PAD,
                 do_to_id=False, tag_to_id=None, char_tokenizer=None,
                 for_bert=False, do_sort=False,
                 add_bigram_feature=False, bigram_tokenizer=None,
                 debug=False):
        self.read_from_file(data_path, max_len, do_pad, pad_token, do_to_id, tag_to_id, char_tokenizer,
                            for_bert, do_sort, add_bigram_feature, bigram_tokenizer, debug)

    @staticmethod
    def get_label_type():
        return 'BIO'

    @staticmethod
    def transform(src_path, tgt_path):
        sents = []
        sent = []
        with codecs.open(src_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line.strip() == "":
                    if len(sent) > 0:
                        sent = WeiboDataset.transform_sent(sent)
                        sents.append(sent)
                        sent = []
                else:
                    word, tag = line.split()
                    word, segTag = word[:-1], int(word[-1:])
                    sent.append([word, tag, segTag])

            if len(sent) > 0:
                sent = WeiboDataset.transform_sent(sent)
                sents.append(sent)

        with codecs.open(tgt_path, 'w', 'utf-8') as fout:
            for sent in sents:
                for item in sent:
                    fout.write('%s\t%s\t%s\n' % (item[0], item[1], item[2]))
                fout.write('\n')

    @staticmethod
    def transform_sent(sent):
        for i in range(len(sent)):
            item = sent[i]
            item_prev = None if i == 0 else sent[i - 1]
            item_back = None if (i == len(sent) - 1) else sent[i + 1]

            if item_prev is None:
                if item[2] == 0 and item_back[2] == 0:
                    item[2] = 'S'
                else:
                    item[2] = 'B'
                continue

            if item_back is None:
                if item[2] == 0:
                    item[2] = 'S'
                else:
                    item[2] = 'E'
                continue

            if item[2] == 0:
                if item_back[2] == 0:
                    item[2] = 'S'
                else:
                    item[2] = 'B'
            else:
                if item_back[2] == 0:
                    item[2] = 'E'
                else:
                    item[2] = 'M'
        return sent


class ResumeDataset(BaseDataset):
    def __init__(self, data_path, max_len=0,
                 do_pad=False, pad_token=TOKEN_PAD,
                 do_to_id=False, tag_to_id=None, char_tokenizer=None,
                 for_bert=False, do_sort=False,
                 add_bigram_feature=False, bigram_tokenizer=None,
                 debug=False):
        self.read_from_file(data_path, max_len, do_pad, pad_token, do_to_id, tag_to_id, char_tokenizer,
                            for_bert, do_sort, add_bigram_feature, bigram_tokenizer, debug)

    @staticmethod
    def get_label_type():
        return 'BMES'

    @staticmethod
    def transform(src_path, tgt_path):
        sents = []
        sent = []
        with codecs.open(src_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line.strip() == "":
                    if len(sent) > 0:
                        sent = ResumeDataset.transform_sent(sent)
                        sents.append(sent)
                        sent = []
                else:
                    word, tag = line.split()
                    sent.append([word, tag])

            if len(sent) > 0:
                sent = ResumeDataset.transform_sent(sent)
                sents.append(sent)

        with codecs.open(tgt_path, 'w', 'utf-8') as fout:
            for sent in sents:
                for item in sent:
                    fout.write('%s\t%s\t%s\n' % (item[0], item[1], item[2]))
                fout.write('\n')

    @staticmethod
    def transform_sent(sent):
        text = ''.join([item[0] for item in sent])
        words = jieba.cut(text)

        seg_tags = []
        for word in words:
            if len(word) == 1:
                seg_tags.append('S')
            else:
                seg_tags.extend(['B'] + ['M'] * (len(word) - 2) + ['E'])
        assert len(sent) == len(seg_tags)

        sent_with_tag = []
        for i in range(len(sent)):
            sent_with_tag.append([sent[i][0], sent[i][1], seg_tags[i]])
        return sent_with_tag


class MSRADataset(BaseDataset):
    def __init__(self, data_path, max_len=0,
                 do_pad=False, pad_token=TOKEN_PAD,
                 do_to_id=False, tag_to_id=None, char_tokenizer=None,
                 for_bert=False, do_sort=False,
                 add_bigram_feature=False, bigram_tokenizer=None,
                 debug=False):
        self.read_from_file(data_path, max_len, do_pad, pad_token, do_to_id, tag_to_id, char_tokenizer,
                            for_bert, do_sort, add_bigram_feature, bigram_tokenizer, debug)

    @staticmethod
    def get_label_type():
        return 'BMES'

    @staticmethod
    def transform_train(src_path, tgt_path):
        sents = []
        with codecs.open(src_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line.strip() != "":
                    line = line.split()
                    sent = []
                    for item in line:
                        item = item.split('/')
                        item = MSRADataset.transform_train_sent(item[0], item[1])
                        sent.extend(item)
                    sents.append(sent)

        with codecs.open(tgt_path, 'w', 'utf-8') as fout:
            for sent in sents:
                for item in sent:
                    fout.write('%s\t%s\t%s\n' % (item[0], item[1], item[2]))
                fout.write('\n')

    @staticmethod
    def transform_train_sent(word, ne):
        tokens = [w for w in word]

        seg_tag = []
        if len(word) == 1:
            seg_tag.append('S')
        else:
            seg_tag.extend(['B'] + ['M'] * (len(word) - 2) + ['E'])

        ne_tag = []
        if ne == 'o':
            ne_tag = ['O'] * len(word)
        else:
            if len(word) == 1:
                ne_tag.append('S-' + ne)
            else:
                ne_tag.extend(['B-' + ne] + ['M-' + ne] * (len(word) - 2) + ['E-' + ne])

        assert len(tokens) == len(ne_tag) == len(seg_tag)
        word_with_tag = []
        for i in range(len(tokens)):
            word_with_tag.append([tokens[i], ne_tag[i], seg_tag[i]])
        return word_with_tag

    @staticmethod
    def transform_test(src_path, tgt_path):
        sents = []
        with codecs.open(src_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line.strip() != "":
                    line = line.split()
                    sent = []
                    for item in line:
                        item = item.split('/')
                        item = MSRADataset.transform_test_sent(item[0], item[1])
                        sent.extend(item)
                    sents.append(sent)

        with codecs.open(tgt_path, 'w', 'utf-8') as fout:
            for sent in sents:
                for item in sent:
                    fout.write('%s\t%s\t%s\n' % (item[0], item[1], item[2]))
                fout.write('\n')

    @staticmethod
    def transform_test_sent(text, ne):
        words = jieba.cut(text)

        tokens = []
        seg_tags = []
        for word in words:
            tokens.extend([w for w in word])
            if len(word) == 1:
                seg_tags.append('S')
            else:
                seg_tags.extend(['B'] + ['M'] * (len(word) - 2) + ['E'])
        assert len(text) == len(seg_tags)

        ne_tag = []
        if ne == 'o':
            ne_tag = ['O'] * len(text)
        else:
            if len(text) == 1:
                ne_tag.append('S-' + ne)
            else:
                ne_tag.extend(['B-' + ne] + ['M-' + ne] * (len(text) - 2) + ['E-' + ne])

        sent_with_tag = []
        for i in range(len(tokens)):
            sent_with_tag.append([tokens[i], ne_tag[i], seg_tags[i]])
        return sent_with_tag


class Tokenizer:
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    def convert_tokens_to_ids(self, tokens, unk_token=TOKEN_UNK):
        ids = []
        for token in tokens:
            if isinstance(token, str):
                ids.append([self.token_to_id.get(token, self.token_to_id[unk_token])])
            else:
                ids.append([self.token_to_id.get(t, self.token_to_id[unk_token]) for t in token])
        return ids

    def convert_ids_to_tokens(self, ids, max_len):
        tokens = [self.id_to_token[i] for i in ids]
        if max_len > 0:
            tokens = tokens[:max_len]
        return tokens


def build_tag_dict(src_path, tgt_path):
    tag_set = set()
    with codecs.open(src_path, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line.strip() != "":
                word, tag, seg = line.split()
                tag_set.add(tag)

    with codecs.open(tgt_path, 'w', 'utf-8') as fout:
        for tag in tag_set:
            fout.write('%s\n' % tag)


def load_tag_dict(file_path):
    tag_to_id = {}
    with codecs.open(file_path, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line != '':
                tag_to_id[line] = len(tag_to_id)

    id_to_tag = {v: k for k, v in tag_to_id.items()}
    return tag_to_id, id_to_tag


def convert_ids_to_tags(id_to_tag, ids, max_len):
    tags = [id_to_tag[i] for i in ids]
    if max_len > 0:
        tags = tags[:max_len]
    return tags


def load_pretrain_embedding(filepath, add_pad=False, pad_token=TOKEN_PAD, add_unk=False, unk_token=TOKEN_UNK,
                            debug=False):
    with codecs.open(filepath, 'r', 'utf-8', errors='ignore') as fin:
        token_to_id = {}
        embed = []

        first_line = fin.readline().strip().split()
        embed_size = len(first_line) - 1

        if add_pad:
            token_to_id[pad_token] = len(token_to_id)
            embed.append([0.] * embed_size)

        if add_unk:
            token_to_id[unk_token] = len(token_to_id)
            embed.append([0.] * embed_size)

        token_to_id[first_line[0]] = len(token_to_id)
        embed.append([float(x) for x in first_line[1:]])

        for line in fin:
            if debug:
                if len(token_to_id) >= 1000:
                    break

            line = line.split()

            if len(line) != embed_size + 1:
                continue
            if line[0] in token_to_id:
                continue

            token_to_id[line[0]] = len(token_to_id)
            embed.append([float(x) for x in line[1:]])

    return token_to_id, embed


def stat_info(file_path):
    sents = []
    sent = []
    with codecs.open(file_path, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line.strip() == "":
                if len(sent) > 0:
                    sents.append(sent)
                    sent = []
            else:
                word, tag, seg = line.split()
                sent.append([word, tag, seg])

        if len(sent) > 0:
            sents.append(sent)

    max_sent_len = 0
    entity_num = 0
    len_map = {50: 0, 100: 0, 150: 0, 200: 0, 300: 0, 400: 0, 500: 0, 1000: 0}
    for sent in sents:
        if len(sent) > max_sent_len:
            max_sent_len = len(sent)
        for item in sent:
            if item[1][0] == 'B':
                entity_num += 1

        for l in len_map:
            if len(sent) < l:
                len_map[l] = len_map[l] + 1
                break

    print('Total sentence num: %s, entity num: %s, max length: %s' % (len(sents), entity_num, max_sent_len))
    print('Length: %s' % len_map)
