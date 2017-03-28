#-*- coding:utf-8 -*-

import codecs
import os
import collections
from six.moves import cPickle,reduce,map
import numpy as np

BEGIN_CHAR = '^'
END_CHAR = '$'
UNKNOWN_CHAR = '*'
MAX_LENGTH = 100

class TextLoader():

    def __init__(self, batch_size, max_vocabsize=3000, encoding='utf-8'):
        self.batch_size = batch_size
        self.max_vocabsize = max_vocabsize
        self.encoding = encoding

        data_dir = './data'

        input_file = os.path.join(data_dir, "poems.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        def handle_poem(line):
            line = line.replace(' ', '')
            if len(line) >= MAX_LENGTH:
                index_end = line.rfind(u'。', 0, MAX_LENGTH)
                index_end = index_end if index_end > 0 else MAX_LENGTH
                line = line[:index_end + 1]

            return BEGIN_CHAR + line + END_CHAR

        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            lines = list(map(handle_poem, f.read().strip().split('\n')))

        # 统计单个字出现的频率,计算出来后是个字典
        counter = collections.Counter(reduce(lambda data, line: line + data, lines, ''))
        # 转成list,每个元素是(单字, 频率)的元组
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        # 所有单字的集合
        chars, _ = zip(*count_pairs)
        # 词库的长度,给未知字符留有一个占位符
        self.vocab_size = min(len(chars), self.max_vocabsize - 1) + 1
        # 词库中所有词的集合,包含一个特殊字符
        self.chars = chars[:self.vocab_size - 1] + (UNKNOWN_CHAR,)
        # 按词在 `self.chars` 集合中出现的顺序编号并存入字典
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        # 未知词的编号
        unknown_char_int = self.vocab.get(UNKNOWN_CHAR)
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)

        get_int = lambda char: self.vocab.get(char, unknown_char_int)
        # 对输入排序
        lines = sorted(lines, key=lambda line: len(line))
        # 生成每个输入序列的张量
        self.tensor = [list(map(get_int, line)) for line in lines]
        with open(tensor_file, 'wb') as f:
            cPickle.dump(self.tensor, f)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        with open(tensor_file,'rb') as f:
            self.tensor = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

    def create_batches(self):
        # 批次总数 = 总张量数 / 每批次个数
        self.num_batches = int(len(self.tensor) / self.batch_size)
        # 剔除掉无法形成批次的多富余出来的数据
        self.tensor = self.tensor[:self.num_batches * self.batch_size]
        unknown_char_int = self.vocab.get(UNKNOWN_CHAR)

        self.x_batches = []
        self.y_batches = []

        for i in range(self.num_batches):
            from_index = i * self.batch_size
            to_index = from_index + self.batch_size
            batches  = self.tensor[from_index:to_index]
            # 取本批次中最长序列的长度
            seq_length = max(map(len, batches))
            # 生成一个 batch_size * seq_length 的矩阵,并填充为未知字符的向量值
            xdata = np.full((self.batch_size, seq_length), unknown_char_int, np.int32)
            for row in range(self.batch_size):
                # 用真实的数据填充
                xdata[row,:len(batches[row])] = batches[row]
            ydata = np.copy(xdata)
            # 矩阵行不变,去掉第一列,重复最后一列
            ydata[:,:-1] = xdata[:,1:]

            self.x_batches.append(xdata)
            self.y_batches.append(ydata)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
