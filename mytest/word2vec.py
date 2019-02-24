""" word2vec complete process
"""

import sys
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm


from model import SkipgramModel
from input_data import InputData


class Word2Vec:
    """ Word2vec complete process
    """

    def __init__(self, infile, outfile, emb_dim=100, batch_size=128,
                 window_size=5, epochs=5, initial_lr=1, min_count=5):
        self.data = InputData(infile, min_count)
        self.outfile = outfile
        self.emb_size = len(self.data.id2word)
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.window_size = window_size
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.wv_model = SkipgramModel(self.emb_size, self.emb_dim)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.wv_model.cuda()
        self.optimizer = optim.SGD(self.wv_model.parameters(), lr=self.initial_lr)

    def train(self, use_neg=False):
        """ Train word2vec """
        pair_count = self.data.estimate_pair_count(self.window_size)
        batch_count = self.epochs * pair_count / self.batch_size
        process_bar = tqdm(range(int(batch_count)))
        for idx in process_bar:
            pos_pairs = self.data.get_batch_pairs(self.batch_size, self.window_size)
            targs = [x[0] for x in pos_pairs]
            conts = [x[1] for x in pos_pairs]
            if use_neg:
                negs = self.data.get_neg_pairs(pos_pairs, self.window_size)
            else:
                negs = None
            targs = Variable(torch.LongTensor(targs))
            conts = Variable(torch.LongTensor(conts))
            if use_neg:
                negs = Variable(torch.LongTensor(negs))
            if self.use_cuda:
                targs = targs.cuda()
                conts = conts.cuda()
                if use_neg:
                    negs = negs.cuda()
            self.optimizer.zero_grad()
            loss = self.wv_model.forward(targs, conts, negs)
            loss.backward()
            self.optimizer.step()

            process_bar.set_description(
                "Loss: %0.8f, lr: %0.6f" %
                #(loss.data[0], self.optimizer.param_groups[0]['lr']))
                (loss.data.item(), self.optimizer.param_groups[0]['lr']))

            if idx * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * idx / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        self.wv_model.save(
            self.data.id2word, self.outfile, self.use_cuda)


if __name__ == '__main__':
    w2v = Word2Vec(infile=sys.argv[1], outfile=sys.argv[2])
    w2v.train()
