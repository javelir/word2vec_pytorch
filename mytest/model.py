import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as Func


class SkipgramModel(nn.Module):
    """ Skipgram model for word2vec
    """

    def __init__(self, emb_size, emb_dim):
        super(SkipgramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.targ_embeddings = nn.Embedding(emb_size, emb_dim, sparse=True)
        self.cont_embeddings = nn.Embedding(emb_size, emb_dim, sparse=True)

    def init_emb(self):
        """ Init embeddings """
        init_range = 0.5 / self.emb_size
        self.targ_embeddings.weight.data.uniform_(-init_range, init_range)
        self.cont_embeddings.weight.data.uniform_(0, 0)

    def forward(self, targ, cont, negs=None):
        """ Forward pass """
        targ_ems = self.targ_embeddings(targ)
        cont_ems = self.cont_embeddings(cont)
        score = torch.mul(targ_ems, cont_ems).squeeze()
        score = torch.sum(score, dim=1)
        score = Func.logsigmoid(score)
        if not negs:
            return -1 * torch.sum(score)
        neg_ems = self.cont_embeddings(negs)
        neg_score = torch.bmm(neg_embs, targ_embs.unsqueeze(2)).squeeze()
        neg_score = Func.logsigmoid(-1 * neg_score)
        return -1 * (torch.sum(score) + torch.sum(neg_score))

    def save(self, id2word, filename, use_cuda=False):
        """ Save data to disk """
        if use_cuda:
            embeddings = self.tag_embeddings.weight.cpu().data.numpy()
        else:
            embeddings = self.targ_embeddings.weight.data.numpy()
        with open(filename, "w+") as fout:
            fout.write("%d %d\n" % (self.emb_size, self.emb_dim))
            for wid, word in id2word.items():
                emb = embeddings[wid]
                emb_line = " ".join(map(lambda x: str(x), emb))
                fout.write("%d %s %s\n"  % (wid, word, emb_line))


def test():
    """ Test function """
    model = SkipgramModel(100, 100)
    id2word = dict()
    for i in range(100):
        id2word[i] = "word" + str(i)
    model.save(id2word, filename="./output_emb.txt")


if __name__ == "__main__":
    test()
