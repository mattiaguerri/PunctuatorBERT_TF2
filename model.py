
from torch import nn

# from pytorch_pretrained_bert import BertForMaskedLM

# from transformers import BertForMaskedLM

from transformers import CamembertForMaskedLM


class BertPunc(nn.Module):
    
    def __init__(self, vocabSize, segment_size, output_size, dropout):
        super(BertPunc, self).__init__()

        # self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        # self.bert = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
        self.bert = CamembertForMaskedLM.from_pretrained('camembert-base')

        self.bert_vocab_size = vocabSize

        self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.bert(x)[0]
        x = x.view(x.shape[0], -1)
        x = self.fc(self.dropout(self.bn(x)))
        return x
