import sys
sys.path.append("..")
from pytorch_pretrained_bert.modeling import BertEmbeddings
from pytorch_pretrained_bert import BertForMaskedLM
from pytorch_pretrained_bert import BertModel, BertOnlyMLMHead
import torch
from torch import nn

class my_BertEmbeddings(BertEmbeddings):
    def __init__(self,config):
        super(BertEmbeddings, self).__init__(config)
        self.word_embed_tensor = self.word_embeddings(torch.arange(config.vocab_size))
    def forward(self, input_weight_tensor, token_type_ids=None):
        # input_weight_list format: [[list of weights for word 1],...,[list of weights for word N]]
        seq_length = input_weight_list.size(0)
        position_ids = torch.arange(seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(seq_length)
        if token_type_ids is None:
            token_type_ids = torch.zeros(seq_length)
        word_embs = []
        for word_weight_tensor in input_weight_tensor:
            word_emb = torch.mean(self.word_embed_tensor*word_weight_tensor,0)
            word_embs.append(word_emb)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        words_embeddings = torch.stack(word_embs)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class my_BertModel(BertModel):
    def __init__(self,config):
        super(my_BertModel, self).__init__(config)
        self.embeddings = my_BertEmbeddings(config)

class my_BertForMaskedLM(BertForMaskedLM):
    def __init__(self,config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = my_BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)