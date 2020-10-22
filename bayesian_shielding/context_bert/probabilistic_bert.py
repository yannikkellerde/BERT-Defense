import sys
sys.path.append("..")
from pytorch_pretrained_bert.modeling import BertEmbeddings,BertOnlyMLMHead
from pytorch_pretrained_bert import BertForMaskedLM,BertModel
import torch
from torch import nn
import pickle

class my_BertEmbeddings(BertEmbeddings):
    def __init__(self,config):
        super(my_BertEmbeddings, self).__init__(config)
        self.vocab_size = config.vocab_size
    def forward(self, input_weight_tensor, token_type_ids=None):
        word_embed_tensor = self.word_embeddings(torch.arange(self.vocab_size,dtype=torch.long).to(self.device))
        seq_length = input_weight_tensor.size(1)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_weight_tensor.size(0),input_weight_tensor.size(1),dtype=torch.long).to(self.device)
        position_ids = torch.arange(seq_length, dtype=torch.long).to(self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_type_ids)
        all_embs = []
        for one in input_weight_tensor:
            word_embs = []
            for i,word_weight_tensor in enumerate(one):
                word_emb = torch.sum((word_embed_tensor.transpose(1,0)*word_weight_tensor).transpose(1,0),0)
                word_embs.append(word_emb)
            all_embs.append(torch.stack(word_embs))

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        words_embeddings = torch.stack(all_embs).to(self.device)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings.to(self.device)

class my_BertModel(BertModel):
    def __init__(self,config):
        super(my_BertModel, self).__init__(config)
        self.embeddings = my_BertEmbeddings(config)
    def forward(self, input_weight_tensor, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones(input_weight_tensor.size(0),input_weight_tensor.size(1),dtype=torch.long).to(self.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_weight_tensor.size(0),input_weight_tensor.size(1),dtype=torch.long).to(self.device)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_weight_tensor, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

class my_BertForMaskedLM(BertForMaskedLM):
    def __init__(self,config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = my_BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)