from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_util import config
from numpy import random

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

### Initializations

# Initialize LSTM weights
def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

# Initialize embedding layer weights (and not use pretrained weights) with normal distribution
def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

# Initialize embedding layer weights (and not use pretrained weights) with uniform distribution
def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

# Initialize Linear layer weights 
def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


# Class Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim) # 50000 * 128, ie 50,000 embeddings each of dimension emb_dim,128
        init_wt_normal(self.embedding.weight)
        
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
    
    def forward(self, input, seq_lens): # seq_lens in descending order input (8,400) seq_length is (1,8)
        # print("IN ENCODER")
        embedded = self.embedding(input) # input_dimensions(b , max_seq_len (=t_k = 400)) x emb_dim = 128, note: max_seq_len is a list [] with indices of the corresponding word in the vocabulary
        # print("Embedded is", embedded.size())
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True) # b x variable_seq_len = sum of all var lens x emb_dim
        # print("Packed is", packed[0].size()) #(3059, 128)
        output, hidden = self.lstm(packed) 
        # output, tuple: b * variable_seq_length(=3059) x 2* hidden_dim, (3059, 2*256)
        # hidden: (h_n,c_n) of (layers(=1)* no_of_dir(=2), b, hidden_dim) for the FINAL token t = seq_len = (2, 8, 256)
        # print("output is ", output[0].size(), "and hidden is a tuple of (h_n, c_n) of size ", hidden[0].size())
        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True) #b x max_seq_len (=t_k) x 2* hidden_dim = (8, 400, 512)
        # print("Padded output encoder_outputs is ", encoder_outputs.size())
        encoder_outputs = encoder_outputs.contiguous()

        encoder_feature = encoder_outputs.view(-1, 2*config.hidden_dim)  # b * max_seq_len (=t_k) x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature) # b * max_seq_len (=t_k) x 2*hidden_dim = (3200, 512)
        # print("encoder_feature is ", encoder_feature)
        return encoder_outputs, encoder_feature, hidden

# Class ReduceState (between encoder and decoder) to handle the dimensions of the encoder features (hidden weights)
class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim


# Class Attention
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        # print("IN ATTENTION")
        # Attention
        # eti = vT tanh(Whhi +Wsst +battn) where hi: encoder hidden state s_t_1, st: decoder state s_t_hat
        # at =softmax(et)
        b, t_k, n = list(encoder_outputs.size())  # (8, 400, 512)

        ## To find Ws (of Wsst) part of the eti 
        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        # print("dec_fea is", dec_fea.size()) #(8, 512)
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim
        # print("dec_fea_expanded is", dec_fea_expanded.size()) #(3200, 512)
        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        # print("att_features is", att_features.size()) # att_features is (3200, 512), attention for all the 400 words of 8 batches based on the first decoding output

        ## eti=vTtanh(Whhi+Wsst+wccti+battn), ie add coverage_feature
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature
        
        e = F.tanh(att_features) # B * t_k x 2*hidden_dim
        # find the weight vT in eti calculation
        scores = self.v(e)  # B * t_k x 1
        # print("scores is before", scores.size()) # (3200, 1)
        scores = scores.view(-1, t_k)  # B x t_k
        # print("scores is after", scores.size()) # (8, 400)

        # convert to probability distribution ie at =softmax(et)
        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor
        # print("attn_dist before", attn_dist.size()) # (8, 400)

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k, (8, 1, 400)
        # print("attn_dist for the matrix mult with encoder_output)", attn_dist.size())

        # Find context ht* =sum(atihi)
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        # print("c_t is before", c_t.size()) # (8, 1, 512)
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim
        # print("c_t is after", c_t.size()) # (8, 512)

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k, change it back after multiplication to its original form
        # print("attn_dist is", attn_dist.size()) #  (8, 400)


        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist
        # print("c_t is ", c_t.size(), "attn_dist is ", attn_dist.size()) # c_t is  (8, 512) attn_dist is  (8, 400)
        return c_t, attn_dist, coverage
            
# Class Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)
    
    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask, c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):
        # print("IN DECODER")
        
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1 # weights of encoder 
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                              enc_padding_mask, coverage)
            coverage = coverage_next
        
        # Decoder target embedding
        y_t_1_embd = self.embedding(y_t_1) 
        # print("y_t_1 is ", y_t_1.size() , "y_t_1_emb is ", y_t_1_embd.size())  # y_t_1 is (8,) and y_t_1_emb (8, 128)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        # print("x is ", x.size()) # (8, 128) 

        # Decoder hidden states, lstm_out IS used . (8, 128) ---> (8, 512) 
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)
        # print(" lstm_out is ", lstm_out.size())
        # Convert decoder hidden state output into the desired form
        h_decoder, c_decoder = s_t
        # print("Decoder weights is h_decoder", h_decoder.size(), " and c_decoder is ", c_decoder.size()) # h_decoder (1, 8, 256)  and c_decoder is  (1, 8, 256)
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        # print("s_t_hat is ", s_t_hat.size()) # s_t_hat is  (8, 512) ie (b, 2 * hidden_dim)

        # Attention
        # eti = vT tanh(Whhi +Wsst +battn) where hi: encoder hidden state s_t_1, st: decoder state s_t_hat
        # at =softmax(et)
        # Output: c_t is  (8, 512) attn_dist is  (8, 400

        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                          enc_padding_mask, coverage)
        if self.training or step > 0:
            coverage = coverage_next

        # finding p_gen if required
        p_gen = None
        if config.pointer_gen:
            # p_gen for  timestep t is calculated from the context vector ht*, the decoder state st and the decoder input xt :
            # pgen=sigmoid(wTh*ht*+wTsst+wTxxt+bptr)
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            # print("p_gen_input is ", p_gen_input.size()) # p_gen_input is  (8, 1152)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)
            # print("p_gen is ", p_gen.size()) # p_gen is  (8, 1)

        # finding p_vocab
        # Pvocab = softmax(V'(V[st,ht*]+b)+b')
        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        # print("output is", output.size()) # output is (8, 768)
        output = self.out1(output) # B x hidden_dim
        # print("out1 is", output.size()) # out1 is (8, 256)
        output = self.out2(output) # B x vocab_size
        # print("out2 is", output.size()) # out2 is (8, 50000)
        vocab_dist = F.softmax(output, dim=1)
        # print("vocab_dist is", vocab_dist.size()) # (8, 50000)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        # print("final_dist is ", final_dist.size()) #final_dist is  (8, 50009)
        return final_dist, s_t, c_t, attn_dist, p_gen, coverage
                                       

# Class Model
class Model(object):
    def __init__(self, model_file_path=None, is_eval=False):
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])

if __name__ == '__main__':
    embedding = nn.Embedding(config.vocab_size, config.emb_dim)
    # print(dir(embedding.weight.data))
    # print(dir(embedding))
    # print(embedding.__dict__)


