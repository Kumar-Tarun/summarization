import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from allennlp.nn.util import sort_batch_by_length

class Encoder(nn.Module):
  def __init__(self, emb_dim, hidden_dim):
    super(Encoder, self).__init__()

    self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

  def forward(self, inputs, lengths):

    batch_size, seq_len, emb_dim = inputs.shape
    (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(inputs, lengths)
    packed_input = pack_padded_sequence(sorted_input, sorted_lengths.data.tolist(), batch_first=True)
    packed_sorted_output, (sorted_h_n, sorted_c_n) = self.lstm(packed_input)
    sorted_output, _ = pad_packed_sequence(packed_sorted_output, batch_first=True)
    output = sorted_output[input_unsort_indices]
    sorted_h_n = sorted_h_n.transpose(0, 1).contiguous().view(batch_size, -1)
    sorted_c_n = sorted_c_n.transpose(0, 1).contiguous().view(batch_size, -1)
    h_n = sorted_h_n[input_unsort_indices]
    c_n = sorted_c_n[input_unsort_indices]
    return output, h_n, c_n

class Decoder(nn.Module):
  def __init__(self, emb_dim, hidden_dim, vocab_size):
    super(Decoder, self).__init__()
    self.lstm = nn.LSTMCell(emb_dim, 2*hidden_dim)

    self.bilinear_weight_encoder = nn.Linear(2*hidden_dim, 2*hidden_dim, bias=False)
    self.bilinear_weight_decoder = nn.Linear(2*hidden_dim, 2*hidden_dim, bias=False)
    self.output_projection = nn.Linear(2*hidden_dim + 2*hidden_dim + 2*hidden_dim, vocab_size)
    self.output_copy = nn.Linear(2*hidden_dim + 2*hidden_dim + 2*hidden_dim, 1)

  def forward(self, t, h_t, c_t, enc_out, dec_out, et_sum, enc_padding_mask, enc_ext_vocab, max_zeros_ext_vocab, inputs):

    (h_next, c_next) = self.lstm(inputs, (h_t, c_t))
    # TEMPORAL ATTENTION
    part2 = self.bilinear_weight_encoder(enc_out).transpose(1, 2)
    part1 = h_next.unsqueeze(1)
    et = torch.bmm(part1, part2).squeeze(1)
    et_exp = et.exp()
    
    if t == 0:
      et_prime = et_exp
      et_sum = et_exp
    else:
      et_prime = et_exp/et_sum
      et_sum = et_sum + et_exp

    # mask out attention weights for padding tokens
    et_prime = et_prime * enc_padding_mask
    alpha_et = et_prime/torch.sum(et_prime, dim=1).view(-1, 1)
    c_et = torch.bmm(alpha_et.unsqueeze(1), enc_out).squeeze(1)

    # DECODER ATTENTION
    if t == 0:
      c_dt = torch.zeros_like(h_next)
      dec_out = h_next.unsqueeze(1)
    else:
      part2 = self.bilinear_weight_decoder(dec_out).transpose(1, 2)
      e_dt = torch.bmm(part1, part2).squeeze(1)
      alpha_dt = F.softmax(e_dt, dim=1)
      c_dt = torch.bmm(alpha_dt.unsqueeze(1), dec_out).squeeze(1)

      dec_out = torch.cat([dec_out, h_next.unsqueeze(1)], dim=1)

    final_concat = torch.cat([h_next, c_et, c_dt], dim=1)
    p_y_u_zero = F.softmax(self.output_projection(final_concat), dim=1)
    p_u_one = torch.sigmoid(self.output_copy(final_concat))
    p_u_zero = 1 - p_u_one
    p_y_u_one = alpha_et
    p_y_part1 = p_y_u_zero*p_u_zero
    p_y_part2 = p_y_u_one*p_u_one
    p_y_part1 = torch.cat([p_y_part1, max_zeros_ext_vocab], dim=1)
    p_y = p_y_part1.scatter_add(1, enc_ext_vocab, p_y_part2)

    return p_y, h_next, c_next, et_sum, dec_out


class Model(nn.Module):
  def __init__(self, emb_dim, hidden_dim, embedding_matrix, vocab_size):
    super(Model, self).__init__()
    self.embedding_matrix = embedding_matrix
    self.encoder = Encoder(emb_dim, hidden_dim)
    self.decoder = Decoder(emb_dim, hidden_dim, vocab_size)
