import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from util import use_cuda, ids2target, Summarizer
from search import beam_search, search
from tqdm import tqdm
from model import Model
from rouge import Rouge
from pprint import pprint

EPS = 1e-8

def get_reward(generated_sentences, original_sentences):
  rouge = Rouge()
  scores = rouge.get_scores(generated_sentences, original_sentences)
  rouge_l_scores = [score['rouge-l']['f'] + score['rouge-1']['f'] + score['rouge-2']['f'] for score in scores]
  rouge_l_scores = torch.Tensor(rouge_l_scores)
  rouge_l_scores = use_cuda(rouge_l_scores)

  return rouge_l_scores

class Summarization(object):
  def __init__(self, emb_matrix, emb_dim, hidden_dim, word2idx, idx2word):

    self.word2idx = word2idx
    self.idx2word = idx2word
    self.batch_size = 64
    self.model =  Model(emb_dim=emb_dim, hidden_dim=hidden_dim, embedding_matrix=emb_matrix, vocab_size=len(self.word2idx))
    self.optimizer = optim.Adam(self.model.parameters(), lr=0.00002)

  def load_weights(self, path):

    checkpoint = torch.load(path)
    print('Model weights found.')

    self.model.load_state_dict(checkpoint['model'])
    self.optimizer = optim.Adam(self.model.parameters(), lr=0.00002)
    self.optimizer.load_state_dict(checkpoint['optimizer'])

    iteration = int(path.split('_')[-1][:-4])

    print('Model weights loaded.')
    return iteration

  def save_weights(self, save_path):

    print('Saving checkpoint...')
    torch.save({
          "model": self.model.state_dict(),
          "optimizer": self.optimizer.state_dict()
      }, save_path)

  def train_RL_part(self, h_t, c_t, enc_out, enc_inp_len, dec_tar, enc_padding_mask, 
                    enc_ext_vocab, max_zeros_ext_vocab, oovs, greedy=False):
    
    inputs = torch.zeros_like(enc_inp_len).fill_(self.word2idx['<SOS>'])
    et_sum = None
    dec_out = None
    sampled_ids = []
    probs = []
    mask = torch.zeros_like(enc_inp_len).fill_(1)
    masks = []
    for t in range(dec_tar.shape[1]):
      inputs = self.model.embedding_matrix(inputs)
      p_y, h_t, c_t, et_sum, dec_out = self.model.decoder(t, h_t, c_t, enc_out, dec_out, et_sum, 
                                                          enc_padding_mask, enc_ext_vocab, 
                                                          max_zeros_ext_vocab, inputs)
      if not greedy:
        prob_dist = Categorical(p_y)
        inputs = prob_dist.sample()
        prob = prob_dist.log_prob(inputs)
        probs.append(prob)
      else:
        _, inputs = torch.max(p_y, dim=1)

      sampled_ids.append(inputs)
      # mask == 1 and inp != EOS => 1 else => 0
      # mask = (mask == 1)*(inputs != word2idx['<EOS>']) == 1
      mask_t = torch.zeros(len(enc_out)).cuda()                                                
      mask_t[mask == 1] = 1
      masks.append(mask_t)
      is_oov = (inputs >= len(self.word2idx)).type(torch.LongTensor)
      is_oov = use_cuda(is_oov)
      inputs = is_oov * self.word2idx['<UNK>'] + (1 - is_oov) * inputs

    sampled_ids = torch.stack(sampled_ids, dim=1)
    masks = torch.stack(masks, dim=1)
    if not greedy:
      probs = torch.stack(probs, dim=1)
      probs = probs * masks
      lengths = masks.sum(dim=1)
      probs = probs.sum(dim=1)/lengths

    decoded_sentences = []
    for j in range(len(enc_out)):
      decoded_words = ids2target(sampled_ids[j].cpu().numpy(), oovs[j], self.idx2word)
      try:
        end_idx = decoded_words.index('<EOS>')
        decoded_words = decoded_words[:end_idx]
      except:
        pass

      if len(decoded_words) < 2:
        decoded_words = "xxx"
      else:
        decoded_words = " ".join(decoded_words)
      decoded_sentences.append(decoded_words)

    return probs, decoded_sentences

  def train_ML_part(self, h_t, c_t, enc_out, enc_inp_len, dec_inp, enc_ext_vocab, dec_tar, 
                    dec_inp_len, max_zeros_ext_vocab, enc_padding_mask, loss_criterion):

    dec_out = None
    et_sum = None
    iter_losses = []
    i = 0
    inputs = torch.zeros_like(enc_inp_len).fill_(self.word2idx['<SOS>'])

    while (i<dec_tar.shape[1]):
      use_ground_truth = (torch.rand(len(enc_inp_len)) > 0.25).type(torch.LongTensor)
      use_ground_truth = use_cuda(use_ground_truth)
      inputs = use_ground_truth * dec_inp[:, i] + (1 - use_ground_truth) * inputs

      inputs = self.model.embedding_matrix(inputs)
      p_y, h_t, c_t, et_sum, dec_out = self.model.decoder(i, h_t, c_t, enc_out, dec_out, et_sum, 
                                                          enc_padding_mask, enc_ext_vocab, 
                                                          max_zeros_ext_vocab, inputs)
      target = dec_tar[:, i]
      log_preds = torch.log(p_y + EPS)
      iter_loss = loss_criterion(log_preds, target)
      iter_losses.append(iter_loss)
      inputs = torch.multinomial(p_y, 1).squeeze(1)
      is_oov = (inputs >= len(self.word2idx)).type(torch.LongTensor)
      is_oov = use_cuda(is_oov)
      inputs = is_oov * self.word2idx['<UNK>'] + (1 - is_oov) * inputs
      i += 1

    # get sum of losses for all steps per batch
    total_loss_batchwise = torch.sum(torch.stack(iter_losses, dim=1), 1)
    avg_loss_batchwise = total_loss_batchwise / dec_inp_len
    avg_loss = torch.mean(avg_loss_batchwise)

    return avg_loss

  def train(self, data, train_ml=True, train_rl=False, use_prev=None):

    train_dataset = Summarizer(*data)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                 collate_fn=Summarizer.collate_fn)

    total_parameters = sum(par.numel() for par in self.model.parameters() if par.requires_grad)
    print('# Total parameters: ', total_parameters)

    self.model = use_cuda(self.model)

    num_iter = 0
    if use_prev is not None:
      num_iter = self.load_weights(use_prev)

    loss_criterion = nn.NLLLoss(reduction='none', ignore_index=self.word2idx['<PAD>'])
    self.model.train()
    num_epochs = 5
    gamma = 0.95

    # training loop
    train_losses = []
    train_rl_losses = []
    train_rewards = []
    for epoch in range(num_epochs):
      print(f'Starting epoch: {epoch+1}')
      epoch_loss = 0
      epoch_rl_loss = 0
      epoch_rewards = 0
      epoch_g_rewards = 0
      factor = 0
      for enc_inp, dec_inp, enc_ext_vocab, dec_tar, enc_inp_len, dec_inp_len, max_zeros_ext_vocab, enc_padding_mask, oovs, original_sentences in train_dataloader:

        enc_inp = self.model.embedding_matrix(enc_inp)
        enc_out, h_n, c_n = self.model.encoder(enc_inp, enc_inp_len)
        h_t = h_n
        c_t = c_n
        
        if train_ml:
          ml_loss = self.train_ML_part(h_t, c_t, enc_out, enc_inp_len, dec_inp, 
                                      enc_ext_vocab, dec_tar, dec_inp_len, max_zeros_ext_vocab, 
                                      enc_padding_mask, loss_criterion)
        else:
          ml_loss = torch.zeros(1)
          ml_loss = use_cuda(ml_loss)

        if train_rl:
          sampled_probs, sampled_sentences = self.train_RL_part(h_t, c_t, enc_out, enc_inp_len, 
                                                                dec_tar, enc_padding_mask, enc_ext_vocab, 
                                                                max_zeros_ext_vocab, oovs, greedy=False)
          with torch.no_grad():
            _, greedy_sentences = self.train_RL_part(h_t, c_t, enc_out, enc_inp_len, dec_tar, 
                                                     enc_padding_mask, enc_ext_vocab, max_zeros_ext_vocab, 
                                                     oovs, greedy=True)
          sampled_reward = get_reward(sampled_sentences, original_sentences)
          greedy_reward = get_reward(greedy_sentences, original_sentences)

          sampled_reward_avg = torch.mean(sampled_reward).item()
          greedy_reward_avg = torch.mean(greedy_reward).item()

          rl_loss = -(sampled_reward - greedy_reward) * sampled_probs
          rl_loss = torch.mean(rl_loss)
        else:
          rl_loss = torch.zeros(1)
          sampled_reward_avg = 0
          greedy_reward_avg = 0
          rl_loss = use_cuda(rl_loss)

        mixed_loss = gamma * rl_loss + (1 - gamma) * ml_loss
        self.optimizer.zero_grad()
        mixed_loss.backward()
        self.optimizer.step()
        num_iter += 1
        epoch_loss += ml_loss.item()
        epoch_rl_loss += rl_loss.item()
        factor += 1
        epoch_rewards += sampled_reward_avg
        epoch_g_rewards += greedy_reward_avg
        if num_iter % 50 == 0:
          print('Iteration: %d, ML Loss: %.3f, RL loss: %.3f, Sampled Reward: %.3f, Greedy Reward: %.3f' % (num_iter, epoch_loss/factor, epoch_rl_loss/factor, epoch_rewards/factor, epoch_g_rewards/factor))

        if num_iter % 100 == 0:
          save_path = "./models/model_%04d.tar"%num_iter
          self.save_weights(save_path)

      num_batches = train_dataloader.__len__()
      train_loss = epoch_loss/num_batches
      train_rl_loss = epoch_rl_loss/num_batches
      train_losses.append(train_loss)
      train_rl_losses.append(train_rl_loss)
      train_reward = epoch_rewards/num_batches
      train_g_reward = epoch_g_rewards/num_batches
      train_rewards.append(train_reward)
      print('Epoch %d, Loss: %.3f, RL Loss: %.3f, Sampled Reward: %.3f, Greedy Reward: %.3f' % (epoch+1, train_loss, train_rl_loss, train_reward, train_g_reward))

  def eval(self, data, eval_df, load_path, evaluation='val', search='BEAM', print_samples=False):
    # evaluation can be val or test
    # search can be BEAM, GREEDY, RANDOM

    self.model = use_cuda(self.model)
    _ = self.load_weights(load_path)
    self.model.eval()
 
    eval_dataset = Summarizer(*data)
    if len(eval_df) >= self.batch_size:
      eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size,
                                 collate_fn=Summarizer.collate_fn)
    else:
      eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=len(eval_df),
                                 collate_fn=Summarizer.collate_fn)
    rouge  = Rouge()
    decoded_sentences = []
    eval_titles = []
    print('Generating summaries ...')
    pbar = tqdm(total=eval_dataloader.__len__())
    for enc_inp, dec_inp, enc_ext_vocab, dec_tar, enc_inp_len, dec_inp_len, max_zeros_ext_vocab, enc_padding_mask, oovs, titles in eval_dataloader:

      enc_inp = self.model.embedding_matrix(enc_inp)
      enc_out, h_n, c_n = self.model.encoder(enc_inp, enc_inp_len)
      if search == 'BEAM':
        prediction_ids = beam_search(h_n, c_n, enc_out, dec_tar, enc_padding_mask, 
                                     enc_ext_vocab, max_zeros_ext_vocab, self.model, 
                                     self.word2idx, hidden_dim=128*2, evaluation=evaluation)
      elif search == 'GREEDY':
        prediction_ids = search(h_n, c_n, enc_out, enc_inp_len, dec_tar, enc_padding_mask, 
                                enc_ext_vocab, max_zeros_ext_vocab, self.model, self.word2idx, 
                                evaluation=evaluation)
      elif search == 'RANDOM':
        prediction_ids = search(h_n, c_n, enc_out, enc_inp_len, dec_tar, enc_padding_mask, 
                                enc_ext_vocab, max_zeros_ext_vocab, self.model, self.word2idx, 
                                greedy=False, evaluation=evaluation)
      else:
        print('Unknown search strategy.')
        return

      eval_titles.extend(titles)
      for j in range(len(prediction_ids)):
        decoded_words = ids2target(prediction_ids[j], oovs[j], self.idx2word)
        try:
          end_idx = decoded_words.index('<EOS>')
          decoded_words = decoded_words[:end_idx]
        except:
          pass
        if len(decoded_words) < 2:
          decoded_words = "xxx"
        else:
          decoded_words = " ".join(decoded_words)
        decoded_sentences.append(decoded_words)

      # update the tqdm progress bar
      pbar.update(1)

    if evaluation == 'val':
      rouge_scores = rouge.get_scores(decoded_sentences, eval_titles, avg=True)
      print('Rouge score on eval set: ')
      print('Rouge-1 F1: ', rouge_scores['rouge-1']['f'])
      print('Rouge-2 F1: ', rouge_scores['rouge-2']['f'])
      print('Rouge-L F1: ', rouge_scores['rouge-l']['f'])

    if print_samples:
      eval_abstracts = [' '.join(ab.replace('\n',' ').split()) for ab in eval_df['abstract'].values]
      i = 0
      for ab, t, pred_t in zip(eval_abstracts, eval_titles, decoded_sentences):
        print('Abstract: ')
        pprint(ab)
        if evaluation == 'val':
          print('Gold Title: ')
          pprint(t)
        print('Generated title: ')
        pprint(pred_t)
        print("************************************************")
        i += 1
        if i == 5:
          break