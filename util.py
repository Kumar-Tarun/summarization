import numpy as np
import pandas as pd
import pickle
import mmap
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset

def use_cuda(obj):

  if torch.cuda.is_available():
    obj = obj.cuda()

  return obj

def read_data(abstract_path, title_path):
  """
  :param abstract_path: path to abstract pickle file
  :param title_path: path to title pickle file
  :returns: dataframe containing abstracts in 1st column and titles in 2nd column
  """
  with open(abstract_path, 'rb') as f:
    abstracts = pickle.load(f)

  with open(title_path, 'rb') as f:
    titles = pickle.load(f)

  df = pd.DataFrame({'abstract': abstracts, 'title': titles})
  return df

def split_data(df, split=0.1):
  """
  :param df: original dataframe containing data
  :param split: split in fraction for both val and test
  :returns: train, val and test dataframes
  """
  np.random.seed(101)
  val_df = df.sample(frac=split, random_state=101)
  train_df = df.drop(val_df.index)

  return train_df, val_df

def get_vocab(train_df, size=50000):
    """
    :param train_df: train dataframe
    :param size: size of the vocabulary to extract
    :return: a list containing upto size number of most frequent words
    """
    vocab = {}
    data = list(train_df['abstract'].values) + list(train_df['title'].values)
    for example in data:
      for word in example.split():
        if word in vocab:
          vocab[word] += 1
        else:
          vocab[word] = 1

    vocab = sorted(vocab.items(), key=lambda x : x[1], reverse=True)
 
    vocab = [w[0] for w in vocab[:size]]
    print("vocab size: ", len(vocab))
    return vocab

def get_word2idx_idx2word(vocab):
    """
    :param vocab: a set of strings: vocabulary
    :return: word2idx: string to an int
             idx2word: int to a string
    """
    # <SOS> and <EOS> are start of sentence and end of sentence respectively
    word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
    for word in vocab:
        assigned_index = len(word2idx)
        word2idx[word] = assigned_index
        idx2word[assigned_index] = word
    return word2idx, idx2word

def source2id(source_words, word2idx):
  """
  :param source_words: list of abstract words
  :param word2idx: dict mapping word to int
  :returns: list of ints, list of OOV words in abstract
  """
  vocab_len = len(word2idx)
  source_ids = []
  oovs = []
  unk_id = word2idx['<UNK>']
  for w in source_words:
    id = word2idx.get(w, 1)
    if id == unk_id:
      if w not in oovs:
        oovs.append(w)
      source_ids.append(vocab_len + oovs.index(w))
    else:
      source_ids.append(id)

  return source_ids, oovs

def target2id(target_words, source_oovs, word2idx):
  """
  :param target_words: list of title words
  :param source_oovs: list of abstract OOV words from above
  :param word2idx: dict mapping word to int
  """
  vocab_len = len(word2idx)
  target_ids = []
  unk_id = word2idx['<UNK>']
  for w in target_words:
    id = word2idx.get(w, 1)
    if id == unk_id:
      if w in source_oovs:
        target_ids.append(vocab_len + source_oovs.index(w))
      else:
        target_ids.append(unk_id)
    else:
      target_ids.append(id)

  return target_ids

def prepare_input_data(data, word2idx):
  """
  :param data: dataframe containing abstracts and titles
  :param word2idx: dict mapping word to int
  :returns: encoder_inps: source words converted to ints
            encoder_ext_vocabs: source words converted to ints and special ints for OOV words (extended vocab)
            decoder_inps: target words converted to ints
            decoder_targets: shifted target words converted to ints and special ints for source OOV words
            source_oovs: list of source OOV words
            target_sentences: original target titles
  """
  encoder_inps = []
  encoder_ext_vocabs = []
  decoder_inps = []
  decoder_targets = []
  source_oovs = []
  target_sentences = []
  for row in data.iterrows():
    ab, ti = row[0], row[1]
    abs = str(ab).lower().replace('\n', ' ').split()
    t = str(ti).lower().replace('\n', ' ').split()
    target_sentences.append(' '.join(t))
    encoder_inp = [word2idx.get(w, 1) for w in abs]
    decoder_inp = [word2idx['<SOS>']] + [word2idx.get(w, 1) for w in t]
    encoder_ext_vocab, source_oov = source2id(abs, word2idx)
    # only decoder target has multiple ids for OOV words, encoder input for multiple ids is 
    # only used to get the final distribution
    decoder_target = target2id(t, source_oov, word2idx) + [word2idx['<EOS>']]
    encoder_inps.append(encoder_inp)
    encoder_ext_vocabs.append(encoder_ext_vocab)
    decoder_inps.append(decoder_inp)
    decoder_targets.append(decoder_target)
    source_oovs.append(source_oov)

  return encoder_inps, encoder_ext_vocabs, decoder_inps, decoder_targets, source_oovs, target_sentences

def prepare_test_data(csv, word2idx):
  """
  :param csv: csv file containing abstracts with column name 'abstract'
  :param word2idx: dict mapping word to int
  :returns: encoder_inps: source words converted to ints
            encoder_ext_vocabs: source words converted to ints and special ints for OOV words (extended vocab)
            decoder_inps: target words converted to ints
            decoder_targets: shifted target words converted to ints and special ints for source OOV words
            source_oovs: list of source OOV words
            target_sentences: original target titles
  """
  df = pd.read_csv(csv)
  abstracts = df['abstract'].values
  encoder_inps = []
  encoder_ext_vocabs = []
  decoder_inps = []
  source_oovs = []
  for row in abstracts:
    ab = row
    abs = str(ab).lower().replace('\n', ' ').split()
    encoder_inp = [word2idx.get(w, 1) for w in abs]
    encoder_ext_vocab, source_oov = source2id(abs, word2idx)

    encoder_inps.append(encoder_inp)
    encoder_ext_vocabs.append(encoder_ext_vocab)
    source_oovs.append(source_oov)

  # make dummy inputs, targets and target sentences
  decoder_inps = [[0]]*len(df)
  decoder_targets = decoder_inps
  target_sentences = list(abstracts)

  return df, (encoder_inps, encoder_ext_vocabs, decoder_inps, decoder_targets, source_oovs, target_sentences)

def ids2target(ids, source_oovs, idx2word):
  """
  :param ids: list of ints corresponding to words in vocab
  :param source_oovs: list of source OOV words
  :param idx2word: dict mapping int to word
  :returns: list of words obtained from ints (also includes source OOV words)
  """
  words = []
  for id in ids:
    try:
      # try if the word is in vocab (including <UNK>)
      word = idx2word[id]
    except:
      # this means the word is in source oovs
      index = id - len(idx2word)
      word = source_oovs[index]

    words.append(word)

  return words

def get_num_lines(file_path):
  """
  :param file_path: path to file of which to count number of lines
  :returns: number of lines in the file
  """
  fp = open(file_path, "r+")
  buf = mmap.mmap(fp.fileno(), 0)
  lines = 0
  while buf.readline():
    lines += 1
  return lines

def get_embedding_matrix(word2idx, idx2word, glove_path, name, normalization=False):
  """
  :param word2idx: dict mapping word to int
  :param idx2word: dict mapping int to word
  :param glove_path: path from where to read the embeddings from
  :param name: name of the file in data folder to save embedding matrix
  :param normalization: whether to normalize the embeddings to norm of one
  :return: an embedding matrix: a nn.Embeddings
  """
  embedding_dim = 300
  try:
    with open('./data/embeddings_' + name + '.pkl', 'rb') as f:
      glove_vectors = pickle.load(f)
  except:
    glove_vectors = {}
    # Load the GloVe vectors into a dictionary, keeping only words in vocab
    with open(glove_path) as glove_file:
      for line in tqdm(glove_file, total=get_num_lines(glove_path)):
        split_line = line.rstrip().split()
        word = split_line[0]
        if len(split_line) != (embedding_dim + 1) or word not in word2idx:
          continue
        assert (len(split_line) == embedding_dim + 1)
        vector = np.array([float(x) for x in split_line[1:]], dtype="float32")
        if normalization:
          vector = vector / np.linalg.norm(vector)
        assert len(vector) == embedding_dim
        glove_vectors[word] = vector

    with open('./data/embeddings_' + name + '.pkl', 'wb+') as f:
      pickle.dump(glove_vectors, f)

  print("Number of pre-trained word vectors loaded: ", len(glove_vectors))

  # Calculate mean and stdev of embeddings
  all_embeddings = np.array(list(glove_vectors.values()))
  embeddings_mean = float(np.mean(all_embeddings))
  embeddings_stdev = float(np.std(all_embeddings))
  print("Embeddings mean: ", embeddings_mean)
  print("Embeddings stdev: ", embeddings_stdev)

  # Randomly initialize an embedding matrix of (vocab_size, embedding_dim) shape
  # with a similar distribution as the pretrained embeddings for words in vocab.
  vocab_size = len(word2idx)
  embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean, embeddings_stdev)
  # Go through the embedding matrix and replace the random vector with a
  # pretrained one if available. Start iteration at 2 since 0, 1 are PAD, UNK
  for i in range(2, vocab_size):
    word = idx2word[i]
    if word in glove_vectors:
      embedding_matrix[i] = torch.FloatTensor(glove_vectors[word])

  if normalization:
    for i in range(vocab_size):
      embedding_matrix[i] = embedding_matrix[i] / float(np.linalg.norm(embedding_matrix[i]))

  embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
  embeddings.weight = nn.Parameter(embedding_matrix)
  return embeddings

# Make sure to subclass torch.utils.data.Dataset
class Summarizer(Dataset):
  def __init__(self, encoder_inputs, encoder_ext_vocabs, decoder_inputs, decoder_targets, source_oovs, target_sentences):
    """
    :param encoder_inputs: a list of lists: each inner list is a sequence of word ids
    :param encoder_ext_vocabs:  a list of lists: each inner list is a sequence of word ids 
                                (including special ids for OOV words)
    :param decoder_inputs: a list of lists: each inner list is a sequence of word ids
    :param decoder_targets: a list of lists: each inner list is a sequence of word ids 
                            (including special ids for OOV words)
    :param sourc_oovs: a list of lists: each inner list is a sequence of source OOV words
    :param target_sentences: a list of target sentences
     
    """
    if len(encoder_inputs) != len(decoder_inputs):
        raise ValueError("Differing number of encoder inputs and decoder inputs")

    self.encoder_inputs = encoder_inputs
    self.encoder_ext_vocabs = encoder_ext_vocabs
    self.decoder_inputs = decoder_inputs
    self.decoder_targets = decoder_targets
    self.source_oovs = source_oovs
    self.target_sentences = target_sentences

  def __getitem__(self, idx):
    """
    :param idx: index into each of the inputs
    :returns: the Dataset example at index `idx`.
    """
    example_enc_input = self.encoder_inputs[idx]
    example_enc_ext_vocab = self.encoder_ext_vocabs[idx]
    example_dec_input = self.decoder_inputs[idx]
    example_dec_target = self.decoder_targets[idx]
    example_source_oov = self.source_oovs[idx]
    example_target_sen = self.target_sentences[idx]
    assert (len(example_enc_input) == len(example_enc_ext_vocab))
    assert (len(example_dec_input) == len(example_dec_target))

    return example_enc_input, example_enc_ext_vocab, example_dec_input, example_dec_target, example_source_oov, example_target_sen

  def __len__(self):
    """
    Return the number of examples in the Dataset.
    """
    return len(self.encoder_inputs)

  @staticmethod
  def collate_fn(batch):
    """
    :param batch: given a list of examples (each from __getitem__),
    combine them to form a single batch by padding.
    :returns:
    -------
    batch_padded_enc_inp: LongTensor
      LongTensor of shape (batch_size, longest_sequence_length) with the
      padded ids for each example in the batch
    batch_padded_dec_inp: LongTensor
      LongTensor of shape (batch_size, longest_sequence_length) with the
      padded ids for each example in the batch
    batch_padded_enc_ext_vocab: LongTensor
      LongTensor of shape (batch_size, longest_sequence_length) with the
      padded ids for each example in the batch
    batch_padded_dec_tar: LongTensor
      LongTensor of shape (batch_size, longest_sequence_length) with the
      padded ids for each example in the batch
    batch_enc_inp_lengths: LongTensor
      LongTensor of shape (batch_size,) with the length of each example in batch
    batch_dec_inp_lengths: LongTensor
      LongTensor of shape (batch_size,) with the length of each example in batch
    max_zeros_ext_vocab: FloatTensor
      FloatTensor of shape (batch_size, max_num_ovvs) with all zeros
      (used while combining pointer distribution and softmax distribution)
    batch_enc_padding_masks: FloatTensor
      FloatTensor of shape (batch_size, longest_sequence_length) with the
      padding mask for each example in the batch
    all_oovs: a list of lists: each inner list contains source oovs corresponding
      to each example in batch
    batch_target_sens: a list of target sentences in the batch
    """
    batch_padded_enc_inp = []
    batch_padded_dec_inp = []
    batch_padded_enc_ext_vocab = []
    batch_padded_dec_tar = []
    batch_enc_padding_masks = []
    batch_enc_inp_lengths = []
    batch_dec_inp_lengths = []
    batch_target_sens = []
    max_oov_len = 0
    max_enc_inp_len = 0
    max_dec_inp_len = 0
    all_oovs = []
    max_zeros_ext_vocab = []
    padding_amounts = []

    for enc_inp, __, dec_inp, __, oov, __ in batch:
      if len(enc_inp) > max_enc_inp_len:
        max_enc_inp_len = len(enc_inp)
      if len(dec_inp) > max_dec_inp_len:
        max_dec_inp_len = len(dec_inp)
      if len(oov) > max_oov_len:
        max_oov_len = len(oov)

    # Iterate over each example in the batch
    for enc_inp, enc_ext_vocab, dec_inp, dec_tar, oov, target_sen in batch:
      # Unpack the example (returned from __getitem__)
      enc_len = len(enc_inp)
      dec_len = len(dec_inp)
      all_oovs.append(oov)
      max_zeros_ext_vocab.append(torch.zeros(max_oov_len))
      batch_target_sens.append(target_sen)
      batch_enc_inp_lengths.append(enc_len)
      batch_dec_inp_lengths.append(dec_len)
      # Calculate the amount of padding for both enc and dec
      amount_to_pad_enc = max_enc_inp_len - enc_len
      amount_to_pad_dec = max_dec_inp_len - dec_len

      enc_padding_mask = [1]*enc_len + [0]*amount_to_pad_enc
      batch_enc_padding_masks.append(enc_padding_mask)

      # append padding to corresponding variables
      padded_enc_inp = enc_inp + [0]*amount_to_pad_enc
      padded_dec_inp = dec_inp + [0]*amount_to_pad_dec
      padded_enc_ext_vocab = enc_ext_vocab + [0]*amount_to_pad_enc
      padded_dec_tar = dec_tar + [0]*amount_to_pad_dec
      # Add the padded example to our batch
      batch_padded_enc_inp.append(padded_enc_inp)
      batch_padded_dec_inp.append(padded_dec_inp)
      batch_padded_enc_ext_vocab.append(padded_enc_ext_vocab)
      batch_padded_dec_tar.append(padded_dec_tar)

    # Stack the list of LongTensors into a single LongTensor
    return (use_cuda(torch.LongTensor(batch_padded_enc_inp)),
            use_cuda(torch.LongTensor(batch_padded_dec_inp)),
            use_cuda(torch.LongTensor(batch_padded_enc_ext_vocab)),
            use_cuda(torch.LongTensor(batch_padded_dec_tar)),
            use_cuda(torch.LongTensor(batch_enc_inp_lengths)),
            use_cuda(torch.LongTensor(batch_dec_inp_lengths)),
            use_cuda(torch.stack(max_zeros_ext_vocab)),
            use_cuda(torch.Tensor(batch_enc_padding_masks)),
            all_oovs,
            batch_target_sens,
            )
