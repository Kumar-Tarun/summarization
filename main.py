from util import read_data, split_data, get_vocab, get_word2idx_idx2word, prepare_input_data, get_embedding_matrix
from train import Summarization
import sys

def preprocess_data(abstracts_path, titles_path):

	print('Reading data ...')
  total_df = read_data(abstracts_path, titles_path)

  train_df, val_df, test_df = split_data(total_df)
  print('Preparing vocabulary ...')
  vocab = get_vocab(train_df, size=80000)
  word2idx, idx2word = get_word2idx_idx2word(vocab)
  return train_df, val_df, test_df, word2idx, idx2word

def main():
	choice = sys.argv[1]
	load_path = sys.argv[2]
	if choice in ['val', 'test'] and load_path is None:
		print('Please specify some path to load model weights from.')
		return

	train_df, val_df, test_df, word2idx, idx2word = preprocess_data('./data/abstracts.pkl', './data/titles.pkl')
	print('Preparing embedding matrix ...')
	emb_matrix = get_embedding_matrix(word2idx, idx2word, './data/glove_vectors.txt', 'glove')
	summarization = Summarization(emb_matrix, emb_dim=300, hidden_dim=128, word2idx=word2idx, idx2word=idx2word)
	print('Preparing Input data ...')
	if choice == 'train':
		train_data = prepare_input_data(train_df, word2idx)
		summarization.train(train_data, use_prev=load_path)
	elif choice == 'val':
		eval_data = prepare_input_data(val_df, word2idx)
		summarization.eval(eval_data, val_df, load_path=load_path, evaluation='val', print_samples=True)
	elif choice == 'test':
		eval_data = prepare_input_data(test_df, word2idx)
		summarization.eval(eval_data, test_df, load_path=load_path, evaluation='test', print_samples=True)

if __name__ == '__main__':
	main()







