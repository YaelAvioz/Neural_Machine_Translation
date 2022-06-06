import os
import sys
import time

import numpy as np
import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from utils import print_epoch, parse_input, convert_data_format, data_to_index, plot_graph, to_tensor, shuffle_data

OUTPUT_DIR_PATH = './Output'
EPOCHS = 10
LR = 0.0006


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_embeddings):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)

    def forward(self, x, hidden):
        return self.lstm(self.embedding(x).view(1, 1, -1), hidden)

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size)


class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, hidden_size, drop=0.25):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_size)
        self.out = nn.Linear(hidden_size, num_embeddings)
        self.dropout = nn.Dropout(drop)
        self.hidden_size = hidden_size

    def forward(self, encoder_out, word_input, hidden):
        embedded = self.embedding(word_input).view(1, 1, -1)
        lstm_input = torch.cat([encoder_out, embedded], dim=2)
        x, hidden = self.lstm(lstm_input, hidden)
        x = self.dropout(x)
        x = F.log_softmax(self.out(x.view(1, -1)), dim=1)

        return x, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size)


def train(train_source, train_target, dev_source, dev_target, encoder, decoder, target_text, i2t, encoder_file,
          decoder_file):
    start_time = time.time()

    # Adam optimizer
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LR)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LR)

    nll_loss = nn.NLLLoss()

    dev_losses = []
    bleu_scores = []
    max_bleu_score = 0

    for epoch in range(1, EPOCHS + 1):

        encoder.train()
        decoder.train()

        # Shuffle data
        source, target = shuffle_data(train_source, train_target)

        sum_loss = 0
        for index, (source_seq, target_seq) in tqdm(enumerate(zip(source, target))):

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss = 0

            encoder_hidden = encoder.init_hidden()
            for i in range(source_seq.size(0)):
                encoder_output, encoder_hidden = encoder(source_seq[i], encoder_hidden)

            decoder_hidden = decoder.init_hidden()
            for i in range(target_seq.size(0) - 1):
                decoder_output, decoder_hidden = decoder(encoder_hidden[0], target_seq[i], decoder_hidden)

                loss += nll_loss(decoder_output, target_seq[i + 1].unsqueeze(0))
            sum_loss += (loss.item() / (target_seq.size(0) - 1))

            loss.backward()

            # Update the weights
            encoder_optimizer.step()
            decoder_optimizer.step()

        dev_loss, bleu = evaluate(dev_source, dev_target, encoder, decoder, nll_loss, i2t, target_text)

        dev_losses.append(dev_loss)
        bleu_scores.append(bleu.score)

        if bleu.score > max_bleu_score:
            max_bleu_score = bleu.score
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)

        print_epoch(epoch, sum_loss / len(train_source), dev_loss, bleu.score)

    print(f'Training Time: {(time.time() - start_time) / 60:.3f} minutes')

    return dev_losses, bleu_scores


def evaluate(source_tensors, target_tensors, encoder, decoder, nll_loss, index_to_target, target_txt):
    encoder.eval()
    decoder.eval()

    sum_loss = 0
    predction = []

    with torch.no_grad():

        # Shuffle data
        source_tensors, target_tensors, target_txt = shuffle_data(source_tensors, target_tensors, target_txt)

        # Calculate the max sentence length
        max_length = max([len(sentence) for sentence in target_tensors])
        for index, (source_seq, target_seq) in enumerate(zip(source_tensors, target_tensors)):

            loss = 0

            encoder_hidden = encoder.init_hidden()
            for i in range(source_seq.size(0)):
                encoder_output, encoder_hidden = encoder(source_seq[i], encoder_hidden)

            hidden = decoder.init_hidden()
            for i in range(target_seq.size(0) - 1):
                output, hidden = decoder(encoder_hidden[0], target_seq[i], hidden)

                loss += nll_loss(output, target_seq[i + 1].unsqueeze(0))
            sum_loss += (loss.item() / (target_seq.size(0) - 1))

            decoder_hidden = decoder.init_hidden()

            words = []
            decoder_input = target_seq[0]

            for i in range(max_length):
                decoder_output, decoder_hidden = decoder(encoder_hidden[0], decoder_input, decoder_hidden)
                decoder_input = np.argmax(decoder_output.data, axis=1)
                words.append(index_to_target[decoder_input.item()])
                if index_to_target[decoder_input.item()] == '</s>':
                    break

            predction.append(' '.join(['<s>'] + words))

        bleu_ref = [' '.join(ref) for ref in target_txt]
        bleu = sacrebleu.corpus_bleu(predction, [bleu_ref])

    # Returns the bleu score
    return sum_loss / len(source_tensors), bleu


def main(train_source_path, train_target_path, dev_source_path, dev_target_path):
    # In case there is not output directory - create new directory
    if not os.path.exists(OUTPUT_DIR_PATH):
        os.makedirs(OUTPUT_DIR_PATH)
    encoder_file = os.path.join(OUTPUT_DIR_PATH, 'encoder_file')
    decoder_file = os.path.join(OUTPUT_DIR_PATH, 'decoder_file')
    dictionary_file = os.path.join(OUTPUT_DIR_PATH, 'dictionary_file')

    # Parse input
    train_data, train_source, train_target = parse_input(train_source_path, train_target_path)

    # Filter non frequent words and return mapping from word to index and index to word
    source_to_index, index_to_source = convert_data_format(train_source)
    target_to_index, index_to_target = convert_data_format(train_target)

    # Replace the words with their indexes.
    train_source_index, train_target_index = data_to_index(train_data, source_to_index, target_to_index)

    # Parse dev input
    dev_data, _, dev_target_data = parse_input(dev_source_path, dev_target_path)

    # Initialize the encoder
    encoder = Encoder(embedding_dim=128, hidden_size=128, num_embeddings=len(source_to_index))

    # Initialize the decoder
    decoder = Decoder(embedding_dim=128, num_embeddings=len(target_to_index), hidden_size=256)

    # Convert to indexes representation
    dev_source_index, dev_target_index = data_to_index(dev_data, source_to_index, target_to_index)

    # Training
    dev_target_txt = [[word if word in target_to_index else '<?>' for word in seq] for seq in dev_target_data]
    losses, bleu_scores = train(to_tensor(train_source_index), to_tensor(train_target_index),
                                to_tensor(dev_source_index), to_tensor(dev_target_index), encoder, decoder,
                                dev_target_txt, index_to_target, encoder_file, decoder_file)

    # Plot loss and BLEU Score graphs
    plot_graph('Dev Loss', losses)
    plot_graph('BLEU Score', bleu_scores)

    # Saving the dictionaries (vocabularies)
    torch.save({'source_to_index': source_to_index,
                'index_to_source': index_to_source,
                'target_to_index': target_to_index,
                'index_to_target': index_to_target}, dictionary_file)


if __name__ == '__main__':
    main(*(sys.argv[i] for i in range(1, 5)))
