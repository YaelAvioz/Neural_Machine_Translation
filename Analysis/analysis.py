import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from utils import parse_input, convert_data_format, data_to_index, to_tensor, shuffle_data, write_epoch, create_heatmap

FILE_NAME = '/weights.json'

EXAMPLE = 13
OUTPUT_DIR_PATH = './Output'
EPOCHS = 10
LR = 0.00020


class Encoder(nn.Module):

    def __init__(self, embedding_dim, hidden_size, num_embeddings):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)

    def forward(self, word_input, hidden):
        return self.lstm(self.embedding(word_input).view(1, 1, -1), hidden)

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size)


class Decoder(nn.Module):

    def __init__(self, embedding_dim, num_embeddings, hidden_size, drop=0.3):
        super(Decoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim + self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(drop)
        self.attention = nn.Linear(self.embedding_dim + self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.v = nn.Parameter(torch.rand(self.hidden_size) * 0.1)
        self.attn_hidden = nn.Linear(self.embedding_dim + self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.num_embeddings)

    def forward(self, encoder_out, word_input, last_context, hidden):
        lstm_input = torch.cat((last_context.unsqueeze(0), (self.embedding(word_input).view(1, 1, -1))), 2)
        x, hidden = self.lstm(lstm_input, hidden)
        x = self.dropout(x)
        attn_weights = torch.zeros(len(encoder_out))
        for i in range(len(encoder_out)):
            attn_weights[i] = self.v.dot(
                self.tanh(self.attention(torch.cat((x[0], encoder_out[i].view(1, -1)), dim=1))).view(-1))

        normalized_weights = F.softmax(attn_weights, dim=0).unsqueeze(0).unsqueeze(0)
        context = normalized_weights.bmm(encoder_out.unsqueeze(0))
        x_t = self.tanh(self.attn_hidden(torch.cat((context, x), dim=2)))
        output = F.log_softmax(self.out(x_t.view(1, -1)), dim=1)
        return output, hidden, x_t[0], normalized_weights[0]

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size)


def train(train_source, train_target, encoder, decoder, source_dev_example,
          trg_dev_example, src_dev_example_txt, trg_dev_example_txt):
    start_time = time.time()

    nll_loss = nn.NLLLoss()

    # Adam optimizer
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LR)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):

        encoder.train()
        decoder.train()

        # Shuffle the data
        source, target = shuffle_data(train_source, train_target)

        sum_loss = 0
        for index, (source_seq, target_seq) in enumerate(zip(source, target)):

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss = 0

            encoder_hidden = encoder.init_hidden()
            encoder_outputs = torch.zeros(source_seq.size(0), encoder.hidden_size)

            for i in range(source_seq.size(0)):
                encoder_output, encoder_hidden = encoder(source_seq[i], encoder_hidden)
                encoder_outputs[i] = encoder_output.view(-1)

            decoder_hidden = decoder.init_hidden()
            decoder_context = torch.zeros(1, decoder.hidden_size)

            for i in range(target_seq.size(0) - 1):
                decoder_output, decoder_hidden, decoder_context, decoder_attention = decoder(
                    encoder_outputs, target_seq[i], decoder_context, decoder_hidden)

                loss += nll_loss(decoder_output, target_seq[i + 1].unsqueeze(0))

            sum_loss += (loss.item() / (target_seq.size(0) - 1))
            loss.backward()

            # Update the weights
            encoder_optimizer.step()
            decoder_optimizer.step()

        # Analyze the attention weights of the model
        analyze_attention(source_dev_example, trg_dev_example, src_dev_example_txt, trg_dev_example_txt, encoder,
                          decoder, epoch)

        print(f'Epoch: {epoch}\t Train Loss: {(sum_loss / len(train_source)):.3f}')

    print(f'Training Time: {(time.time() - start_time) / 60:.3f} minutes')


def analyze_attention(dev_source_example, dev_target_example, dev_source_example_txt, dev_target_example_txt, encoder,
                      decoder, epoch):
    with torch.no_grad():

        encoder_hidden = encoder.init_hidden()
        encoder_outputs = torch.zeros(dev_source_example.size(0), encoder.hidden_size)

        for i in range(dev_source_example.size(0)):
            encoder_output, encoder_hidden = encoder(dev_source_example[i], encoder_hidden)
            encoder_outputs[i] = encoder_output.view(-1)

        decoder_hidden = decoder.init_hidden()
        decoder_context = torch.zeros(1, decoder.hidden_size)
        attentions = []
        decoder_input = dev_target_example[0]

        for i in range(dev_target_example.size(0) - 1):
            decoder_output, decoder_hidden, decoder_context, decoder_attention = decoder(
                encoder_outputs, decoder_input, decoder_context, decoder_hidden)
            decoder_input = np.argmax(decoder_output.data, axis=1)
            attentions.append(decoder_attention.data)

        create_heatmap(attentions, epoch, dev_source_example, dev_source_example_txt, dev_target_example_txt, OUTPUT_DIR_PATH)
        write_epoch(attentions, epoch, OUTPUT_DIR_PATH, FILE_NAME)


def main(train_source_path, train_target_path, dev_source_path, dev_target_path):
    # In case there is not output directory - create new directory
    if not os.path.exists(OUTPUT_DIR_PATH):
        os.makedirs(OUTPUT_DIR_PATH)

    if os.path.exists(OUTPUT_DIR_PATH + FILE_NAME):
        os.remove(OUTPUT_DIR_PATH + FILE_NAME)

    # Parse input
    train_data, train_src, train_trg = parse_input(train_source_path, train_target_path)

    # Filter non frequent words and return mapping from word to index and index to word
    source_to_index, index_to_source = convert_data_format(train_src)
    target_to_index, index_to_target = convert_data_format(train_trg)

    # Replace the words with their indexes.
    train_source_index, train_target_index = data_to_index(train_data, source_to_index, target_to_index)

    # Parse dev input
    dev_data, dev_source, dev_target = parse_input(dev_source_path, dev_target_path)

    # Initialize the encoder
    encoder = Encoder(embedding_dim=128, hidden_size=128, num_embeddings=len(source_to_index))

    # Initialize the decoder
    decoder = Decoder(embedding_dim=128, num_embeddings=len(target_to_index), hidden_size=256)

    # Convert to indexes representation
    dev_src_idx, dev_trg_idx = data_to_index(dev_data, source_to_index, target_to_index)

    # Training
    dev_src_text = [[word if word in source_to_index else '<?>' for word in seq] for seq in dev_source]
    dev_trg_text = [[word if word in target_to_index else '<?>' for word in seq] for seq in dev_target]
    train(to_tensor(train_source_index), to_tensor(train_target_index), encoder, decoder,
          to_tensor(dev_src_idx)[EXAMPLE], to_tensor(dev_trg_idx)[EXAMPLE], dev_src_text[EXAMPLE],
          dev_trg_text[EXAMPLE])


if __name__ == '__main__':
    main(*(sys.argv[i] for i in range(1, 5)))
