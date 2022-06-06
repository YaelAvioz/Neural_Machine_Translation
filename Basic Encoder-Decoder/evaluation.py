import sys

import sacrebleu

from training import Encoder, Decoder
from utils import *

FILE_PATH = './Output/Test.pred'


def evaluate(src_tensors, trg_tensors, encoder, decoder, index_to_target, target_text):
    predction = []

    with torch.no_grad():

        # Calculate the max sentence length
        max_length = max([len(sentence) for sentence in trg_tensors])
        for index, (source_seq, target_seq) in enumerate(zip(src_tensors, trg_tensors)):

            input_length = source_seq.size(0)
            encoder_hidden = encoder.init_hidden()

            for i in range(input_length):
                encoder_output, encoder_hidden = encoder(source_seq[i], encoder_hidden)

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

        bleu_ref = [' '.join(ref) for ref in target_text]
        bleu = sacrebleu.corpus_bleu(predction, [bleu_ref])

    # Create Test.pred file
    if os.path.exists(FILE_PATH):
        os.remove(FILE_PATH)

    f = open(FILE_PATH, 'a+')

    for prediction in predction:
        f.write("%s\n" % prediction[len('<s>') + 1:])

    f.close()

    # Returns the bleu score
    return bleu


def main(encoder_file_path, decoder_file_path, dictionary_file_path, test_source_path, test_target_path):
    dictionary = torch.load(dictionary_file_path)

    source_to_index = dictionary['source_to_index']
    target_to_index = dictionary['target_to_index']
    index_to_target = dictionary['index_to_target']

    # Initialize the encoder
    encoder = Encoder(embedding_dim=128, hidden_size=128, num_embeddings=len(source_to_index))

    # Initialize the decoder
    decoder = Decoder(num_embeddings=len(target_to_index), embedding_dim=128, hidden_size=256)

    # Load the state of the encoder and the decoder
    encoder.load_state_dict(torch.load(encoder_file_path))
    decoder.load_state_dict(torch.load(decoder_file_path))

    encoder.eval()
    decoder.eval()

    test_data, _, test_trg = parse_input(test_source_path, test_target_path)

    dev_trg_text = [[word if word in target_to_index else '<?>' for word in seq] for seq in test_trg]

    # Convert the data to index
    test_src_idx, test_trg_idx = data_to_index(test_data, source_to_index, target_to_index)

    bleu = evaluate(to_tensor(test_src_idx), to_tensor(test_trg_idx), encoder, decoder, index_to_target, dev_trg_text)

    print(f'BLEU Score is: {bleu.score:.3f}\t')


if __name__ == '__main__':
    main(*(sys.argv[i] for i in range(1, 6)))
