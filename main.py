import os

from data_preprocess import loadPrepareData, trimRareWords
import torch
import torch.nn as nn
from torch import optim
from model import EncoderRNN, LuongAttnDecoderRNN
from train import trainIters, GreedySearchDecoder, evaluateInput
from values import device, Hidden_Size, Encoder_n_layers, \
    Decoder_n_layers, Dropout_rate, Batch_size, MIN_COUNT, Teacher_Forcing_Ratio

if __name__ == "__main__":

    mode = "train"  ## "train" or "eval"
    # attn_model = 'dot'
    attn_model = 'general'
    # attn_model = 'concat'
    # 'concat' will result nan loss, didn't try it whether could get right model

    # import the message data
    voc, qa_pairs = loadPrepareData("QQbot", "formatted.txt")

    # show some content in the pairs
    # printsample(qa_pairs)

    if mode == "train":

        # print(torch.mean(torch.Tensor(list(map(len,sentences))))) ##tensor(9.0420)
        # mean length of sentence is 9.0420, then let the max_length become 15

        # Trim voc and pairs
        pairs = trimRareWords(voc, qa_pairs, min_count=MIN_COUNT)

        # Configure models
        model_name = 'cb_model'
        hidden_size = Hidden_Size
        encoder_n_layers = Encoder_n_layers
        decoder_n_layers = Decoder_n_layers
        dropout = Dropout_rate
        batch_size = Batch_size

        # Configure training/optimization
        clip = 50.0
        teacher_forcing_ratio = Teacher_Forcing_Ratio
        learning_rate = 0.0001
        decoder_learning_ratio = 3.0
        n_iteration = 5000
        print_every = 100
        save_every = 500

        save_dir = os.path.join("model_save")
        corpus_name = "formatted"

        # Set checkpoint to load from; set to None if starting from scratch
        loadFilename = None
        checkpoint_iter = 4000
        # loadFilename = os.path.join(save_dir, model_name, corpus_name,
        #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
        #                            '{}_checkpoint.tar'.format(checkpoint_iter))

        # Load model if a loadFilename is provided
        if loadFilename:
            # If loading on same machine the model was trained on
            checkpoint = torch.load(loadFilename)
            # If loading a model trained on GPU to CPU
            # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
            encoder_sd = checkpoint['en']
            decoder_sd = checkpoint['de']
            encoder_optimizer_sd = checkpoint['en_opt']
            decoder_optimizer_sd = checkpoint['de_opt']
            embedding_sd = checkpoint['embedding']
            voc.__dict__ = checkpoint['voc_dict']

        print('Building encoder and decoder ...')
        # Initialize word embeddings
        embedding = nn.Embedding(voc.num_words, hidden_size)
        if loadFilename:
            embedding.load_state_dict(embedding_sd)
        # Initialize encoder & decoder models
        encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
        if loadFilename:
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)
        # Use appropriate device
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        print('Models built and ready to go!')

        # Ensure dropout layers are in train mode
        encoder.train()
        decoder.train()

        # Initialize optimizers
        print('Building optimizers ...')
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
        if loadFilename:
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        # Run training iterations
        print("Starting Training!")
        trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
                   decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers,
                   save_dir, n_iteration, batch_size, print_every, save_every, clip,
                   corpus_name, loadFilename)

        torch.save(encoder, 'encoder.pkl')
        torch.save(decoder, 'decoder.pkl')
        print("saved models")


        encoder = torch.load('encoder.pkl')
        decoder = torch.load('decoder.pkl')

        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()

        # Initialize search module
        searcher = GreedySearchDecoder(encoder, decoder)

        # Begin chatting (uncomment and run the following line to begin)
        evaluateInput(encoder, decoder, searcher, voc)


    elif mode == "eval":
        encoder = torch.load('encoder.pkl')
        decoder = torch.load('decoder.pkl')

        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()

        # Initialize search module
        searcher = GreedySearchDecoder(encoder, decoder)

        # Begin chatting (uncomment and run the following line to begin)
        evaluateInput(encoder, decoder, searcher, voc)

    else:
        print("wrong mode choice input!")
