import torch

MAX_LENGTH = 15  # Maximum sentence length to consider
MIN_COUNT = 3    # Minimum word count threshold for trimming
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

device = torch.device("cpu")

Hidden_Size = 500  # Hidden cell number, here means the model contain 500 cell
Encoder_n_layers = 2
Decoder_n_layers = 2
Dropout_rate = 0.2
Batch_size = 64
Teacher_Forcing_Ratio = 0.1
