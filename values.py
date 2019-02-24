import torch

MAX_LENGTH = 15  # Maximum sentence length to consider
MIN_COUNT = 3    # Minimum word count threshold for trimming
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

device = torch.device("cpu")

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

clip = 50.0
Teacher_Forcing_Ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 100
save_every = 500