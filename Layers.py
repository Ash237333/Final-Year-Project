import torch
from torch import nn

EMBEDDING_DIMENSION = 128

def positional_encoder(input):
    """
    Defines an encoder for adding positional information to vector embeddings

    :param input: Tensor after batch embeddings (seq_length, batch_size, embedding_dim)
    :return: Positional encoding added to embedding dim (seq_length, batch_size, embedding_dim)
    """

    # Initialises the result tensor as an empty tensor of shape (seq_length, embedding_dim)
    seq_length, _, embedding_dim = input.shape
    result = torch.empty(seq_length, embedding_dim)

    # Calculates the denominator taking into account the difference for even and odd embedding positions
    positions = torch.arange(start=0, end=seq_length, step=1, dtype=torch.float).unsqueeze(1)
    embedding_positions = torch.arange(start=0, end=embedding_dim, step=1, dtype=torch.float)
    calculate_i = (embedding_positions // 2) * 2
    divisor = 10000 ** (calculate_i / embedding_dim)

    # Calculates the full term within the sin/cos function
    positions = positions / divisor

    # Applies sin/cos based on if the number is odd or even
    result[:, ::2] = torch.sin(positions[:, ::2])
    result[:, 1::2] = torch.cos(positions[:, 1::2])

    # Adds the positional encoding to the embedded vectors
    # Broadcasts up in batch dim as pos info doesn't depend on batch index
    result = input + result.unsqueeze(1)

    return result

class Encoder_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=EMBEDDING_DIMENSION, num_heads=8, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
        )
        self.layer_norm = nn.LayerNorm(EMBEDDING_DIMENSION)
        self.layer_norm2 = nn.LayerNorm(EMBEDDING_DIMENSION)


    def forward(self, input, padded_mask):
        #Multi-headed attention
        #input.shape = (B, Seq_length, Embedding_dim)
        #Padding mask.shape = (B, seq_length)
        mha, _ = self.mha(input, input, input, key_padding_mask=padded_mask)
        mha = self.layer_norm(input + mha)

        #Feed Forward network
        feed_forward = self.feed_forward(mha)
        feed_forward = self.layer_norm2(mha + feed_forward)
        print(feed_forward.shape)

        # Output same shape as input
        return feed_forward

class Decoder_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=EMBEDDING_DIMENSION, num_heads=8, batch_first=True)
        self.mha2 = nn.MultiheadAttention(embed_dim=EMBEDDING_DIMENSION, num_heads=8,  batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
        )
        self.layer_norm = nn.LayerNorm(EMBEDDING_DIMENSION)
        self.layer_norm2 = nn.LayerNorm(EMBEDDING_DIMENSION)
        self.layer_norm3 = nn.LayerNorm(EMBEDDING_DIMENSION)


    def forward(self, target_seq, encoder_output, enc_padding_mask, target_padding_mask, target_subsequent_mask):
        #Self Attention layer
        #,key_padding_mask=target_padding_mask

        #target_seq.shape = (B, seq_length, Embedding_dim) - english labels
        #target_padding_mask.shape = (B, seq_length)
        #target_subsequent_mask.shape = (seq_length, seq_length)
        #encoder_output.shape = (B, seq_length, Embedding_dim) - german input
        #enc_padding_mask.shape = (B, seq_length)
        #German and english seq lengths will differ


        mha, _ = self.mha(target_seq, target_seq, target_seq, key_padding_mask=target_padding_mask)
        mha = self.layer_norm(target_seq + mha)

        #Encoder output attention layer
        mha2, _ = self.mha2(mha, encoder_output, encoder_output, key_padding_mask=enc_padding_mask, attn_mask=target_subsequent_mask)
        mha2 = self.layer_norm2(mha + mha2)

        #Feed Forward network
        feed_forward = self.feed_forward(mha2)
        feed_forward = self.layer_norm3(feed_forward + mha2)
        return feed_forward
