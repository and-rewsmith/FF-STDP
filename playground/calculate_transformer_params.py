input_dim = 1
dim_feedforward = 8
base_sequence_length = 300
pos_dim = 3
nhead = 2
num_decoder_layers = 2

# Input embedding layer parameters
input_embedding_params = input_dim * dim_feedforward + dim_feedforward

# Positional encoding parameters
positional_encoding_params = base_sequence_length * pos_dim

# Self-attention block parameters
num_attention_heads = nhead
num_attention_matrices_per_head = 4  # Query, Key, Value, and Output matrices
attention_head_size = (dim_feedforward + pos_dim) // num_attention_heads
self_attention_params = num_attention_heads * num_attention_matrices_per_head * attention_head_size ** 2

# Multihead attention block parameters
multihead_attention_params = num_attention_heads * num_attention_matrices_per_head * attention_head_size ** 2

# Feedforward network parameters
feedforward_params = (dim_feedforward + pos_dim) * dim_feedforward + dim_feedforward + \
    dim_feedforward * (dim_feedforward + pos_dim) + (dim_feedforward + pos_dim)

# Layer normalization parameters
num_norm_params = 2  # Scale and bias parameters for layer normalization
layer_norm_params = 3 * num_norm_params * (dim_feedforward + pos_dim)  # Three layer norms in the decoder layer

# Transformer decoder layer parameters
decoder_layer_params = self_attention_params + multihead_attention_params + feedforward_params + layer_norm_params

# Total parameters in all decoder layers
total_decoder_params = num_decoder_layers * decoder_layer_params

# Output linear layer parameters
output_layer_params = (dim_feedforward + pos_dim) * input_dim + input_dim

# Total parameters in the model
total_params = input_embedding_params + positional_encoding_params + total_decoder_params + output_layer_params

print(total_params)
