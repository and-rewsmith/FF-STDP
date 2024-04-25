import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def generate_waveforms(num_sequences, sequence_length, num_modes, freq_range, amp_range, phase_range):
    """
    Generates a dataset of waveforms, each composed of a sum of sinusoidal modes.

    Parameters:
    - num_sequences (int): Number of waveform sequences to generate.
    - sequence_length (int): Number of samples in each waveform sequence.
    - num_modes (int): Number of sinusoidal modes per sequence.
    - freq_range (tuple): Range of frequencies (min_freq, max_freq).
    - amp_range (tuple): Range of amplitudes (min_amp, max_amp).
    - phase_range (tuple): Range of phases (min_phase, max_phase).

    Returns:
    - np.array: Array of shape (num_sequences, sequence_length) containing the generated waveforms.
    """
    waveforms = np.zeros((num_sequences, sequence_length))
    # Create a time array from 0 to 1, not inclusive, scaled by sequence length for better frequency handling
    t = np.linspace(0, 1, sequence_length, endpoint=False)
    for i in range(num_sequences):
        for _ in range(num_modes):
            frequency = np.random.uniform(*freq_range)
            amplitude = np.random.uniform(*amp_range)
            phase = np.random.uniform(*phase_range)
            # Adjust frequency to fit within the time array scale
            waveforms[i] += amplitude * np.sin(2 * np.pi * frequency * t + phase)

    return waveforms


def create_batches(data, batch_size):
    """
    Splits the data into batches.

    Parameters:
    - data (np.array): Input data to be batched.
    - batch_size (int): Size of each batch.

    Returns:
    - list of np.array: List containing batches of data.
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, pos_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(sequence_length, pos_dim))

    def forward(self, x):
        # x is the input features, shape: (batch_size, sequence_length, feature_dim)
        # Expand pos_embedding to match batch size and concatenate
        pos_embedding = self.pos_embedding.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat((x, pos_embedding), dim=2)


class TransformerDecoderModel(nn.Module):
    def __init__(self, input_dim, pos_dim, sequence_length, nhead, num_decoder_layers, dim_feedforward):
        super(TransformerDecoderModel, self).__init__()
        self.input_dim = input_dim
        self.pos_dim = pos_dim
        self.sequence_length = sequence_length
        self.pos_encoder = PositionalEncoding(sequence_length, pos_dim)
        self.embedding = nn.Linear(input_dim, dim_feedforward)  # Project to dim_feedforward
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=dim_feedforward + pos_dim, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_decoder_layers
        )
        self.fc_out = nn.Linear(dim_feedforward + pos_dim, input_dim)  # Project back to input_dim

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, src):
        # src shape should be (batch_size, sequence_length, 1)
        src = src.unsqueeze(-1) if src.dim() == 2 else src
        src = self.embedding(src)
        src = self.pos_encoder(src)
        # Transformer expects shape (seq_len, batch_size, feature)
        src = src.permute(1, 0, 2)
        tgt_mask = self.generate_square_subsequent_mask(src.size(0))
        # Forward through the transformer decoder
        output = self.transformer_decoder(tgt=src, memory=src, tgt_mask=tgt_mask)
        # Convert back to (batch_size, seq_len, feature)
        output = output.permute(1, 0, 2)
        # Project back to input_dim
        output = self.fc_out(output)
        return output


def train(model, dataloader, loss_fn, optimizer, num_epochs):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            print("batch")
            # print(len(batch))
            # print(batch[0].shape)
            # input()
            input_batch = batch[0].unsqueeze(-1)
            target_batch = batch[0].unsqueeze(-1)

            optimizer.zero_grad()
            output_batch = model(input_batch)
            loss = loss_fn(output_batch, target_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')
    return losses

# Evaluation and plotting function


def evaluate_and_plot(model, dataloader, num_sequences_to_plot=3):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_sequences_to_plot:
                break
            input_batch = batch[0].unsqueeze(-1)
            output_batch = model(input_batch)
            input_waveform = input_batch.squeeze().numpy()
            output_waveform = output_batch.squeeze().numpy()

            plt.figure(figsize=(10, 4))
            plt.plot(input_waveform[i], label='Original')
            plt.plot(output_waveform[i], label='Predicted', linestyle='--')
            plt.legend()
            plt.title(f'Waveform Sequence {i+1}')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.savefig(f'waveform_sequence_{i+1}.png')


# Main script
if __name__ == "__main__":
    # set torch random seed
    torch.manual_seed(0)

    # Example parameters
    num_sequences = 1024  # Changed to a larger number for more training data
    sequence_length = 1024
    num_modes = 5
    freq_range = (1, 5)
    amp_range = (0.5, 1.0)
    phase_range = (0, 2 * np.pi)

    # Generate waveforms
    waveforms = generate_waveforms(num_sequences, sequence_length, num_modes, freq_range, amp_range, phase_range)
    waveforms_tensor = torch.tensor(waveforms, dtype=torch.float32)

    print("generated waveforms: ", waveforms_tensor.shape)

    # Split data into train and test sets (80-20 split)
    split_index = int(num_sequences * 0.8)
    train_waveforms = waveforms_tensor[:split_index]
    test_waveforms = waveforms_tensor[split_index:]

    # Create datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(train_waveforms)
    test_dataset = torch.utils.data.TensorDataset(test_waveforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model instantiation
    model = TransformerDecoderModel(
        input_dim=1,
        pos_dim=16,
        sequence_length=sequence_length,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=512
    )

    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # Train the model
    losses = train(model, train_dataloader, loss_fn, optimizer, num_epochs=5)

    # Plot the training loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig('loss_curve.png')

    # Evaluate the model on the test data and plot predictions
    evaluate_and_plot(model, test_dataloader)
