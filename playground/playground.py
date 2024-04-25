import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Waveform Generation


def generate_waveforms(num_sequences, sequence_length, num_modes, freq_range, amp_range, phase_range):
    waveforms = np.zeros((num_sequences, sequence_length))
    t = np.linspace(0, 3, sequence_length, endpoint=False)
    for i in range(num_sequences):
        for _ in range(num_modes):
            frequency = np.random.uniform(*freq_range)
            amplitude = np.random.uniform(*amp_range)
            phase = np.random.uniform(*phase_range)
            waveforms[i] += amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return waveforms

# Positional Encoding Module


class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, pos_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(sequence_length, pos_dim))

    def forward(self, x):
        pos_embedding = self.pos_embedding.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat((x, pos_embedding), dim=2)

# Transformer Decoder Model with Positional Encoding


class TransformerDecoderModel(nn.Module):
    def __init__(self, input_dim, pos_dim, sequence_length, nhead, num_decoder_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        self.pos_encoder = PositionalEncoding(sequence_length, pos_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=dim_feedforward + pos_dim, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_decoder_layers
        )
        self.fc_out = nn.Linear(dim_feedforward + pos_dim, input_dim)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # seq_len, batch, feature
        output = self.transformer_decoder(src, src)  # No mask needed for full sequence in training
        output = output.permute(1, 0, 2)  # Convert back to batch, seq_len, feature
        return self.fc_out(output)

# Training Function using Teacher Forcing


def train(model, dataloader, loss_fn, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for input_batch, target_batch in dataloader:
            optimizer.zero_grad()
            output_batch = model(input_batch)
            loss = loss_fn(output_batch, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}: Loss {total_loss / len(dataloader)}')

# Autoregressive Inference and Visualization


def autoregressive_inference(model, initial_input, total_length):
    model.eval()
    with torch.no_grad():
        input_sequence = initial_input
        predictions = []
        for _ in range(total_length - len(initial_input)):
            output = model(input_sequence)
            next_value = output[:, -1:, :]
            predictions.append(next_value)
            input_sequence = torch.cat((input_sequence, next_value), dim=1)
        return torch.cat(predictions, dim=1)


# Main Script
if __name__ == "__main__":
    # Parameters
    num_epochs = 1
    num_sequences = 50
    sequence_length = 300  # Total length including input and predicted part
    num_modes = 5
    freq_range = (1, 5)
    amp_range = (0.5, 1.0)
    phase_range = (0, 2 * np.pi)

    # Generate Data
    waveforms = generate_waveforms(num_sequences, sequence_length, num_modes, freq_range, amp_range, phase_range)
    inputs = torch.tensor(waveforms, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension

    # Plot the first few waveforms
    plt.figure(figsize=(10, 8))
    for i in range(min(num_sequences, 5)):  # Plot only the first 5 sequences
        plt.plot(waveforms[i], label=f'Sequence {i+1}')
    plt.title('Generated Waveforms')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig('waveforms.png')

    print("inputs shape: ", inputs.shape)
    input()

    # Create Dataset
    dataset = torch.utils.data.TensorDataset(inputs[:, :-1, :], inputs[:, 1:, :])  # Shifted by one for teacher forcing
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    # Initialize Model
    model = TransformerDecoderModel(
        input_dim=1,
        pos_dim=10,
        sequence_length=sequence_length-1,  # Minus one because we're shifting for teacher forcing
        nhead=4,
        num_decoder_layers=3,
        dim_feedforward=50
    )

    # Move model to appropriate device
    device = torch.device("mps")
    model.to(device)

    # Setup Loss and Optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Train the Model
    train(model, dataloader, loss_fn, optimizer, num_epochs)

    # Autoregressive Inference for Visualization
    test_input = inputs[0:1, :int(sequence_length * 0.2), :]  # Use the first 20% as starting input
    predicted_output = autoregressive_inference(model, test_input, sequence_length)
    predicted_output = predicted_output.squeeze().cpu().numpy()

    # Visualization of the results
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(sequence_length), waveforms[0], label='Original Full Waveform')
    plt.plot(np.arange(int(sequence_length * 0.2), sequence_length),
             predicted_output, label='Predicted Waveform', linestyle='--')
    plt.axvline(x=int(sequence_length * 0.2), color='r', linestyle=':', label='Start of Prediction')
    plt.legend()
    plt.title('Comparison of Original and Predicted Waveforms')
    plt.xlabel('Time Steps')
    plt.ylabel('Waveform Amplitude')
    plt.savefig('waveform_comparison.png')
    plt.show()
