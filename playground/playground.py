import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Check for MPS availability and use it
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class TransformerDecoderModel(nn.Module):
    def __init__(self, input_dim, pos_dim, sequence_length, nhead, num_decoder_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward).to(device)
        self.pos_encoder = PositionalEncoding(sequence_length, pos_dim).to(device)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=dim_feedforward + pos_dim, nhead=nhead,
                                       dim_feedforward=dim_feedforward).to(device),
            num_layers=num_decoder_layers
        ).to(device)
        self.fc_out = nn.Linear(dim_feedforward + pos_dim, input_dim).to(device)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # Shape should be (seq_len, batch_size, feature)
        tgt_mask = self.generate_square_subsequent_mask(src.size(0))
        output = self.transformer_decoder(tgt=src, memory=src, tgt_mask=tgt_mask)
        output = output.permute(1, 0, 2)
        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

# Ensure all data tensors are sent to the appropriate device


def prepare_data(waveforms, window_size):
    inputs = []
    targets = []
    for waveform in waveforms:
        for start in range(waveform.shape[0] - window_size):
            inputs.append(waveform[start:start+window_size])
            targets.append(waveform[start+window_size])
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1).to(device)
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1).to(device)
    return inputs, targets


def generate_waveforms(num_sequences, sequence_length, num_modes, freq_range, amp_range, phase_range):
    """
    Generates a dataset of waveforms, each composed of a sum of sinusoidal modes.
    """
    waveforms = np.zeros((num_sequences, sequence_length))
    t = np.linspace(0, 3, sequence_length, endpoint=False)  # Assume 3 full cycles in the extended sequence
    for i in range(num_sequences):
        for _ in range(num_modes):
            frequency = np.random.uniform(*freq_range)
            amplitude = np.random.uniform(*amp_range)
            phase = np.random.uniform(*phase_range)
            waveforms[i] += amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return waveforms


class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, pos_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(sequence_length, pos_dim))

    def forward(self, x):
        pos_embedding = self.pos_embedding.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat((x, pos_embedding), dim=2)


def train(model, dataloader, loss_fn, optimizer, num_epochs):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        print("epoch")
        print("len dataloader: ", len(dataloader))
        total_loss = 0
        for input_batch, target_batch in dataloader:
            print("batch")
            print("input_batch: ", input_batch.shape)
            optimizer.zero_grad()
            output_batch = model(input_batch)
            print("output_batch: ", output_batch.shape)
            input()
            loss = loss_fn(output_batch[:, -1], target_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')
    return losses


# Example of adjusting the main script:
if __name__ == "__main__":
    base_sequence_length = 256
    num_sequences = 10
    sequence_length = base_sequence_length * 3  # Extended sequence length
    num_modes = 5
    freq_range = (1, 5)
    amp_range = (0.5, 1.0)
    phase_range = (0, 2 * np.pi)

    waveforms = generate_waveforms(num_sequences, sequence_length, num_modes, freq_range, amp_range, phase_range)
    inputs, targets = prepare_data(waveforms, base_sequence_length)

    print("prepared data")
    print("num inputs: ", inputs.shape)
    print("num targets: ", targets.shape)

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = TransformerDecoderModel(
        input_dim=1,
        pos_dim=16,
        sequence_length=base_sequence_length,  # Input window size for the model
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=512
    ).to(device)

    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    print("training model")
    losses = train(model, dataloader, loss_fn, optimizer, num_epochs=1)
    print("trained model")

    # Plot the training loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig('loss_curve.png')

    # Function to perform autoregressive inference
    def autoregressive_inference(model, initial_sequence, total_length=base_sequence_length):
        model.eval()
        with torch.no_grad():
            generated_sequence = initial_sequence.clone()
            current_input = initial_sequence
            while generated_sequence.shape[1] < total_length:
                next_step = model(current_input)[:, -1:, :]  # Predict the next step
                generated_sequence = torch.cat((generated_sequence, next_step), dim=1)  # Append to the sequence
                current_input = torch.cat((current_input[:, 1:, :], next_step), dim=1)  # Slide the window
            return generated_sequence

    print("autoregressive_inference")

    # Select a sample from the dataset to demonstrate autoregressive inference
    sample = inputs[:1]  # Take the first sample from the dataset
    predicted_waveform = autoregressive_inference(model, sample)

    print("predicted_waveform")

    # Plot the original and the predicted waveform
    plt.figure(figsize=(15, 5))
    plt.plot(inputs[0].squeeze().cpu().numpy(), label='Original')
    plt.plot(predicted_waveform.squeeze().cpu().numpy(), label='Predicted', linestyle='--')
    plt.legend()
    plt.title('Comparison of Original and Predicted Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.savefig('predicted_waveform.png')
