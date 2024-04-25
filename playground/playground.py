import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def generate_waveforms(num_sequences, full_sequence_length, num_modes, freq_range, amp_range, phase_range):
    waveforms = np.zeros((num_sequences, full_sequence_length))
    t = np.linspace(0, 3, full_sequence_length, endpoint=False)
    for i in range(num_sequences):
        for _ in range(num_modes):
            frequency = np.random.uniform(*freq_range)
            amplitude = np.random.uniform(*amp_range)
            phase = np.random.uniform(*phase_range)
            waveforms[i] += amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return waveforms


class PositionalEncoding(nn.Module):
    def __init__(self, max_length, pos_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(max_length, pos_dim))

    def forward(self, x):
        return torch.cat((x, self.pos_embedding[:x.size(1), :].unsqueeze(0).expand(x.size(0), -1, -1)), dim=2)


class TransformerDecoderModel(nn.Module):
    def __init__(self, input_dim, pos_dim, base_sequence_length, nhead, num_decoder_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward).to(device)
        self.pos_encoder = PositionalEncoding(base_sequence_length, pos_dim).to(device)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=dim_feedforward + pos_dim, nhead=nhead,
                                       dim_feedforward=dim_feedforward).to(device),
            num_layers=num_decoder_layers
        ).to(device)
        self.fc_out = nn.Linear(dim_feedforward + pos_dim, input_dim).to(device)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        output = self.transformer_decoder(src, src)
        output = output.permute(1, 0, 2)
        return self.fc_out(output)


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


def prepare_data(waveforms, base_sequence_length):
    inputs = []
    targets = []
    for waveform in waveforms:
        for start in range(waveform.shape[0] - base_sequence_length):
            end = start + base_sequence_length
            inputs.append(waveform[start:end])
            targets.append(waveform[start+1:end+1])  # Shifted by one for targets
    return torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1).to(device), torch.tensor(targets, dtype=torch.float32).unsqueeze(-1).to(device)


if __name__ == "__main__":
    num_epochs = 1
    num_sequences = 1
    base_sequence_length = 300  # Consistent sequence length for training and inference
    full_sequence_length = base_sequence_length * 3  # Longer sequence to enable sliding window
    num_modes = 1
    freq_range = (1, 5)
    amp_range = (0.5, 1.0)
    phase_range = (0, 2 * np.pi)

    waveforms = generate_waveforms(num_sequences, full_sequence_length, num_modes, freq_range, amp_range, phase_range)
    inputs, targets = prepare_data(waveforms, base_sequence_length)

    print(inputs.shape)
    print(targets.shape)

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    model = TransformerDecoderModel(
        input_dim=1,
        pos_dim=10,
        base_sequence_length=base_sequence_length,
        nhead=4,
        num_decoder_layers=3,
        dim_feedforward=50
    ).to(device)

    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train(model, dataloader, loss_fn, optimizer, num_epochs)

    # Autoregressive inference function to generate predictions after training
    def autoregressive_inference(model, initial_input, total_length):
        model.eval()
        with torch.no_grad():
            input_sequence = initial_input.clone()
            predictions = []
            while len(predictions) < (total_length - initial_input.size(1)):
                output = model(input_sequence)
                next_value = output[:, -1:, :]  # Taking the last output as the next input
                predictions.append(next_value)
                input_sequence = torch.cat((input_sequence, next_value), dim=1)  # Append to the input sequence
            return torch.cat(predictions, dim=1)

    # Visualizing the predictions
    test_input = inputs[:1, :, :]  # Take the first sample for testing
    predicted_output = autoregressive_inference(model, test_input, full_sequence_length)
    predicted_output = predicted_output.squeeze().cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(full_sequence_length), waveforms[0], label='Original Full Waveform')
    plt.plot(np.arange(base_sequence_length, full_sequence_length),
             predicted_output[base_sequence_length:], label='Predicted Waveform', linestyle='--')
    plt.axvline(x=base_sequence_length, color='r', linestyle=':', label='Start of Prediction')
    plt.legend()
    plt.title('Comparison of Original and Predicted Waveforms')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.savefig('waveform_comparison.png')
    plt.show()
