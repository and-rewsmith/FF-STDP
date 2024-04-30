import logging
import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb


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
    t = np.linspace(0, 2 * np.pi, sequence_length, endpoint=False)
    for i in range(num_sequences):
        for _ in range(num_modes):
            frequency = np.random.uniform(*freq_range)
            amplitude = np.random.uniform(*amp_range)
            phase = np.random.uniform(*phase_range)
            waveforms[i] += amplitude * np.sin(frequency * t + phase)
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
        return self.fc_out(output[:, -1, :])  # Predict only the next token


def train(model, dataloader, loss_fn, optimizer, num_epochs, saved_loss_curve_filename):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
            print(f"batch: {batch_idx}")
            print(f"input_batch: {input_batch.shape}")
            # input()
            optimizer.zero_grad()
            output_batch = model(input_batch)
            target_batch = target_batch.squeeze(2)  # Remove the extra dimension
            loss = loss_fn(output_batch, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}: Loss {total_loss / len(dataloader)}')
        # wandb.log({"loss": total_loss / len(dataloader)})
        losses.append(total_loss / len(dataloader))

    # Plot loss curve after training
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(losses)), losses, label='Training Loss')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig(saved_loss_curve_filename)  # Save loss curve


def prepare_data(waveforms, base_sequence_length, split_ratio=0.8):
    inputs = []
    targets = []
    for waveform in waveforms:
        for start in range(waveform.shape[0] - base_sequence_length):
            end = start + base_sequence_length
            inputs.append(waveform[start:end])
            targets.append(waveform[end])
    inputs = np.array(inputs)
    targets = np.array(targets)

    # Splitting the data
    split_index = int(len(inputs) * split_ratio)
    train_inputs, test_inputs = inputs[:split_index], inputs[split_index:]
    train_targets, test_targets = targets[:split_index], targets[split_index:]

    # Convert to tensors
    train_inputs = torch.tensor(train_inputs, dtype=torch.float32).unsqueeze(-1).to(device)
    train_targets = torch.tensor(train_targets, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).to(device)
    test_inputs = torch.tensor(test_inputs, dtype=torch.float32).unsqueeze(-1).to(device)
    test_targets = torch.tensor(test_targets, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).to(device)

    return (train_inputs, train_targets), (test_inputs, test_targets)


def train_model_and_plot(num_heads, num_decoder_layers, embedding_dim, num_modes=3, base_sequence_length=300, full_sequence_multiplier=3):
    num_epochs = 3
    num_sequences = 40
    full_sequence_length = base_sequence_length * full_sequence_multiplier
    freq_range = (25, 75)
    amp_range = (0.5, 1.0)
    phase_range = (0, 2 * np.pi)
    batch_size = 256
    plot_trained_autoregressive_inference = False  # If false make sure we have enough sequences
    pos_dim = 2

    waveforms = generate_waveforms(num_sequences, full_sequence_length, num_modes, freq_range, amp_range, phase_range)
    (train_inputs, train_targets), (test_inputs, test_targets) = prepare_data(waveforms, base_sequence_length)

    plt.figure(figsize=(12, 6))

    for i, waveform in enumerate(waveforms):
        plt.plot(np.arange(full_sequence_length), waveform, label=f'Waveform {i+1}')

    plt.legend()
    plt.title('Input Waveforms')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.savefig('input_waveforms.png')

    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerDecoderModel(input_dim=1, pos_dim=pos_dim, base_sequence_length=base_sequence_length,
                                    nhead=num_heads, num_decoder_layers=num_decoder_layers, dim_feedforward=embedding_dim).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    # print(count_parameters(model))

    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    saved_loss_curve_filename = f'loss_curve_{num_params}_{num_heads}_{num_decoder_layers}_{embedding_dim}.png'
    train(model, train_loader, loss_fn, optimizer, num_epochs, saved_loss_curve_filename)

    def evaluate(model, test_loader, loss_fn):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for input_batch, target_batch in test_loader:
                output_batch = model(input_batch)
                target_batch = target_batch.squeeze(2)  # Remove the extra dimension
                loss = loss_fn(output_batch, target_batch)
                total_loss += loss.item()
        average_loss = total_loss / len(test_loader)
        print(f'Test Loss: {average_loss}')
        return average_loss

    # Evaluate the model on the test data
    test_loss = evaluate(model, test_loader, loss_fn)

    # Autoregressive inference function to generate predictions after training
    def autoregressive_inference(model, initial_input, total_length, base_sequence_length):
        model.eval()
        with torch.no_grad():
            input_sequence = initial_input.clone()
            predictions = []
            while len(predictions) < (total_length - initial_input.size(1)):
                output = model(input_sequence)
                predictions.append(output)
                output = output.unsqueeze(1)
                # Append the predicted token to the input sequence
                input_sequence = torch.cat((input_sequence, output), dim=1)
                # Sliding window: remove the oldest token if sequence length exceeds base_sequence_length
                if input_sequence.size(1) > base_sequence_length:
                    input_sequence = input_sequence[:, -base_sequence_length:, :]
            return torch.cat(predictions, dim=1)

    waveform_to_plot = 0 if plot_trained_autoregressive_inference else -1

    # Construct the input to plot up until the prediction point
    sampling_input = waveforms[waveform_to_plot][0:base_sequence_length]
    sampling_input = torch.tensor(sampling_input, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)

    # Autoregressive inference to generate the rest of the waveform
    predicted_output = autoregressive_inference(model,
                                                sampling_input, full_sequence_length, base_sequence_length)

    # Massage into combined output for plotting
    sampling_input = sampling_input.cpu().numpy()
    predicted_output = predicted_output.squeeze().cpu().numpy()
    combined_output = np.concatenate((waveforms[waveform_to_plot][0:base_sequence_length], predicted_output))

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(full_sequence_length), waveforms[waveform_to_plot], label='Original Full Waveform')
    plt.plot(np.arange(full_sequence_length), combined_output, label='Predicted Waveform', linestyle='--')
    plt.axvline(x=base_sequence_length, color='r', linestyle=':', label='Start of Prediction')
    plt.legend()
    plt.title('Comparison of Original and Predicted Waveforms')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.savefig(f'waveform_comparison_{num_params}_{num_decoder_layers}_{num_heads}_{num_decoder_layers}.png')


def set_logging() -> None:
    """
    Must be called after argparse.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    set_logging()

    # torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(precision=10, sci_mode=False)
    rand = random.randint(1000, 9999)
    torch.manual_seed(rand)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # sweep_configuration
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="transformer-poc")
    # wandb.init(project="transformer-poc", config={"architecture": "initial", "dataset": "waves"})

    train_model_and_plot(num_heads=2, num_decoder_layers=2, embedding_dim=8)
