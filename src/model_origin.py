import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class RepeatedDNNSNNModel(nn.Module):
    def __init__(self, num_steps=10):
        super().__init__()

        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid()

        self.dnn1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.snn1_fc = nn.Linear(16 * 14 * 14, 64)
        self.lif1 = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad
        )

        self.dnn2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.snn2_fc = nn.Linear(32, 10)
        self.lif2 = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad
        )

    def forward(self, x):
        batch_size = x.size(0)

        dnn1_out = self.dnn1(x)
        dnn1_flat = dnn1_out.view(batch_size, -1)

        snn1_input = self.snn1_fc(dnn1_flat)

        mem1 = self.lif1.init_leaky()
        spk1_recording = []

        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(snn1_input, mem1)
            spk1_recording.append(spk1)

        snn2_input_series = []

        for spk1 in spk1_recording:
            dnn2_out = self.dnn2(spk1)
            snn2_in = self.snn2_fc(dnn2_out)
            snn2_input_series.append(snn2_in)

        mem2 = self.lif2.init_leaky()
        spk2_recording = []

        for snn2_in in snn2_input_series:
            spk2, mem2 = self.lif2(snn2_in, mem2)
            spk2_recording.append(spk2)

        spk2_recording = torch.stack(spk2_recording)
        output_spikes = spk2_recording.sum(dim=0)

        return output_spikes


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RepeatedDNNSNNModel(num_steps=10).to(device)

    mock_images = torch.randn(4, 1, 28, 28).to(device)
    mock_targets = torch.randint(0, 10, (4,)).to(device)

    criterion = nn.CrossEntropyLoss()

    outputs = model(mock_images)
    loss = criterion(outputs, mock_targets)

    loss.backward()

    print("Output shape:", outputs.shape)
    print("Loss:", loss.item())
