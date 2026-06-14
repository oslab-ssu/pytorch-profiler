import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.profiler import profile, record_function, ProfilerActivity

class RepeatedDNNSNNModel(nn.Module):
    def __init__(self, num_steps=10):
        super(RepeatedDNNSNNModel, self).__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid()

        self.dnn1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  
        )
        self.snn1_fc = nn.Linear(16 * 14 * 14, 64)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.dnn2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.snn2_fc = nn.Linear(32, 10)  
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, x):
        batch_size = x.size(0)

        with record_function("1_DNN_Layer"):
            dnn1_out = self.dnn1(x)
            dnn1_flat = dnn1_out.view(batch_size, -1)
            snn1_input = self.snn1_fc(dnn1_flat)

        with record_function("2_SNN_Layer"):
            mem1 = self.lif1.init_leaky()
            spk1_recording = []
            for step in range(self.num_steps):
                spk1, mem1 = self.lif1(snn1_input, mem1)
                spk1_recording.append(spk1)


        with record_function("3_SNN_Layer"):
            snn2_input_series = []
            for step in range(self.num_steps):
                current_spk1 = spk1_recording[step]
                dnn2_out = self.dnn2(current_spk1)
                snn2_in = self.snn2_fc(dnn2_out)
                snn2_input_series.append(snn2_in)

        with record_function("4_SNN_Layer"):
            mem2 = self.lif2.init_leaky()
            spk2_recording = []
            for step in range(self.num_steps):
                spk2, mem2 = self.lif2(snn2_input_series[step], mem2)
                spk2_recording.append(spk2)

            spk2_recording = torch.stack(spk2_recording)
            output_spikes = spk2_recording.sum(dim=0)

        return output_spikes

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mock_images = torch.randn(4, 1, 28, 28).to(device)  
    mock_targets = torch.randint(0, 10, (4,)).to(device)

    model = RepeatedDNNSNNModel(num_steps=10).to(device)
    criterion = nn.CrossEntropyLoss()

    activities = [ProfilerActivity.CPU,]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,  
        profile_memory=True,  
        with_stack=True,       
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./result') 
    ) as prof:

        with record_function("model_forward"):
            outputs = model(mock_images)
            loss = criterion(outputs, mock_targets)

        with record_function("model_backward"):
            loss.backward()
        torch.cuda.synchronize()

    print("최종 출력 텐서 크기:", outputs.shape)

    # 3. 콘솔에 요약 테이블 출력 (CPU 총 시간 기준 상위 15개 연산)
    print("\n[프로파일링 요약 (CPU 총 시간 기준)]")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    if torch.cuda.is_available():
        print("\n[프로파일링 요약 (CUDA 총 시간 기준)]")
        print(prof.key_averages().table(sort_by="device_time_total", row_limit=15))
