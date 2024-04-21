import torch.nn as nn


class NLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, input_seq, output_seq):
        super(NLinear, self).__init__()
        self.input_len = input_seq
        self.output_len = output_seq
        self.Linear = nn.Linear(self.input_len, self.output_len)

    def forward(self, x):
        # x: [배치, 입력 길이, 채널]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x.view(-1, 1)  # [배치, 출력 길이, 채널]
