# dl.models.onwDim.mix_cnn_lstm.yaml
import torch
from torch import nn
from src.utils.dim_conv import MultiKernelConv1d


# from torchsummary import summary


class MIX_CNN_LSTM(nn.Module):
    def __init__(
            self,
            input_size: int = 41,
            lin1_size: int = 640,
            lin2_size: int = 256,
            lin3_size: int = 128,
            output_size: int = 40,
            dropout_proba: float = 0.2,
            kernel_sizes=None,
    ):
        super(MIX_CNN_LSTM, self).__init__()

        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5, 7, 9]
            # kernel_sizes = [1, 3]

        self.conv_layer = nn.Sequential(
            MultiKernelConv1d(in_channels=1, out_channels=8, kernel_sizes=kernel_sizes),
        )

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=8 * len(kernel_sizes), out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.lstm_layer1 = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, bidirectional=True)
        self.lstm_layer2 = nn.LSTM(input_size=64, hidden_size=16, num_layers=2, bidirectional=True)
        self.lstm_layer3 = nn.LSTM(input_size=1, hidden_size=16, num_layers=5, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_proba),

            # nn.Linear(32 * 3 * 3, lin1_size),
            nn.Linear(3 * 32 * input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):  # input(1, 8)
        batch_size, feature = x.size()
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1)
        # print(x.shape)

        y = self.conv_layer(x)
        y = self.conv_layer1(y)

        x = x.permute(2, 0, 1)
        z, _ = self.lstm_layer1(x)
        z, _ = self.lstm_layer2(z)
        z3, _ = self.lstm_layer3(x)
        z = z.permute(1, 2, 0)
        z3 = z3.permute(1, 2, 0)

        x = torch.cat((y, z, z3), dim=1)
        # print(x.shape)
        x = self.fc(x)
        return x


class CNN_LSTM(nn.Module):
    def __init__(
            self,
            input_size: int = 41,
            lin1_size: int = 640,
            lin2_size: int = 256,
            lin3_size: int = 128,
            output_size: int = 40,
            dropout_proba: float = 0.2,
            kernel_sizes=None,
    ):
        super(CNN_LSTM, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.lstm_layer1 = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, bidirectional=True)
        self.lstm_layer2 = nn.LSTM(input_size=64, hidden_size=16, num_layers=2, bidirectional=True)
        self.lstm_layer3 = nn.LSTM(input_size=1, hidden_size=16, num_layers=5, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_proba),

            # nn.Linear(32 * 3 * 3, lin1_size),
            nn.Linear(3 * 32 * input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):  # input(1, 8)
        batch_size, feature = x.size()
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1)
        # print(x.shape)

        y = self.conv_layer1(x)

        x = x.permute(2, 0, 1)
        z, _ = self.lstm_layer1(x)
        z, _ = self.lstm_layer2(z)
        z3, _ = self.lstm_layer3(x)
        z = z.permute(1, 2, 0)
        z3 = z3.permute(1, 2, 0)

        x = torch.cat((y, z, z3), dim=1)
        # print(x.shape)
        x = self.fc(x)
        return x


class CNN(nn.Module):
    def __init__(
            self,
            input_size: int = 41,
            lin1_size: int = 640,
            lin2_size: int = 256,
            lin3_size: int = 128,
            output_size: int = 40,
            dropout_proba: float = 0.2,
    ):
        super(CNN, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_proba),

            nn.Linear(32 * input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):  # input(1, 8)
        batch_size, feature = x.size()
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1)
        # print(x.shape)

        x = self.conv_layer1(x)

        x = self.fc(x)
        return x


class LSTM(nn.Module):
    def __init__(
            self,
            input_size: int = 41,
            lin1_size: int = 640,
            lin2_size: int = 256,
            lin3_size: int = 128,
            output_size: int = 40,
            dropout_proba: float = 0.2,
            kernel_sizes=None,
    ):
        super(LSTM, self).__init__()

        self.lstm_layer1 = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, bidirectional=True)
        self.lstm_layer2 = nn.LSTM(input_size=64, hidden_size=16, num_layers=2, bidirectional=True)
        self.lstm_layer3 = nn.LSTM(input_size=1, hidden_size=16, num_layers=5, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_proba),

            # nn.Linear(32 * 3 * 3, lin1_size),
            nn.Linear(2 * 32 * input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):  # input(1, 8)
        batch_size, feature = x.size()
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1)
        # print(x.shape)

        x = x.permute(2, 0, 1)
        z, _ = self.lstm_layer1(x)
        z, _ = self.lstm_layer2(z)
        z3, _ = self.lstm_layer3(x)
        z = z.permute(1, 2, 0)
        z3 = z3.permute(1, 2, 0)

        x = torch.cat((z, z3), dim=1)
        print(x.shape)
        x = self.fc(x)
        return x


class MIX_LSTM(nn.Module):
    def __init__(
            self,
            input_size: int = 41,
            lin1_size: int = 640,
            lin2_size: int = 256,
            lin3_size: int = 128,
            output_size: int = 40,
            dropout_proba: float = 0.2,
            kernel_sizes=None,
    ):
        super(MIX_LSTM, self).__init__()

        self.lstm_layer1 = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, bidirectional=True)
        self.lstm_layer2 = nn.LSTM(input_size=64, hidden_size=16, num_layers=2, bidirectional=True)
        self.lstm_layer3 = nn.LSTM(input_size=32, hidden_size=16, num_layers=2, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_proba),

            # nn.Linear(32 * 3 * 3, lin1_size),
            nn.Linear(4 * 32 * input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):  # input(1, 8)
        batch_size, feature = x.size()
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1)
        # print(x.shape)  # torch.Size([128, 1, 196])

        x = x.permute(2, 0, 1)
        z1, _ = self.lstm_layer1(x)
        z2, _ = self.lstm_layer2(z1)
        z3, _ = self.lstm_layer3(z2)

        z1 = z1.permute(1, 2, 0)
        z2 = z2.permute(1, 2, 0)
        z3 = z3.permute(1, 2, 0)

        x = torch.cat((z1, z2, z3), dim=1)
        # print(x.shape)  # torch.Size([128, 128, 196])
        x = self.fc(x)
        return x

    def tsne_forward(self, x):
        batch_size, feature = x.size()
        x = x.view(batch_size, -1).unsqueeze(1).permute(2, 0, 1)

        z1, _ = self.lstm_layer1(x)
        z2, _ = self.lstm_layer2(z1)
        z3, _ = self.lstm_layer3(z2)

        x_concat = torch.cat((z1.permute(1, 2, 0), z2.permute(1, 2, 0), z3.permute(1, 2, 0)), dim=1)
        x_flat = x_concat.view(x_concat.size(0), -1)

        sequential_outputs = [x_flat]
        for i in range(0, len(self.fc), 3):  # Assuming there's always a triplet: linear -> relu -> dropout
            x_in = sequential_outputs[-1]
            x_out = self.fc[i + 2](self.fc[i + 1](self.fc[i](x_in)))
            sequential_outputs.append(x_out)

        results = {
            'x_input': x,
            'x_lsmt1': z1,
            'x_lstm2': z2,
            'x_lstm3': z3,
            # 'x_concat': x_concat, some wrong
            'x_flat': x_flat,
            'x_lin1': sequential_outputs[1],
            'x_lin2': sequential_outputs[2],
            'x_lin3': sequential_outputs[3],
            'x_output': sequential_outputs[4]
        }

        return results

class MLP(nn.Module):
    def __init__(
            self,
            input_size: int = 41,
            lin1_size: int = 640,
            lin2_size: int = 256,
            lin3_size: int = 128,
            output_size: int = 40,
            dropout_proba: float = 0.2,
    ):
        super(MLP, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Dropout(dropout_proba),

            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == "__main__":
    batch_size = 128
    input_size = 41
    X = torch.rand(batch_size, input_size)
    # print(X)
    # print(MLP()(X))
    # print(CNN()(X))
    # print(LSTM()(X))
    print(MIX_LSTM())
    # print(CNN_LSTM()(X))

    # y = summary(MLP(), input_size=(input_size, ), device='cpu')
