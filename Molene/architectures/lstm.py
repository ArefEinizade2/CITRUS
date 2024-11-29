import torch


class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, window, hidden_size, output_size=None, device='cpu'):
        super(MV_LSTM, self).__init__()
        if not output_size:
            output_size = n_features
        self.n_features = n_features
        self.seq_len = window

        self.n_hidden = hidden_size  # number of hidden states
        self.n_layers = 1  # number of LSTM layers (stacked)

        self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                    hidden_size=self.n_hidden,
                                    num_layers=self.n_layers,
                                    batch_first=True)
        self.l_linear = torch.nn.Linear(self.n_hidden, output_size)

        self.hidden = None
        self.device = device


    def init_hidden(self, b_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, b_size, self.n_hidden).float().to(self.device)
        cell_state = torch.zeros(self.n_layers, b_size, self.n_hidden).float().to(self.device)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        b_size = x.size()[0]

        self.init_hidden(b_size)

        lstm_out, self.hidden = self.l_lstm(x, self.hidden)

        last_output = lstm_out[:, -1, :]
        return self.l_linear(last_output)