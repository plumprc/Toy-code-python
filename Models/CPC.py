import torch
import torch.nn as nn
import numpy as np

class CPC(nn.Module):
    def __init__(self, seq_len, K, in_dim, d_model):
        """
        seq_len: sequence length
        K: future K steps to predict
        """
        super(CPC, self).__init__()

        self.seq_len = seq_len
        self.K = K
        self.z_dim = in_dim
        self.c_dim = d_model

        self.encoder = nn.Sequential( 
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, seq_len)
        )
        self.gru = nn.GRU(in_dim, self.c_dim, num_layers=1, bidirectional=False, batch_first=True)
        
        # Predictions
        self.Wk = nn.ModuleList([nn.Linear(self.c_dim, self.z_dim) for _ in range(self.K)])
        self.softmax = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size, device, use_gpu=True):
        if use_gpu: return torch.zeros(1, batch_size, self.c_dim).to(device)
        else: return torch.zeros(1, batch_size, self.c_dim)

    def forward(self, x, hidden):
        batch_size = x.size()[0]
        # z: [batch_size, seq_len, z_dim]
        z = self.encoder(x.transpose(1, 2)).transpose(1, 2)

        # Pick timestep to be the last in the context, time_C, later ones are targets 
        highest = self.seq_len - self.K # 96 - 3 = 93
        time_C = torch.randint(highest, size=(1,)).long() # between 0 and 93

        # z_t_k: [K, batch_size, z_dim]
        z_t_k = z[:, time_C + 1:time_C + self.K + 1, :].clone().cpu().float()
        z_t_k = z_t_k.transpose(1, 0)

        z_0_T = z[:,:time_C + 1,:]
        output, hidden = self.gru(z_0_T, hidden)
        
        # Historical context information
        c_t = output[:, time_C, :].view(batch_size, self.c_dim)
        
        # For the future K timesteps, predict their z_t+k, 
        pred_c_k = torch.empty((self.K, batch_size, self.z_dim)).float() # e.g. size 12*8*512
        
        for k, proj in enumerate(self.Wk):
            pred_c_k[k] = proj(c_t)
        
        nce = 0
        for k in np.arange(0, self.K):
            # [batch_size, z_dim] x [z_dim, batch_size]
            zWc = torch.mm(z_t_k[k], torch.transpose(pred_c_k[k],0,1))         
            logsof_zWc = self.lsoftmax(zWc)
            nce += torch.sum(torch.diag(logsof_zWc))
            
        nce /= -1. * batch_size * self.K
        
        argmax = torch.argmax(self.softmax(zWc), dim=0)
        correct = torch.sum(torch.eq(argmax, torch.arange(0, batch_size))) 
        accuracy = 1. * correct.item() / batch_size

        return accuracy, nce, hidden

    def predict(self, x, hidden):        
        z = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        output, hidden = self.gru(z, hidden)

        return output, hidden


if __name__ == '__main__':
    import torch.optim as optim
    
    x = torch.rand(8, 96, 7)
    model = CPC(96, 3, 7, 512)
    model_optim = optim.Adam(model.parameters(), lr=0.001)
    hidden = model.init_hidden(len(x), 'cpu', use_gpu=False)
    for epoch in range(100):
        acc, nce, _ = model(x, hidden)
        print('acc:', acc, 'loss:', nce.item())
        nce.backward()
        model_optim.step()
        model_optim.zero_grad()
    
    print(acc)
