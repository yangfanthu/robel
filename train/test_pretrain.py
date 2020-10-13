import os 
import datetime
import numpy as np
import torch

from utils import ReplayBuffer
from modules import DynamicModel
from tensorboardX import SummaryWriter
import pdb

class Dataset(torch.utils.data.Dataset):
    def __init__(self, buffer, mode="train", train_ratio = 0.9):
        self.buffer = buffer
        self.current_state = buffer.state
        self.next_state = buffer.next_state
        self.action = buffer.action
        total_size = self.current_state.shape[0]
        num_train = int(total_size * train_ratio)
        if mode == "train":
            self.current_state = self.current_state[:num_train]
            self.next_state = self.next_state[:num_train]
            self.action = self.action[:num_train]
        elif mode == "validation":
            self.current_state = self.current_state[num_train:]
            self.next_state = self.next_state[num_train:]
            self.action = self.action[num_train:]
        else:
            raise ValueError

    def __getitem__(self, index):
        return self.current_state[index], self.action[index], self.next_state[index]
    def __len__(self):
        return len(self.current_state)


if __name__ == "__main__":
    outdir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = 'pretrain' + outdir
    outdir = os.path.join('./saved_models', outdir)
    os.system('mkdir ' + outdir)
    writer = SummaryWriter(logdir=('logs/pretrain{}').format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    input_state_dim = 12
    output_state_dim = 9
    action_dim = 9
    hidden_dim = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicModel(input_state_dim=input_state_dim, action_dim=action_dim, output_state_dim=output_state_dim, hidden_dim=hidden_dim)
    model = model.to(device)
    model.load_state_dict(torch.load('./saved_models/model_033.ckpt'))
    buffer = ReplayBuffer(state_dim = input_state_dim, action_dim = action_dim)
    buffer.restore()
    val_set = Dataset(buffer, mode="validation")
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                             batch_size=1,
                                             shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    max_epoch = 100
    train_index = 0
    for epoch in range(max_epoch):
        with torch.no_grad():
            loss_list = []
            for i, (current_state, action, next_state) in enumerate(val_loader):
                current_state = current_state.float().to(device)
                action = action.float().to(device)
                next_state = next_state[:,:output_state_dim].float().to(device)
                predict_state = model(current_state, action)
                pdb.set_trace()
                loss = criterion(predict_state, next_state)
                loss_list.append(loss.item())
            print("Epoch [{}/{}], Validation Loss: {:06f}"
            .format(epoch, max_epoch, np.array(loss_list).mean()))
            writer.add_scalar('val/loss', np.array(loss_list).mean(), epoch)
            writer.close()