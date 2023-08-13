import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import create_mirror_masks
    
class BasicActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(BasicActor, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = torch.FloatTensor(max_action).to(device)
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)

    def forward(self, x):
        res = F.relu(self.layer1(x))
        res = F.relu(self.layer2(res))
        res = torch.tanh(self.layer3(res))
        return self.max_action * res


class BasicCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BasicCritic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x, a):
        sa = torch.hstack((x, a))
        res = F.relu(self.layer1(sa))
        res = F.relu(self.layer2(res))
        ll_features = res
        res = self.layer3(res)
        return res, ll_features
    

class MirrorActor(BasicActor):
    def __init__(self, state_dim, action_dim, max_action):
        super(MirrorActor, self).__init__(state_dim, action_dim, max_action)

    def forward(self, x):
        state_mirror_mask, action_mirror_mask = create_mirror_masks(x)
        res = super().forward(x)
        return res * action_mirror_mask

class MirrorCritic(BasicCritic):
    def __init__(self, state_dim, action_dim):
        super(MirrorCritic, self).__init__(state_dim, action_dim)

    def forward(self, x, a):
        state_mirror_mask, action_mirror_mask = create_mirror_masks(x)
        x = x * state_mirror_mask
        a = a * action_mirror_mask
        res, ll_features = super().forward(x, a)
        return res, ll_features
