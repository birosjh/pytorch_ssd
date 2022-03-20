from simple_ssd import SSD

import torch

net = SSD(num_classes=1)

my_tensor = torch.ones(32, 3, 256, 256)

bbox_preds, cls_preds = net(my_tensor)
