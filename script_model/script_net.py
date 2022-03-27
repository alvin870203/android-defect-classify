# Use torch=1.7.0 torchvision=0.8.0

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.mobile_optimizer import optimize_for_mobile

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 173 * 173, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def script_Net():
    with torch.no_grad():
        orig_model = Net()
        orig_model.load_state_dict(torch.load("./script_model/net-0327_0135-weight.pt"))
        orig_model.eval()
        scripted_orig_model = torch.jit.script(orig_model)
        print(scripted_orig_model.code)
        scripted_orig_model.save("./script_model/net-0327_0135-mobile.pt")


def test_export_torchvision_format():

    from typing import List, Dict
    class Wrapper(torch.nn.Module):
        def __init__(self, orig_model):
            super().__init__()
            self.model = orig_model
            
            # must create transforms in __init__, cannot create them in forward
            # because transforms belong to nn.Module
            self.center_crop = T.CenterCrop(size=704)
            self.normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        def forward(self, inputs: List[torch.Tensor]):
            with torch.no_grad():
                x = inputs[0].unsqueeze(0) * 255  # multiply 255 to increase floating point precision during interpolate
                scale = 705.0 / min(x.shape[-2], x.shape[-1])  # larger target size to prevent index out-of-bound after rounding
                x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=True, recompute_scale_factor=True)
                x = self.center_crop(x)
                x = x / 255
                x = self.normalize(x)
                out = F.softmax(self.model(x), dim=1) * 100  # out%: confidence percentage 
                result : Dict[str, torch.Tensor] = {}
                result["round"] = out[0][0]
                result["sharp"] = out[0][1]
                return inputs, [result]

    with torch.no_grad():        
        orig_model = torch.jit.load("./script_model/net-0327_0135-mobile.pt")
        orig_model.eval()
        
        wrapped_model = Wrapper(orig_model)
        wrapped_model.eval()
        
        # optionally do a forward
        print(wrapped_model([torch.rand(3, 4032, 1728)]))
        
        scripted_model = torch.jit.script(wrapped_model)
        scripted_model.save("./DefectClassification/app/src/main/assets/net-0327_0135-wrap.pt")
        # optimized_scripted_model = optimize_for_mobile(scripted_model)
        # optimized_scripted_model.save("./DefectClassification/app/src/main/assets/net-0327_0135-wrap.pt")


if __name__ == "__main__":
    assert torch.__version__ == "1.7.0" and torchvision == "0.8.0"
    
    # script_Net()
    test_export_torchvision_format()
    
    # validate saved scripted model
    wrapped_model = torch.jit.load("./DefectClassification/app/src/main/assets/net-0327_0135-wrap.pt")
    print(wrapped_model([torch.rand(3, 4032, 1728)]))
    
    print(wrapped_model.code)