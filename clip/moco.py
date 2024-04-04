import torchvision.models as models
import torch
import torch.nn as nn
import os
from torchvision.models import resnet50

def load_moco(pretrain_path):
    print("=> creating model")
    model = resnet50()
    linear_keyword = 'fc'
    if os.path.isfile(pretrain_path):
        print("=> loading checkpoint '{}'".format(pretrain_path))
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        assert set(msg.missing_keys) == {f"{linear_keyword}.weight", f"{linear_keyword}.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrain_path))
    else:
        print("=> no checkpoint found at '{}'".format(pretrain_path))
        raise FileNotFoundError
    model.fc = nn.Identity()
    return model, 2048

if __name__ == "__main__":
    
    model = load_moco("resnet50", ).cuda()
    #print(model)
    print(model(torch.rand(32,3,224,224).cuda()).shape)

    
    
    