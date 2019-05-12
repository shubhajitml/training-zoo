import torch.nn as nn

extras = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
#‘S’ means a stride=2 and padding=1 convolution layer,
# the number of filters comes next in the list (for example, the first ‘S’ has 512 filters).

def add_extras(cfg, i, batch_norm=False):
    # extra layers added to vgg for feature scaling 
    layers = []
    in_channels = i
    flag = False

    for k,v in enumerate(cfg):
        if in_channels != 'S':
            if v=='S':
                layers += [nn.Conv2d(in_channels, cfg[k+1], 
                            kernel_size=(1,3)[flag], stride=2, padding=1 )]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1,3)[flag])]
            flag = not flag
        in_channels = v
    return layers

