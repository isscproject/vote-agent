import os
import torch

def load_checkpoint(filename,model):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location='cpu') # ,map_location='cpu'
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    return model
