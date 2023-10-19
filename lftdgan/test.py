"""
 > Script for testing .pth models  
    * set model_name ('funiegan'/'ugan') and  model path
    * set data_dir (input) and sample_dir (output) 
"""
# py libs
import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from collections import OrderedDict

## options
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/home/zlh/data/zsj/data/UIBE/test/LQ/")
# parser.add_argument("--data_dir", type=str, default="/home/zlh/data/zsj/data/UIQS/E/")
parser.add_argument("--sample_dir", type=str, default="result/LFT-DGAN/")
parser.add_argument("--model_name", type=str, default="lftdgan") #
parser.add_argument("--model_path", type=str, default="models/model_best.pth")
parser.add_argument("--no_spectral", default=True, help="")
opt = parser.parse_args()

## checks
assert exists(opt.model_path), "model not found"
os.makedirs(opt.sample_dir, exist_ok=True)
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor 

## model arch

if opt.model_name.lower()=='lftdgan':
    from nets.LFT_DGAN import LFT_DGAN
    model = LFT_DGAN(base_model='lft-dgan').netG
else: 

    pass

#################################3
def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

## load weights
# model.load_state_dict(torch.load(opt.model_path))

load_checkpoint(model, opt.model_path)

if is_cuda: model.cuda()
model.eval()
print ("Loaded model from %s" % (opt.model_path))

## data pipeline
img_width, img_height, channels = 256, 256, 3
transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),             # 对图像进行裁剪
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)

def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 256

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]



## testing loop
times = []
test_files = sorted(glob(join(opt.data_dir, "*.*")))
for path in test_files:
    inp_img = transform(Image.open(path))
    inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
    # generate enhanced image
    s = time.time()
    input_noisy, pad_left_noisy, pad_right_noisy, pad_top_noisy, pad_bottom_noisy = pad_tensor(inp_img)  # 这个是处理unet特征不对齐问题 调整图像大小
    gen_img = model(input_noisy)
    gen_img = pad_tensor_back(gen_img, pad_left_noisy, pad_right_noisy, pad_top_noisy, pad_bottom_noisy)
    times.append(time.time()-s)
    # save output
    # img_sample = torch.cat((inp_img.data, gen_img.data), -1)
    save_image(gen_img.data, join(opt.sample_dir, basename(path)), normalize=True)
    print ("Tested: %s" % path)

## run-time    
if (len(times) > 1):
    print ("\nTotal samples: %d" % len(test_files)) 
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
    print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
    print("Saved generated images in in %s\n" %(opt.sample_dir))



