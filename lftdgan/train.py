import os
import yaml
import argparse
import time
from tqdm import tqdm
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler

from PIL import Image
# pytorch libs
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
# local libs
from nets.LFT_DGAN import LFT_DGAN
from utils.data_utils import GetTrainingPairs, MixUp_AUG, DCToperator
from nets.loss import VGG19_PercepLoss, Gradient_Difference_Loss, Weights_Normal, Gradient_Penalty


## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="configs/train_UIBE.yaml")                    # 文件配置
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--n_critic", type=int, default=10, help="training steps for D per iter w.r.t G")
parser.add_argument("--no_spectral", default=False, help="which epoch to start from")
args = parser.parse_args()

## training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size =  args.batch_size
lr_rate = args.lr
num_critic = args.n_critic

model_v = "LFT-DGAN"
# load the data config file
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# get info from config file
dataset_name = cfg["dataset_name"]
dataval_name = cfg["dataval_name"]
dataset_path = cfg["dataset_path"]
channels = cfg["chans"]
img_width = cfg["im_width"]
img_height = cfg["im_height"] 
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]

## create dir for model and validation data
samples_dir = "samples/%s/%s" % (model_v, dataset_name)
checkpoint_dir = "checkpoints/%s/%s/" % (model_v, dataset_name)
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


""" loss functions
-------------------------------------------------"""
L1_G  = torch.nn.L1Loss() # l1 loss term
L1_gp = Gradient_Penalty() # wgan_gp loss term
L_gdl = Gradient_Difference_Loss() # GDL loss term
L_per = VGG19_PercepLoss()   # per


# Initialize generator and discriminator

lftdgan_ = LFT_DGAN(base_model='lft-dgan')
generator = lftdgan_.netG
discriminator = lftdgan_.netD
if not args.no_spectral:
    discriminator_s = lftdgan_.netD_s     # 频域鉴别器


# see if cuda is available
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    if not args.no_spectral:
        discriminator_s = discriminator_s.cuda()  # 频域鉴别器

    frequen = DCToperator().cuda()  # frequency_image
    L1_gp.cuda()
    L1_G = L1_G.cuda()
    L_gdl = L_gdl.cuda()
    L_per = VGG19_PercepLoss().cuda()

    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# Initialize weights or load pretrained models
if args.epoch == 0:
    # generator.apply(Weights_Normal)
    discriminator.apply(Weights_Normal)
    if not args.no_spectral:
        discriminator_s.apply(Weights_Normal)
else:
    generator.load_state_dict(torch.load("checkpoints/%s/%s/generator_%d.pth" % (model_v, dataset_name, args.epoch)))
    discriminator.load_state_dict(torch.load("checkpoints/%s/%s/discriminator_%d.pth" % (model_v, dataset_name, epoch)))
    if not args.no_spectral:
        discriminator_s.load_state_dict(torch.load("checkpoints/%s/%s/discriminator_s_%d.pth" % (model_v, dataset_name, epoch)))
    print ("Loaded model from epoch %d" %(epoch))

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr_rate, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)

if not args.no_spectral:
    optimizer_Ds = optim.Adam(discriminator_s.parameters(), lr=lr_rate, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)

## Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

dataloader = DataLoader(
    GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
    batch_size = batch_size,
    shuffle = True,
    num_workers = 1,
)

val_dataloader = DataLoader(
    GetTrainingPairs(dataset_path, dataval_name, transforms_=transforms_),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)

mixup = MixUp_AUG()

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

best_loss = 0
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(dataloader)//3 - 1
print(f"\nEval after every {eval_now} Iterations !!!\n")

######### Scheduler ###########
warmup_epochs = 5
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, num_epochs-warmup_epochs+40, eta_min=1e-6)
scheduler = GradualWarmupScheduler(optimizer_G, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Training pipeline
for epoch in range(epoch, num_epochs):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    for i, batch in enumerate(tqdm(dataloader), 0):
        # Model inputs
        imgs_distorted = Variable(batch["A"].type(Tensor))
        imgs_good_gt = Variable(batch["B"].type(Tensor))

        if epoch>5:
            imgs_good_gt, imgs_distorted = mixup.aug(imgs_good_gt, imgs_distorted)

        ##################################################
        ## Train Discriminator and discriminator_s
        #################################################

        optimizer_D.zero_grad()
        imgs_fake = generator(imgs_distorted)
        pred_real = discriminator(imgs_good_gt)
        pred_fake = discriminator(imgs_fake.detach())                                  #.detach()
        loss_D = -torch.mean(pred_real) + torch.mean(pred_fake) # wgan 
        gradient_penalty = L1_gp(discriminator, imgs_good_gt.data, imgs_fake.data)
        loss_D += 10 * gradient_penalty # Eq.2 paper
        loss_D.backward()
        optimizer_D.step()
        if not args.no_spectral:
            optimizer_Ds.zero_grad()

            imgs_good_gt1 = frequen(imgs_good_gt)
            imgs_fake1 = frequen(imgs_fake)

            pred_real_s1 = discriminator_s(imgs_good_gt1)
            pred_fake_s1 = discriminator_s(imgs_fake1.detach())
            loss_Ds = -torch.mean(pred_real_s1) + torch.mean(pred_fake_s1) # wgan

            loss_Ds.backward()
            optimizer_Ds.step()

        optimizer_G.zero_grad()
        ## Train Generator at 1:num_critic rate 
        if i % num_critic == 0:
            imgs_fake = generator(imgs_distorted)
            pred_fake = discriminator(imgs_fake.detach())
            pred_fake_real = discriminator(imgs_good_gt)
            loss_gen = -torch.mean(pred_fake) + torch.mean(pred_fake_real)

            if not args.no_spectral:

                imgs_good_gt2 = frequen(imgs_good_gt)
                imgs_fake2 = frequen(imgs_fake)
                pred_fake_s2 = discriminator_s(imgs_fake2.detach())
                pred_real_s2 = discriminator_s(imgs_good_gt2)
                loss_gen_s = -torch.mean(pred_real_s2) + torch.mean(pred_fake_s2)  # wgan


            loss_1 = L1_G(imgs_fake, imgs_good_gt)
            loss_gdl = L_gdl(imgs_fake, imgs_good_gt)
            loss_per = L_per(imgs_fake, imgs_good_gt)


            if not args.no_spectral:
                loss_G = loss_gen + loss_gen_s + 10 * loss_1 + loss_gdl + 100 *loss_per
            else:
                loss_G =  loss_gen + 10 * loss_1 +  loss_gdl +  loss_per
            loss_G.backward()
            optimizer_G.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % val_interval == 0:
            imgs1 = next(iter(val_dataloader))
            imgs_val1 = Variable(imgs1["A"].type(Tensor))
            imgs_gen1 = generator(imgs_val1)
            imgs_valR1 = Variable(imgs1["B"].type(Tensor))
            img_sample1 = torch.cat((imgs_val1.data, imgs_gen1.data,imgs_valR1.data), -2)

            imgs2 = next(iter(val_dataloader))
            imgs_val2 = Variable(imgs2["A"].type(Tensor))
            imgs_gen2 = generator(imgs_val2)
            imgs_valR2 = Variable(imgs2["B"].type(Tensor))
            img_sample2 = torch.cat((imgs_val2.data, imgs_gen2.data,imgs_valR2.data), -2)

            imgs3 = next(iter(val_dataloader))
            imgs_val3 = Variable(imgs3["A"].type(Tensor))
            imgs_gen3 = generator(imgs_val3)
            imgs_valR3 = Variable(imgs3["B"].type(Tensor))
            img_sample3 = torch.cat((imgs_val3.data, imgs_gen3.data, imgs_valR3.data), -2)

            imgs4 = next(iter(val_dataloader))
            imgs_val4 = Variable(imgs4["A"].type(Tensor))
            imgs_gen4 = generator(imgs_val4)
            imgs_valR4 = Variable(imgs4["B"].type(Tensor))
            img_sample4 = torch.cat((imgs_val4.data, imgs_gen4.data, imgs_valR4.data), -2)

            img_sample = torch.cat((img_sample1, img_sample2, img_sample3, img_sample4), dim=0)
            save_image(img_sample, "samples/%s/%s/%s.png" % (model_v, dataset_name, batches_done), nrow=5, normalize=True)

        #### Evaluation ####
        if i % eval_now == 0 and  epoch > 0:
            generator.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate((val_dataloader), 0):
                input_ = Variable(data_val["A"].type(Tensor))
                target = Variable(data_val["B"].type(Tensor))


                with torch.no_grad():
                    restored = generator(input_)
                restored = restored[0]

                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(torchPSNR(res, tar))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()


            print("[epoch %d it %d PSNR: %.4f --- best_epoch %d best_iter %d Best_PSNR %.4f]" % (
            epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))

            torch.save({'epoch': epoch,
                        'state_dict': generator.state_dict(),
                        'optimizer': optimizer_G.state_dict()
                        }, os.path.join("checkpoints/LFT-DGAN/train", f"model_epoch_{epoch}.pth"))

            generator.train()


    scheduler.step()
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, loss_G, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch,
                'state_dict': generator.state_dict(),
                'optimizer': optimizer_G.state_dict()
                }, os.path.join("checkpoints/LFT-DGAN/train", "last_model.pth"))
