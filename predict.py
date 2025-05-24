import glob
import os.path

import cv2
import numpy as np
# import rawpy
import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.metrics import structural_similarity as compare_ssim

# from datasets import BatchDataset, get_images
from model import UNetSeeInDark, Model, Model2

if __name__ == "__main__":
    img_size = 256
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size, img_size]),
    ])

    model = Model()
    gpus = [0,1]  #这里填写你想用的GPU编号
    # model = nn.DataParallel(model, device_ids=gpus)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=gpus)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    m_path = 'saved_model_age/checkpoint_0014.pth'
    #m_path = 'saved_model_res18_reg/checkpoint_0010.pth'
    checkpoint = torch.load(m_path, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
    #model.load_state_dict(torch.load(m_path, map_location=device))



    # model = model.cuda(device=gpus[0])
    model = model.to(device)
    model.eval()

    files = glob.glob("./testset/*.jpg")

    # image_dir = 'valset\\valset'
    # file_txt = 'annotations\\annotations\\val.txt'
    # files = get_images(image_dir, file_txt)
    print(len(files))

    f = open('predict_res50_14.txt', 'w')
    st = ''

    ret = []
    for file in files:
        # file, label = file
        image = Image.open(file).convert('RGB')
        # image = cv2.imread(file, 1).astype(np.float32) / 255
        image = np.array(image)
        input = transform2(image).unsqueeze(0).to(device)
        #print(input.shape)

        out = model(input)
        out = out.detach().cpu().numpy().reshape(-1)

        pred_age = out[0]
        #pred_age = np.sum(out * np.arange(0, 100).reshape(1, -1)) * 2 + 1
        #print(int(label), pred_age, np.abs(pred_age -int(label)))
        #ret.append([int(label), pred_age, pred_age -int(label), np.abs(pred_age -int(label))])
        #print(out)
        st = os.path.basename(file)+'\t%.2f\n' % (pred_age.item())
        f.write(st)

    # ret = np.array(ret)
    # print(ret)
    # print(np.mean(ret, axis=0))
    #np.savetxt('ret54.txt', ret+2, fmt='%.1f', delimiter=' ')
