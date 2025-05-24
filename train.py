
import glob
import os.path

# import cv2
import numpy as np
# import rawpy
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


from datasets import BatchDataset
from model import Model, Model2

import torchvision


if __name__ == "__main__":
    # 1.当前版本信息
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.get_device_name(0))

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 2. 设置device信息 和 创建model
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = Model()

    gpus = [0,1] #这里填写你想用的GPU编号
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device:', device)
    model.to(device)

    # 3. dataset 和 data loader, num_workers设置线程数目，pin_memory设置固定内存
    img_size = 256
    transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2, hue=0.1),
        #transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        transforms.RandomRotation(degrees=(-90, 90)),
    ])
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size, img_size]),
    ])
    train_dataset = BatchDataset('train', transform1)
    train_dataset_loader = DataLoader(train_dataset, batch_size=32*4, shuffle=True, num_workers=8, pin_memory=True)

    eval_dataset = BatchDataset('eval', transform2)
    eval_dataset_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    print('load dataset !', len(train_dataset), len(eval_dataset))

    # 4. 损失函数 和  优化器
    age_criterion = nn.MSELoss()
    gender_criterion = nn.CrossEntropyLoss().to(device)
    loss_fn = nn.L1Loss().to(device)
    loss_fn2 = nn.SmoothL1Loss().to(device)

    learning_rate = 1 * 1e-4
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    lr_step = 50
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, gamma=0.5)

    # 5. hyper para 设置
    epochs = 800

    save_epoch = 100
    save_model_dir = 'saved_model_age'

    eval_epoch = 100
    save_sample_dir = 'saved_sample_age'
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # 6. 是否恢复模型
    resume = 0
    last_epoch = 0
    if resume and last_epoch > 1:
        model.load_state_dict(torch.load(
            save_model_dir + '/checkpoint_%04d.pth' % (last_epoch),
            map_location=device))
        print('resume ' , save_model_dir + '/checkpoint_%04d.pth' % (last_epoch))

    # 7. 训练epoch

    f1 = open('traininfo1.txt', 'a')
    f2 = open('evalinfo1.txt', 'a')

    for epoch in range(last_epoch + 1, epochs + 1):
        print('current epoch:', epoch, 'current lr:', optimizer.state_dict()['param_groups'][0]['lr'])
        if epoch < last_epoch + 101:
            save_epoch = 2
            eval_epoch = 2
        else:
            save_epoch = 10
            eval_epoch = 10
        # 8. train loop
        model.train()
        g_loss = []
        g_mae = []
        for data in tqdm(train_dataset_loader):
            image, age, filename = data
            # print(image.shape, age, filename)
            image = image.to(device)
            age = age.to(device)

            pred_age = model(image)
            #print(image.shape, pred_age.shape)
            loss = loss_fn(age, pred_age)
            #loss = age_criterion(age, pred_age)
            #print('dd:', age.detach().cpu().numpy().reshape(-1), pred_age.detach().cpu().numpy().reshape(-1))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # training result
            g_loss.append(loss.item())
            mae = np.sum(np.abs(age.detach().cpu().numpy().reshape(-1) - pred_age.detach().cpu().numpy().reshape(-1))) / len(age)
            g_mae.append(mae)
            #print( loss.item(), mae)
        #print(len(g_loss), len(g_mae))
        mean_loss = np.mean(np.array(g_loss))
        mean_mae = np.mean(np.array(g_mae))
        print(f'epoch{epoch:04d} ,train loss: {mean_loss},train mae: {mean_mae}')
        f1.write("%d, %.6f, %.4f\n" % (epoch, mean_loss, mean_mae))

        # 9. save model
        if epoch % save_epoch == 0:
            save_model_path = os.path.join(save_model_dir, f'checkpoint_{epoch:04d}.pth')
            torch.save(model.state_dict(), save_model_path)
        # 10. eval test and save some samples if needed

        if epoch % eval_epoch == 0:
            model.eval()
            maes = []
            with torch.no_grad():

                for data in  tqdm(eval_dataset_loader):
                    image, age, filename = data
                    image = image.to(device)
                    age = age.to(device)

                    out = model(image)
                    mae = loss_fn(out, age)
                    #print( age.detach().cpu().numpy().reshape(-1), out.detach().cpu().numpy().reshape(-1), mae.item())
                    maes.append(mae.item())

                print('eval dataset  mae: ', np.array(maes).mean())
                f2.write("%d, %.6f\n" % (epoch,  np.array(maes).mean()))
        scheduler.step()  # 更新学习率

