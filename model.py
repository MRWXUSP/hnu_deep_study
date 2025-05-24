import torch
import torch.nn as nn
from torchinfo import summary

import timm

class base_net(nn.Module):
    def __init__(self, input_features, num_features=64):
        super().__init__()
        self.num_features = num_features
        self.conv = nn.Sequential(
            nn.Conv2d(input_features, num_features, kernel_size=3, padding=3//2),
            #nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features*2, kernel_size=3, padding=3//2),
            #nn.BatchNorm2d(num_features*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_features*2, num_features, kernel_size=3, padding=3 // 2),
            #nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2),
            #nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_features, num_features, kernel_size=3, padding=3//2),
        )

    def forward(self, x):
        x = self.conv(x)

        return x
class Predictor(nn.Module):
    """ The header to predict age (regression branch) """

    def __init__(self, num_features, num_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features // 4, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(num_features // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(num_features // 4, num_features // 8, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(num_features // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(num_features // 8, num_features // 16, kernel_size=3, padding=3 // 2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(num_features//16, num_classes, kernel_size=1, bias=True)
        #self.dp = nn.Dropout(0.5)
    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)
        #x = self.dp(x)
        x = self.fc(x)
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)
        return x


class Classifier(nn.Module):
    """ The header to predict gender (classification branch) """

    def __init__(self, num_features, num_classes=100):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features // 4, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(num_features // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(num_features // 4, num_features // 8, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(num_features // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(num_features // 8, num_features // 16, kernel_size=3, padding=3 // 2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(num_features//16, num_classes, kernel_size=1, bias=True)
        self.dp = nn.Dropout(0.4)


    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)

        x = self.dp(x)
        x = self.fc(x)
        x = x.squeeze(-1).squeeze(-1)
        # x = nn.functional.softmax(x, dim=1)
        return x

#https://github.com/NICE-FUTURE/predict-gender-and-age-from-camera/tree/master
class Model(nn.Module):
    """ A model to predict age and gender """

    def __init__(self, timm_pretrained=False):
        super().__init__()

        self.backbone = timm.create_model(
            "resnet18", 
            pretrained=timm_pretrained,
            checkpoint_path='./weights/resnet18/model.safetensors'
        )  # base_net(3, 64) #

        self.predictor = Predictor(self.backbone.num_features)
        # self.classifier = Classifier(self.backbone.num_features)


    def forward(self, x):

        x = self.backbone.forward_features(x)  # shape: B, D, H, W
        age = self.predictor(x)
        #gender = self.classifier(x)

        return age

class Model2(nn.Module):
    """ A model to predict age and gender """

    def __init__(self, timm_pretrained=True):
        super().__init__()

        self.backbone = timm.create_model("resnet18", pretrained=timm_pretrained)  #base_net(3, 64) #

        # self.predictor = Predictor(self.backbone.num_features)
        self.classifier = Classifier(self.backbone.num_features) # 100类概率


    def forward(self, x):

        x = self.backbone.forward_features(x)  # shape: B, D, H, W
        #x = self.backbone.forward(x)  # shape: B, D, H, W
        prob = self.classifier(x)
        #gender = self.classifier(x)

        return prob

class Model3(nn.Module):
    """ A model to predict age and gender """

    def __init__(self, timm_pretrained=False):
        super().__init__()

        self.backbone = base_net(3, 64) # timm.create_model("resnet18", pretrained=timm_pretrained)  #

        # self.predictor = Predictor(self.backbone.num_features)
        self.classifier = Classifier(self.backbone.num_features) # 100类概率


    def forward(self, x):

        #x = self.backbone.forward_features(x)  # shape: B, D, H, W
        x = self.backbone.forward(x)  # shape: B, D, H, W
        prob = self.classifier(x)
        #gender = self.classifier(x)

        return prob
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)
    modelviz = Model().to(device)
    # 打印模型结构
    print(modelviz)
    summary(modelviz, input_size=(2, 3, 256, 256), col_names=["kernel_size", "output_size", "num_params", "mult_adds"])
    # for p in modelviz.parameters():
    #     if p.requires_grad:
    #         print(p.shape)

    input = torch.rand(2, 3, 256, 256).to(device)
    out = modelviz(input)


    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(modelviz, (3, 256, 256), verbose=True, print_per_layer_stat=True)
    print(macs, params)
    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs * 2, params)  # 8个图像的 FLOPs, 这里的结果 和 其他方法应该一致
    print('out:', out.shape, out)
