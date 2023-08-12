# 包
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from config import *
from VAE import *
import utils


# 设备配置
torch.cuda.set_device(1) # 这句用来设置pytorch在哪块GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 如果没有文件夹就创建一个文件夹
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


args = get_config()

# dataset = torchvision.datasets.MNIST(root='data/minist',
#                                      train=True,
#                                      transform=transforms.ToTensor(),
#                                      download=True)

# 数据加载器
# data_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                           batch_size=args.batch_size,
#                                           shuffle=True)

train_loader, test_loader, train_set = utils.make_dataloader(args)

# 实例化一个模型
model = VAE(args.image_size, args.h_dim, args.z_dim).to(device)

# 创建优化器
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

for epoch in range(args.num_epochs):
    for i, (x, _) in enumerate(train_loader):
        # 获取样本，并前向传播
        x = x.to(device).view(-1, args.image_size)
        x_reconst, mu, log_var = model(x)

        # 计算重构损失和KL散度（KL散度用于衡量两种分布的相似程度）
        # KL散度的计算可以参考论文或者文章开头的链接
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # 反向传播和优化
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 5 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                  .format(epoch + 1, args.num_epochs, i + 1, len(train_loader), reconst_loss.item(), kl_div.item()))

    # 利用训练的模型进行测试
    with torch.no_grad():
        # 随机生成的图像
        z = torch.randn(args.batch_size, args.z_dim).to(device)
        out = model.decode(z).view(-1, 1, 128, 128)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))

        # 重构的图像
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 1, 128, 128), out.view(-1, 1, 128, 128)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))



