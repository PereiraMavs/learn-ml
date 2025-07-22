import argparse
import os  # Make sure this is included
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from model.generator import Generator
from model.discriminator import Discriminator

manual_seed = 999

random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.use_deterministic_algorithms(True)

# Update these global variables for CIFAR-10
dataroot = './data'  # CIFAR-10 will be downloaded here
image_size = 64      # Change from 32 to 64
batch_size = 128
workers = 2  # Add this missing variable
nc = 3  # Number of channels in the training images. For color images, this is 3.
nz = 100  # Size of the latent z vector.
ngf = 64  # Size of feature maps in the generator.
ndf = 64  # Size of feature maps in the discriminator.
num_epochs = 5
lr = 0.0002  # Learning rate for optimizers.
beta1 = 0.5  # Beta1 hyperparameter for Adam optimizers.
ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def get_data():
    # Use CIFAR-10 dataset instead of ImageFolder
    dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.Resize(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]))

    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    # Optional: Show sample images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("CIFAR-10 Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True), (1, 2, 0)))
    
    return dataloader



#plt.show()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def create_generator():
    netG = Generator(ngpu).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    netG.apply(weights_init)
    #print(netG)
    return netG


def create_discriminator():
    netD = Discriminator(ngpu, ndf, nc).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netD.apply(weights_init)
    #print(netD)
    return netD

def loss_function_optimizer(netD, netG):
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    return criterion, fixed_noise, optimizerD, optimizerG


def save_models(netG, netD, epoch, G_loss, D_loss, model_dir='saved_models'):
    """Save generator and discriminator models"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save generator
    torch.save({
        'epoch': epoch,
        'model_state_dict': netG.state_dict(),
        'loss': G_loss,
    }, os.path.join(model_dir, f'generator_epoch_{epoch}.pth'))
    
    # Save discriminator
    torch.save({
        'epoch': epoch,
        'model_state_dict': netD.state_dict(),
        'loss': D_loss,
    }, os.path.join(model_dir, f'discriminator_epoch_{epoch}.pth'))
    
    print(f'Models saved at epoch {epoch}')

def save_best_models(netG, netD, epoch, G_loss, D_loss, model_dir='best_models'):
    """Save the best performing models"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save best generator
    torch.save({
        'epoch': epoch,
        'model_state_dict': netG.state_dict(),
        'loss': G_loss,
    }, os.path.join(model_dir, 'best_generator.pth'))
    
    # Save best discriminator
    torch.save({
        'epoch': epoch,
        'model_state_dict': netD.state_dict(),
        'loss': D_loss,
    }, os.path.join(model_dir, 'best_discriminator.pth'))
    
    print(f'Best models updated at epoch {epoch}')


def main():
    dataloader = get_data()
    print("Data loaded successfully.")

    netG = create_generator()
    print("Generator created successfully.")

    netD = create_discriminator()
    print("Discriminator created successfully.")

    criterion, fixed_noise, optimizerD, optimizerG = loss_function_optimizer(netD, netG)
    print("Loss function and optimizers initialized successfully.")

    real_label = 1
    fake_label = 0
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    
    # Track best models
    best_G_loss = float('inf')
    best_D_loss = float('inf')
    best_combined_loss = float('inf')

    print("Starting training...")
    num_epochs = 20
    print("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        epoch_G_losses = []
        epoch_D_losses = []
        
        # for each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device, dtype=torch.float)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item() # Calculate D(x), the discriminator's output for real images

            ## Train with fake images
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\t'
                      f'Loss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\t'
                      f'D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            epoch_G_losses.append(errG.item())
            epoch_D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        
        # Calculate average losses for this epoch
        avg_G_loss = sum(epoch_G_losses) / len(epoch_G_losses)
        avg_D_loss = sum(epoch_D_losses) / len(epoch_D_losses)
        avg_combined_loss = avg_G_loss + avg_D_loss
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Avg G_loss: {avg_G_loss:.4f}, Avg D_loss: {avg_D_loss:.4f}')
        
        # Save models every epoch
        save_models(netG, netD, epoch+1, avg_G_loss, avg_D_loss)
        
        # Save best models based on combined loss
        if avg_combined_loss < best_combined_loss:
            best_combined_loss = avg_combined_loss
            best_G_loss = avg_G_loss
            best_D_loss = avg_D_loss
            save_best_models(netG, netD, epoch+1, avg_G_loss, avg_D_loss)
            print(f'New best models found! Combined loss: {best_combined_loss:.4f}')
    
    print(f'Training completed. Best combined loss: {best_combined_loss:.4f}')
    print(f'Best models saved in ./best_models/')
    print(f'All epoch models saved in ./saved_models/')


if __name__ == "__main__":
    main()
    print("Training completed successfully.")