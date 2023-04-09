import os
from tqdm import tqdm
import numpy as np
import torch
import PIL
from torch import nn
from models import Generator, Discriminator
from losses import get_discriminator_loss, get_generator_loss
from utils import show_tensor_images, dehaze_patchMap

torch.cuda.empty_cache()

adversarial_criterion = nn.MSELoss()
# adversarial_criterion = nn.KLDivLoss(reduction="batchmean")
recon_criterion = nn.L1Loss()
EPOCHS = 1
dim_A = 3
dim_B = 3
DISPLAY_STEP = 200
BATCH_SIZE = 12
learning_rate = 0.000197
save_dir = "Saved_models"
# target_shape = 256
device = 'cuda'


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def train(data_loaded,
          gen_AB,
          gen_BA,
          disc_A,
          disc_B,
          gen_opt,
          disc_A_opt,
          disc_B_opt,
          target_shape=64,
          do_save_model=False,
          do_visualize=True):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = data_loaded
    current_step = 0

    for epoch in range(EPOCHS):
        if epoch == (EPOCHS - 1):
            do_save_model = True

        for real_A, real_B in tqdm(dataloader):
            curr_batch_size = len(real_A)
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            disc_A_opt.zero_grad()
            with torch.no_grad():
                fake_A = gen_BA(real_B)

            disc_A_loss = get_discriminator_loss(real_A, fake_A, disc_A, adversarial_criterion)
            disc_A_loss.backward(retain_graph=True)
            disc_A_opt.step()

            disc_B_opt.zero_grad()
            with torch.no_grad():
                fake_B = gen_AB(real_A)

            disc_B_loss = get_discriminator_loss(real_B, fake_B, disc_B, adversarial_criterion)
            disc_B_loss.backward(retain_graph=True)
            disc_B_opt.step()

            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_generator_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B,
                                                          adversarial_criterion, recon_criterion, recon_criterion)
            gen_loss.backward(retain_graph=True)
            gen_opt.step()

            mean_discriminator_loss += disc_A_loss.item() / DISPLAY_STEP
            mean_generator_loss += gen_loss.item() / DISPLAY_STEP

            # Code for Visualization
            if current_step % DISPLAY_STEP == 0:
                print()
                print(
                    f"Epoch {epoch}: Step {current_step}: Generator (U-Net) loss: {mean_generator_loss}, "
                    f"Discriminator loss: {mean_discriminator_loss}")
                if do_visualize:
                    show_tensor_images(torch.cat([real_A, real_B]), size=(dim_A, target_shape, target_shape))
                    show_tensor_images(torch.cat([fake_B, fake_A]), size=(dim_B, target_shape, target_shape))
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if do_save_model:
                    torch.save({
                        'gen_AB': gen_AB.state_dict(),
                        'gen_BA': gen_BA.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc_A': disc_A.state_dict(),
                        'disc_A_opt': disc_A_opt.state_dict(),
                        'disc_B': disc_B.state_dict(),
                        'disc_B_opt': disc_B_opt.state_dict()
                    }, os.path.join(save_dir, f"cycleGAN_{current_step}.pth"))
            current_step += 1
    return gen_AB, gen_BA


def do_train(data_loaded, photo_data_loaded, pretrained_path=None, pretrained=False, do_visualize=True):
    # ________________________________________________________________________________________________ #
    gen_AB = Generator(dim_A, dim_B).to(device)
    gen_BA = Generator(dim_B, dim_A).to(device)
    gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=learning_rate,
                               betas=(0.5, 0.999))
    disc_A = Discriminator(dim_A).to(device)
    disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    disc_B = Discriminator(dim_B).to(device)
    disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    # _________________________________________________________________________________________________ #
    if pretrained:
        pre_dict = torch.load(pretrained_path)  # Give the path of the pre-saved file
        gen_AB.load_state_dict(pre_dict['gen_AB'])
        gen_BA.load_state_dict(pre_dict['gen_BA'])
        gen_opt.load_state_dict(pre_dict['gen_opt'])
        disc_A.load_state_dict(pre_dict['disc_A'])
        disc_A_opt.load_state_dict(pre_dict['disc_A_opt'])
        disc_B.load_state_dict(pre_dict['disc_B'])
        disc_B_opt.load_state_dict(pre_dict['disc_B_opt'])
    else:
        gen_AB = gen_AB.apply(weights_init)
        gen_BA = gen_BA.apply(weights_init)
        disc_A = disc_A.apply(weights_init)
        disc_B = disc_B.apply(weights_init)
    generator_AB, generator_BA = train(data_loaded, gen_AB, gen_BA, disc_A, disc_B, gen_opt, disc_A_opt, disc_B_opt,
                                       do_save_model=False, do_visualize=do_visualize)
    get_results(generator_AB, photo_data_loaded)
    return generator_AB, generator_BA


def get_results(generator, photo_data_loaded):
    i = 1
    for real_A in tqdm(photo_data_loaded):
        real_A = real_A.to(device)
        with torch.no_grad():
            prediction = generator(real_A)[0].cpu().numpy()
        prediction = (prediction * 127.5 + 127.5).T.astype(np.uint8)
        result_image = PIL.Image.fromarray(prediction)
        output = result_image.rotate(270)
        output.save("./Results/dehazed{}.jpg".format(i))
        torch.cuda.empty_cache()
        i += 1
    return
