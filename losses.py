import torch


def get_discriminator_loss(real_x, fake_x, disc_X, adversarial_criterion):
    real_y = disc_X(real_x.detach())
    real_pred_y = torch.ones_like(real_y)
    fake_y = disc_X(fake_x.detach())
    fake_pred_y = torch.zeros_like(fake_y)
    real_pred_loss = adversarial_criterion(real_y, real_pred_y)
    fake_pred_loss = adversarial_criterion(fake_y, fake_pred_y)
    disc_loss = (real_pred_loss + fake_pred_loss) / 2.0
    return disc_loss


# Loss: Generator_loss - Adversarial_loss
def get_generator_adversarial_loss(real_x, gen_XY, disc_X, gen_adv_criterion):
    fake_y = gen_XY(real_x)
    fake_pred_y = disc_X(fake_y)
    gen_adv_loss = gen_adv_criterion(fake_pred_y, torch.ones_like(fake_pred_y))
    return gen_adv_loss, fake_y


# Loss: Generator_loss - Cycle_Consistency_Loss
def get_cycle_consistency_loss(real_x, fake_y, gen_YX, cycle_criterion):
    real_pred_x = gen_YX(fake_y)
    cycle_consistency_loss = cycle_criterion(real_x, real_pred_x)
    return cycle_consistency_loss, real_pred_x


# Loss: Generator_loss - Identity_Loss
def get_identity_loss(real_x, gen_YX, identity_criterion):
    identity_x = gen_YX(real_x)
    identity_loss = identity_criterion(real_x, identity_x)
    return identity_loss, identity_x


# Loss: Generator_loss
def get_generator_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adversarial_criterion, cycle_criterion,
                       identity_criterion, lambda_adversarial=1, lambda_cycle=9, lambda_identity=2.10
                       ):
    adversarial_loss_AB, fake_B = get_generator_adversarial_loss(real_A, gen_AB, disc_A, adversarial_criterion)
    adversarial_loss_BA, fake_A = get_generator_adversarial_loss(real_B, gen_BA, disc_B, adversarial_criterion)
    adversarial_loss = (adversarial_loss_AB + adversarial_loss_BA)

    cycle_loss_ABA, real_pred_A = get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
    cycle_loss_BAB, real_pred_B = get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
    cycle_loss = cycle_loss_ABA + cycle_loss_BAB

    identity_loss_A, identity_A = get_identity_loss(real_A, gen_BA, identity_criterion)
    identity_loss_B, identity_B = get_identity_loss(real_B, gen_AB, identity_criterion)
    identity_loss = identity_loss_A + identity_loss_B

    generator_loss = adversarial_loss * lambda_adversarial + cycle_loss * lambda_cycle + identity_loss * lambda_identity
    return generator_loss, fake_A, fake_B
