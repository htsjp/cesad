import torch
import torch.nn.functional as F


def hard_label_loss(image_posterior, patch_posterior, image_label, patch_mask=None):
    image_target = torch.tensor([int(image_label)], device=image_posterior.device)
    image_loss = F.nll_loss(torch.log(image_posterior.clamp_min(1e-6)), image_target)
    patch_loss = patch_posterior[..., 1].mean() * 0.0
    if patch_mask is not None:
        target = patch_mask.reshape(1, -1).to(image_posterior.device).float()
        patch_loss = F.binary_cross_entropy(patch_posterior[..., 1].clamp(1e-6, 1 - 1e-6), target)
    return image_loss + 0.1 * patch_loss


def preserve_loss(student_image, teacher_image, student_patch, teacher_patch):
    return (1.0 - F.cosine_similarity(student_image, teacher_image.detach(), dim=-1).mean()) + (
        1.0 - F.cosine_similarity(student_patch.reshape(-1, student_patch.shape[-1]), teacher_patch.detach().reshape(-1, teacher_patch.shape[-1]), dim=-1).mean()
    )


def memory_distillation_loss(student_img_p, teacher_img_p, student_patch_p, teacher_patch_p, student_img_margin, teacher_img_margin, student_patch_margin, teacher_patch_margin):
    p_loss = F.mse_loss(student_img_p[..., 1], teacher_img_p[..., 1].detach())
    p_loss = p_loss + F.mse_loss(student_patch_p[..., 1], teacher_patch_p[..., 1].detach())
    m_loss = F.mse_loss(student_img_margin, teacher_img_margin.detach())
    m_loss = m_loss + F.mse_loss(student_patch_margin, teacher_patch_margin.detach())
    return p_loss + 0.1 * m_loss
