import torch

def fgsm(model, loss_fn, images, labels, epsilon):

    images = images.clone().detach().to(images.device)

    images.requires_grad = True

    outputs = model(images)
    loss = loss_fn(outputs, labels)

    model.zero_grad()
    loss.backward()

    grad_sign = images.grad.sign()

    atk_images = images + epsilon * grad_sign
    
    return atk_images
