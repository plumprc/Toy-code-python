import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from mnist import MLP

device = 'cpu'
transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root='datasets', train=False, transform=transform, download=False)
mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=1, shuffle=True)

def deepfool_attack(image, model, num_classes=10, max_iter=50, epsilon=0.02):
    image = image.to(device).requires_grad_()
    label = torch.argmax(model(image))

    perturbation = torch.zeros_like(image).to(device)
    f_i = model(image)[0]

    for _ in range(max_iter):
        output = model(image + perturbation)
        _, adversarial_label = torch.max(output, dim=1)

        if adversarial_label != label:
            break

        loss = -output[0, label]  # Define a loss for the target class
        grads = torch.autograd.grad(loss, image, retain_graph=True)[0]
        grads_norm = torch.norm(grads)

        if grads_norm < 1e-10:
            break

        perturbation += epsilon * grads / grads_norm
        f_i = model(image + perturbation)[0]

    adversarial_image = torch.clamp(image + perturbation, 0.0, 1.0)
    
    return adversarial_image, adversarial_label, label


if __name__ == '__main__':
    model = torch.load('checkpoints/mlp.pth').to(device)
    model.eval()

    # Choose an example from the dataset
    example_index = 0
    original_image, label = next(iter(mnist_loader))
    original_image = original_image.to(device)

    # Generate an adversarial image using DeepFool
    adversarial_image, adversarial_label, label = deepfool_attack(original_image, model)

    # Display the original and adversarial images
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image[0][0].cpu().detach().numpy(), cmap='gray')
    plt.title("Original Image:" + str(label.item()))
    plt.subplot(1, 2, 2)
    plt.imshow(adversarial_image[0][0].cpu().detach().numpy(), cmap='gray')
    plt.title("Adversarial Image:" + str(adversarial_label.item()))
    plt.show()
