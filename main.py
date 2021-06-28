import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torch import nn
from skimage.transform import resize

from utils import reformat_image, resize_image, load_data
from DataGeneration import generate_img, data_generator
from model import VGGModel

from config import num_classes,num_backgrounds,img_dim,epochs,lr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#The loss calculation
def custom_loss(y_pred, y_true):
    locations_loss = nn.BCELoss()(y_pred[:, :4], y_true[:, :4])

    class_loss = nn.CrossEntropyLoss()(y_pred[:, 4:4+num_classes], y_true[:, 4].long())

    appeared_loss = nn.BCELoss()(y_pred[:, -1], y_true[:, -1])

    return torch.mean(locations_loss * y_true[:, -1] + class_loss * y_true[:, -1] + 0.5 * appeared_loss)

#the training loop
def train():
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_losses = []
        batch_count = 0

        for img_batch, target_batch in data_generator(animals,backgrounds):
            optimizer.zero_grad()
            input_batch = torch.cat(list(map(reformat_image, img_batch))).to(device)
            input_batch = input_batch.to(torch.float32)

            target_batch = torch.tensor(target_batch).to(device)
            target_batch = target_batch.float()

            outputs = model(input_batch).float()

            loss = custom_loss(outputs, target_batch)

            epoch_losses.append(loss.item())

            print(f'Epoch: {epoch + 1}/{epochs}, batch: {batch_count + 1}/{40}, loss: {loss.item():.4f}')

            batch_count += 1
            loss.backward()
            optimizer.step()

        avg_epoch_los = np.mean(epoch_losses)
        losses.append(avg_epoch_los)

    return losses

#Final stage. predict and plot the bounding box
def plot_and_predict(orig_img):

    img_H, img_W = orig_img.shape[:-1]
    img = resize(orig_img, (img_dim, img_dim), preserve_range=True)

    model.eval()
    input_img = reformat_image(img).to(device)
    input_img = input_img.to(torch.float32)

    #fid the image into the model and get the output
    outputs = model(input_img)[0].cpu().detach().numpy()

    coord = outputs[:4]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(orig_img)

    appear = np.round(outputs[-1])

    #if object appear, plot the rectangle
    if appear == 1:
        rect = Rectangle((coord[0] * img_W, coord[1] * img_H), coord[2] * img_W, coord[3] * img_H,
                         linewidth=1, edgecolor='r', facecolor='none')

        selected_animal = np.argmax(outputs[4:4+num_classes])
        animal_name = idx2animal[selected_animal]

        plt.title("Predicted Animal: " + str(animal_name))

        ax.add_patch(rect)

    else:
        plt.title("No Animal Detected")

    plt.show()


if __name__ == '__main__':
    animals, idx2animal, backgrounds = load_data()
    model = VGGModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = train()

    plt.plot(losses)
    plt.title("Loss per epoch")
    plt.show()

    gen_img, _, _, _ = generate_img(animals,backgrounds)

    input_value='y'

    while input_value == 'y':

        plot_and_predict(gen_img)
        input_value = input("Enter 'y' to predict new image, or 'n' to exit")

