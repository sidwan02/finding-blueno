# class TrainModel():
#     def __init__(self, num_epochs, train_image_path, train_mask_path, test_image_path, test_mask_path, optimizer, loss_func):

#         for epoch in range(num_epochs):
#             running_loss = 0
#             running_accuracy = 0

from model_known_good import UNET
import time
import torch
import torch.nn as nn
from PIL import Image
import os
import numpy as np

my_path = os.path.dirname(__file__)


def train(model, loss_fn, optimizer, acc_fn, epochs=1):
    # start = time.time()
    model.cuda()

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                # dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                # dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            x_arr = np.array(Image.open(
                my_path + '\processed_images\images\processed_0000047.jpg').convert('RGB'))
            # moves the channels so image is 3, ..., ...
            # x_arr = np.moveaxis(x_arr, -1, 0)
            # print('x_arr: ', x_arr)
            # add batch dimension using .unsqueeze
            #
            x = torch.from_numpy(x_arr).unsqueeze(0)
            print(x.shape)
            # make in the format of in_channels, batches, dim1, dim2
            # float conversion because conv2d cannot support byte
            # cuda because model is on GPU
            # x = x.permute(3, 0, 2, 1).float().cuda()
            x = x.permute(0, 3, 2, 1).float().cuda()
            print(x.shape)
            outputs = model(x)

            # you can give axis attribute if you wanna squeeze in specific dimension
            arr = outputs.detach().cpu().numpy()
            arr = np.squeeze(arr)
            img = Image.fromarray(arr, 'RGB')
            img.save('my.png')
            # plt.imshow(arr)
            # plt.show()

            # iterate over data
            # for x, y in dataloader:
            #     x = x.cuda()
            #     y = y.cuda()
            #     step += 1

            # forward pass
            # if phase == 'train':
            #     # zero the gradients
            #     optimizer.zero_grad()
            #     outputs = model(x)
            #     loss = loss_fn(outputs, y)

            #     # the backward pass frees the graph memory, so there is no
            #     # need for torch.no_grad in this training pass
            #     loss.backward()
            #     optimizer.step()
            #     # scheduler.step()

            # else:
            #     with torch.no_grad():
            #         outputs = model(x)
            #         loss = loss_fn(outputs, y.long())

            #     # stats - whatever is the phase
            #     acc = acc_fn(outputs, y)

            #     running_acc += acc*dataloader.batch_size
            #     running_loss += loss*dataloader.batch_size

            #     if step % 10 == 0:
            #         # clear_output(wait=True)
            #         print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(
            #             step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
            #         # print(torch.cuda.memory_summary())

            # epoch_loss = running_loss / 1
            # epoch_acc = running_acc / 1

            # print('{} Loss: {:.4f} Acc: {}'.format(
            #     phase, epoch_loss, epoch_acc))

            # train_loss.append(
            #     epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)

    # time_elapsed = time.time() - start
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss


def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()


unet = UNET()

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(unet.parameters(), lr=0.1)
train_loss, valid_loss = train(unet, loss_fn, opt, acc_metric, 1)
