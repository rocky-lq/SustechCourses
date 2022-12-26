import matplotlib.pyplot as plt
import torch
from torchsummary import summary

from cnn_model import CNNSpeakerIdentificationModel


def vis_model():
    # model summary
    model = CNNSpeakerIdentificationModel()
    model.load_state_dict(torch.load('model_single.pth'))
    summary(model.cuda(), input_size=(1, 64, 586), batch_size=-1)

    # convert to onnx, then use netron to visualize
    x = torch.rand((8, 1, 64, 586), dtype=torch.float).cuda()
    with torch.no_grad():
        torch.onnx.export(
            model.cuda(),
            x,
            'model.onnx',
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
        )


def show_train_record():
    # read the train.log file
    with open('train.log', 'r') as f:
        train_acc_list = f.readline().split(',')
        train_loss_list = f.readline().split(',')
        val_acc_list = f.readline().split(',')
        val_loss_list = f.readline().split(',')

    # convert string to float
    train_acc_list = [float(i) for i in train_acc_list]
    train_loss_list = [float(i) for i in train_loss_list]
    val_acc_list = [float(i) for i in val_acc_list]
    val_loss_list = [float(i) for i in val_loss_list]

    # plot
    plt.figure(dpi=300)
    plt.plot(train_acc_list, label='train_acc')
    plt.plot(val_acc_list, label='val_acc')
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(val_loss_list, label='val_loss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epoch')
    # plt.show()
    plt.savefig('train_record.png')


if __name__ == '__main__':
    # vis()
    show_train_record()
