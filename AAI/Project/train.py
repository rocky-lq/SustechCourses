import logging

import torch
import torch.nn as nn

from cnn_model import CNNSpeakerIdentificationModel
from config import device, device_ids
from load_data import load_training_data

logger = logging.getLogger(__name__)


def training(model, train_dl, val_dl, num_epochs=100):
    logger.info('Start training...')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs, anneal_strategy='linear')
    # Define the accuracy and loss list
    train_acc_list, train_loss_list = [], []
    val_acc_list, val_loss_list = [], []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data['waveform'].cuda(), data['target'].cuda()

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        train_acc_list.append(acc)
        train_loss_list.append(avg_loss)
        logger.info(f'Epoch: {epoch}, Loss: {avg_loss:.3f}, Accuracy: {acc:.3f}')

        # Testing the model on the validation set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            val_loss = 0
            for data in val_dl:
                inputs, labels = data['waveform'].cuda(), data['target'].cuda()

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                # Get the predicted class with the highest score
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # Count of predictions that matched the target label
                correct += (predicted == labels).sum().item()
            val_acc = correct / total
            val_loss = val_loss / len(val_dl)
            logger.info(f'Epoch: {epoch}, Val Accuracy: {val_acc:.3f}, Val Loss: {val_loss:.3f}')
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)

    # save the model
    if len(device_ids) > 1:
        logger.info('Saving the model...')
        torch.save(model.state_dict(), 'model_multi.pth')
        logger.info('Model saved in model_multi.pth')
    else:
        logger.info('Saving the model...')
        torch.save(model.state_dict(), 'model_single.pth')
        logger.info('Model saved in model_single.pth')

    # print(train_acc_list)
    # print(train_loss_list)
    # print(val_acc_list)
    # print(val_loss_list)
    logger.info('Finished Training')


if __name__ == '__main__':

    model = CNNSpeakerIdentificationModel().to(device)
    train_loader, val_loader = load_training_data()

    # Wrapping models in multiple GPUs
    if len(device_ids) > 1:
        logger.info('Using multiple GPUs')
        model = nn.DataParallel(model, device_ids=device_ids)

    model.cuda()
    training(model, train_loader, val_loader, num_epochs=30)
