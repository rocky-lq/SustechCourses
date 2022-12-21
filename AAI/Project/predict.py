import os

import torch

from cnn_model import CNNSpeakerIdentificationModel
from config import SAMPLE_RATE, TEST_DATA_DIR
from load_data import open, pre_processing_audio, index_to_label


def predict():
    # load the model.pth
    model = CNNSpeakerIdentificationModel()
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('model.pth', map_location=torch.device('cpu')).items()})
    model.eval()

    sig, file = load_test_data()

    # predict
    with torch.no_grad():
        # sig = sig.cuda()
        sig_m, sig_s = sig.mean(), sig.std()
        sig = (sig - sig_m) / sig_s
        output = model(sig)
        _, predicted = torch.max(output.data, 1)

    # tensor to int
    predicted = predicted.numpy()

    for i in range(len(predicted)):
        print(file[i], index_to_label[predicted[i]])

    # print the result to predict.txt
    with open('predict.txt', 'w') as f:
        for i in range(len(predicted)):
            f.write(file[i] + ' ' + index_to_label[predicted[i]] + '\n')


def load_test_data():
    # get all files in TEST_DATA_DIR
    test_sigs = []
    file_list = os.listdir(TEST_DATA_DIR)
    file_list.sort()

    for file in file_list:
        test_audio = open(os.path.join(TEST_DATA_DIR, file), SAMPLE_RATE)
        sig, sr = pre_processing_audio(test_audio)
        test_sigs.append(sig)

    # convert to batch
    test_sigs = torch.stack(test_sigs)
    print(test_sigs.shape)
    return test_sigs, file_list


if __name__ == '__main__':
    predict()
    # load_test_data()
