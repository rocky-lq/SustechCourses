import logging
import os

import torch

from cnn_model import CNNSpeakerIdentificationModel
from config import SAMPLE_RATE, TEST_DATA_DIR, PROJECT_DIR
from load_data import open_audio_file, pre_processing_audio, index_to_label

logger = logging.getLogger(__name__)


def predict():
    logger.info('start predict...')
    # load the model.pth
    model = CNNSpeakerIdentificationModel()
    # multi-gpu
    # model.load_state_dict(
    #     {k.replace('module.', ''): v for k, v in torch.load('model_multi.pth').items()})

    model.load_state_dict(torch.load('model_single.pth'))
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

    # print the result to predict.txt
    with open(os.path.join(PROJECT_DIR, 'predict.txt'), 'w') as f:
        for i in range(len(predicted)):
            f.write(file[i] + ' ' + index_to_label[predicted[i]] + '\n')

    logger.info('predict finished!')


def load_test_data():
    # get all files in TEST_DATA_DIR
    logger.info('load test data...')
    test_sigs = []
    file_list = os.listdir(TEST_DATA_DIR)
    file_list.sort()

    for file in file_list:
        test_audio = open_audio_file(os.path.join(TEST_DATA_DIR, file), SAMPLE_RATE)
        sig, sr = pre_processing_audio(test_audio)
        test_sigs.append(sig)

    # convert to batch
    test_sigs = torch.stack(test_sigs)
    logger.info(f'{test_sigs.shape}')
    return test_sigs, file_list


if __name__ == '__main__':
    predict()
