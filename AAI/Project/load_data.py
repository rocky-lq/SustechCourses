import os
import random

import torch
import torchaudio

# 当前文件的目录
project_dir = os.path.abspath(os.path.dirname(__file__))

train_data_path = os.path.join(project_dir, 'data', 'LibriSpeech-SI', 'train')
# print(train_data_path)
# train_labels是spk001-spk250的列表
train_labels = os.listdir(train_data_path)
# 去除不以spk开头的
train_labels = [x for x in train_labels if x.startswith('spk')]
train_labels.sort()
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# 定义train_labels到index的映射
label_to_index = {label: index for index, label in enumerate(train_labels)}


def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor_minusmean.max()


# 规则化waveform，如果长度大于200000，随机截取200000，如果小于200000，随机在前后补0
def regular_waveform(waveform):
    max_length = 200000
    rows, len = waveform.shape
    if len > max_length:
        start = random.randint(0, len - max_length)
        waveform = waveform[:, start:start + max_length]
    elif len < max_length:
        pad_begin_len = random.randint(0, max_length - len)
        pad_end_len = max_length - len - pad_begin_len
        pad_begin = torch.zeros((rows, pad_begin_len))
        pad_end = torch.zeros((rows, pad_end_len))
        waveform = torch.cat((pad_begin, waveform, pad_end), 1)
    return waveform


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, root, all_labels, type='train'):
        self.root = root
        self.type = type
        self.all_labels = all_labels
        self.set = []
        self.max_wave_time = 0

        for root, dirs, files in os.walk(self.root):
            target = root.split(os.sep)[-1]
            print(target)
            if target == 'spk010':
                break
            if target in self.all_labels:
                for file in files:
                    # audio_data = torchaudio.load(os.path.join(root, file))
                    waveform, sample_rate = torchaudio.load(os.path.join(root, file), normalize=True)
                    waveform = regular_waveform(waveform)

                    information = {
                        'audio': waveform,
                        'target': label_to_index[target]
                    }
                    self.set.append(information)

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):
        return self.set[idx]


def load_data():
    train_set = AudioDataset(train_data_path, train_labels, 'train')
    print('max_wave_time:', train_set.max_wave_time)
    # 將train_set分为训练集和验证集
    train_size = int(0.7 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader

# 采样率为16000
# waveform, sample_rate = torchaudio.load(os.path.join(train_data_path, 'spk001', 'spk001_003.flac'))
# print(waveform, sample_rate)
