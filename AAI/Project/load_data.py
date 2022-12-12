import glob
import os
import torch
import torchaudio

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_data_path = f'{project_dir}/data/LibriSpeech-SI/train'
# all_type是spk001-spk250的列表
train_labels = os.listdir(train_data_path)
# 去除不以spk开头的
train_labels = [x for x in train_labels if x.startswith('spk')]
train_labels.sort()
# 排序
print(train_labels)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, root, all_labels, type='train'):
        self.root = root
        self.type = type
        self.all_labels = all_labels
        self.set = []

        for root, dirs, files in os.walk(self.root):
            target = root.split('\\')[-1]
            if target in self.all_labels:
                for file in files:
                    audio_data = torchaudio.load(os.path.join(root, file))
                    information = {
                        'audio': audio_data,
                        'target': target
                    }
                    self.set.append(information)

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):
        return self.set[idx]


if __name__ == '__main__':
    train_set = AudioDataset(train_data_path, train_labels, 'train')
    print(len(train_set))
    ...
