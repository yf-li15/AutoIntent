from utils import *
from concat import ConcatDataset
import world


class DataProcessor(object):
    """Base class for data converters for intent classification data sets."""
    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """userID, locationID, timeID, cateID, intentID."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "train.csv")), world.config['num_activities'])
        elif mode == 'eval':
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "val.csv")), world.config['num_activities'])
        elif mode == 'test':
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "test.csv")), world.config['num_activities'])

    def get_labels(self, data_dir):
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.csv"), sep=",")
        labels = np.unique(np.array(test['intentID']))
            
        return labels

    def _create_examples(self, lines, num_cates):
        """Creates examples for the training and dev sets."""
        examples = []
        targets = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 5:
                continue
            uid = int(line[0])
            lid = int(line[1])
            tid = int(line[2])
            cid = int(line[3])
            label = int(line[4])
            while True:
                neg = np.random.randint(0, num_cates)   # [0, num_cates)
                if neg == cid:
                    continue
                else:
                    break
            negid = neg # sample negative category
            
            examples.append(
                (uid, lid, tid, cid, negid))
            targets.append(label)

        return examples, targets

class Data(data.Dataset):
    def __init__(self, mode, target_list = None):
        self.data_dir = world.DATA_PATH
        processor = DatasetProcessor()
        self.data_dir = world.DATA_PATH
        self.all_label_list = processor.get_labels(self.data_dir)
        
        data_tmp, targets_tmp = processor.get_examples(self.data_dir, mode)
        if target_list is not None:
            self.data = []
            self.targets = []
            for i in range(len(data_tmp)):
                if targets_tmp[i] in target_list:
                    self.data.append(data_tmp[i])
                    self.targets.append(targets_tmp[i])
            # self.label_map = {label:i for i,label in enumerate(target_list)}
        else:
            self.data, self.targets = data_tmp, targets_tmp

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (data, target) where target is index of the target class.
            data: uid, lid, tid, cid, negid
        """
        data, target= self.data[index], self.targets[index]

        # return data, self.label_map[target], index
        return data, target, index

    def __len__(self):
        return len(self.data)


def Loader(batch_size, mode='train', num_workers=4, shuffle=True, target_list=range(5)):
    dataset = Data(mode, target_list)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


def LoaderMix(batch_size, mode='train',num_workers=4, shuffle=True, labeled_list=range(5), unlabeled_list=range(5, 10), new_labels=None):
    dataset_labeled = Data(mode, target_list=labeled_list)
    dataset_unlabeled = Data(mode, target_list=unlabeled_list)
    if new_labels is not None:
        dataset_unlabeled.targets = new_labels
    dataset_labeled.targets = dataset_labeled.targets + dataset_unlabeled.targets
    dataset_labeled.data = dataset_labeled.data + dataset_unlabeled.data
    loader = data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def LoaderTwoStream(batch_size, mode='train',num_workers=4, shuffle=True, labeled_list=range(5), unlabeled_list=range(5, 10), unlabeled_batch_size=64):
    dataset_labeled = Data(mode, target_list=labeled_list)
    dataset_unlabeled = Data(mode, target_list=unlabeled_list)
    dataset = ConcatDataset((dataset_labeled, dataset_unlabeled))
    labeled_idxs = range(len(dataset_labeled))
    unlabeled_idxs = range(len(dataset_labeled), len(dataset_labeled)+len(dataset_unlabeled))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, unlabeled_batch_size)
    loader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    loader.labeled_length = len(dataset_labeled)
    loader.unlabeled_length = len(dataset_unlabeled)
    return loader


if __name__ == '__main__':
    print('Data and Parameters Initialization...')
    from parse import parse_args
    args = parse_args()
    # setting
    '''
    if args.dataset == 'Beijing':
        labeled_list = list([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14])
        unlabeled_list = list([4, 12, 13, 15])
    '''
    labeled_list = range(12)
    unlabeled_list = range(12, 16)
    mix_train_loader = LoaderMix(batch_size=args.train_batch_size, mode='train', shuffle=True, labeled_list=labeled_list, unlabeled_list=unlabeled_list)
    labeled_train_loader = Loader(batch_size=args.train_batch_size, mode='train', shuffle=True, target_list = labeled_list)
   
