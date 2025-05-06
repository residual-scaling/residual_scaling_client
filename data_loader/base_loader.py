import random


class BaseLoader:
    """
    Dataloader for loading data in to the standard format.
    Standard format: list of json objects
    JSON object: {
        "question": str,
        "truth": str,
        "options": list of str, # optional
        "reasoning": str, # optional
    }
    """
    data: list
    size: int

    def load_data(self):
        return self.data
    
    def load_data_percent(self, percent: float):
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")
        if percent < 0 or percent > 1:
            raise ValueError("Percent must be between 0 and 1.")
        num_samples = int(self.size * percent)
        return self.data[:num_samples]
    
    def load_data_percent_back(self, percent: float):
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")
        if percent < 0 or percent > 1:
            raise ValueError("Percent must be between 0 and 1.")
        num_samples = int(self.size * percent)
        return self.data[-num_samples:]
    
    def load_train_test_set(self, train_percent: float, random_shuffle: bool = False):
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")
        if train_percent < 0 or train_percent > 1:
            raise ValueError("Train percent must be between 0 and 1.")
        if random_shuffle:
            random.shuffle(self.data)
        num_train_samples = int(self.size * train_percent)
        train_set = self.data[:num_train_samples]
        test_set = self.data[num_train_samples:]
        return train_set, test_set
    
    def load_n_from_front(self, n: int):
        if n > self.size:
            raise ValueError("n is larger than the size of the data.")
        if n < 0:
            raise ValueError("n must be a positive integer.")
        return self.data[:n]
    
    def load_n_from_back(self, n: int):
        if n > self.size:
            raise ValueError("n is larger than the size of the data.")
        if n < 0:
            raise ValueError("n must be a positive integer.")
        return self.data[-n:]

    def load_n_offeset(self, n: int, offset: int):
        if n > self.size:
            raise ValueError("n is larger than the size of the data.")
        if n < 0:
            raise ValueError("n must be a positive integer.")
        if offset < 0:
            raise ValueError("offset must be a positive integer.")
        return self.data[offset:offset + n]

    