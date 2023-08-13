import os
import numpy as np

class Logger:
    def __init__(self, dir_path):
        train_logger_path = os.path.join(dir_path, 'train_logs.txt')
        self.LOGGER_FD = os.open(train_logger_path, os.O_RDWR | os.O_CREAT | os.O_APPEND)
        self.METRICS_PATH = os.path.join(dir_path, 'metrics')
        os.makedirs(self.METRICS_PATH)
        self.metrics = dict()

    def update_metric(self, metric_name, x, y):
        new_entry = [[x], [y]]
        if metric_name not in self.metrics:
            self.metrics[metric_name] = np.array(new_entry)
        else:
            self.metrics[metric_name] = np.hstack((self.metrics[metric_name], new_entry))
        
    def register(self, msg):
        if not msg.endswith('\n'): msg = msg + '\n'
        print(msg, end="")
        os.write(self.LOGGER_FD, str.encode(msg))

    def close(self):
        for m in self.metrics.keys():
            metric = self.metrics[m]
            with open(os.path.join(self.METRICS_PATH, f'{m}.npy'), 'wb') as f:
                np.save(f, metric)
        os.close(self.LOGGER_FD)
        
