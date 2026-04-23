import time
import torch

from latent_space_explorer.config import Config

class Profiler():
    def __init__(self, config: Config):
        self.start = None
        self.end = None
        self.message = None
        self.device = config.device

    def start_timer(self, message : str) -> None:
        self.message = message

        if self.device != 'cpu':
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.start = time.time()
        
    def report_elapsed_time(self):
        if self.device != 'cpu':    
            self.end.record()
            torch.cuda.synchronize()

            elapsed_time = self.start.elapsed_time(self.end)
        else:
            elapsed_time = 1000*(time.time() - self.start)

        print(f"{self.message} ({elapsed_time:.2f}ms)")