import torch

class Profiler():
    def __init__(self):
        self.start = None
        self.end = None
        self.message = None

    def start_timer(self, message : str) -> None:
        self.message = message

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        
    def report_elapsed_time(self):
        self.end.record()
        torch.cuda.synchronize()

        elapsed_time = self.start.elapsed_time(self.end)
        print(f"{self.message} ({elapsed_time:.2f}ms)")