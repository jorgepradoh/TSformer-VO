import os
from datetime import datetime

class LossLogger:
    def __init__(self, log_dir="./data", log_filename="loss_weights_log"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_filename)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_filename = f"{log_filename}_{timestamp}.csv"
        self.log_path = os.path.join(log_dir, log_filename)

        # Write header on creation
        with open(self.log_path, "w") as f:
            f.write("timestamp,epoch,sigma_t,sigma_r\n")

    def log(self, epoch, sigma_t, sigma_r):
        timestamp = datetime.now().isoformat(timespec='seconds')
        log_line = f"{timestamp},{epoch},{sigma_t},{sigma_r}\n"
        with open(self.log_path, "a") as f:
            f.write(log_line)
    
    def get_log_path(self):
        return self.log_path