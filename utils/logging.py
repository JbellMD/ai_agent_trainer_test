import logging
from datetime import datetime

class AutoTrainerLogger:
    def __init__(self):
        self.logger = logging.getLogger('auto_trainer')
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            filename=f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def log(self, message, level='info'):
        getattr(self.logger, level)(message)