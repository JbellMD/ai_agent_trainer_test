import time
import psutil
from ..utils.logging import AutoTrainerLogger

class PerformanceMonitor:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        
    def monitor_system(self):
        """Monitor system performance metrics"""
        self.logger.log("Monitoring system performance")
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
    def check_resource_limits(self, limits: dict):
        """Check if resource usage exceeds limits"""
        self.logger.log("Checking resource limits")
        stats = self.monitor_system()
        for resource, limit in limits.items():
            if stats[resource] > limit:
                self.logger.log(
                    f"Resource limit exceeded: {resource} ({stats[resource]} > {limit})",
                    level='warning'
                )
                return False
        return True