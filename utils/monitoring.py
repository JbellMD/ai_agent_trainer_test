import time
import psutil
from typing import Dict, Any
from .logging import AutoTrainerLogger

class ResourceMonitor:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        self.start_time = time.time()
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        stats = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'uptime': time.time() - self.start_time
        }
        self.logger.log(f"System stats: {stats}")
        return stats
        
    def check_resource_limits(self, limits: Dict[str, float]) -> bool:
        """Check if resource usage exceeds limits"""
        stats = self.get_system_stats()
        for resource, limit in limits.items():
            if stats.get(resource, 0) > limit:
                self.logger.log(
                    f"Resource limit exceeded: {resource} ({stats[resource]} > {limit})",
                    level='warning'
                )
                return False
        return True