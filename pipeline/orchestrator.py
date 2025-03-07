from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, validator
from ..utils.logging import AutoTrainerLogger

class PipelineConfig(BaseModel):
    """Schema for pipeline configuration"""
    tasks: List[Dict[str, Any]]
    max_workers: int = 4
    retries: int = 3
    timeout: Optional[int] = None
    
    @validator('tasks')
    def validate_tasks(cls, tasks):
        """Validate task dependencies"""
        task_names = [task['name'] for task in tasks]
        for task in tasks:
            if 'depends_on' in task:
                for dep in task['depends_on']:
                    if dep not in task_names:
                        raise ValueError(f"Task {task['name']} depends on unknown task {dep}")
        return tasks

class PipelineOrchestrator:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        self.task_registry = {}
        
    def register_task(self, name: str, function, dependencies: List[str] = []):
        """Register a new pipeline task"""
        self.task_registry[name] = {
            'function': function,
            'dependencies': dependencies
        }
        
    def run_pipeline(self, config: Dict[str, Any]):
        """Run the complete ML pipeline with enhanced features"""
        try:
            validated_config = PipelineConfig(**config)
            self.logger.log("Starting enhanced ML pipeline")
            
            # Build task execution graph
            execution_graph = self._build_execution_graph(validated_config.tasks)
            
            # Execute pipeline
            results = {}
            with ThreadPoolExecutor(max_workers=validated_config.max_workers) as executor:
                for stage in execution_graph:
                    futures = {
                        executor.submit(
                            self._execute_task_with_retries,
                            task,
                            results,
                            validated_config.retries,
                            validated_config.timeout
                        ): task['name']
                        for task in stage
                    }
                    
                    for future in as_completed(futures):
                        task_name = futures[future]
                        try:
                            results[task_name] = future.result()
                        except Exception as e:
                            self.logger.log(f"Task {task_name} failed: {e}", level='error')
                            raise
                            
            self.logger.log("Pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.log(f"Pipeline execution failed: {e}", level='error')
            raise
            
    def _build_execution_graph(self, tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Build execution stages based on task dependencies"""
        execution_graph = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            # Find tasks with all dependencies satisfied
            stage = []
            for task in remaining_tasks:
                dependencies = task.get('depends_on', [])
                if all(dep in [t['name'] for t in execution_graph] for dep in dependencies):
                    stage.append(task)
                    
            if not stage:
                raise ValueError("Circular dependency detected in pipeline tasks")
                
            execution_graph.append(stage)
            remaining_tasks = [t for t in remaining_tasks if t not in stage]
            
        return execution_graph
        
    def _execute_task_with_retries(self, task: Dict[str, Any], results: Dict[str, Any], retries: int, timeout: Optional[int]):
        """Execute a task with retry logic"""
        task_name = task['name']
        task_func = self.task_registry[task_name]['function']
        
        for attempt in range(retries + 1):
            try:
                self.logger.log(f"Executing task {task_name} (attempt {attempt + 1})")
                return task_func(results)
            except Exception as e:
                if attempt == retries:
                    self.logger.log(f"Task {task_name} failed after {retries} attempts", level='error')
                    raise
                self.logger.log(f"Task {task_name} failed, retrying: {e}", level='warning')