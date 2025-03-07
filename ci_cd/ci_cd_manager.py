import subprocess
import sys
from pathlib import Path
from ..utils.logging import AutoTrainerLogger

class CICDManager:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        self.project_root = Path(__file__).parent.parent.parent
        
    def run_tests(self):
        """Run all tests in the test suite"""
        self.logger.log("Running test suite")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(self.project_root / "tests")],
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.log("All tests passed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.log(f"Tests failed:\n{e.stderr}", level='error')
            return False
            
    def run_linting(self):
        """Run code linting and style checks"""
        self.logger.log("Running code linting")
        try:
            subprocess.run(
                ["flake8", str(self.project_root)],
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.log("Code linting passed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.log(f"Linting failed:\n{e.stderr}", level='error')
            return False
            
    def build_docker_image(self, tag: str = "latest"):
        """Build Docker image for deployment"""
        self.logger.log(f"Building Docker image with tag: {tag}")
        try:
            subprocess.run(
                ["docker", "build", "-t", f"ai_trainer:{tag}", "."],
                cwd=str(self.project_root),
                check=True
            )
            self.logger.log("Docker image built successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.log(f"Docker build failed: {e}", level='error')
            return False
            
    def deploy_to_kubernetes(self, config_path: str):
        """Deploy to Kubernetes cluster"""
        self.logger.log(f"Deploying to Kubernetes using config: {config_path}")
        try:
            subprocess.run(
                ["kubectl", "apply", "-f", config_path],
                check=True
            )
            self.logger.log("Kubernetes deployment successful")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.log(f"Kubernetes deployment failed: {e}", level='error')
            return False
            
    def run_full_pipeline(self, tag: str = "latest", k8s_config: str = "k8s/deployment.yaml"):
        """Run full CI/CD pipeline"""
        if not self.run_tests():
            return False
        if not self.run_linting():
            return False
        if not self.build_docker_image(tag):
            return False
        if not self.deploy_to_kubernetes(k8s_config):
            return False
        self.logger.log("CI/CD pipeline completed successfully")
        return True