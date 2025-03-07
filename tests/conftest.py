import pytest

@pytest.fixture(scope="session", autouse=True)
def setup_tests():
    # Setup code that runs once for all tests
    pass