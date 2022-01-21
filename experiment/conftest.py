# Configure pytest settings to take in specific algorithm as argument

def pytest_addoption(parser):
    # Required option:
    parser.addoption("--ml", action="store", default="", help="algorithm to test")