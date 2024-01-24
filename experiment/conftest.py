# configures pytest to allow CLI arguments. 

def pytest_addoption(parser):
    parser.addoption("--ml",  action="store", default="AFPRegressor")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.ml
    if 'ml' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("ml", [option_value])
