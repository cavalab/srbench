"""
This Logging module is just a simplified interface to the python built-in
logging library.  Its sets up default logging options which are typical of most
bingo runs.
"""
import logging
import warnings

INFO = 25
DETAILED_INFO = 20

try:
    import mpi4py
    MPISIZE = mpi4py.MPI.COMM_WORLD.Get_size()
    MPIRANK = mpi4py.MPI.COMM_WORLD.Get_rank()
    USING_MPI = MPISIZE > 1
except (ImportError, AttributeError):
    USING_MPI = False


def configure_logging(verbosity="standard", module=False, timestamp=False,
                      stats_file=None, logfile=None):
    """Configuration of Bingo logging

    Parameters
    ----------
    verbosity : str or int
        verbosity options are "quiet", "standard", "detailed", "debug", or an
        integer (0 - 100) that corresponds to typical python log level.
    module : bool
        whether to show the module name on logging output. Default False
    timestamp :
        whether to show a time stamp on logging output. Default False
    stats_file : str
        (optional) file name for evolution statistics to be logged to
    logfile : str
        (optional) file name for a copy of the log to be saved
    """
    level = _get_log_level_from_verbosity(verbosity)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    root_logger.handlers=[] # remove current handlers

    console_handler = _make_console_handler(level, module, timestamp)
    root_logger.addHandler(console_handler)

    if logfile is not None:
        logfile_handler = _make_logfile_handler(logfile, level, module,
                                                timestamp)
        root_logger.addHandler(logfile_handler)

    if stats_file is not None:
        stats_file_handler = _make_stats_file_handler(stats_file)
        root_logger.addHandler(stats_file_handler)


def _make_console_handler(level, module, timestamp):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    format_string = _get_console_format_string(module, timestamp)
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)

    console_handler.addFilter(StatsFilter(filter_out=True))
    console_handler.addFilter(MpiFilter())
    return console_handler


def _make_logfile_handler(filename, level, module, timestamp):
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(level)

    format_string = _get_console_format_string(module, timestamp)
    formatter = logging.Formatter(format_string)
    file_handler.setFormatter(formatter)

    file_handler.addFilter(StatsFilter(filter_out=True))
    file_handler.addFilter(MpiFilter())
    return file_handler


def _get_log_level_from_verbosity(verbosity):
    verbosity_map = {"quiet": logging.WARNING,
                     "standard": INFO,
                     "detailed": DETAILED_INFO,
                     "debug": logging.DEBUG}
    if isinstance(verbosity, str):
        return verbosity_map[verbosity]
    if isinstance(verbosity, int):
        return verbosity
    warnings.warn("Unrecognized verbosity level provided. "
                  "Using standard verbosity.")
    return INFO


def _get_console_format_string(module, timestamp):
    format_string = "%(message)s"
    if module:
        format_string = "%(module)s\t" + format_string
    if timestamp:
        format_string = "%(asctime)s\t" + format_string
    return format_string


def _make_stats_file_handler(stats_file):
    file_handler = logging.FileHandler(stats_file)
    file_handler.setLevel(INFO)

    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)

    file_handler.addFilter(StatsFilter(filter_out=False))
    file_handler.addFilter(MpiFilter())
    return file_handler


class MpiFilter(logging.Filter):
    """
    This is a filter which filters out messages from auxiliary processes at the
    INFO level

    Parameters
    ----------
    add_proc_number : bool (optional)
        Add processor identifier to multi-processor log messages. default True.
    """
    def __init__(self, add_proc_number=True):
        super().__init__()
        self._add_proc_number = add_proc_number

    def filter(self, record):
        if USING_MPI:
            if record.levelno == INFO:
                return MPIRANK == 0
            if self._add_proc_number:
                record.msg = "{}>\t".format(MPIRANK) + record.msg
        return True


class StatsFilter(logging.Filter):
    """This is a filter which filters based on the identifier "<stats>" at the
    beginning of a log message

    Parameters
    ----------
    filter_out : bool
        Whether to filter-out or filter-in stats messages
    """
    def __init__(self, filter_out):
        super().__init__()
        self._filter_out = filter_out

    def filter(self, record):
        if "stats" in record.__dict__:
            return not self._filter_out == record.stats
        return self._filter_out
