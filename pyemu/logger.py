"""module for logging pyemu progress
"""
from __future__ import annotations

import copy
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple, Type, TypeVar, Union, cast

from .pyemu_warnings import PyemuWarning

_old_style_logging = False  # Flag to enable old style logging


def set_old_style_logging(**kwargs) -> None:
    """Configure the logging system to mimic the old style logs.

    Args:
        **kwargs (Any): passed directly to `logging.basicConfig`.
    """
    global _old_style_logging
    _old_style_logging = True
    options = dict(
        format="{asctime} [{name}] {levelname}: {message}", level=logging.INFO
    )
    options.update(kwargs)
    logging.basicConfig(**options)
    logging.captureWarnings(True)


ET = TypeVar("ET", bound=Exception)


@dataclass
class SuperLogger:
    logger: logging.Logger

    def getChild(self, suffix: str) -> SuperLogger:
        """
        Get a logger which is a descendant to this one.

        This is a convenience method, such that

        `logging.getLogger('abc').getChild('def.ghi')`

        is the same as

        `logging.getLogger('abc.def.ghi')`

        It's useful, for example, when the parent logger is named using
        __name__ rather than a literal string.

        Args:
            suffix (str): name of the child logger, to be concatenated with the name
                of the parent logger.

        Returns:
            SuperLogger: child logger.
        """
        child = self.logger.getChild(suffix)
        return SuperLogger(child)

    def logged_exception(self, message: str, exc_type: Type[ET] = Exception) -> ET:
        """Log an error message and return an Exception with the same message.

        Note that this does not raise the error to ensure the exception can be raised from
        the location where the error occurred.

        Args:
            message (str): Description of the error condition.
            exc_type (Type[Exception], optional): Type of exception to return. Defaults to Exception.

        Returns:
            Exception: Exception with the given message.
        """
        self.error(message)
        return exc_type(message)

    def logged_warning(
        self,
        message: str,
        category: Type[Warning] = PyemuWarning,
        stacklevel: int = 1,
        source: Optional[Any] = None,
    ) -> None:
        """Log a warning message and raise a warning with the same message.

        If `logging.captureWarnings(True)` is used this will cause duplicate log
        entries. This is because there is no safe way to check whether this is set
        in the logging module.

        Args:
            message (str): Description of the warning condition.
            category (Type[Warning], optional): Category of the warning to be raised.
                Defaults to PyemuWarning.
            stacklevel (int, optional): Stacklevel of the source of the warning.
                Higher levels place the source higher up the call stack. Defaults to 1.
            source (Any, optional): In case of a ResourceWarning, the destroyed object
                which the warning. Defaults to None.
        """
        self.warning(message)
        warnings.warn(message, category, stacklevel=stacklevel + 1, source=source)

    def timed_info(self, message: str) -> TimerLog:
        return TimerLog(self.info, message)

    def statement(self, message: str) -> None:
        warnings.warn(
            "Logger.statement is deprecated, use Logger.info instead",
            DeprecationWarning,
        )
        self.info(message)

    def warn(self, message: str) -> None:
        warnings.warn(
            "Logger.warn is deprecated, use Logger.logged_warning instead",
            DeprecationWarning,
        )
        self.logged_warning(message, stacklevel=2)

    def lraise(self, message):
        warnings.warn(
            "Logger.lraise is deprecated, use Logger.logged_exception instead",
            DeprecationWarning,
        )
        raise self.logged_exception(message)

    # Wrappers around logging.Logger
    @property
    def handlers(self) -> List[logging.Handler]:
        """
        List of the handlers attached to this logger.
        """
        return self.logger.handlers

    def addHandler(self, handler: logging.Handler) -> None:
        """
        Add the specified handler to this logger.
        """
        self.logger.addHandler(handler)

    def removeHandler(self, handler: logging.Handler) -> None:
        """
        Remove the specified handler from this logger.
        """
        self.logger.removeHandler(handler)

    @property
    def filters(self) -> List[logging.Filter]:
        """
        List of the filters attached to this logger.
        """
        return self.logger.filters

    def addFilter(self, filter_: logging.Filter) -> None:
        """
        Add the specified filter to this logger.
        """
        self.logger.addFilter(filter_)

    def removeFilter(self, filter_: logging.Filter) -> None:
        """
        Remove the specified filter from this logger.
        """
        self.logger.removeFilter(filter_)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Log 'message % args' with severity 'DEBUG'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.debug("Houston, we have a %s", "thorny problem", exc_info=1)
        """
        self.logger.debug(message, args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Log 'message % args' with severity 'INFO'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.info("Houston, we have a %s", "interesting problem", exc_info=1)
        """
        self.logger.info(message, args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Log 'message % args' with severity 'WARNING'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        self.logger.warning(message, args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Log 'message % args' with severity 'ERROR'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.error("Houston, we have a %s", "major problem", exc_info=1)
        """
        self.logger.error(message, args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Log 'message % args' with severity 'CRITICAL'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.critical("Houston, we have a %s", "major disaster", exc_info=1)
        """
        self.logger.critical(message, args, **kwargs)


def get_logger(name: str, filename: str = "", echo: bool = False) -> SuperLogger:
    logger = SuperLogger(logging.getLogger(name))
    if _old_style_logging:
        # getLogger will return a previous instance if called with the same name,
        # remove the old handlers to configure the handlers according to the current call.
        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.close()
        if not filename:
            filename = "{}.log".format(name)
        logger.addHandler(logging.FileHandler(filename, mode="w"))
        if echo:
            logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


class TimerLog:
    def __init__(self, log_func: Callable[[str], None], message: str):
        """Logger with builtin timer to calculate the time taken.

        Args:
            log_func (Callable[[str], None]): Function to be used to log the message.
                Called like `log_func(message)`.
            message (str): message to log
        """
        self.log_func = log_func
        self.message = message
        self._finished = False
        self._error: Union[
            Tuple[None, None, None], Tuple[Type[Exception], object, object]
        ] = (None, None, None)
        self.tic = datetime.now()

        self.log_func("starting {}".format(message))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._error = (exc_type, exc_value, traceback)
        self.finish()

    def finish(self) -> None:
        """Finish the current timer and log the time taken."""
        if self._finished:
            warnings.warn(
                "{cls}({message}) already finished".format(
                    cls=self.__class__.__name__, message=self.message
                ),
                stacklevel=2,
            )
        toc = datetime.now()
        self._finished = True
        exc_type, exc_value, _ = self._error
        if exc_type:
            err_msg = f" with error {exc_type.__name__}({exc_value})"
        else:
            err_msg = ""
        self.log_func(f"finished {self.message}{err_msg} took: {toc - self.tic}")


class Logger(object):
    """a basic class for logging events during the linear analysis calculations
        if filename is passed, then a file handle is opened.

    Args:
        filename (`str`): Filename to write logged events to. If False, no file will be created,
            and logged events will be displayed on standard out.
        echo (`bool`):  Flag to cause logged events to be echoed to the screen.

    """

    def __init__(self, filename, echo=False):
        self.items = {}
        self.echo = bool(echo)
        if filename == True:
            self.echo = True
            self.filename = None
        elif filename:
            self.filename = filename
            self.f = open(filename, "w")
            self.t = datetime.now()
            self.log("opening " + str(filename) + " for logging")
        else:
            self.filename = None

    def statement(self, phrase):
        """log a one-time statement

        Arg:
            phrase (`str`): statement to log

        """
        t = datetime.now()
        s = str(t) + " " + str(phrase) + "\n"
        if self.echo:
            print(s, end="")
        if self.filename:
            self.f.write(s)
            self.f.flush()

    def log(self, phrase):
        """log something that happened.

        Arg:
            phrase (`str`): statement to log

        Notes:
            The first time phrase is passed the start time is saved.
                The second time the phrase is logged, the elapsed time is written
        """
        pass
        t = datetime.now()
        if phrase in self.items.keys():
            s = (
                str(t)
                + " finished: "
                + str(phrase)
                + " took: "
                + str(t - self.items[phrase])
                + "\n"
            )
            if self.echo:
                print(s, end="")
            if self.filename:
                self.f.write(s)
                self.f.flush()
            self.items.pop(phrase)
        else:
            s = str(t) + " starting: " + str(phrase) + "\n"
            if self.echo:
                print(s, end="")
            if self.filename:
                self.f.write(s)
                self.f.flush()
            self.items[phrase] = copy.deepcopy(t)

    def warn(self, message):
        """write a warning to the log file.

        Arg:
            phrase (`str`): warning statement to log


        """
        s = str(datetime.now()) + " WARNING: " + message + "\n"
        if self.echo:
            print(s, end="")
        if self.filename:
            self.f.write(s)
            self.f.flush
        warnings.warn(s, PyemuWarning)

    def lraise(self, message):
        """log an exception, close the log file, then raise the exception

        Arg:
            phrase (`str`): exception statement to log and raise

        """
        s = str(datetime.now()) + " ERROR: " + message + "\n"
        print(s, end="")
        if self.filename:
            self.f.write(s)
            self.f.flush
            self.f.close()
        raise Exception(message)
