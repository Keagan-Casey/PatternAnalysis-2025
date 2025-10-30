from wasabi import Printer


class Logger:
    """A custom logger that wraps the wasabi Printer for enhanced logging.

    Supports logging methods from the wasabi Printer. Configurable display based on the debug_mode.
    Direct access to wasabi Printer methods is available.

    Examples:
         >>> logger = Logger(debug_mode=True)
         >>> logger.info("This is an info message.")
         >>> logger.warn("This is a warning message.")
         >>> logger.fail("This is a failure message.")
         >>> logger.good("This is a good message.")
         >>> logger.text("This is a text message.")
         >>> logger.divider("Sample Divider")
    """

    def __init__(self, debug_mode: bool = False):
        """Initializes the Logger with optional debug mode."""
        self._printer = Printer(
            colors={"info": 5},
            timestamp=True,
            pretty=debug_mode,
            hide_animation=(not debug_mode),
        )

    def __getattr__(self, name):
        """Provides access to the wasabi Printer methods."""
        return getattr(self._printer, name)
