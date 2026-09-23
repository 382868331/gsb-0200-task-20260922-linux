"""Exception types for the nonzero-winding fill library.

All errors raised by the library derive from :class:`WindingFillError`, so
callers can catch a single base class.  Every error message embeds a
*location* string (e.g. ``contours[2].segments[1].p0.x``) that pinpoints
the offending value; invalid input is never silently corrected.
"""

from __future__ import annotations


class WindingFillError(Exception):
    """Base class for every error raised by :mod:`winding_fill`."""


class InvalidInputError(WindingFillError, ValueError):
    """Raised for malformed geometry or canvas parameters.

    The first argument is always the human-readable *location* of the
    offending value; the message itself is the second argument.
    """

    def __init__(self, location: str, message: str = ""):
        if message:
            text = f"{location}: {message}"
        else:
            text = location
        super().__init__(text)
        self.location = location
        self.message = message
