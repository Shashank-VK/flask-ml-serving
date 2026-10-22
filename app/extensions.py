"""Flask extensions instantiated for application factory pattern."""

from flask import current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


def get_default_rate_limit() -> str:
    """Dynamically fetch default rate limit string from active app config."""
    if current_app:
        return str(current_app.config.get("RATELIMIT_DEFAULT", "60 per minute"))
    return "60 per minute"


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[get_default_rate_limit],
    default_limits_per_method=False,
)
