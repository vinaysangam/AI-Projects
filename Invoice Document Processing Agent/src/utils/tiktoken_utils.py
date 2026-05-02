# This module has been removed — the invoice pipeline does not use tiktoken directly.

import tiktoken

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Cache the encoding instance at module level.
_encoding: tiktoken.Encoding | None = None


def _get_encoding(model: str = "gpt-4o") -> tiktoken.Encoding:
    """Return a cached tiktoken encoding for *model*."""
    global _encoding
    if _encoding is None:
        try:
            _encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in *text*.

    Args:
        text: The string to tokenise.
        model: Model name used to select the tokeniser.

    Returns:
        Token count.
    """
    enc = _get_encoding(model)
    return len(enc.encode(text))


def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """Truncate *text* to fit within *max_tokens*.

    Args:
        text: Input text.
        max_tokens: Maximum number of tokens allowed.
        model: Model name for tokeniser selection.

    Returns:
        Truncated (or original) text.
    """
    enc = _get_encoding(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    logger.warning("Truncating text from %d to %d tokens", len(tokens), max_tokens)
    return enc.decode(tokens[:max_tokens])
