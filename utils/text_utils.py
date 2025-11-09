"""Text processing utilities."""

import hashlib


def clamp_text(text: str | None, max_chars: int) -> str:
    """Clamp text to reduce token usage.
    
    Args:
        text: Text to clamp
        max_chars: Maximum characters to keep
        
    Returns:
        Clamped text or empty string if text is None
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text.
    
    Args:
        text: Text to hash
        
    Returns:
        Hex digest of the hash
    """
    if not text:
        return ""
    # normalize whitespace to ensure stable hash
    norm = "\n".join(line.strip() for line in text.splitlines() if line is not None)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()
