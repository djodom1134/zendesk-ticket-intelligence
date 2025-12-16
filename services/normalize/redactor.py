"""
PII Redaction Module
Removes personally identifiable information from ticket text
"""

import re
from typing import Pattern

import structlog

logger = structlog.get_logger()


class PIIRedactor:
    """
    Redacts personally identifiable information from text.
    
    Patterns redacted:
    - Email addresses
    - Phone numbers (various formats)
    - IP addresses
    - Credit card numbers
    - Social Security Numbers
    - URLs with credentials
    - API keys / tokens (common patterns)
    """

    def __init__(self, strict: bool = True):
        """
        Args:
            strict: If True, use aggressive pattern matching
        """
        self.strict = strict
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> list[tuple[str, Pattern]]:
        """Compile regex patterns for PII detection"""
        patterns = [
            # Email addresses
            ("EMAIL", re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            )),
            
            # Phone numbers (various formats)
            ("PHONE", re.compile(
                r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
            )),
            
            # International phone numbers
            ("PHONE_INTL", re.compile(
                r'\b\+[0-9]{1,3}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}\b'
            )),
            
            # IP addresses (IPv4)
            ("IP", re.compile(
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            )),
            
            # Credit card numbers (basic pattern)
            ("CC", re.compile(
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|'
                r'3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
            )),
            
            # SSN (US Social Security Number)
            ("SSN", re.compile(
                r'\b[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b'
            )),
            
            # URLs with potential credentials
            ("URL_CREDS", re.compile(
                r'https?://[^:]+:[^@]+@[^\s]+'
            )),
            
            # API keys / tokens (common patterns)
            ("API_KEY", re.compile(
                r'\b(?:api[_-]?key|token|secret|password|auth)[=:]\s*["\']?[A-Za-z0-9_-]{20,}["\']?',
                re.IGNORECASE
            )),
            
            # License keys (product-specific patterns)
            ("LICENSE", re.compile(
                r'\b[A-Z0-9]{4,5}[-\s][A-Z0-9]{4,5}[-\s][A-Z0-9]{4,5}[-\s][A-Z0-9]{4,5}\b',
                re.IGNORECASE
            )),
        ]
        
        if self.strict:
            # Add more aggressive patterns
            patterns.extend([
                # Names after common prefixes (aggressive)
                ("NAME_PREFIX", re.compile(
                    r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'
                )),
            ])
        
        return patterns

    def redact(self, text: str) -> str:
        """
        Redact PII from text.
        
        Args:
            text: Input text potentially containing PII
            
        Returns:
            Text with PII replaced by [REDACTED_TYPE]
        """
        if not text:
            return text
        
        redacted = text
        for name, pattern in self._patterns:
            redacted = pattern.sub(f'[REDACTED_{name}]', redacted)
        
        return redacted

    def redact_with_stats(self, text: str) -> tuple[str, dict[str, int]]:
        """
        Redact PII and return statistics.
        
        Returns:
            Tuple of (redacted_text, {pattern_name: count})
        """
        if not text:
            return text, {}
        
        stats = {}
        redacted = text
        
        for name, pattern in self._patterns:
            matches = pattern.findall(redacted)
            if matches:
                stats[name] = len(matches)
            redacted = pattern.sub(f'[REDACTED_{name}]', redacted)
        
        return redacted, stats


# Default redactor instance
default_redactor = PIIRedactor(strict=True)


def redact_pii(text: str) -> str:
    """Convenience function to redact PII from text"""
    return default_redactor.redact(text)

