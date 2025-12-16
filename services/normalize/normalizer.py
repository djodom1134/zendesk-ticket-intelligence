"""
Ticket Normalizer
Converts raw Zendesk ticket payloads to canonical TicketDocument format
"""

import re
from datetime import datetime
from typing import Any, Optional

import structlog

from shared.schemas.ticket import (
    Attachment,
    AuthorRole,
    Comment,
    TicketDocument,
    TicketPriority,
    TicketStatus,
)

from .redactor import PIIRedactor

logger = structlog.get_logger()


class TicketNormalizer:
    """
    Normalizes raw Zendesk ticket data to TicketDocument schema.

    Features:
    - Maps Zendesk fields to canonical schema
    - Extracts platform/version from tags
    - Computes fulltext for embedding
    - Applies PII redaction
    """

    # Platform detection patterns
    PLATFORM_PATTERNS = {
        "windows": re.compile(r'\b(?:windows|win|win10|win11|w10|w11)\b', re.I),
        "linux": re.compile(r'\b(?:linux|ubuntu|debian|centos|rhel|fedora)\b', re.I),
        "macos": re.compile(r'\b(?:macos|mac|osx|darwin|macbook|imac)\b', re.I),
        "arm": re.compile(r'\b(?:arm|arm64|aarch64|raspberry\s*pi)\b', re.I),
        "ios": re.compile(r'\b(?:ios|iphone|ipad)\b', re.I),
        "android": re.compile(r'\b(?:android)\b', re.I),
    }

    # Version pattern (e.g., 6.0.5, v2.1.0, version 3.2)
    VERSION_PATTERN = re.compile(r'\b(?:v(?:ersion)?\.?\s*)?(\d+\.\d+(?:\.\d+)?)\b', re.I)

    def __init__(self, redactor: Optional[PIIRedactor] = None):
        self.redactor = redactor or PIIRedactor()

    def normalize(self, raw: dict) -> TicketDocument:
        """
        Normalize a raw ticket payload to TicketDocument.

        Args:
            raw: Raw ticket dict from Zendesk API

        Returns:
            Normalized TicketDocument
        """
        # Parse timestamps
        created_at = self._parse_datetime(raw.get("created_at"))
        updated_at = self._parse_datetime(raw.get("updated_at")) or created_at
        solved_at = self._parse_datetime(raw.get("solved_at"))

        # Parse status
        status = self._parse_status(raw.get("status", "open"))
        priority = self._parse_priority(raw.get("priority"))

        # Extract content
        subject = raw.get("subject", "") or ""
        description = raw.get("description", "") or ""
        tags = raw.get("tags", []) or []

        # Normalize comments
        comments = self._normalize_comments(raw.get("comments", []))

        # Extract platform/version from tags and text
        all_text = f"{subject} {description} {' '.join(tags)}"
        platform = self._detect_platform(all_text, tags)
        version = self._detect_version(tags)

        # Build fulltext for embedding
        comment_texts = [c.body_text for c in comments]
        fulltext = self._build_fulltext(subject, description, comment_texts)

        # Apply PII redaction
        description_redacted = self.redactor.redact(description)
        pii_redacted_text = self.redactor.redact(fulltext)

        # Redact comments
        for comment in comments:
            comment.body_redacted = self.redactor.redact(comment.body_text)

        return TicketDocument(
            ticket_id=str(raw.get("id", "")),
            source_url=raw.get("url"),
            created_at=created_at,
            updated_at=updated_at,
            solved_at=solved_at,
            status=status,
            priority=priority,
            ticket_type=raw.get("type"),
            platform=platform,
            version=version,
            subject=subject,
            description=description,
            description_redacted=description_redacted,
            comments=comments,
            tags=tags,
            custom_fields=self._flatten_custom_fields(raw.get("custom_fields", [])),
            attachments=self._normalize_attachments(raw.get("attachments", [])),
            assignee_id=raw.get("assignee_id"),
            requester_id=raw.get("requester_id"),
            group_id=raw.get("group_id"),
            ticket_fulltext=fulltext,
            pii_redacted_text=pii_redacted_text,
        )

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from various formats"""
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            # ISO format
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def _parse_status(self, status: str) -> TicketStatus:
        """Parse ticket status"""
        try:
            return TicketStatus(status.lower())
        except (ValueError, AttributeError):
            return TicketStatus.OPEN

    def _parse_priority(self, priority: Any) -> Optional[TicketPriority]:
        """Parse ticket priority"""
        if not priority:
            return None
        try:
            return TicketPriority(priority.lower())
        except (ValueError, AttributeError):
            return None

    def _normalize_comments(self, comments: list) -> list[Comment]:
        """Normalize comment list"""
        normalized = []
        for c in comments or []:
            # Determine author role
            role = AuthorRole.END_USER
            if c.get("public") is False:
                role = AuthorRole.SYSTEM
            # Note: Would need user data to determine if agent

            normalized.append(Comment(
                id=c.get("id"),
                author_id=c.get("author_id"),
                author_role=role,
                created_at=self._parse_datetime(c.get("created_at")) or datetime.utcnow(),
                body_text=c.get("body", "") or "",
                is_public=c.get("public", True),
            ))
        return normalized

    def _detect_platform(self, text: str, tags: list[str]) -> Optional[str]:
        """Detect platform from text and tags"""
        all_text = f"{text} {' '.join(tags)}"
        for platform, pattern in self.PLATFORM_PATTERNS.items():
            if pattern.search(all_text):
                return platform
        return None

    def _detect_version(self, tags: list[str]) -> Optional[str]:
        """Extract version number from tags"""
        for tag in tags:
            match = self.VERSION_PATTERN.search(tag)
            if match:
                return match.group(1)
        return None

    def _flatten_custom_fields(self, custom_fields: list) -> dict[str, Any]:
        """Flatten custom fields array to dict"""
        if not custom_fields:
            return {}
        result = {}
        for field in custom_fields:
            if isinstance(field, dict) and "id" in field:
                result[str(field["id"])] = field.get("value")
        return result

    def _normalize_attachments(self, attachments: list) -> list[Attachment]:
        """Normalize attachment metadata"""
        result = []
        for att in attachments or []:
            if isinstance(att, dict):
                result.append(Attachment(
                    id=att.get("id"),
                    filename=att.get("file_name", att.get("filename", "")),
                    content_type=att.get("content_type", ""),
                    size=att.get("size", 0),
                    url=att.get("content_url"),
                ))
        return result

    def _build_fulltext(
        self,
        subject: str,
        description: str,
        comment_texts: list[str],
    ) -> str:
        """Build fulltext for embedding"""
        parts = [subject, description]
        parts.extend(comment_texts)
        # Remove empty parts and join with newlines
        return "\n\n".join(p for p in parts if p and p.strip())


# Default normalizer instance
default_normalizer = TicketNormalizer()


def normalize_ticket(raw: dict) -> TicketDocument:
    """Convenience function to normalize a ticket"""
    return default_normalizer.normalize(raw)
