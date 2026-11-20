"""Website database models (SQLite).

This database stores website accounts and analysis records only.  It is not
read by the model and cannot affect training, features or predictions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from . import db, login_manager


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    __tablename__ = "vh_users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)

    # Preferences
    theme = db.Column(db.String(10), default="dark", nullable=False)
    notify_on_analysis = db.Column(db.Boolean, default=True, nullable=False)
    retain_audio = db.Column(db.Boolean, default=False, nullable=False)

    analyses = db.relationship(
        "Analysis",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def initials(self) -> str:
        parts = [part for part in (self.name or "").split() if part]
        if not parts:
            return (self.email or "?")[:1].upper()
        if len(parts) == 1:
            return parts[0][:2].upper()
        return (parts[0][0] + parts[-1][0]).upper()

    @property
    def total_analyses(self) -> int:
        return self.analyses.count()


@login_manager.user_loader
def load_user(user_id: str):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None


class Analysis(db.Model):
    __tablename__ = "vh_analyses"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("vh_users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    # File metadata (as reported by the file, not invented)
    filename = db.Column(db.String(255), nullable=False)
    file_format = db.Column(db.String(16), default="—", nullable=False)
    size_bytes = db.Column(db.Integer, default=0, nullable=False)
    duration_seconds = db.Column(db.Float)
    sample_rate = db.Column(db.Integer)
    channels = db.Column(db.Integer)

    # Actual values returned by the existing model pipeline
    prediction = db.Column(db.String(16), nullable=False, index=True)  # BONAFIDE/SPOOF/UNCERTAIN
    spoof_score = db.Column(db.Float)          # 0..1, calibrated as returned
    raw_spoof_score = db.Column(db.Float)      # 0..1, raw model score as returned
    confidence = db.Column(db.Float)           # as returned (may be None)
    confidence_label = db.Column(db.String(32))
    reason = db.Column(db.Text)
    bonafide_threshold = db.Column(db.Float)
    spoof_threshold = db.Column(db.Float)
    chunk_count = db.Column(db.Integer)
    chunk_seconds = db.Column(db.Float)
    chunk_disagreement = db.Column(db.Float)
    chunk_json = db.Column(db.Text)
    truncated = db.Column(db.Boolean, default=False, nullable=False)

    # Website bookkeeping
    processing_ms = db.Column(db.Float)
    model_name = db.Column(db.String(64))
    model_version = db.Column(db.String(64))
    feature_set = db.Column(db.String(32))
    n_features = db.Column(db.Integer)
    audio_file = db.Column(db.String(255))  # only when the user opts in to retention

    user = db.relationship("User", back_populates="analyses")

    @property
    def verdict_slug(self) -> str:
        value = (self.prediction or "").upper()
        if value == "SPOOF":
            return "spoof"
        if value == "UNCERTAIN":
            return "uncertain"
        return "bonafide"

    @property
    def verdict_title(self) -> str:
        return {
            "bonafide": "Human Voice Detected",
            "spoof": "Potential AI-Generated or Manipulated Voice",
            "uncertain": "Inconclusive Audio Analysis",
        }[self.verdict_slug]

    @property
    def spoof_percent(self) -> str:
        if self.spoof_score is None:
            return "Not available"
        return f"{self.spoof_score * 100:.2f}%"

    @property
    def confidence_percent(self) -> str:
        if self.confidence is None:
            return "Not available"
        return f"{self.confidence * 100:.0f}%"

    @property
    def chunks(self) -> list[dict]:
        if not self.chunk_json:
            return []
        try:
            value = json.loads(self.chunk_json)
            return value if isinstance(value, list) else []
        except (TypeError, ValueError):
            return []

    @property
    def report_id(self) -> str:
        return f"VH-{self.created_at:%Y%m%d}-{self.id:06d}"

    @property
    def duration_label(self) -> str:
        if not self.duration_seconds:
            return "Not available"
        seconds = float(self.duration_seconds)
        minutes, rest = divmod(seconds, 60)
        if minutes >= 1:
            return f"{int(minutes)}m {rest:04.1f}s"
        return f"{rest:.1f}s"

    @property
    def size_label(self) -> str:
        size = float(self.size_bytes or 0)
        if size >= 1024 * 1024:
            return f"{size / (1024 * 1024):.2f} MB"
        if size >= 1024:
            return f"{size / 1024:.1f} KB"
        return f"{int(size)} B"
