"""VeriHuman website layer.

This package contains ONLY the website: pages, authentication, analysis history,
reports and a thin adapter that calls the existing prediction interface in
`audio_features.py`.

The machine-learning implementation is not imported for modification and is not
retrained or altered here.  `predict_audio_files()` is called exactly as the
original Flask route calls it.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template
from flask_login import LoginManager, current_user, login_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect

db = SQLAlchemy()
login_manager = LoginManager()
csrf = CSRFProtect()


def _load_or_create_secret(instance_dir: Path) -> str:
    """Keep sessions valid across restarts without hard-coding a secret."""
    key_file = instance_dir / "verihuman_secret_key"
    if key_file.exists():
        value = key_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    value = secrets.token_hex(32)
    key_file.write_text(value, encoding="utf-8")
    try:
        key_file.chmod(0o600)
    except OSError:
        pass
    return value


def init_website(app: Flask, model_bundle=None, model_error=None) -> None:
    """Attach the website to the existing Flask app.

    `model_bundle` and `model_error` are passed in from app.py, so the website
    uses the already-loaded model object.  Nothing about loading or prediction
    changes.
    """
    project_root = Path(app.root_path)
    instance_dir = project_root / "instance"
    instance_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = instance_dir / "retained_audio"
    upload_dir.mkdir(parents=True, exist_ok=True)

    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = _load_or_create_secret(instance_dir)
    app.config.setdefault(
        "SQLALCHEMY_DATABASE_URI", f"sqlite:///{(instance_dir / 'verihuman.db').as_posix()}"
    )
    app.config.setdefault("SQLALCHEMY_TRACK_MODIFICATIONS", False)
    app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
    app.config.setdefault("SESSION_COOKIE_SAMESITE", "Lax")
    app.config.setdefault("REMEMBER_COOKIE_HTTPONLY", True)
    app.config.setdefault("WTF_CSRF_TIME_LIMIT", None)
    app.config.setdefault("VERIHUMAN_MAX_FILES", 10)
    app.config.setdefault("VERIHUMAN_MAX_FILE_MB", 50)

    app.config["VERIHUMAN_MODEL_BUNDLE"] = model_bundle
    app.config["VERIHUMAN_MODEL_ERROR"] = model_error
    app.config["VERIHUMAN_UPLOAD_DIR"] = upload_dir

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please sign in to open that page."
    login_manager.login_message_category = "info"

    from .auth import bp as auth_bp
    from .views import bp as website_bp
    from .api import bp as api_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(website_bp)
    app.register_blueprint(api_bp)

    # The original JSON/multipart endpoints predate the website and do not send
    # CSRF tokens.  They stay byte-for-byte functional.
    for endpoint in ("predict", "health"):
        view = app.view_functions.get(endpoint)
        if view is not None:
            csrf.exempt(view)

    with app.app_context():
        from . import models  # noqa: F401  (register tables)

        db.create_all()
        _upgrade_existing_schema(app)
        from .models import User

        guest = db.session.execute(
            db.select(User).where(User.email == "guest@local.verihuman")
        ).scalar_one_or_none()
        if guest is None:
            guest = User(name="VeriHuman Guest", email="guest@local.verihuman")
            guest.set_password(secrets.token_hex(32))
            guest.retain_audio = True
            db.session.add(guest)
            db.session.commit()
        elif not guest.retain_audio:
            guest.retain_audio = True
            db.session.commit()
    @app.before_request
    def open_guest_workspace():
        if not current_user.is_authenticated:
            from .models import User

            guest = db.session.execute(
                db.select(User).where(User.email == "guest@local.verihuman")
            ).scalar_one_or_none()
            if guest is not None:
                if not guest.retain_audio:
                    guest.retain_audio = True
                    db.session.commit()
                login_user(guest)

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def server_error(_error):
        return render_template("errors/500.html"), 500

    @app.template_filter("dt")
    def format_datetime(value: datetime | None, fmt: str = "%d %b %Y, %H:%M"):
        if not value:
            return "—"
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.strftime(fmt)

    @app.template_filter("pct")
    def format_percent(value, digits: int = 1):
        if value is None:
            return "—"
        try:
            return f"{float(value) * 100:.{digits}f}%"
        except (TypeError, ValueError):
            return "—"

    @app.context_processor
    def inject_brand():
        return {
            "brand_name": "VeriHuman",
            "brand_tagline": "Trust Every Voice. Verify Every Sound.",
            "current_year": datetime.now(timezone.utc).year,
        }


def _upgrade_existing_schema(app: Flask) -> None:
    """Add website columns introduced after an existing SQLite database."""
    from sqlalchemy import inspect, text

    required_columns = {
        "model_prediction": "VARCHAR(16)",
        "acoustic_json": "TEXT",
    }
    existing_columns = {
        column["name"]
        for column in inspect(db.engine).get_columns("vh_analyses")
    }
    missing_columns = [
        (name, sql_type)
        for name, sql_type in required_columns.items()
        if name not in existing_columns
    ]
    for name, sql_type in missing_columns:
        db.session.execute(
            text(f'ALTER TABLE "vh_analyses" ADD COLUMN "{name}" {sql_type}')
        )
        app.logger.info("Added missing database column vh_analyses.%s", name)
    if missing_columns:
        db.session.commit()
