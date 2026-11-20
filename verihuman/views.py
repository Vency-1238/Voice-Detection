"""Website pages: landing, dashboard, detection, history, reports, settings."""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy import func

from . import db
from .forms import DeleteAccountForm, PasswordForm, ProfileForm, SettingsForm
from .models import Analysis, User
from .reports import build_csv, build_pdf
from .services import (
    delete_retained_audio,
    delete_user_audio,
    get_bundle,
    model_error,
    model_info,
    retained_audio_path,
)

bp = Blueprint(
    "website",
    __name__,
    static_folder="static",
    static_url_path="/assets",
)

VERDICTS = {"BONAFIDE", "SPOOF", "UNCERTAIN"}


# ---------------------------------------------------------------------------
# Public pages
# ---------------------------------------------------------------------------
@bp.route("/")
def landing():
    info = model_info()
    return render_template("landing.html", info=info, page_title=None)


@bp.route("/about")
def about():
    return render_template("about.html", info=model_info(), page_title="About VeriHuman")


@bp.route("/model")
def model_page():
    return render_template(
        "model.html",
        info=model_info(),
        error=model_error(),
        page_title="Model Information",
    )


@bp.route("/help")
def help_page():
    return render_template("help.html", page_title="Help & Documentation")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@bp.route("/dashboard")
@login_required
def dashboard():
    base_query = Analysis.query.filter_by(user_id=current_user.id)
    total = base_query.count()

    counts = {
        "BONAFIDE": 0,
        "SPOOF": 0,
        "UNCERTAIN": 0,
    }
    for prediction, count in (
        db.session.query(Analysis.prediction, func.count(Analysis.id))
        .filter(Analysis.user_id == current_user.id)
        .group_by(Analysis.prediction)
        .all()
    ):
        if prediction in counts:
            counts[prediction] = int(count)

    avg_ms_row = (
        db.session.query(func.avg(Analysis.processing_ms))
        .filter(Analysis.user_id == current_user.id)
        .scalar()
    )
    avg_ms = round(float(avg_ms_row), 1) if avg_ms_row is not None else None

    recent = (
        base_query.order_by(Analysis.created_at.desc()).limit(6).all()
    )

    # Activity for the last 14 days, grouped by prediction.
    today = datetime.now(timezone.utc).date()
    start_day = today - timedelta(days=13)
    day_labels = [(start_day + timedelta(days=offset)) for offset in range(14)]
    day_keys = [day.isoformat() for day in day_labels]
    activity = {key: {"BONAFIDE": 0, "SPOOF": 0, "UNCERTAIN": 0} for key in day_keys}
    rows = (
        db.session.query(
            func.date(Analysis.created_at), Analysis.prediction, func.count(Analysis.id)
        )
        .filter(
            Analysis.user_id == current_user.id,
            Analysis.created_at >= datetime.combine(start_day, time.min),
        )
        .group_by(func.date(Analysis.created_at), Analysis.prediction)
        .all()
    )
    for day_value, prediction, count in rows:
        key = str(day_value)
        if key in activity and prediction in activity[key]:
            activity[key][prediction] = int(count)

    # Distribution by calibrated spoof-score bucket (actual stored scores only).
    score_rows = (
        db.session.query(Analysis.spoof_score)
        .filter(Analysis.user_id == current_user.id, Analysis.spoof_score.isnot(None))
        .all()
    )
    buckets = [0, 0, 0, 0, 0]
    for (score,) in score_rows:
        index = min(4, max(0, int(float(score) * 5)))
        buckets[index] += 1

    trend_rows = (
        db.session.query(Analysis.created_at, Analysis.spoof_score)
        .filter(Analysis.user_id == current_user.id, Analysis.spoof_score.isnot(None))
        .order_by(Analysis.created_at.desc())
        .limit(20)
        .all()
    )
    trend_rows = list(reversed(trend_rows))

    chart_data = {
        "hasData": total > 0,
        "trend": {
            "labels": [row[0].strftime("%d %b %H:%M") for row in trend_rows],
            "values": [round(float(row[1]) * 100, 2) for row in trend_rows],
        },
        "days": {
            "labels": [day.strftime("%d %b") for day in day_labels],
            "bonafide": [activity[key]["BONAFIDE"] for key in day_keys],
            "spoof": [activity[key]["SPOOF"] for key in day_keys],
            "uncertain": [activity[key]["UNCERTAIN"] for key in day_keys],
        },
        "verdicts": {
            "bonafide": counts["BONAFIDE"],
            "spoof": counts["SPOOF"],
            "uncertain": counts["UNCERTAIN"],
        },
        "buckets": {
            "labels": ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
            "values": buckets,
        },
    }

    return render_template(
        "dashboard.html",
        page_title="Overview",
        total=total,
        counts=counts,
        avg_ms=avg_ms,
        recent=recent,
        chart_data=chart_data,
        active="overview",
        info=model_info(),
    )


# ---------------------------------------------------------------------------
# Detection page
# ---------------------------------------------------------------------------
@bp.route("/detect")
@login_required
def detect():
    info = model_info()
    return render_template(
        "detect.html",
        page_title="Detect Audio",
        active="detect",
        info=info,
        error=model_error(),
        max_files=current_app.config["VERIHUMAN_MAX_FILES"],
        max_file_mb=current_app.config["VERIHUMAN_MAX_FILE_MB"],
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def _filtered_history():
    query = Analysis.query.filter_by(user_id=current_user.id)

    search = (request.args.get("q") or "").strip()
    if search:
        query = query.filter(Analysis.filename.ilike(f"%{search}%"))

    verdict = (request.args.get("verdict") or "").upper()
    if verdict in VERDICTS:
        query = query.filter(Analysis.prediction == verdict)

    date_from = request.args.get("from") or ""
    date_to = request.args.get("to") or ""
    if date_from:
        try:
            start = datetime.combine(date.fromisoformat(date_from), time.min)
            query = query.filter(Analysis.created_at >= start)
        except ValueError:
            date_from = ""
    if date_to:
        try:
            end = datetime.combine(date.fromisoformat(date_to), time.max)
            query = query.filter(Analysis.created_at <= end)
        except ValueError:
            date_to = ""

    sort = request.args.get("sort", "newest")
    if sort == "oldest":
        query = query.order_by(Analysis.created_at.asc())
    else:
        sort = "newest"
        query = query.order_by(Analysis.created_at.desc())

    return query, {
        "q": search,
        "verdict": verdict if verdict in VERDICTS else "",
        "from": date_from,
        "to": date_to,
        "sort": sort,
    }


@bp.route("/history")
@login_required
def history():
    query, filters = _filtered_history()
    page = max(1, request.args.get("page", type=int) or 1)
    per_page = 10
    total = query.count()
    pages = max(1, math.ceil(total / per_page))
    page = min(page, pages)
    items = query.offset((page - 1) * per_page).limit(per_page).all()

    return render_template(
        "history.html",
        page_title="Analysis History",
        active="history",
        items=items,
        total=total,
        page=page,
        pages=pages,
        filters=filters,
    )


@bp.route("/history/export.csv")
@login_required
def history_export():
    query, _filters = _filtered_history()
    records = query.all()
    csv_text = build_csv(records)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="verihuman-history-{stamp}.csv"'
        },
    )


@bp.route("/reports")
@login_required
def reports():
    query, filters = _filtered_history()
    page = max(1, request.args.get("page", type=int) or 1)
    per_page = 12
    total = query.count()
    pages = max(1, math.ceil(total / per_page))
    page = min(page, pages)
    items = query.offset((page - 1) * per_page).limit(per_page).all()

    return render_template(
        "reports.html",
        page_title="Reports",
        active="reports",
        items=items,
        total=total,
        page=page,
        pages=pages,
        filters=filters,
    )


@bp.post("/history/clear")
@login_required
def history_clear():
    records = Analysis.query.filter_by(user_id=current_user.id).all()
    for record in records:
        delete_retained_audio(record)
        db.session.delete(record)
    db.session.commit()
    flash("Analysis history cleared.", "info")
    return redirect(url_for("website.history"))


@bp.post("/history/<int:analysis_id>/delete")
@login_required
def history_delete(analysis_id: int):
    record = _owned_analysis(analysis_id)
    delete_retained_audio(record)
    db.session.delete(record)
    db.session.commit()
    flash("Analysis deleted.", "info")
    return redirect(url_for("website.history", **_back_args()))


def _back_args() -> dict:
    args = {}
    for key in ("q", "verdict", "from", "to", "sort", "page"):
        value = request.args.get(key)
        if value:
            args[key] = value
    return args


def _owned_analysis(analysis_id: int) -> Analysis:
    record = db.session.get(Analysis, analysis_id)
    if record is None or record.user_id != current_user.id:
        abort(404)
    return record


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
@bp.route("/report/<int:analysis_id>")
@login_required
def report(analysis_id: int):
    record = _owned_analysis(analysis_id)
    return render_template(
        "report.html",
        page_title=f"Report {record.report_id}",
        active="reports",
        analysis=record,
        info=model_info(),
        has_audio=retained_audio_path(record) is not None,
    )


@bp.route("/report/<int:analysis_id>/pdf")
@login_required
def report_pdf(analysis_id: int):
    record = _owned_analysis(analysis_id)
    payload = build_pdf(record, current_user, model_info())
    return Response(
        payload,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{record.report_id}.pdf"'
        },
    )


@bp.route("/report/<int:analysis_id>/csv")
@login_required
def report_csv(analysis_id: int):
    record = _owned_analysis(analysis_id)
    csv_text = build_csv([record])
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{record.report_id}.csv"'
        },
    )


@bp.route("/report/<int:analysis_id>/audio")
@login_required
def report_audio(analysis_id: int):
    record = _owned_analysis(analysis_id)
    path = retained_audio_path(record)
    if path is None:
        abort(404)
    return send_file(path, as_attachment=True, download_name=record.filename)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
@bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    form = ProfileForm(obj=current_user)
    password_form = PasswordForm()

    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        other = (
            db.session.query(User)
            .filter(func.lower(User.email) == email, User.id != current_user.id)
            .first()
        )
        if other is not None:
            form.email.errors.append("That email address is already in use.")
        else:
            current_user.name = form.name.data.strip()
            current_user.email = email
            db.session.commit()
            flash("Profile updated.", "success")
            return redirect(url_for("website.profile"))

    return render_template(
        "profile.html",
        page_title="Profile",
        active="profile",
        form=form,
        password_form=password_form,
        total=current_user.analyses.count(),
        info=model_info(),
    )


@bp.post("/profile/password")
@login_required
def change_password():
    password_form = PasswordForm()
    if password_form.validate_on_submit():
        if not current_user.check_password(password_form.current_password.data):
            flash("Current password is incorrect.", "danger")
        else:
            current_user.set_password(password_form.new_password.data)
            db.session.commit()
            flash("Password updated.", "success")
    else:
        for errors in password_form.errors.values():
            for error in errors:
                flash(error, "danger")
    return redirect(url_for("website.profile"))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    form = SettingsForm(obj=current_user)
    delete_form = DeleteAccountForm()

    if form.validate_on_submit():
        current_user.theme = form.theme.data if form.theme.data in {"dark", "light"} else "dark"
        current_user.notify_on_analysis = bool(form.notify_on_analysis.data)
        was_retaining = bool(current_user.retain_audio)
        current_user.retain_audio = bool(form.retain_audio.data)
        if was_retaining and not current_user.retain_audio:
            for record in current_user.analyses:
                delete_retained_audio(record)
                record.audio_file = None
        db.session.commit()
        flash("Preferences saved.", "success")
        return redirect(url_for("website.settings"))

    return render_template(
        "settings.html",
        page_title="Settings",
        active="settings",
        form=form,
        delete_form=delete_form,
    )


@bp.post("/settings/delete-account")
@login_required
def delete_account():
    form = DeleteAccountForm()
    if not form.validate_on_submit():
        flash("Could not verify the password. Account was not deleted.", "danger")
        return redirect(url_for("website.settings"))

    if not current_user.check_password(form.password.data):
        flash("Password is incorrect. Account was not deleted.", "danger")
        return redirect(url_for("website.settings"))

    user = current_user
    delete_user_audio(user)
    db.session.delete(user)
    db.session.commit()
    flash("Your account and analysis history have been deleted.", "info")
    return redirect(url_for("website.landing"))


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------
@bp.app_context_processor
def nav_context():
    if not current_user.is_authenticated:
        return {}
    recent = (
        Analysis.query.filter_by(user_id=current_user.id)
        .order_by(Analysis.created_at.desc())
        .limit(5)
        .all()
    )
    return {
        "nav_recent": recent,
        "nav_theme": current_user.theme if current_user.theme in {"dark", "light"} else "dark",
    }
