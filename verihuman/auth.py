"""Authentication routes: registration, login, logout."""

from __future__ import annotations

from flask import Blueprint, flash, redirect, url_for
from flask_login import current_user, logout_user

bp = Blueprint("auth", __name__)


@bp.route("/register", methods=["GET", "POST"])
def register():
    return redirect(url_for("website.landing"))


@bp.route("/login", methods=["GET", "POST"])
def login():
    return redirect(url_for("website.landing"))


@bp.post("/logout")
def logout():
    if current_user.is_authenticated:
        logout_user()
    flash("You have been signed out.", "info")
    return redirect(url_for("website.landing"))
