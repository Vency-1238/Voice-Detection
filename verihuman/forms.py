"""Website forms with server-side validation and CSRF protection."""

from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import BooleanField, PasswordField, RadioField, StringField, SubmitField
from wtforms.validators import DataRequired, EqualTo, Length, Regexp

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$"


class RegisterForm(FlaskForm):
    name = StringField(
        "Full name",
        validators=[DataRequired(), Length(min=2, max=120)],
        render_kw={"autocomplete": "name", "placeholder": "Ada Lovelace"},
    )
    email = StringField(
        "Email address",
        validators=[
            DataRequired(),
            Length(max=255),
            Regexp(EMAIL_PATTERN, message="Enter a valid email address."),
        ],
        render_kw={"autocomplete": "email", "placeholder": "you@example.com"},
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired(), Length(min=8, max=128, message="Use at least 8 characters.")],
        render_kw={"autocomplete": "new-password", "placeholder": "At least 8 characters"},
    )
    confirm = PasswordField(
        "Confirm password",
        validators=[DataRequired(), EqualTo("password", message="Passwords do not match.")],
        render_kw={"autocomplete": "new-password", "placeholder": "Repeat your password"},
    )
    submit = SubmitField("Create account")


class LoginForm(FlaskForm):
    email = StringField(
        "Email address",
        validators=[DataRequired(), Length(max=255)],
        render_kw={"autocomplete": "email", "placeholder": "you@example.com"},
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired()],
        render_kw={"autocomplete": "current-password", "placeholder": "Your password"},
    )
    remember = BooleanField("Keep me signed in")
    submit = SubmitField("Sign in")


class ProfileForm(FlaskForm):
    name = StringField(
        "Full name",
        validators=[DataRequired(), Length(min=2, max=120)],
    )
    email = StringField(
        "Email address",
        validators=[
            DataRequired(),
            Length(max=255),
            Regexp(EMAIL_PATTERN, message="Enter a valid email address."),
        ],
    )
    submit = SubmitField("Save profile")


class PasswordForm(FlaskForm):
    current_password = PasswordField(
        "Current password",
        validators=[DataRequired()],
        render_kw={"autocomplete": "current-password"},
    )
    new_password = PasswordField(
        "New password",
        validators=[DataRequired(), Length(min=8, max=128, message="Use at least 8 characters.")],
        render_kw={"autocomplete": "new-password"},
    )
    confirm_password = PasswordField(
        "Confirm new password",
        validators=[DataRequired(), EqualTo("new_password", message="Passwords do not match.")],
        render_kw={"autocomplete": "new-password"},
    )
    submit = SubmitField("Update password")


class SettingsForm(FlaskForm):
    theme = RadioField(
        "Theme",
        choices=[("dark", "Dark"), ("light", "Light")],
        default="dark",
        validators=[DataRequired()],
    )
    notify_on_analysis = BooleanField("Show a notification when an analysis finishes")
    retain_audio = BooleanField("Keep a private copy of my uploaded audio with each analysis")
    submit = SubmitField("Save preferences")


class DeleteAccountForm(FlaskForm):
    password = PasswordField(
        "Confirm your password",
        validators=[DataRequired()],
        render_kw={"autocomplete": "current-password"},
    )
    submit = SubmitField("Delete my account")
