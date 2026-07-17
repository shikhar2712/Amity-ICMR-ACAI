"""
Authentication gate for the Streamlit app: Clerk email/password sign-up and
sign-in (via the Clerk *backend* Python SDK, `clerk-backend-api`) plus the
existing Google OIDC option (Streamlit's built-in st.login / st.user).

Why the backend SDK and not Clerk's JS components: Streamlit renders custom
components inside sandboxed iframes, which breaks Clerk's browser JS SDK
(component-caching / cross-frame issues). The backend SDK avoids all of that:
every Clerk call happens server-side over REST, and the browser only ever
talks to Streamlit.

Roles
-----
Two roles exist: "user" and "admin". The single source of truth is the Clerk
user record's `public_metadata["role"]`. Admins are promoted manually in the
Clerk dashboard (Users -> select user -> Metadata -> public ->
`{"role": "admin"}`). Role is resolved *by email address* for every login
method, so a Google-authenticated visitor is an admin if (and only if) a
Clerk user with the same email carries `role: admin`. No Clerk record, or no
role in metadata, means "user".

The page-access matrix lives in ROLE_PAGES below; app.py builds its sidebar
from it and every restricted page re-checks it server-side (defense in
depth: hiding a nav button alone would not stop a visitor who writes
st.session_state["navigation_page"] directly).

Required Streamlit secrets (Streamlit Cloud: App settings -> Secrets)
---------------------------------------------------------------------
Clerk (email/password sign-up + sign-in):

    [clerk]
    secret_key = "sk_..."        # Clerk dashboard -> API keys -> Secret key

(Also accepted: top-level CLERK_SECRET_KEY in secrets, or a CLERK_SECRET_KEY
environment variable.)

Google (optional "Continue with Google" button; omit the whole [auth] block
to hide the button):

    [auth]
    redirect_uri = "https://<your-app>.streamlit.app/oauth2callback"
    cookie_secret = "<a long random string, e.g. `openssl rand -hex 32`>"
    client_id = "<Google OAuth client ID>"
    client_secret = "<Google OAuth client secret>"
    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

SMTP (required for the "Forgot password?" email-code flow; omit the block to
disable that flow -- sign-in itself still works without it). For Gmail create
an App Password (Google Account -> Security -> 2-Step Verification -> App
passwords) and use it as `password`:

    [smtp]
    host = "smtp.gmail.com"
    port = 587
    username = "yourproject@gmail.com"
    password = "<16-char Gmail app password>"
    from_email = "yourproject@gmail.com"   # optional, defaults to username

Security notes
--------------
- The secret key is read only from secrets/env, never hardcoded.
- Login/sign-up errors shown to visitors are deliberately generic ("Invalid
  email or password") so the form can't be used to probe which emails exist;
  only actionable non-sensitive detail (weak password, account exists) is
  surfaced.
- Failed sign-ins are throttled per session (short lockout after repeated
  failures) as a brake on password guessing.
- The authenticated identity and role live in st.session_state, which is
  held server-side by Streamlit -- the browser cannot read or forge it. The
  trade-off is that a full browser reload starts a fresh session and asks
  for sign-in again.
"""
import base64
import hashlib
import hmac
import os
import re
import secrets as pysecrets
import smtplib
import time
from email.message import EmailMessage
from io import BytesIO

import streamlit as st

# ---------------------------------------------------------------------------
# Role / page-access model (single source of truth, imported by app.py)
# ---------------------------------------------------------------------------

ROLE_PAGES = {
    "user": ["Home", "Prediction", "About"],
    "admin": ["Home", "Dashboard", "Prediction", "View Records", "About"],
}
DEFAULT_PAGE = "Home"
VALID_ROLES = set(ROLE_PAGES)

# st.session_state keys owned by this module. app.py's reset flow clears
# session state wholesale and must preserve these, so they're exported.
AUTH_SESSION_KEY = "auth_session"
AUTH_THROTTLE_KEY = "auth_throttle"
AUTH_RESET_KEY = "auth_pw_reset"
AUTH_STATE_KEYS = (AUTH_SESSION_KEY, AUTH_THROTTLE_KEY, AUTH_RESET_KEY)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MIN_PASSWORD_LEN = 8

# Sign-in throttling: after this many consecutive failures, refuse further
# attempts for the lockout window. Per Streamlit session.
_MAX_FAILURES = 5
_LOCKOUT_SECONDS = 60

# Forgot-password email codes: 6 digits, valid 10 minutes, at most 5 wrong
# guesses per code, and a cooldown between sends. Per Streamlit session.
_OTP_DIGITS = 6
_OTP_TTL_SECONDS = 600
_OTP_MAX_ATTEMPTS = 5
_OTP_RESEND_COOLDOWN = 60


# ---------------------------------------------------------------------------
# Clerk client / configuration
# ---------------------------------------------------------------------------

def _clerk_secret_key():
    """Resolve the Clerk secret key from secrets or environment (never code)."""
    try:
        section = st.secrets.get("clerk", None)
        if section and section.get("secret_key"):
            return section["secret_key"]
        if st.secrets.get("CLERK_SECRET_KEY"):
            return st.secrets["CLERK_SECRET_KEY"]
    except Exception:
        pass  # no secrets.toml at all -- fall through to the environment
    return os.environ.get("CLERK_SECRET_KEY")


@st.cache_resource(show_spinner=False)
def _clerk_client(secret_key: str):
    from clerk_backend_api import Clerk
    return Clerk(bearer_auth=secret_key)


def _get_clerk():
    key = _clerk_secret_key()
    return _clerk_client(key) if key else None


def _google_configured() -> bool:
    """True when Streamlit's native OIDC login ([auth] secrets) is set up."""
    try:
        auth = st.secrets.get("auth", None)
        return bool(auth and auth.get("client_id") and auth.get("client_secret"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Clerk lookups
# ---------------------------------------------------------------------------

def _find_clerk_user(clerk, email: str):
    """Exact-match a Clerk user by email; None if no account exists."""
    from clerk_backend_api.models import GetUserListRequest
    users = clerk.users.list(
        request=GetUserListRequest(email_address=[email], limit=1),
        timeout_ms=10000,
    )
    return users[0] if users else None


def _role_from_user(clerk_user) -> str:
    meta = getattr(clerk_user, "public_metadata", None) or {}
    role = str(meta.get("role", "user")).lower()
    return role if role in VALID_ROLES else "user"


def _resolve_role(email: str) -> str:
    """Role for an authenticated email: Clerk public_metadata, else 'user'."""
    clerk = _get_clerk()
    if clerk is None:
        return "user"
    try:
        user = _find_clerk_user(clerk, email)
        return _role_from_user(user) if user else "user"
    except Exception:
        # Fail closed: an API hiccup must never grant elevated access.
        return "user"


def _clerk_error_codes(exc) -> list:
    """Extract machine-readable error codes from a ClerkErrors exception."""
    try:
        return [e.code for e in exc.data.errors if getattr(e, "code", None)]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _session():
    return st.session_state.get(AUTH_SESSION_KEY)


def _set_session(email: str, name: str, role: str, login_method: str):
    st.session_state[AUTH_SESSION_KEY] = {
        "authenticated": True,
        "email": email,
        "name": name,
        "role": role,
        "login_method": login_method,
        "signed_in_at": time.time(),
    }
    st.session_state.pop(AUTH_THROTTLE_KEY, None)


def get_current_role() -> str:
    """Role of the signed-in visitor ('user' when unknown -- fail closed)."""
    sess = _session()
    role = (sess or {}).get("role", "user")
    return role if role in VALID_ROLES else "user"


def allowed_pages(role: str = None) -> list:
    """Pages the given role (default: current visitor) may open."""
    return ROLE_PAGES.get(role or get_current_role(), ROLE_PAGES["user"])


def page_allowed(page: str, role: str = None) -> bool:
    return page in allowed_pages(role)


def require_page_access(page: str):
    """Server-side guard for restricted pages: call first inside the page
    handler. Shows an access-denied notice and halts the run when the
    current role may not open `page` (defense in depth behind the nav)."""
    if not page_allowed(page):
        st.error("🚫 **Access denied** — this section is restricted to administrators.")
        st.info("If you believe you need access, contact your system administrator.")
        st.stop()


# ---------------------------------------------------------------------------
# Sign-in throttling (per Streamlit session)
# ---------------------------------------------------------------------------

def _throttle():
    return st.session_state.setdefault(
        AUTH_THROTTLE_KEY, {"failures": 0, "locked_until": 0.0}
    )


def _throttle_seconds_left() -> int:
    return max(0, int(_throttle()["locked_until"] - time.time()))


def _record_failure():
    t = _throttle()
    t["failures"] += 1
    if t["failures"] >= _MAX_FAILURES:
        t["locked_until"] = time.time() + _LOCKOUT_SECONDS
        t["failures"] = 0


# ---------------------------------------------------------------------------
# Email (SMTP) for the forgot-password flow
# ---------------------------------------------------------------------------

def _smtp_config():
    """SMTP settings from secrets, or None when the block is absent."""
    try:
        smtp = st.secrets.get("smtp", None)
        if smtp and smtp.get("host") and smtp.get("username") and smtp.get("password"):
            return smtp
    except Exception:
        pass
    return None


def _send_email(to_addr: str, subject: str, body: str) -> bool:
    smtp = _smtp_config()
    if not smtp:
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp.get("from_email") or smtp["username"]
    msg["To"] = to_addr
    msg.set_content(body)
    try:
        with smtplib.SMTP(smtp["host"], int(smtp.get("port", 587)), timeout=15) as srv:
            srv.starttls()
            srv.login(smtp["username"], smtp["password"])
            srv.send_message(msg)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Forgot-password (email one-time code)
# ---------------------------------------------------------------------------

def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def _reset_state():
    return st.session_state.get(AUTH_RESET_KEY)


def _start_password_reset(email: str):
    """Email a one-time code. Returns (ok, message) -- the message is the
    same whether or not an account exists, so the form can't be used to
    probe which emails are registered."""
    generic = ("If an account exists for that email, a 6-digit code has been "
               "sent to it. The code is valid for 10 minutes.")
    state = _reset_state()
    if state and time.time() - state.get("last_sent", 0) < _OTP_RESEND_COOLDOWN:
        wait = int(_OTP_RESEND_COOLDOWN - (time.time() - state["last_sent"]))
        return False, f"A code was sent recently. Please wait {wait}s before requesting another."

    clerk = _get_clerk()
    user = None
    if clerk is not None:
        try:
            user = _find_clerk_user(clerk, email)
        except Exception:
            user = None

    # Generate/store state only when the account is real, but always claim a
    # send: the visitor sees identical output either way.
    if user is not None:
        code = "".join(str(pysecrets.randbelow(10)) for _ in range(_OTP_DIGITS))
        sent = _send_email(
            email,
            "Your password reset code - Virus Detection System",
            f"Your password reset code is: {code}\n\n"
            f"It is valid for 10 minutes. If you did not request this, you can "
            f"safely ignore this email -- your password has not been changed.",
        )
        if not sent:
            return False, ("Could not send the reset email. Please try again "
                           "later or contact your administrator.")
        st.session_state[AUTH_RESET_KEY] = {
            "email": email,
            "user_id": user.id,
            "code_hash": _hash_code(code),
            "expires_at": time.time() + _OTP_TTL_SECONDS,
            "attempts": 0,
            "last_sent": time.time(),
        }
    return True, generic


def _complete_password_reset(email: str, code: str, new_password: str):
    """Verify the emailed code and set the new password. Returns (ok, error)."""
    state = _reset_state()
    generic = "Invalid or expired code. Please request a new one."
    if (not state or state.get("email") != email
            or time.time() > state.get("expires_at", 0)):
        return False, generic
    if state["attempts"] >= _OTP_MAX_ATTEMPTS:
        st.session_state.pop(AUTH_RESET_KEY, None)
        return False, "Too many incorrect codes. Please request a new one."
    if not hmac.compare_digest(_hash_code(code.strip()), state["code_hash"]):
        state["attempts"] += 1
        return False, "Incorrect code. Please check the email and try again."

    clerk = _get_clerk()
    try:
        user = clerk.users.update(
            user_id=state["user_id"],
            password=new_password,
            sign_out_of_other_sessions=True,
            timeout_ms=15000,
        )
    except Exception:
        # Most likely Clerk rejected the password (too weak / breached).
        return False, ("This password can't be used (it may be too weak or "
                       "found in a known data breach). Please choose a "
                       "longer, unique password.")

    st.session_state.pop(AUTH_RESET_KEY, None)  # single use
    name = " ".join(p for p in (user.first_name, user.last_name) if p) or email
    _set_session(email, name, _role_from_user(user), "clerk")
    return True, None


# ---------------------------------------------------------------------------
# Change / set password for signed-in users
# ---------------------------------------------------------------------------

def _change_password(current_password: str, new_password: str):
    """Change the signed-in user's password. Identity proof: their current
    password -- except for Google-authenticated users without one, whose
    identity was already proven by the Google login. Returns (ok, error)."""
    sess = _session() or {}
    email = sess.get("email")
    clerk = _get_clerk()
    if not email or clerk is None:
        return False, "Password management is not available right now."

    wait = _throttle_seconds_left()
    if wait:
        return False, f"Too many failed attempts. Please wait {wait} seconds and try again."

    try:
        user = _find_clerk_user(clerk, email)
    except Exception:
        return False, "Temporarily unavailable. Please try again shortly."

    if user is not None and getattr(user, "password_enabled", False):
        try:
            clerk.users.verify_password(
                user_id=user.id, password=current_password, timeout_ms=10000)
        except Exception:
            _record_failure()
            return False, "Current password is incorrect."
    elif sess.get("login_method") != "google":
        return False, "Password management is not available for this account."

    weak = ("This password can't be used (it may be too weak or found in a "
            "known data breach). Please choose a longer, unique password.")
    try:
        if user is None:
            # Google-authenticated visitor with no Clerk record yet: create
            # one so they can also sign in with email/password. Standard
            # access -- roles are only ever granted in the Clerk dashboard.
            clerk.users.create(
                email_address=[email],
                password=new_password,
                first_name=(sess.get("name") or "").split(" ")[0] or None,
                public_metadata={"role": "user"},
                timeout_ms=15000,
            )
        else:
            clerk.users.update(
                user_id=user.id,
                password=new_password,
                sign_out_of_other_sessions=True,
                timeout_ms=15000,
            )
    except Exception:
        return False, weak
    return True, None


# ---------------------------------------------------------------------------
# Auth flows
# ---------------------------------------------------------------------------

def _sign_in(email: str, password: str):
    """Verify credentials against Clerk. Returns (ok, error_message)."""
    clerk = _get_clerk()
    if clerk is None:
        return False, ("Email/password sign-in is not configured. "
                       "Set the Clerk secret key in Streamlit secrets.")

    wait = _throttle_seconds_left()
    if wait:
        return False, f"Too many failed attempts. Please wait {wait} seconds and try again."

    # One generic message for every credential problem (unknown email, wrong
    # password, passwordless/Google-only account, banned/locked) so the form
    # doesn't leak which accounts exist or why a login was refused.
    generic = "Invalid email or password."
    try:
        user = _find_clerk_user(clerk, email)
    except Exception:
        return False, "Sign-in is temporarily unavailable. Please try again shortly."

    if (user is None or not getattr(user, "password_enabled", False)
            or getattr(user, "banned", False) or getattr(user, "locked", False)):
        _record_failure()
        return False, generic

    try:
        clerk.users.verify_password(user_id=user.id, password=password, timeout_ms=10000)
    except Exception:
        _record_failure()
        return False, generic

    name = " ".join(p for p in (user.first_name, user.last_name) if p) or email
    _set_session(email, name, _role_from_user(user), "clerk")
    return True, None


def _sign_up(first_name: str, last_name: str, email: str, password: str):
    """Create a Clerk account (role: user) and sign it in. Returns (ok, error)."""
    clerk = _get_clerk()
    if clerk is None:
        return False, ("Sign-up is not configured. "
                       "Set the Clerk secret key in Streamlit secrets.")
    try:
        from clerk_backend_api.models import ClerkErrors
        user = clerk.users.create(
            email_address=[email],
            password=password,
            first_name=first_name,
            last_name=last_name or None,
            public_metadata={"role": "user"},
            timeout_ms=15000,
        )
    except ClerkErrors as e:
        codes = _clerk_error_codes(e)
        if any(c in ("form_identifier_exists", "email_address_exists") for c in codes):
            return False, ("An account with this email already exists. "
                           "Please sign in instead.")
        if any(c.startswith("form_password") for c in codes):
            return False, ("This password can't be used (it may be too weak or "
                           "found in a known data breach). Please choose a "
                           "longer, unique password.")
        return False, "Sign-up failed. Please check your details and try again."
    except Exception:
        return False, "Sign-up is temporarily unavailable. Please try again shortly."

    name = " ".join(p for p in (first_name, last_name) if p) or email
    _set_session(email, name, _role_from_user(user), "clerk")
    return True, None


def _validate_signup(first_name, email, password, confirm):
    if not first_name.strip():
        return "Please enter your first name."
    if not _EMAIL_RE.match(email):
        return "Please enter a valid email address."
    if len(password) < _MIN_PASSWORD_LEN:
        return f"Password must be at least {_MIN_PASSWORD_LEN} characters long."
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        return "Password must contain at least one letter and one number."
    if password != confirm:
        return "Passwords do not match."
    return None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_AUTH_CSS = """
<style>
/* Center the auth card and give it a professional, elevated look */
.auth-hero { text-align: center; margin: 1.2rem 0 0.4rem 0; }
.auth-hero h1 { font-size: 1.9rem; margin-bottom: 0.2rem; }
.auth-hero p { color: #5f6b7a; margin-top: 0; }
.st-key-auth_card {
    max-width: 460px;
    margin: 0 auto;
    padding: 1.6rem 1.8rem 1.8rem 1.8rem;
    border-radius: 14px;
    border: 1px solid rgba(49, 51, 63, 0.15);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.10);
    background: var(--background-color, #ffffff);
}
.st-key-auth_card [data-testid="stForm"] { border: none; padding: 0; }
.st-key-auth_card button[kind="primary"],
.st-key-auth_card button[data-testid="stBaseButton-primaryFormSubmit"] {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
}
.auth-divider {
    display: flex; align-items: center; gap: 0.8rem;
    color: #8a93a1; font-size: 0.85rem; margin: 0.9rem 0 0.5rem 0;
}
.auth-divider::before, .auth-divider::after {
    content: ""; flex: 1; height: 1px; background: rgba(49, 51, 63, 0.2);
}
.auth-footnote { text-align: center; color: #8a93a1; font-size: 0.8rem; margin-top: 1rem; }
</style>
"""


def _render_forgot_password():
    """Two-step 'Forgot password?' flow inside the sign-in tab: request an
    emailed code, then enter it with a new password."""
    with st.expander("Forgot password?"):
        if not _smtp_config():
            st.info("Password reset by email is not configured on this "
                    "deployment. Please contact your administrator to reset "
                    "your password.")
            return

        awaiting_code = _reset_state() is not None

        with st.form("pw_reset_request_form", border=False):
            reset_email = st.text_input(
                "Account email", key="reset_email",
                placeholder="you@example.com", autocomplete="email",
            )
            sent = st.form_submit_button(
                "Resend code" if awaiting_code else "Email me a code")
        if sent:
            email = reset_email.strip().lower()
            if not _EMAIL_RE.match(email):
                st.error("Please enter a valid email address.")
            else:
                with st.spinner("Sending code…"):
                    ok, msg = _start_password_reset(email)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()

        if awaiting_code:
            st.caption("Enter the 6-digit code from the email, then choose "
                       "a new password.")
            with st.form("pw_reset_complete_form", border=False):
                code = st.text_input("6-digit code", key="reset_code",
                                     max_chars=_OTP_DIGITS, placeholder="123456")
                new_pw = st.text_input(
                    "New password", key="reset_new_pw", type="password",
                    placeholder=f"At least {_MIN_PASSWORD_LEN} characters, with a letter and a number",
                    autocomplete="new-password",
                )
                confirm_pw = st.text_input(
                    "Confirm new password", key="reset_confirm_pw",
                    type="password", autocomplete="new-password",
                )
                submitted = st.form_submit_button("Reset password", type="primary")
            if submitted:
                email = (st.session_state.get("reset_email") or "").strip().lower()
                err = None
                if len(new_pw) < _MIN_PASSWORD_LEN:
                    err = f"Password must be at least {_MIN_PASSWORD_LEN} characters long."
                elif not (re.search(r"[A-Za-z]", new_pw) and re.search(r"\d", new_pw)):
                    err = "Password must contain at least one letter and one number."
                elif new_pw != confirm_pw:
                    err = "Passwords do not match."
                if err is None:
                    with st.spinner("Resetting your password…"):
                        ok, err = _complete_password_reset(email, code, new_pw)
                    if ok:
                        st.toast("Password reset. You're signed in!", icon="✅")
                        st.rerun()
                if err:
                    st.error(err)


@st.cache_data(show_spinner=False)
def _auth_logos_html():
    """Co-branded logo strip for the top of the auth card: the ICMR/NIE,
    Department of Health Research, and Amity logos in one centred row (same
    order as the Home page), balanced to a common height with subtle dividers
    between them. Images are downscaled and inlined (base64). Returns '' if the
    files can't be read (the header then simply omits the logos)."""
    from PIL import Image

    def _enc(path, display_h):
        im = Image.open(path)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        target_h = display_h * 2  # 2x for crisp HiDPI rendering
        if im.height > target_h:
            im = im.resize((round(im.width * target_h / im.height), target_h))
        is_png = path.lower().endswith(".png")
        buf = BytesIO()
        im.save(buf, format="PNG" if is_png else "JPEG", quality=88, optimize=True)
        return (f"data:image/{'png' if is_png else 'jpeg'};base64,"
                + base64.b64encode(buf.getvalue()).decode())

    try:
        icmr = _enc("logo_1.jpeg", 96)
        dhr = _enc("logo_2.jpeg", 96)
        amity = _enc("Amity_logo2.png", 96)
    except Exception:
        return ""

    divider = ("<span style='width:1px;height:74px;"
               "background:rgba(49,51,63,0.18);'></span>")
    return (
        "<div style='display:flex;align-items:center;justify-content:center;"
        "gap:1.75rem;flex-wrap:wrap;margin:0.6rem 0 0.4rem;'>"
        f"<img src='{icmr}' alt='ICMR-NIE' style='height:96px;width:auto;'>"
        f"{divider}"
        f"<img src='{dhr}' alt='Department of Health Research' "
        "style='height:96px;width:auto;'>"
        f"{divider}"
        f"<img src='{amity}' alt='Amity Centre for Artificial Intelligence' "
        "style='height:96px;width:auto;'>"
        "</div>"
    )


def _render_login_screen():
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)

    # Co-branded ICMR/NIE + Amity logos, centred at the top of the sign-in card.
    _logos_html = _auth_logos_html()
    if _logos_html:
        st.markdown(_logos_html, unsafe_allow_html=True)
    else:
        col1, col2, col3 = st.columns([2, 3, 2])
        with col2:
            try:
                st.image("Amity_logo2.png", use_container_width=True)
            except Exception:
                pass
    st.markdown(
        "<div class='auth-hero'>"
        "<h1>🦠 Personalized Laboratory Test Recommendation System</h1>"
        "<p>Sign in to continue to the diagnostic workspace</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    clerk_ready = _get_clerk() is not None
    google_ready = _google_configured()

    with st.container(key="auth_card"):
        if not clerk_ready and not google_ready:
            st.error(
                "Authentication is not configured. An administrator must set "
                "the Clerk secret key (`[clerk] secret_key`) and/or Google "
                "OAuth (`[auth]`) in Streamlit secrets — see auth.py for the "
                "exact format."
            )
            st.stop()

        tab_signin, tab_signup = st.tabs(["🔐  Sign in", "✨  Create account"])

        with tab_signin:
            if clerk_ready:
                with st.form("clerk_signin_form", border=False):
                    email = st.text_input(
                        "Email address", key="signin_email",
                        placeholder="you@example.com", autocomplete="email",
                    )
                    password = st.text_input(
                        "Password", key="signin_password", type="password",
                        placeholder="Your password", autocomplete="current-password",
                    )
                    submitted = st.form_submit_button("Sign in", type="primary")
                if submitted:
                    email = email.strip().lower()
                    if not _EMAIL_RE.match(email) or not password:
                        st.error("Please enter your email and password.")
                    else:
                        with st.spinner("Signing you in…"):
                            ok, err = _sign_in(email, password)
                        if ok:
                            st.toast("Signed in successfully. Welcome back!", icon="✅")
                            st.rerun()
                        else:
                            st.error(err)
                _render_forgot_password()
            else:
                st.info("Email/password sign-in is not configured on this deployment.")

            if google_ready:
                if clerk_ready:
                    st.markdown("<div class='auth-divider'>or</div>", unsafe_allow_html=True)
                if st.button("Continue with Google", key="google_signin_btn",
                             use_container_width=True):
                    st.login()

        with tab_signup:
            if clerk_ready:
                with st.form("clerk_signup_form", border=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        first_name = st.text_input(
                            "First name", key="signup_first_name",
                            placeholder="First name", autocomplete="given-name",
                        )
                    with c2:
                        last_name = st.text_input(
                            "Last name", key="signup_last_name",
                            placeholder="Last name (optional)", autocomplete="family-name",
                        )
                    email = st.text_input(
                        "Email address", key="signup_email",
                        placeholder="you@example.com", autocomplete="email",
                    )
                    password = st.text_input(
                        "Password", key="signup_password", type="password",
                        placeholder=f"At least {_MIN_PASSWORD_LEN} characters, with a letter and a number",
                        autocomplete="new-password",
                    )
                    confirm = st.text_input(
                        "Confirm password", key="signup_confirm", type="password",
                        placeholder="Re-enter your password", autocomplete="new-password",
                    )
                    st.caption("New accounts start with standard (user) access. "
                               "Administrator access is granted separately.")
                    submitted = st.form_submit_button("Create account", type="primary")
                if submitted:
                    email = email.strip().lower()
                    err = _validate_signup(first_name, email, password, confirm)
                    if err is None:
                        with st.spinner("Creating your account…"):
                            ok, err = _sign_up(first_name.strip(), last_name.strip(),
                                               email, password)
                        if ok:
                            st.toast("Account created. Welcome!", icon="🎉")
                            st.rerun()
                    if err:
                        st.error(err)
            else:
                st.info("Account creation is not configured on this deployment.")

    st.markdown(
        "<p class='auth-footnote'>Authorized ICMR/NIE personnel only. "
        "All access is monitored.</p>",
        unsafe_allow_html=True,
    )
    st.stop()


# ---------------------------------------------------------------------------
# Public API (same signatures app.py always used)
# ---------------------------------------------------------------------------

def require_login():
    """
    Auth gate for the app. Call this as the very first line of main().
    Renders the sign-in/sign-up screen and stops the script if the visitor
    is not authenticated; otherwise returns and lets the app render.
    """
    sess = _session()
    if sess and sess.get("authenticated"):
        return

    # Returning from Streamlit's Google OAuth flow: adopt the identity and
    # resolve the role from Clerk by email (single source of truth).
    try:
        google_logged_in = st.user.is_logged_in
    except AttributeError:
        google_logged_in = False
    if google_logged_in:
        email = (getattr(st.user, "email", None) or "").strip().lower()
        name = getattr(st.user, "name", None) or email or "Signed in"
        with st.spinner("Checking your access…"):
            role = _resolve_role(email) if email else "user"
        _set_session(email, name, role, "google")
        return

    _render_login_screen()


def _render_change_password(sess):
    """Sidebar 'Change password' expander. Requires the current password;
    Google-authenticated users without one can set a first password instead
    (their identity is already proven by the Google login)."""
    if _get_clerk() is None:
        return
    has_password = sess.get("login_method") == "clerk"
    label = "🔑 Change password" if has_password else "🔑 Set a password"
    with st.sidebar.expander(label):
        with st.form("change_pw_form", border=False):
            current = ""
            if has_password:
                current = st.text_input(
                    "Current password", type="password",
                    key="chpw_current", autocomplete="current-password",
                )
            else:
                st.caption("Add a password so you can also sign in without "
                           "Google.")
            new_pw = st.text_input(
                "New password", type="password", key="chpw_new",
                placeholder=f"At least {_MIN_PASSWORD_LEN} characters, with a letter and a number",
                autocomplete="new-password",
            )
            confirm_pw = st.text_input(
                "Confirm new password", type="password", key="chpw_confirm",
                autocomplete="new-password",
            )
            submitted = st.form_submit_button("Update password", type="primary")
        if submitted:
            err = None
            if has_password and not current:
                err = "Please enter your current password."
            elif len(new_pw) < _MIN_PASSWORD_LEN:
                err = f"Password must be at least {_MIN_PASSWORD_LEN} characters long."
            elif not (re.search(r"[A-Za-z]", new_pw) and re.search(r"\d", new_pw)):
                err = "Password must contain at least one letter and one number."
            elif new_pw != confirm_pw:
                err = "Passwords do not match."
            elif has_password and current == new_pw:
                err = "The new password must be different from the current one."
            if err is None:
                with st.spinner("Updating password…"):
                    ok, err = _change_password(current, new_pw)
                if ok:
                    st.toast("Password updated.", icon="🔑")
                    st.success("Password updated.")
            if err:
                st.error(err)


def render_sign_out_control():
    """Sidebar identity + role badge + sign-out. Call after require_login()."""
    sess = _session() or {}
    role = get_current_role()
    badge = "🛡️ Admin" if role == "admin" else "👤 User"

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='display:flex;align-items:center;justify-content:space-between;"
        f"gap:0.5rem;padding:0.15rem 0.1rem;'>"
        f"<span style='font-size:0.85rem;overflow:hidden;text-overflow:ellipsis;"
        f"white-space:nowrap;' title='{sess.get('email', '')}'>"
        f"{sess.get('name') or sess.get('email') or 'Signed in'}</span>"
        f"<span style='font-size:0.72rem;font-weight:600;padding:2px 8px;"
        f"border-radius:10px;white-space:nowrap;"
        f"background:{'#e8f1fd' if role == 'admin' else '#eef2f6'};"
        f"color:{'#1a63c9' if role == 'admin' else '#4a5563'};'>{badge}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    _render_change_password(sess)

    if st.sidebar.button("Sign out", use_container_width=True, key="auth_sign_out_btn"):
        was_google = sess.get("login_method") == "google"
        for key in AUTH_STATE_KEYS:
            st.session_state.pop(key, None)
        if was_google:
            st.logout()
        st.rerun()
