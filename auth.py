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
import os
import re
import time

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
AUTH_STATE_KEYS = (AUTH_SESSION_KEY, AUTH_THROTTLE_KEY)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MIN_PASSWORD_LEN = 8

# Sign-in throttling: after this many consecutive failures, refuse further
# attempts for the lockout window. Per Streamlit session.
_MAX_FAILURES = 5
_LOCKOUT_SECONDS = 60


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


def _render_login_screen():
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        try:
            st.image("Amity_logo2.png", use_container_width=True)
        except Exception:
            pass
    st.markdown(
        "<div class='auth-hero'>"
        "<h1>🦠 Virus Detection and Classification System</h1>"
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
    if st.sidebar.button("Sign out", use_container_width=True, key="auth_sign_out_btn"):
        was_google = sess.get("login_method") == "google"
        for key in AUTH_STATE_KEYS:
            st.session_state.pop(key, None)
        if was_google:
            st.logout()
        st.rerun()
