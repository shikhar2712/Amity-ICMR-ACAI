"""
Clerk-based authentication gate for the Streamlit app.

Streamlit runs entirely server-side in Python: it has no direct way to read a
browser-side Clerk session cookie or receive an OAuth-style redirect callback.
This module bridges that gap with two pieces:

1. A plain HTML link (via st.components.v1.html) that sends the browser to
   Clerk's hosted "Account Portal" sign-in page and back -- no JavaScript
   bridge needed for this half, since it's just a normal link click.
2. `streamlit_js_eval` (a small community component) to run the Clerk
   JavaScript SDK in the browser after the redirect back, so we can read the
   resulting session token and hand it to Python for verification.

Required Streamlit secrets (Streamlit Cloud: App settings -> Secrets), or the
equivalent environment variables:

    [clerk]
    publishable_key    = "pk_live_..."                                  # Clerk Dashboard -> API Keys
    frontend_api        = "your-app.clerk.accounts.dev"                  # Clerk Dashboard -> API Keys -> Show API URLs
    jwks_url            = "https://your-app.clerk.accounts.dev/.well-known/jwks.json"  # same page
    sign_in_url         = "https://your-app.accounts.dev/sign-in"        # Clerk Dashboard -> Account Portal
    app_url             = "https://amityicmrnie.streamlit.app"           # this app's deployed URL
    authorized_parties  = "https://amityicmrnie.streamlit.app"           # comma-separated allow-list for the azp claim

Environment variable fallbacks use the same names upper-cased and prefixed
with CLERK_, e.g. CLERK_PUBLISHABLE_KEY, CLERK_JWKS_URL, CLERK_SIGN_IN_URL,
CLERK_APP_URL, CLERK_AUTHORIZED_PARTIES.
"""
import os
from urllib.parse import quote

import jwt
import streamlit as st
from streamlit_js_eval import streamlit_js_eval


def _clerk_config():
    try:
        secrets_cfg = st.secrets.get('clerk', {})
    except Exception:
        # No secrets.toml at all (e.g. local dev without Clerk configured yet) --
        # fall back to environment variables below instead of crashing.
        secrets_cfg = {}

    def _get(key, env_name):
        return secrets_cfg.get(key) or os.getenv(env_name, '')

    return {
        'publishable_key': _get('publishable_key', 'CLERK_PUBLISHABLE_KEY'),
        'frontend_api': _get('frontend_api', 'CLERK_FRONTEND_API'),
        'jwks_url': _get('jwks_url', 'CLERK_JWKS_URL'),
        'sign_in_url': _get('sign_in_url', 'CLERK_SIGN_IN_URL'),
        'app_url': _get('app_url', 'CLERK_APP_URL'),
        'authorized_parties': [
            p.strip() for p in _get('authorized_parties', 'CLERK_AUTHORIZED_PARTIES').split(',') if p.strip()
        ],
    }


def _config_is_complete(config: dict) -> bool:
    return all(config[k] for k in ('publishable_key', 'frontend_api', 'jwks_url', 'sign_in_url', 'app_url'))


@st.cache_resource
def _jwks_client(jwks_url: str):
    return jwt.PyJWKClient(jwks_url)


def _verify_session_token(token: str, config: dict):
    """Verify a Clerk session JWT's signature/expiry/authorized-party. Returns claims or None."""
    try:
        signing_key = _jwks_client(config['jwks_url']).get_signing_key_from_jwt(token)
        claims = jwt.decode(token, signing_key.key, algorithms=['RS256'], options={'verify_aud': False})
    except jwt.PyJWTError:
        return None
    if config['authorized_parties'] and claims.get('azp') not in config['authorized_parties']:
        return None
    return claims


def _clerk_loader_js(config: dict) -> str:
    """JS snippet that ensures window.Clerk is loaded, as a string to splice into a larger expression."""
    return f"""
        if (!window.Clerk) {{
            await new Promise((resolve, reject) => {{
                const s = document.createElement('script');
                s.async = true;
                s.crossOrigin = 'anonymous';
                s.setAttribute('data-clerk-publishable-key', '{config['publishable_key']}');
                s.src = 'https://{config['frontend_api']}/npm/@clerk/clerk-js@5/dist/clerk.browser.js';
                s.addEventListener('load', resolve);
                s.addEventListener('error', reject);
                document.head.appendChild(s);
            }});
        }}
        await window.Clerk.load();
    """


def _fetch_browser_session_token(config: dict):
    """
    Run Clerk JS in the browser and report back whether the visitor already
    has an active Clerk session (e.g. just arrived back from the sign-in
    redirect, or returned in a session that's still valid).

    Returns:
        None             -- the browser-side check hasn't reported back yet
        "__no_session__" -- Clerk loaded and confirmed there is no active session
        "__load_error__" -- the Clerk script failed to load (network/ad-blocker/etc.)
        <token string>   -- an active session's token, ready to verify

    "__load_error__" is treated the same as "__no_session__" by the caller --
    without a loaded SDK we can't auto-detect an existing session, but the
    sign-in link itself doesn't depend on the SDK, so the visitor can still
    sign in manually.
    """
    js = f"""
    (async () => {{
        try {{
            {_clerk_loader_js(config)}
            if (window.Clerk.session) {{
                return await window.Clerk.session.getToken();
            }}
            return "__no_session__";
        }} catch (e) {{
            return "__load_error__";
        }}
    }})()
    """
    return streamlit_js_eval(js_expressions=js, key='clerk_session_check')


def _sign_out_in_browser(config: dict):
    js = f"""
    (async () => {{
        {_clerk_loader_js(config)}
        await window.Clerk.signOut();
        return true;
    }})()
    """
    streamlit_js_eval(js_expressions=js, key='clerk_sign_out')


def _render_sign_in_screen(config: dict):
    # NOTE: this must be a native st.link_button, not a link embedded via
    # st.components.v1.html(). Streamlit renders components.html() content in
    # a sandboxed iframe without the allow-top-navigation flag, so a plain
    # <a target="_top"> click gets silently blocked by the browser -- it looks
    # like the "Sign in" button does nothing. st.link_button renders directly
    # in the real page (no sandboxed iframe), so it isn't affected.
    sign_in_url = f"{config['sign_in_url']}?redirect_url={quote(config['app_url'], safe='')}"
    st.markdown(
        "<div style='display:flex;flex-direction:column;align-items:center;"
        "justify-content:center;height:40vh;gap:0.5rem;text-align:center;'>"
        "<h2>🦠 Virus Detection and Classification System</h2>"
        "<p>Please sign in with your ICMR/NIE account to continue.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.link_button("Sign in", sign_in_url, use_container_width=True, type="primary")
        st.caption("Opens in a new tab. Once you've signed in there, come back to this tab and click below.")
        if st.button("I've signed in", use_container_width=True):
            st.rerun()


def require_login():
    """
    Auth gate for the app. Call this as the very first line of main().
    Renders a sign-in screen and stops the script if the visitor is not
    authenticated; otherwise returns and lets the rest of the app render.
    """
    if st.session_state.get('clerk_authenticated'):
        return

    config = _clerk_config()
    if not _config_is_complete(config):
        st.error(
            "Clerk authentication is not configured. Set publishable_key, frontend_api, "
            "jwks_url, sign_in_url, and app_url under [clerk] in Streamlit secrets."
        )
        st.stop()

    result = _fetch_browser_session_token(config)

    if result is None:
        st.markdown(
            "<div style='display:flex;align-items:center;justify-content:center;height:50vh;'>"
            "Checking session…</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    if result not in ('__no_session__', '__load_error__'):
        claims = _verify_session_token(result, config)
        if claims:
            st.session_state['clerk_authenticated'] = True
            st.session_state['clerk_user'] = {
                'id': claims.get('sub'),
                'email': claims.get('email') or claims.get('primary_email_address'),
            }
            st.rerun()

    _render_sign_in_screen(config)
    st.stop()


def render_sign_out_control():
    """Sidebar sign-out control. Call from within main(), after require_login()."""
    user = st.session_state.get('clerk_user') or {}
    st.sidebar.markdown("---")
    st.sidebar.caption(f"👤 {user.get('email') or 'Signed in'}")
    if st.sidebar.button("Sign out", use_container_width=True, key="clerk_sign_out_btn"):
        _sign_out_in_browser(_clerk_config())
        st.session_state.pop('clerk_authenticated', None)
        st.session_state.pop('clerk_user', None)
        st.rerun()
