"""
Google-based authentication gate for the Streamlit app, using Streamlit's own
built-in OIDC login (st.login / st.logout / st.user) instead of a third-party
JS SDK. Streamlit's server handles the OAuth redirect and callback itself and
stores a signed cookie -- there's no browser-side JavaScript bridge involved,
so this doesn't hit the sandboxed-iframe / component-caching issues a JS-SDK
based provider (e.g. Clerk) runs into inside a server-rendered app like this.

Required Streamlit secrets (Streamlit Cloud: App settings -> Secrets):

    [auth]
    redirect_uri = "https://amityicmrnie.streamlit.app/oauth2callback"
    cookie_secret = "<a long random string, e.g. `openssl rand -hex 32`>"
    client_id = "<Google OAuth client ID>"
    client_secret = "<Google OAuth client secret>"
    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

`client_id`/`client_secret` come from a Google Cloud "OAuth 2.0 Client ID"
(APIs & Services -> Credentials), with `redirect_uri` added to that client's
"Authorized redirect URIs". `redirect_uri` must be your app's exact deployed
URL plus the fixed path `/oauth2callback` -- Streamlit owns that route.

Requires Authlib>=1.3.2 (listed in requirements.txt as `Authlib`).
"""
import streamlit as st


def require_login():
    """
    Auth gate for the app. Call this as the very first line of main().
    Renders a sign-in screen and stops the script if the visitor is not
    authenticated; otherwise returns and lets the rest of the app render.
    """
    try:
        if st.user.is_logged_in:
            return
    except AttributeError:
        # st.user has no attributes at all when [auth] isn't configured in
        # secrets.toml -- rather than "not logged in", that's "not set up".
        st.error(
            "Google sign-in is not configured. Set client_id, client_secret, "
            "server_metadata_url, redirect_uri, and cookie_secret under [auth] "
            "in Streamlit secrets."
        )
        st.stop()

    st.markdown(
        "<div style='display:flex;flex-direction:column;align-items:center;"
        "justify-content:center;height:40vh;gap:0.5rem;text-align:center;'>"
        "<h2>🦠 Virus Detection and Classification System</h2>"
        "<p>Please sign in with your ICMR/NIE Google account to continue.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Sign in with Google", use_container_width=True, type="primary"):
            st.login()
    st.stop()


def render_sign_out_control():
    """Sidebar sign-out control. Call from within main(), after require_login()."""
    st.sidebar.markdown("---")
    st.sidebar.caption(f"👤 {st.user.email or st.user.name or 'Signed in'}")
    if st.sidebar.button("Sign out", use_container_width=True, key="auth_sign_out_btn"):
        st.logout()
