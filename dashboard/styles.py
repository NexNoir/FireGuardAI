import streamlit as st


def inject_styles():

    st.markdown(
        """
        <style>

        .block-container {
            max-width: 1600px;
            padding-top: 1rem;
        }

        .hero {
            padding: 1.4rem;
            border-radius: 18px;
            background: linear-gradient(
                135deg,
                #161b22,
                #0d1117
            );
            border: 1px solid #30363d;
            margin-bottom: 1rem;
        }

        .status-card {
            padding: 1rem;
            border-radius: 14px;
            background: #161b22;
            border: 1px solid #30363d;
        }

        .status-live {
            color: #3fb950;
            font-weight: 800;
        }

        .status-stale {
            color: #f85149;
            font-weight: 900;
        }

        .status-warning {
            color: #d29922;
            font-weight: 800;
        }

        .status-unavailable {
            color: #8b949e;
            font-weight: 800;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )