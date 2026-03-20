import streamlit as st
from utils.helpers import init_session_state
from utils.agent import build_copilot
from components.overview import display_data_overview
from components.cleaning import display_cleaning
from components.visualization import display_visualizations
from components.analysis import display_advanced_analytics
from components.time_series import display_time_series_analysis
from components.filter import render_global_filter_widgets

st.set_page_config(
    page_title="Data Spectra",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("✨ Data Spectra")
st.write("**By: Arnab** | An Interactive EDA Tool")
st.markdown("---")

with st.container(border=True):
    st.header("Get Started: Upload Your Dataset")
    st.markdown("""
Welcome to DataSpectra! Upload a CSV file to begin analysis.
""")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/256/8242/8242984.png", width=200)

    with col2:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            label_visibility="collapsed"
        )
        if uploaded_file is None:
            st.info("📂 Waiting for file...")
        else:
            st.success("✅ File uploaded")

if uploaded_file is not None:
    if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.clear()
        init_session_state(uploaded_file)
        st.session_state.uploaded_file_name = uploaded_file.name
        st.rerun()

    if st.session_state.get("original_data") is not None and not st.session_state.original_data.empty:

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if "agent" not in st.session_state:
            st.session_state.agent = build_copilot(st.session_state.processed_data)

        with st.sidebar:
            with st.expander("💬 Data Chat Assistant", expanded=True):

                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                user_input = st.chat_input("Ask something about your data...")

                if user_input:
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input
                    })

                    with st.chat_message("user"):
                        st.markdown(user_input)

                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing..."):
                            try:
                                response = st.session_state.agent.invoke({
                                    "messages": st.session_state.chat_history
                                })

                                final_reply = response["response"]

                                st.markdown(final_reply)

                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": final_reply
                                })

                            except Exception as e:
                                error_msg = f"Agent error: {e}"
                                st.error(error_msg)
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": error_msg
                                })

                if st.button("🧹 Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
        globaly_filtered_data = render_global_filter_widgets(st.session_state.processed_data)
        display_data_overview(globaly_filtered_data)
        display_cleaning()
        display_visualizations(globaly_filtered_data)
        display_advanced_analytics(globaly_filtered_data)
        display_time_series_analysis(globaly_filtered_data)

    else:
        st.error("File processing failed")