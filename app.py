import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import time
from typing import Dict, Any

# Page config
st.set_page_config(
    page_title="VAERS AI Agent",
    page_icon="🤖",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 18px;
        margin: 5px 0;
        margin-left: 20%;
        text-align: right;
    }
    
    .agent-message {
        background-color: #e9ecef;
        color: #333;
        padding: 10px 15px;
        border-radius: 18px;
        margin: 5px 0;
        margin-right: 20%;
    }
    
    .agent-thinking {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px 15px;
        border-radius: 18px;
        margin: 5px 0;
        margin-right: 20%;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

def load_vaers_data():
    """Load VAERS data with timeout and error handling"""
    try:
        with st.spinner("📡 Loading data from GitHub..."):
            # GitHub URLs
            symptoms_url = "https://raw.githubusercontent.com/aysannazarmohamady/agentic_pharma/main/2024VAERSSYMPTOMS.csv"
            vax_url = "https://raw.githubusercontent.com/aysannazarmohamady/agentic_pharma/main/2024VAERSVAX.csv"
            
            # Load with timeout simulation
            symptoms_df = pd.read_csv(symptoms_url, dtype=str, nrows=1000)  # Limit rows for testing
            vax_df = pd.read_csv(vax_url, dtype=str, nrows=1000)  # Limit rows for testing
            
            st.success(f"✅ Loaded: {len(symptoms_df)} symptoms, {len(vax_df)} vaccines")
            return symptoms_df, vax_df
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None

def simple_analysis(query: str, symptoms_df: pd.DataFrame, vax_df: pd.DataFrame):
    """Simple analysis without hanging"""
    
    try:
        query_lower = query.lower()
        
        # Manufacturer analysis
        if "manufacturer" in query_lower:
            if vax_df is not None and 'VAX_MANU' in vax_df.columns:
                manufacturer_counts = vax_df['VAX_MANU'].value_counts().head(5)
                
                result_df = pd.DataFrame({
                    'Manufacturer': manufacturer_counts.index,
                    'Count': manufacturer_counts.values
                })
                
                return "🏭 **Top Vaccine Manufacturers:**", result_df
            else:
                return "❌ Manufacturer data not available", None
        
        # Statistics
        elif "statistic" in query_lower:
            stats_data = [
                {"Metric": "Symptom Records", "Value": len(symptoms_df) if symptoms_df is not None else 0},
                {"Metric": "Vaccine Records", "Value": len(vax_df) if vax_df is not None else 0},
                {"Metric": "Manufacturers", "Value": vax_df['VAX_MANU'].nunique() if vax_df is not None and 'VAX_MANU' in vax_df.columns else 0}
            ]
            
            result_df = pd.DataFrame(stats_data)
            return "📊 **Dataset Statistics:**", result_df
        
        # Neurological symptoms
        elif "neurological" in query_lower or "symptom" in query_lower:
            if symptoms_df is not None and 'SYMPTOM1' in symptoms_df.columns:
                # Simple symptom count
                symptom_counts = symptoms_df['SYMPTOM1'].value_counts().head(10)
                
                result_df = pd.DataFrame({
                    'Symptom': symptom_counts.index,
                    'Count': symptom_counts.values
                })
                
                return "🧠 **Top Symptoms:**", result_df
            else:
                return "❌ Symptom data not available", None
        
        # Default
        else:
            summary_data = [
                {"Type": "Symptoms", "Records": len(symptoms_df) if symptoms_df is not None else 0},
                {"Type": "Vaccines", "Records": len(vax_df) if vax_df is not None else 0}
            ]
            
            result_df = pd.DataFrame(summary_data)
            return "📋 **Data Summary:**", result_df
            
    except Exception as e:
        return f"❌ Analysis error: {str(e)}", None

def display_message(message: str, is_user: bool = False, is_thinking: bool = False):
    """Display chat message"""
    if is_thinking:
        css_class = "agent-thinking"
    elif is_user:
        css_class = "user-message"
    else:
        css_class = "agent-message"
    
    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)

def main():
    st.title("🤖 VAERS AI Agent (Debug Version)")
    st.markdown("*Simple and fast VAERS data analysis*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key (for future use)
        groq_api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            value="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"
        )
        
        st.markdown("---")
        
        # Load data button
        if st.button("🔄 Load VAERS Data", type="primary"):
            symptoms_df, vax_df = load_vaers_data()
            if symptoms_df is not None:
                st.session_state.symptoms_df = symptoms_df
                st.session_state.vax_df = vax_df
                st.session_state.data_loaded = True
                st.rerun()
        
        # Data status
        if 'data_loaded' in st.session_state and st.session_state.data_loaded:
            st.success("✅ Data Loaded")
            if 'symptoms_df' in st.session_state:
                st.metric("Symptoms", len(st.session_state.symptoms_df))
            if 'vax_df' in st.session_state:
                st.metric("Vaccines", len(st.session_state.vax_df))
        else:
            st.warning("⚠️ Please load data first")
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Tests:")
        st.markdown("""
        - "Show me manufacturers"
        - "What are the statistics?"
        - "Show me symptoms"
        """)
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append({
            "message": "👋 Hello! I'm a simplified VAERS AI Agent. Click 'Load VAERS Data' in the sidebar first, then ask me questions!",
            "is_user": False
        })
    
    # Chat interface
    st.markdown("### 💬 Chat")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        display_message(chat["message"], chat.get("is_user", False), chat.get("is_thinking", False))
    
    # Chat input
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        user_input = st.chat_input("Ask me about VAERS data...")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({
                "message": user_input,
                "is_user": True
            })
            
            # Show thinking
            with st.spinner("🤖 Analyzing..."):
                time.sleep(1)  # Small delay for UX
                
                # Get data
                symptoms_df = st.session_state.get('symptoms_df', None)
                vax_df = st.session_state.get('vax_df', None)
                
                # Perform analysis
                message, result_df = simple_analysis(user_input, symptoms_df, vax_df)
                
                # Prepare response
                if result_df is not None:
                    response = f"{message}\n\n{result_df.to_markdown(index=False)}"
                else:
                    response = message
                
                # Add AI response
                st.session_state.chat_history.append({
                    "message": response,
                    "is_user": False
                })
            
            st.rerun()
    
    else:
        st.info("👆 Please load VAERS data from the sidebar first!")
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()
