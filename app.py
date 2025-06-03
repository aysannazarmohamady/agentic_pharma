import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import json
from typing import Dict, Any

# Page config
st.set_page_config(
    page_title="VAERS AI Agent",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for ChatGPT-like interface
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .chat-container {
        height: 400px;
        overflow-y: scroll;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    
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
    
    .stChatInput {
        position: fixed;
        bottom: 0;
        width: 100%;
        background: white;
        padding: 1rem;
        border-top: 1px solid #ddd;
    }
    
    h1 {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
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

class VAERSAgent:
    def __init__(self, groq_api_key: str):
        """Initialize VAERS AI Agent"""
        self.client = Groq(api_key=groq_api_key) if groq_api_key else None
        self.symptoms_df = None
        self.vax_df = None
        self.data_loaded = False
        
        # Neurological symptoms of interest
        self.neurological_symptoms = [
            "Guillain-Barre syndrome", "Miller Fisher syndrome",
            "Acute motor axonal neuropathy", "Motor neurone disease",
            "Muscle weakness", "Demyelinating polyneuropathy",
            "Chronic inflammatory demyelinating polyradiculoneuropathy",
            "Autoimmune neuropathy", "Peripheral motor neuropathy"
        ]
    
    def load_data(self):
        """Load VAERS data from GitHub repository"""
        try:
            # Try to load from session state first
            if 'symptoms_df' in st.session_state and 'vax_df' in st.session_state:
                self.symptoms_df = st.session_state.symptoms_df
                self.vax_df = st.session_state.vax_df
                self.data_loaded = True
                return True
            
            # Load from GitHub repository
            try:
                # GitHub raw URLs for your uploaded files
                github_base_url = "https://raw.githubusercontent.com/aysannazarmohamady/agentic_pharma/main/"
                
                symptoms_url = github_base_url + "2024VAERSSYMPTOMS.csv"
                vax_url = github_base_url + "2024VAERSVAX.csv"
                
                st.info("🔄 Loading VAERS data from GitHub repository...")
                
                # Load symptoms data
                self.symptoms_df = pd.read_csv(
                    symptoms_url,
                    dtype=str,
                    encoding='utf-8',
                    on_bad_lines='skip'
                )
                
                # Load vaccine data  
                self.vax_df = pd.read_csv(
                    vax_url,
                    dtype=str,
                    encoding='latin-1', 
                    on_bad_lines='skip'
                )
                
                # Store in session state
                st.session_state.symptoms_df = self.symptoms_df
                st.session_state.vax_df = self.vax_df
                
                self.data_loaded = True
                st.success(f"✅ Data loaded from GitHub! Symptoms: {len(self.symptoms_df):,} records, Vaccines: {len(self.vax_df):,} records")
                return True
                
            except Exception as github_error:
                st.warning(f"Could not load from GitHub: {str(github_error)}")
                return False
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """Analyze user query and determine what action to take"""
        
        # Define query types and corresponding actions
        query_patterns = {
            "manufacturer": ["manufacturer", "company", "pfizer", "moderna", "j&j", "johnson"],
            "symptoms": ["symptom", "adverse", "effect", "reaction", "neurological"],
            "statistics": ["count", "number", "total", "average", "percentage", "rate"],
            "comparison": ["compare", "versus", "vs", "difference", "between"],
            "trend": ["trend", "over time", "year", "monthly", "increase", "decrease"],
            "severity": ["death", "severe", "serious", "hospitalization", "disability"]
        }
        
        query_lower = user_query.lower()
        detected_types = []
        
        for query_type, keywords in query_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_types.append(query_type)
        
        # Determine specific analysis based on query content
        analysis_plan = {
            "query_types": detected_types,
            "needs_table": True,
            "needs_chart": "trend" in detected_types or "comparison" in detected_types,
            "focus_neurological": any(symptom.lower() in query_lower for symptom in self.neurological_symptoms)
        }
        
        return analysis_plan
    
    def execute_analysis(self, user_query: str, analysis_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the analysis based on the plan"""
        
        results = {
            "thinking": "🤖 Analyzing VAERS data...",
            "data_summary": {},
            "main_table": None,
            "additional_info": "",
            "chart_data": None
        }
        
        if not self.data_loaded:
            results["thinking"] = "❌ No VAERS data loaded. Please reload data from GitHub."
            return results
        
        try:
            # Get basic data summary
            unique_reports = len(self.symptoms_df['VAERS_ID'].unique()) if self.symptoms_df is not None else 0
            total_vaccines = len(self.vax_df) if self.vax_df is not None else 0
            
            results["data_summary"] = {
                "total_reports": unique_reports,
                "total_vaccine_records": total_vaccines
            }
            
            # Simple manufacturer analysis
            if "manufacturer" in analysis_plan["query_types"] or "manufacturer" in user_query.lower():
                results["thinking"] = "🔍 Analyzing vaccine manufacturers..."
                
                if self.vax_df is not None and 'VAX_MANU' in self.vax_df.columns:
                    # Limit to prevent hanging
                    manufacturer_counts = self.vax_df['VAX_MANU'].value_counts().head(10)
                    
                    results["main_table"] = pd.DataFrame({
                        'Manufacturer': manufacturer_counts.index,
                        'Report_Count': manufacturer_counts.values,
                        'Percentage': (manufacturer_counts.values / len(self.vax_df) * 100).round(2)
                    })
                    
                    results["additional_info"] = f"Top 10 vaccine manufacturers by report count"
                else:
                    results["main_table"] = pd.DataFrame({"Error": ["Manufacturer data not available"]})
            
            # Simple neurological symptoms analysis  
            elif "neurological" in user_query.lower() or "symptoms" in user_query.lower():
                results["thinking"] = "🧠 Analyzing neurological symptoms..."
                
                if self.symptoms_df is not None:
                    # Quick neurological symptom check
                    neuro_found = {}
                    for symptom in self.neurological_symptoms[:5]:  # Limit to prevent hanging
                        count = 0
                        for col in ['SYMPTOM1', 'SYMPTOM2']:  # Check only first 2 columns
                            if col in self.symptoms_df.columns:
                                symptom_data = self.symptoms_df[col].fillna('').astype(str)
                                count += symptom_data.str.contains(symptom, case=False, na=False).sum()
                        
                        if count > 0:
                            neuro_found[symptom] = count
                    
                    if neuro_found:
                        results["main_table"] = pd.DataFrame([
                            {"Neurological_Symptom": k, "Report_Count": v} 
                            for k, v in sorted(neuro_found.items(), key=lambda x: x[1], reverse=True)
                        ])
                        results["additional_info"] = f"Found {len(neuro_found)} neurological symptoms"
                    else:
                        results["main_table"] = pd.DataFrame({"Message": ["No specific neurological symptoms found in search"]})
                else:
                    results["main_table"] = pd.DataFrame({"Error": ["Symptoms data not available"]})
            
            # Simple statistics
            else:
                results["thinking"] = "📊 Calculating basic statistics..."
                
                stats_data = []
                
                if self.symptoms_df is not None:
                    stats_data.append({
                        "Metric": "Total Symptom Reports",
                        "Value": f"{len(self.symptoms_df):,}",
                        "Description": "Individual symptom records"
                    })
                    
                    stats_data.append({
                        "Metric": "Unique VAERS Reports",
                        "Value": f"{len(self.symptoms_df['VAERS_ID'].unique()):,}",
                        "Description": "Unique adverse event reports"
                    })
                
                if self.vax_df is not None:
                    stats_data.append({
                        "Metric": "Total Vaccine Records",
                        "Value": f"{len(self.vax_df):,}",
                        "Description": "Vaccine administration records"
                    })
                    
                    if 'VAX_MANU' in self.vax_df.columns:
                        stats_data.append({
                            "Metric": "Vaccine Manufacturers",
                            "Value": f"{self.vax_df['VAX_MANU'].nunique()}",
                            "Description": "Unique manufacturers"
                        })
                
                results["main_table"] = pd.DataFrame(stats_data)
                results["additional_info"] = "Basic VAERS dataset statistics"
        
        except Exception as e:
            results["thinking"] = f"❌ Error during analysis: {str(e)}"
            results["main_table"] = pd.DataFrame({"Error": [f"Analysis failed: {str(e)}"]})
        
        return results
    
    def generate_ai_insights(self, query: str, results: Dict[str, Any]) -> str:
        """Generate AI insights using Groq"""
        
        if not self.client:
            return "AI insights unavailable (no API key provided)"
        
        try:
            # Prepare context for AI
            context = f"""
            User Query: {query}
            
            Analysis Results:
            - Total Reports: {results['data_summary'].get('total_reports', 'N/A')}
            - Total Vaccine Records: {results['data_summary'].get('total_vaccine_records', 'N/A')}
            - Analysis Type: {results['additional_info']}
            
            Table Data: {results['main_table'].to_string() if results['main_table'] is not None else 'No table data'}
            
            Please provide concise insights about this VAERS data analysis in 2-3 sentences.
            Focus on the most important findings and their potential implications for vaccine safety.
            """
            
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"AI analysis error: {str(e)}"

def display_chat_message(message: str, is_user: bool = False, is_thinking: bool = False):
    """Display a chat message with appropriate styling"""
    if is_thinking:
        css_class = "agent-thinking"
    elif is_user:
        css_class = "user-message"
    else:
        css_class = "agent-message"
    
    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)

def main():
    st.title("🤖 VAERS AI Agent")
    st.markdown("*Ask me anything about VAERS data and I'll analyze it for you*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        groq_api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            value="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",
            help="Your Groq API key for AI insights"
        )
        
        st.markdown("---")
        
        # Data status
        st.subheader("📊 Data Status")
        
        if 'symptoms_df' in st.session_state and 'vax_df' in st.session_state:
            st.success("✅ Data Loaded from GitHub")
            st.metric("Symptoms Records", f"{len(st.session_state.symptoms_df):,}")
            st.metric("Vaccine Records", f"{len(st.session_state.vax_df):,}")
        else:
            st.warning("⏳ Loading data...")
        
        # Manual reload button
        if st.button("🔄 Reload Data from GitHub"):
            if 'data_auto_loaded' in st.session_state:
                del st.session_state['data_auto_loaded']
            if 'symptoms_df' in st.session_state:
                del st.session_state['symptoms_df']
            if 'vax_df' in st.session_state:
                del st.session_state['vax_df']
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 💭 Example Questions")
        st.markdown("""
        - "Show me top vaccine manufacturers"
        - "What are the most common neurological symptoms?"
        - "Compare Pfizer vs Moderna adverse events"
        - "Statistics on Guillain-Barre syndrome"
        - "Total number of reports by manufacturer"
        """)
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append({
            "message": "👋 Hello! I'm your VAERS AI Agent. I'm loading the latest VAERS data from GitHub. Ask me questions about vaccine adverse events, manufacturers, symptoms, or statistics.",
            "is_user": False,
            "is_thinking": False
        })
    
    # Initialize agent
    agent = VAERSAgent(groq_api_key)
    
    # Auto-load data on startup
    if 'data_auto_loaded' not in st.session_state:
        with st.spinner("🔄 Loading VAERS data from GitHub..."):
            data_loaded = agent.load_data()
            if data_loaded:
                st.session_state.chat_history.append({
                    "message": f"✅ **VAERS data loaded successfully!**\n\n📊 **Dataset Ready:**\n- Symptoms: {len(agent.symptoms_df):,} records\n- Vaccines: {len(agent.vax_df):,} records\n\n🤖 **Ready to answer your questions!**\n\nTry asking:\n- *'Show me top vaccine manufacturers'*\n- *'What are the most common neurological symptoms?'*\n- *'Statistics on adverse events'*",
                    "is_user": False,
                    "is_thinking": False
                })
            else:
                st.session_state.chat_history.append({
                    "message": "❌ Could not load VAERS data from GitHub. Please check the repository and file names.",
                    "is_user": False,
                    "is_thinking": False
                })
        st.session_state.data_auto_loaded = True
    
    # Chat interface
    st.markdown("### 💬 Chat with VAERS AI Agent")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            display_chat_message(
                chat["message"], 
                chat["is_user"], 
                chat.get("is_thinking", False)
            )
    
    # Chat input
    user_input = st.chat_input("Ask me about VAERS data...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_history.append({
            "message": user_input,
            "is_user": True,
            "is_thinking": False
        })
        
        # Add thinking message
        st.session_state.chat_history.append({
            "message": "🤖 Thinking and analyzing your request...",
            "is_user": False,
            "is_thinking": True
        })
        
        st.rerun()
        
        # Process the query
        try:
            # Analyze query and execute
            analysis_plan = agent.analyze_query(user_input)
            results = agent.execute_analysis(user_input, analysis_plan)
            
            # Remove thinking message
            st.session_state.chat_history.pop()
            
            # Create response message
            response_parts = []
            
            # Add data summary
            if results["data_summary"]:
                response_parts.append(f"📈 **Analysis completed for {results['data_summary'].get('total_reports', 'N/A')} reports**")
            
            # Add main findings
            if results["additional_info"]:
                response_parts.append(f"🔍 **{results['additional_info']}**")
            
            # Add table
            if results["main_table"] is not None and not results["main_table"].empty:
                response_parts.append("📋 **Results Table:**")
                response_parts.append(results["main_table"].to_markdown(index=False))
            
            # Add AI insights
            if groq_api_key:
                ai_insights = agent.generate_ai_insights(user_input, results)
                response_parts.append(f"🧠 **AI Insights:** {ai_insights}")
            
            # Combine response
            full_response = "\n\n".join(response_parts)
            
            # Add agent response to chat
            st.session_state.chat_history.append({
                "message": full_response,
                "is_user": False,
                "is_thinking": False
            })
            
        except Exception as e:
            # Remove thinking message
            if st.session_state.chat_history and st.session_state.chat_history[-1].get("is_thinking"):
                st.session_state.chat_history.pop()
            
            # Add error message
            st.session_state.chat_history.append({
                "message": f"❌ Sorry, I encountered an error: {str(e)}",
                "is_user": False,
                "is_thinking": False
            })
        
        st.rerun()
    
    # Clear chat button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()
