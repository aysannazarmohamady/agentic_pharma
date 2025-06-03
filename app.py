import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
from groq import Groq
import json
from typing import List, Dict, Any
import io

# Page config
st.set_page_config(
    page_title="VAERS Data Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class VAERSProcessor:
    def __init__(self, groq_api_key: str):
        """Initialize VAERS processor with Groq API key"""
        if groq_api_key:
            self.client = Groq(api_key=groq_api_key)
        else:
            self.client = None
        
        self.symptoms_of_interest = [
            "Acute motor axonal neuropathy",
            "Acute motor-sensory axonal neuropathy",
            "Guillain-Barre syndrome", 
            "Miller Fisher syndrome",
            "Subacute inflammatory demyelinating polyneuropathy",
            "Autoimmune nodopathy",
            "Chronic inflammatory demyelinating polyradiculoneuropathy",
            "Demyelinating polyneuropathy",
            "Lewis-Sumner syndrome",
            "Multifocal motor neuropathy",
            "Polyneuropathy idiopathic",
            "Polyneuropathy idiopathic progressive",
            "Autoimmune neuropathy",
            "Immune-mediated neuropathy",
            "Peripheral motor neuropathy",
            "Lower motor neurone lesion",
            "Motor neurone disease",
            "Upper motor neurone lesion",
            "Neuromuscular block prolonged",
            "Neuromuscular blockade",
            "Sensorimotor disorder",
            "Muscle weakness",
            "Neuromuscular pain",
            "Neuromyopathy",
            "Neuropathic muscular atrophy",
            "Autoimmune demyelinating disease"
        ]
    
    def clean_vaers_id(self, vaers_id):
        """Clean VAERS_ID by removing leading zeros and whitespace"""
        if pd.isna(vaers_id):
            return vaers_id
        
        # Convert to string and strip whitespace
        cleaned = str(vaers_id).strip()
        
        # Remove leading zeros
        cleaned = cleaned.lstrip('0')
        
        # Return original if all zeros were removed
        return cleaned if cleaned else '0'
    
    def process_symptoms_file(self, file_content) -> pd.DataFrame:
        """Process VAERS symptoms file similar to R function"""
        try:
            st.info("🔄 Processing symptoms file...")
            
            # Try different encodings and handle CSV parsing issues
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    # Reset file pointer
                    file_content.seek(0)
                    
                    # Read CSV with error handling
                    df = pd.read_csv(file_content, 
                                   dtype={
                                       'VAERS_ID': str,
                                       'SYMPTOM1': str,
                                       'SYMPTOMVERSION1': str,
                                       'SYMPTOM2': str,
                                       'SYMPTOMVERSION2': str,
                                       'SYMPTOM3': str,
                                       'SYMPTOMVERSION3': str,
                                       'SYMPTOM4': str,
                                       'SYMPTOMVERSION4': str,
                                       'SYMPTOM5': str,
                                       'SYMPTOMVERSION5': str
                                   },
                                   na_values=['', 'NA', 'NULL'],
                                   encoding=encoding,
                                   on_bad_lines='skip',
                                   quoting=1,
                                   escapechar='\\')
                    st.success(f"✅ Successfully read with {encoding} encoding")
                    break
                except Exception as parse_error:
                    if encoding == encodings_to_try[-1]:
                        # Try with more lenient settings
                        try:
                            file_content.seek(0)
                            df = pd.read_csv(file_content,
                                           encoding=encoding,
                                           on_bad_lines='skip',
                                           sep=',',
                                           quotechar='"',
                                           escapechar=None,
                                           dtype=str)
                            st.success(f"✅ Read with lenient settings using {encoding}")
                            break
                        except:
                            continue
                    continue
            
            if df is None:
                raise Exception("Could not read file with any encoding")
            
            # Clean VAERS_ID
            if 'VAERS_ID' in df.columns:
                df['VAERS_ID'] = df['VAERS_ID'].apply(self.clean_vaers_id)
            
            # Add year column
            df['YEAR'] = '2024'  # Default year
            
            st.success(f"✅ Successfully processed symptoms file - Shape: {df.shape}")
            return df
            
        except Exception as e:
            st.error(f"❌ Error processing symptoms file: {str(e)}")
            return pd.DataFrame()
    
    def process_vax_file(self, file_content) -> pd.DataFrame:
        """Process VAERS vax file similar to R function"""
        try:
            st.info("🔄 Processing vaccine file...")
            
            # Try different encodings
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    # Reset file pointer
                    file_content.seek(0)
                    
                    # Read CSV with error handling
                    df = pd.read_csv(file_content,
                                   dtype={
                                       'VAERS_ID': str,
                                       'VAX_TYPE': str,
                                       'VAX_MANU': str,
                                       'VAX_LOT': str,
                                       'VAX_DOSE_SERIES': str,
                                       'VAX_ROUTE': str,
                                       'VAX_SITE': str,
                                       'VAX_NAME': str,
                                       'ORDER': str
                                   },
                                   na_values=['', 'NA', 'NULL'],
                                   encoding=encoding,
                                   on_bad_lines='skip',
                                   quoting=1,
                                   escapechar='\\')
                    st.success(f"✅ Successfully read with {encoding} encoding")
                    break
                except Exception as parse_error:
                    if encoding == encodings_to_try[-1]:
                        try:
                            file_content.seek(0)
                            df = pd.read_csv(file_content,
                                           encoding=encoding,
                                           on_bad_lines='skip',
                                           sep=',',
                                           quotechar='"',
                                           escapechar=None,
                                           dtype=str)
                            st.success(f"✅ Read with lenient settings using {encoding}")
                            break
                        except:
                            continue
                    continue
            
            if df is None:
                raise Exception("Could not read file with any encoding")
            
            # Clean VAERS_ID
            if 'VAERS_ID' in df.columns:
                df['VAERS_ID'] = df['VAERS_ID'].apply(self.clean_vaers_id)
            
            # Clean string columns
            string_cols = ['VAX_TYPE', 'VAX_MANU', 'VAX_LOT', 'VAX_DOSE_SERIES', 
                          'VAX_ROUTE', 'VAX_SITE', 'VAX_NAME']
            
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).apply(lambda x: x.strip() if pd.notna(x) and x != 'nan' else np.nan)
            
            # Convert ORDER to numeric if it exists
            if 'ORDER' in df.columns:
                df['ORDER'] = pd.to_numeric(df['ORDER'], errors='coerce')
            
            # Add year column
            df['YEAR'] = '2024'
            
            st.success(f"✅ Successfully processed vaccine file - Shape: {df.shape}")
            return df
            
        except Exception as e:
            st.error(f"❌ Error processing vaccine file: {str(e)}")
            return pd.DataFrame()
    
    def analyze_with_groq(self, symptoms_df: pd.DataFrame, vax_df: pd.DataFrame) -> str:
        """Use Groq LLM to analyze the VAERS data"""
        
        if self.client is None:
            return "⚠️ Groq API key not provided. Analysis skipped."
        
        # Prepare summary statistics for the prompt
        total_reports = len(symptoms_df['VAERS_ID'].unique()) if not symptoms_df.empty else 0
        total_vaccines = len(vax_df) if not vax_df.empty else 0
        
        # Get top manufacturers
        top_manufacturers = {}
        if not vax_df.empty and 'VAX_MANU' in vax_df.columns:
            top_manufacturers = vax_df['VAX_MANU'].value_counts().head(5).to_dict()
        
        # Check for symptoms of interest
        symptoms_found = []
        if not symptoms_df.empty:
            all_symptoms = []
            for col in ['SYMPTOM1', 'SYMPTOM2', 'SYMPTOM3', 'SYMPTOM4', 'SYMPTOM5']:
                if col in symptoms_df.columns:
                    all_symptoms.extend(symptoms_df[col].dropna().tolist())
            
            for symptom in self.symptoms_of_interest:
                if any(symptom.lower() in s.lower() for s in all_symptoms if isinstance(s, str)):
                    symptoms_found.append(symptom)
        
        # Create prompt for Groq
        prompt = f"""
        Analyze the following VAERS (Vaccine Adverse Event Reporting System) data summary:

        Dataset Overview:
        - Total adverse event reports: {total_reports}
        - Total vaccine administrations recorded: {total_vaccines}
        - Top vaccine manufacturers: {top_manufacturers}
        - Neurological symptoms of interest found: {symptoms_found[:10]}

        Key neurological conditions being monitored:
        {', '.join(self.symptoms_of_interest[:15])}

        Please provide:
        1. A brief analysis of the data quality and completeness
        2. Key insights about adverse event patterns
        3. Any notable findings regarding neurological symptoms
        4. Recommendations for further analysis
        5. Limitations of this data

        Keep the analysis professional, objective, and focused on public health insights.
        """
        
        try:
            with st.spinner("🤖 Analyzing data with Groq AI..."):
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    temperature=0.3,
                    max_tokens=1500
                )
                
                return chat_completion.choices[0].message.content
            
        except Exception as e:
            return f"❌ Error in Groq analysis: {str(e)}"

# Main Streamlit App
def main():
    st.title("🏥 VAERS Data Analyzer with Groq AI")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        groq_api_key = st.text_input(
            "🔑 Groq API Key", 
            type="password",
            value="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",
            help="Enter your Groq API key for AI analysis"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Upload** VAERS CSV files
        2. **Process** data automatically  
        3. **View** AI analysis results
        4. **Download** processed files
        """)
        
        st.markdown("---")
        st.markdown("### 🔍 Monitored Symptoms")
        st.markdown("""
        - Guillain-Barre syndrome
        - Miller Fisher syndrome
        - Motor neurone disease
        - Muscle weakness
        - And 22 more neurological conditions
        """)
    
    # Initialize processor
    processor = VAERSProcessor(groq_api_key)
    
    # File upload section
    st.header("📁 Upload VAERS Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🩺 Symptoms File")
        symptoms_file = st.file_uploader(
            "Upload VAERS Symptoms CSV",
            type=['csv'],
            key="symptoms",
            help="Upload the VAERSSYMPTOMS.csv file"
        )
    
    with col2:
        st.subheader("💉 Vaccine File")
        vax_file = st.file_uploader(
            "Upload VAERS Vaccine CSV", 
            type=['csv'],
            key="vax",
            help="Upload the VAERSVAX.csv file"
        )
    
    # Process files when both are uploaded
    if symptoms_file is not None and vax_file is not None:
        st.markdown("---")
        st.header("⚡ Processing Data")
        
        # Process files
        symptoms_df = processor.process_symptoms_file(symptoms_file)
        vax_df = processor.process_vax_file(vax_file)
        
        if not symptoms_df.empty or not vax_df.empty:
            
            # Display summary
            st.markdown("---")
            st.header("📊 Data Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "📋 Symptom Records", 
                    f"{len(symptoms_df):,}" if not symptoms_df.empty else "0"
                )
            
            with col2:
                st.metric(
                    "💉 Vaccine Records", 
                    f"{len(vax_df):,}" if not vax_df.empty else "0"
                )
            
            with col3:
                unique_reports = len(symptoms_df['VAERS_ID'].unique()) if not symptoms_df.empty else 0
                st.metric("🆔 Unique Reports", f"{unique_reports:,}")
            
            with col4:
                if not vax_df.empty and 'VAX_MANU' in vax_df.columns:
                    manufacturers = vax_df['VAX_MANU'].nunique()
                    st.metric("🏭 Manufacturers", f"{manufacturers}")
                else:
                    st.metric("🏭 Manufacturers", "0")
            
            # Data preview tabs
            st.markdown("---")
            st.header("👀 Data Preview")
            
            tab1, tab2 = st.tabs(["🩺 Symptoms Data", "💉 Vaccine Data"])
            
            with tab1:
                if not symptoms_df.empty:
                    st.dataframe(
                        symptoms_df.head(100),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download processed symptoms file
                    csv_symptoms = symptoms_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Processed Symptoms CSV",
                        data=csv_symptoms,
                        file_name="DomesticVAERSSYMPTOMS.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No symptoms data to display")
            
            with tab2:
                if not vax_df.empty:
                    st.dataframe(
                        vax_df.head(100),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download processed vax file
                    csv_vax = vax_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Processed Vaccine CSV",
                        data=csv_vax,
                        file_name="DomesticVAERSVAX.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No vaccine data to display")
            
            # AI Analysis section
            st.markdown("---")
            st.header("🤖 AI Analysis Results")
            
            if groq_api_key:
                analysis = processor.analyze_with_groq(symptoms_df, vax_df)
                
                st.markdown("### 📈 Groq AI Analysis Report")
                st.markdown(analysis)
                
                # Download analysis report
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=analysis,
                    file_name="VAERS_Analysis_Report.txt",
                    mime="text/plain"
                )
            else:
                st.warning("⚠️ Please enter your Groq API key in the sidebar to enable AI analysis.")
            
            # Manufacturer analysis
            if not vax_df.empty and 'VAX_MANU' in vax_df.columns:
                st.markdown("---")
                st.header("🏭 Manufacturer Analysis")
                
                manufacturer_counts = vax_df['VAX_MANU'].value_counts().head(10)
                st.bar_chart(manufacturer_counts)
                
                # Show top manufacturers table
                st.subheader("Top 10 Vaccine Manufacturers")
                manufacturer_df = pd.DataFrame({
                    'Manufacturer': manufacturer_counts.index,
                    'Report Count': manufacturer_counts.values,
                    'Percentage': (manufacturer_counts.values / len(vax_df) * 100).round(2)
                })
                st.dataframe(manufacturer_df, use_container_width=True)
        
        else:
            st.error("❌ No data could be processed from the uploaded files. Please check file formats and content.")
    
    else:
        st.info("👆 Please upload both VAERS files to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        🏥 VAERS Data Analyzer | Powered by Groq AI | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
