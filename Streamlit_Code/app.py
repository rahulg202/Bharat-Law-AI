import streamlit as st
import pandas as pd
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ip_law_agent import answer_query, detect_language, classify_ip_domain

st.set_page_config(
    page_title="Multilingual IP Law Expert",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .domain-badge {
        background-color: #4B5563;
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 10px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .source-list {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .language-indicator {
        font-size: 0.9rem;
        color: #4B5563;
        font-style: italic;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

st.markdown("<h1 class='main-header'>Multilingual IP Law Expert</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.title("About")
    st.write("""
    This application provides expert answers to questions about Intellectual Property Law
    across multiple domains:
    
    - Copyright Law
    - Geographical Indications
    - Design Law
    - Patent Law
    - Trademark Law
    
    The system supports multiple languages and automatically detects and responds in the language of your query.
    """)
    
    st.subheader("How It Works")
    st.write("""
    1. Enter your question in any language
    2. Our system detects the language and IP domain
    3. Retrieves relevant legal information
    4. Provides an authoritative answer with source references
    5. Translates the response back to your original language
    """)
    
    if st.button("Clear Conversation History"):
        st.session_state.conversation_history = []
        st.success("Conversation history cleared!")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask Your Intellectual Property Law Question")
    query = st.text_area("Enter your question in any language:", height=150)
    with st.expander("See Example Questions"):
        examples = {
            "Copyright": "What are the requirements for copyright protection of software?",
            "Geographical Indications": "How can a local producer group register a new geographical indication?",
            "Design": "What is the difference between registered and unregistered design rights?",
            "Patent": "What is the standard for non-obviousness in patent applications?",
            "Trademark": "Can sounds be registered as trademarks? What are the requirements?"
        }
        col_a, col_b = st.columns(2)
        
        for i, (domain, example) in enumerate(examples.items()):
            with col_a if i % 2 == 0 else col_b:
                if st.button(f"{domain} Example", key=domain):
                    st.session_state.query = example
    if 'query' in st.session_state:
        query = st.session_state.query
        st.session_state.query = ""

with col2:
    st.subheader("Domain Prediction")
    if query:
        try:
            with st.spinner("Predicting domain..."):
                domain_prediction = classify_ip_domain(query)
            domain_colors = {
                "Copyright": "#1E40AF", 
                "GI": "#047857",  
                "Design": "#9D174D", 
                "Patent": "#B45309",  
                "Trademark": "#4338CA" 
            }
            color = domain_colors.get(domain_prediction, "#4B5563")  
            
            st.markdown(f"<div class='domain-badge' style='background-color: {color};'>Predicted Domain: {domain_prediction}</div>", 
                    unsafe_allow_html=True)
            domain_descriptions = {
                "Copyright": "Protection for original works of authorship including literary, dramatic, musical, and artistic works.",
                "GI": "Geographical Indications identify products with specific geographical origin and qualities or reputation due to that origin.",
                "Design": "Protection for the visual appearance of products, including shape, configuration, pattern, or ornament.",
                "Patent": "Exclusive rights granted for inventions that are new, useful, and non-obvious.",
                "Trademark": "Protection for brands, logos, symbols, words, or designs that distinguish products or services."
            }
            st.write(domain_descriptions.get(domain_prediction, ""))
        except Exception as e:
            st.error(f"Error predicting domain: {str(e)}")
    else:
        st.info("Enter a question to see the predicted IP law domain.")

def process_query(query_text):
    try:
        return answer_query(query_text)
    except Exception as e:
        return {
            "result": f"Error processing query: {str(e)}",
            "domain": "Error",
            "sources": []
        }

if st.button("Submit", type="primary") and query:
    st.session_state.conversation_history.append({"role": "user", "content": query})
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Detecting language...")
    progress_bar.progress(20)
    time.sleep(0.5)

    status_text.text("Classifying domain...")
    progress_bar.progress(40)
    time.sleep(0.5)
    
    status_text.text("Retrieving knowledge...")
    progress_bar.progress(60)
    time.sleep(0.5)
    
    status_text.text("Generating response...")
    progress_bar.progress(80)
    time.sleep(0.5)
    
    response = process_query(query)
    
    progress_bar.progress(100)
    status_text.text("Done!")
    time.sleep(0.3)
    
    progress_bar.empty()
    status_text.empty()
    
    st.session_state.conversation_history.append({
        "role": "assistant", 
        "content": response["result"],
        "domain": response["domain"],
        "sources": response["sources"]
    })

st.subheader("Conversation History")
if not st.session_state.conversation_history:
    st.info("No conversation yet. Start by asking a question!")
else:
    for i, message in enumerate(st.session_state.conversation_history):
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            
            domain_colors = {
                "Copyright": "#1E40AF", 
                "GI": "#047857",  
                "Design": "#9D174D",  
                "Patent": "#B45309",  
                "Trademark": "#4338CA"  
            }
            color = domain_colors.get(message["domain"], "#4B5563")  
            st.markdown(f"<div class='domain-badge' style='background-color: {color};'>Domain: {message['domain']}</div>", 
                      unsafe_allow_html=True)
            
            st.markdown(f"**Assistant:** {message['content']}")
            
            if message.get("sources") and len(message["sources"]) > 0:
                with st.expander("View Sources"):
                    sources_df = pd.DataFrame({"Source": message["sources"]})
                    st.dataframe(sources_df, hide_index=True)
            
            st.markdown("---")
