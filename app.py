import streamlit as st
import os
import json
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import pdfplumber




load_dotenv()

# Initialize OpenAI client
def get_openai_client():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    openai_client = OpenAI(api_key=openai_api_key)
    return openai_client
    
def read_file_content(uploaded_file):
    """Read content from uploaded file based on its type"""
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            # Save the uploaded file to a temporary location
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Read the PDF file
            with pdfplumber.open(tmp_file_path) as pdf:
                pages = [page.extract_text() for page in pdf.pages]
                return "\n".join(filter(None, pages))  # Remove any None values from pages
            
            # Clean up the temporary file
            import os
            os.unlink(tmp_file_path)
            
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None
    

def analyze_document(content):
    """Analyze document content using OpenAI API"""
    #Load system prompt
    with open("prompts/analyze_doc_prompts.txt", "r") as f:
        SYSTEM_PROMPT = f.read()
    try:
        openai_client = get_openai_client()
        response = openai_client.responses.create(
                    model="gpt-4.1",
                    instructions=SYSTEM_PROMPT,
                    input=content
)
        
        # Parse the response
        analysis = response.output_text
        return json.loads(analysis)
    except Exception as e:
        st.error(f"Error analyzing document: {str(e)}")
        return None

def display_analysis(analysis):
    """Display the analysis results in a user-friendly format"""
    if not analysis:
        return
        
    st.header("Document Analysis Results")
    
    # Document Type & Structure in a button
    with st.expander("📄 Document Type & Structure"):
        st.write(f"**Document Type:** {analysis['document_analysis'].get('document_type', 'N/A')}")
        st.write("**Sections:**")
        for section in analysis['document_analysis'].get('structure', {}).get('sections', []):
            st.write(f"- {section}")
        
        special_elements = analysis['document_analysis'].get('structure', {}).get('special_elements', [])
        if special_elements:
            st.write("**Special Elements:**")
            for element in special_elements:
                st.write(f"- {element}")
    
    # Chunking Recommendations
    with st.expander("🔍 Chunking Recommendations"):
        rec = analysis['document_analysis'].get('chunking_recommendation', {})
        st.write(f"**Recommended Approach:** {rec.get('recommended_approach', 'N/A')}")
        st.write(f"**Suggested Chunk Size:** {rec.get('suggested_chunk_size', 'N/A')} characters")
        st.write(f"**Suggested Overlap:** {rec.get('suggested_overlap', 'N/A')} characters")
        st.write("**Reasoning:**")
        st.write(rec.get('reasoning', 'No reasoning provided.'))
    
    # Special Considerations
    considerations = analysis['document_analysis'].get('special_considerations', [])
    if considerations:
        with st.expander("⚠️ Special Considerations"):
            for item in considerations:
                st.write(f"- {item}")

def main():
    st.set_page_config(
        page_title="RAG Chunking Strategy Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🔍 RAG Chunking Strategy Analyzer")
    st.write("Upload a document to get recommendations for the best chunking strategy for RAG applications.")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["txt", "pdf"],  # Limiting to supported types for now
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Convert single file to list for consistent handling
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
            
        st.success(f"✅ Successfully uploaded {len(uploaded_files)} file(s)")
        
        # Display files info
        with st.expander("📄 Uploaded Files"):
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"**File {i}:** {file.name}")
                st.write(f"  - Type: {file.type}")
                st.write(f"  - Size: {len(file.getvalue()) / 1024:.2f} KB")
        
        # Analyze button
        if st.button("Analyze Documents"):
            all_analysis = []
            
            for file in uploaded_files:
                with st.spinner(f"Analyzing {file.name}..."):
                    # Read file content
                    content = read_file_content(file)
                    
                    if not content:
                        st.error(f"Could not read file: {file.name}")
                        continue
                    
                    # Analyze document
                    analysis = analyze_document(content)
                    
                    if analysis:
                        all_analysis.append({
                            'filename': file.name,
                            'analysis': analysis
                        })
            
            # Display results for all analyzed files
            for analysis in all_analysis:
                st.divider()
                st.subheader(f"Analysis for: {analysis['filename']}")
                display_analysis(analysis['analysis'])
                
                # Show raw JSON for debugging
                with st.expander(f"View Raw Analysis for {analysis['filename']}"):
                    st.json(analysis['analysis'])

if __name__ == "__main__":
    main()