# RAG Chunking Strategy Analyzer

A Streamlit-based application that helps analyze documents and recommend optimal chunking strategies for Retrieval-Augmented Generation (RAG) applications.

## Features

- **Document Analysis**: Upload PDF or text documents for analysis
- **AI-Powered Recommendations**: Get intelligent chunking strategy suggestions
- **Multiple Chunking Strategies**: Supports various chunking approaches
- **Visual Analysis**: View document structure and special considerations
- **Easy Integration**: Simple API for document processing

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd chunking-strategy
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Upload one or more documents (PDF or TXT)

3. Click "Analyze Documents" to get chunking recommendations

4. Review the analysis including:
   - Document type and structure
   - Recommended chunking approach
   - Suggested chunk size and overlap
   - Special considerations

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── chunking_logic.py      # Core chunking strategies
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── prompts/              # System prompts for AI analysis
│   └── analyze_doc_prompts.txt
└── README.md             # This file
```

## Available Chunking Strategies

- **Recursive**: Splits text by multiple separators recursively
- **Semantic**: Groups text by semantic meaning
- **Character**: Splits by character count
- **Token**: Splits by token count
- **Markdown**: Preserves markdown structure
- **Python Code**: Specialized for Python source code
- **Table**: Handles tabular data
- **HTML**: Preserves HTML structure

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in `requirements.txt`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Streamlit
- Uses OpenAI's GPT models for document analysis
- Inspired by best practices in RAG applications
