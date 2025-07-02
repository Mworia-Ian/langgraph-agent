# LangGraph Agent

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful document analysis and processing agent built with LangGraph, designed to work with Google Cloud Storage and various document formats including PDF, CSV, and text files. This project provides both a Python API and a RESTful service for document analysis.

## 🌟 Features

- **Multi-format Support**: Process PDFs, CSVs, and text documents with ease
- **GCS Integration**: Seamlessly work with files stored in Google Cloud Storage
- **AI-Powered Analysis**: Utilizes DeepSeek's language models for intelligent document processing
- **REST API**: Built-in Flask server for easy integration with other services
- **Secure**: API key authentication for protected access
- **Extensible**: Modular design for easy customization and extension

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Google Cloud SDK (for GCS access)
- DeepSeek API key

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Mworia-Ian/sokohela-langgraph-agent.git
   cd sokohela-langgraph-agent
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with your configuration:
   ```
   GCS_BUCKET_NAME=your-bucket-name
   FLASK_API_SECRET_KEY=your-secure-api-key
   DEEPSEEK_API_KEY=your-deepseek-api-key
   ```

## 🛠 Usage

### Running the API Server

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Example API Request

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@document.pdf"
```

### Using the Python API

```python
from agent import analyze_document

# Analyze a file from GCS
result = analyze_document("gs://your-bucket/document.pdf")
print(result)
```

## 📚 Documentation

### Available Tools

- **PDF Processing**: Extract and analyze text from PDF documents
- **CSV Analysis**: Process and summarize tabular data
- **Text Processing**: Work with plain text documents
- **GCS Integration**: List, read, and analyze files directly from Google Cloud Storage

### API Endpoints

- `POST /analyze` - Analyze a document
  - Requires: File upload
  - Returns: Analysis results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- Built with [LangGraph](https://langchain-ai.github.io/langgraph/)
- Powered by [DeepSeek](https://deepseek.com/) AI models
- Icons by [Shields.io](https://shields.io/)
