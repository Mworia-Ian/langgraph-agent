import os
import io
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, Optional
from functools import wraps 

import fitz 

from flask import Flask, request, jsonify

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_node import ToolNode
from langchain_deepseek.chat_models import ChatDeepSeek

load_dotenv()

app = Flask(__name__)

API_SECRET_KEY = os.getenv("FLASK_API_SECRET_KEY")
if not API_SECRET_KEY:
    raise ValueError("FLASK_API_SECRET_KEY environment variable not set!")

analyst_llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
summarizer_llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

def require_api_key(f):
    """
    A decorator to protect routes with an API key.
    Checks for 'Authorization: Bearer <YOUR_KEY>' header.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "Authorization header is missing"}), 401
        
        try:
            auth_type, provided_key = auth_header.split()
            if auth_type.lower() != 'bearer':
                return jsonify({"error": "Authorization type must be Bearer"}), 401
        except ValueError:
            return jsonify({"error": "Invalid Authorization header format"}), 401

        if provided_key != API_SECRET_KEY:
            return jsonify({"error": "Invalid API Key"}), 401
        
        return f(*args, **kwargs)
    return decorated_function

class FileContentSchema(BaseModel):
    file_bytes: bytes = Field(description="The raw byte content of the file.")
    original_filename: str = Field(description="The original name of the file, including its extension.")
    password: Optional[str] = Field(None, description="The password for the file, if it is a protected PDF.")
    
def extract_content_from_file(file_bytes: bytes, original_filename: str, password: Optional[str] = None) -> str:
    """Extracts and summarizes text content from a file's raw bytes based on its extension."""
    filename_lower = original_filename.lower()
    print(f"--- [Tool] Extracting content from '{original_filename}' ---")
    try:
        if filename_lower.endswith('.pdf'):
            doc = None
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                if doc.is_encrypted:
                    if not password or doc.authenticate(password) == 0: return "Error: This PDF is password-protected and the provided password was incorrect or missing."
                all_text = "".join(page.get_text() for page in doc)
                if not all_text.strip(): return "The PDF was opened, but no text was found."
                summary_prompt = f"Concisely summarize the key data points from the following document text:\n\n{all_text[:4000]}"
                return summarizer_llm.invoke(summary_prompt).content
            finally:
                if doc: doc.close()
        elif filename_lower.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
            summary = (f"CSV File Summary for '{original_filename}':\n- Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n- Columns: {', '.join(df.columns)}\n- Data Sample (first 3 rows):\n{df.head(3).to_string()}")
            return summary
        elif filename_lower.endswith(('.txt', '.md')):
            return file_bytes.decode('utf-8')
        else:
            return f"Error: Unsupported file type for filename '{original_filename}'. Supported types are .pdf, .csv, .txt, .md."
    except Exception as e:
        return f"Error processing file '{original_filename}': {e}"

class GraphState(TypedDict):
    analysis_prompt: str
    file_bytes: bytes
    original_filename: str
    password: Optional[str]
    extracted_content: str
    final_analysis: str

def content_extractor_node(state: GraphState):
    print("--- [Graph] Running Content Extractor Node ---")
    content = extract_content_from_file(file_bytes=state["file_bytes"], original_filename=state["original_filename"], password=state["password"])
    return {"extracted_content": content}

def final_analysis_node(state: GraphState):
    print("--- [Graph] Running Final Analysis Node ---")
    if "Error:" in state["extracted_content"]: return {"final_analysis": state["extracted_content"]}
    prompt = f"You are a professional analyst. Your task is to answer the user's request based on the provided file content.\n\n**User's Request:**\n\"{state['analysis_prompt']}\"\n\n**Extracted File Content:**\n---\n{state['extracted_content']}\n---\n\nProvide a clear, direct, and professional response to the user's request."
    response = analyst_llm.invoke(prompt)
    return {"final_analysis": response.content}

workflow = StateGraph(GraphState)
workflow.add_node("extractor", content_extractor_node)
workflow.add_node("analyst", final_analysis_node)
workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "analyst")
workflow.add_edge("analyst", END)
app_graph = workflow.compile()


@app.route('/analyze-file', methods=['POST'])
@require_api_key
def analyze_file_endpoint():
    print("--- [API] Authorized request received for /analyze-file ---")
    if 'file' not in request.files: return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']

    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    password = request.form.get('password', None)
    analysis_prompt = request.form.get('analysis_prompt', "Please provide a brief summary of this file.")

    file_bytes = file.read()

    original_filename = file.filename
    initial_state = {"analysis_prompt": analysis_prompt, "file_bytes": file_bytes, "original_filename": original_filename, "password": password}
    final_state = app_graph.invoke(initial_state)
    return jsonify({"analysis_prompt": analysis_prompt, "filename": original_filename, "analysis_result": final_state.get("final_analysis", "An unknown error occurred.")})