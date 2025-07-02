
import os
import io
import csv 
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Optional
import fitz
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_node import ToolNode
from langchain_deepseek.chat_models import ChatDeepSeek
from google.cloud import storage


load_dotenv()
CONFIGURED_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
if not CONFIGURED_BUCKET_NAME:
    raise ValueError("GCS_BUCKET_NAME environment variable not set in .env file!")

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
summarizer_llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
analyst_llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)

AGENT_SYSTEM_PROMPT = """You are a helpful AI assistant for analyzing files in Google Cloud Storage. Your goal is to fully answer the user's request, which may require a sequence of tool calls. If you need a password, ask for it, then use the user's next message to retry the tool."""

class ReportSaverSchema(BaseModel):
    filename: str = Field(description="The local file name for saving the report. The extension (.txt, .csv, or .pdf) determines the format.")
    content: str = Field(description="The text content or analysis to be saved in the report.")

def save_report(filename: str, content: str) -> str:
    """
    Saves the provided text content to a local file in the format specified by the filename's extension (.txt, .csv, or .pdf).
    """
    try:
        file_ext = os.path.splitext(filename)[1].lower()
        
        print(f"--- [Tool] Attempting to save report to '{filename}' as {file_ext} ---")

        if file_ext == '.txt':
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
        
        elif file_ext == '.pdf':
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            formatted_content = content.replace('\n', '<br/>')
            story = [Paragraph(formatted_content, styles['Normal'])]
            doc.build(story)
            
        elif file_ext == '.csv':
            lines = content.strip().split('\n')

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for line in lines:
                    if ': ' in line:
                        writer.writerow([part.strip() for part in line.split(': ', 1)])
                    else:
                        writer.writerow([line])
        else:
            return f"Error: Unsupported file format '{file_ext}'. Please use .txt, .csv, or .pdf."

        return f"Successfully saved report to '{filename}'."
    except Exception as e:
        return f"An error occurred while saving the file '{filename}': {e}"

class LoanViabilitySchema(BaseModel):
    statement_summary: str = Field(description="A detailed summary of a financial statement, including income, expenses, and transaction patterns.")

def assess_loan_viability(statement_summary: str) -> str:
    """Analyzes a financial statement summary to provide a professional loan viability assessment."""
    print("--- [Tool] Performing final loan viability analysis... ---")
    analysis_prompt = f"You are a professional credit analyst. Based on the following summary of an M-Pesa statement, provide a professional analysis of the individual's loan viability. Structure your analysis with these sections:\n1. **Income Analysis:** Comment on the consistency and amount of incoming funds.\n2. **Spending Habits:** Analyze the expenditure patterns. Are they frivolous or responsible?\n3. **Risk Assessment:** Identify any red flags (e.g., gambling transactions, significant debt payments, erratic cash flow).\n4. **Final Recommendation:** Conclude with a clear recommendation on whether to grant a loan and why.\n\n**Statement Summary:**\n---\n{statement_summary}\n---"
    response = analyst_llm.invoke(analysis_prompt)
    return response.content

class ListFilesSchema(BaseModel):
    bucket_name: Optional[str] = Field(None, description="Optional: The GCS bucket. Defaults to the configured bucket.")

def list_files_in_bucket(bucket_name: Optional[str] = None) -> str:
    """Lists all the files in the configured Google Cloud Storage bucket."""
    bucket_to_use = bucket_name or CONFIGURED_BUCKET_NAME
    try:
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_to_use)
        file_list = [blob.name for blob in blobs]
        if not file_list: return f"The bucket '{bucket_to_use}' is empty."
        return f"Files in bucket '{bucket_to_use}':\n" + "\n".join(file_list)
    except Exception as e: return f"An error occurred while listing files: {e}"

class ReadTextFileSchema(BaseModel):
    blob_name: str = Field(description="The full name of the text-based file (e.g., 'notes.txt', 'report.md').")
    bucket_name: Optional[str] = Field(None, description="Optional: The GCS bucket name.")

def read_generic_text_file(blob_name: str, bucket_name: Optional[str] = None) -> str:
    """Reads the content of any generic text file (like .txt, .md, .json) from GCS."""
    bucket_to_use = bucket_name or CONFIGURED_BUCKET_NAME
    if blob_name.lower().endswith(('.csv', '.pdf')): return f"Error: '{blob_name}' is not a generic text file. Use a more specific tool."
    try:
        storage_client = storage.Client()
        blob = storage_client.bucket(bucket_to_use).blob(blob_name)
        print(f"--- [Tool] Reading text file gs://{bucket_to_use}/{blob_name} ---")
        return blob.download_as_text()
    except Exception as e: return f"An error occurred reading '{blob_name}': {e}"

class GCSAnalyzerSchema(BaseModel):
    blob_name: str = Field(description="The name of the CSV file (e.g., 'loans.csv') to analyze.")
    bucket_name: Optional[str] = Field(None, description="Optional: The GCS bucket name.")

def analyze_gcs_csv(blob_name: str, bucket_name: Optional[str] = None) -> str:
    """Reads, analyzes, and summarizes a CSV file. Use this for data analysis on tabular data ending in .csv."""
    bucket_to_use = bucket_name or CONFIGURED_BUCKET_NAME
    if not blob_name.lower().endswith('.csv'): return f"Error: This tool is only for CSV files. '{blob_name}' is not a CSV."
    try:
        storage_client = storage.Client()
        blob = storage_client.bucket(bucket_to_use).blob(blob_name)
        print(f"--- [Tool] Reading CSV file gs://{bucket_to_use}/{blob_name} ---")
        content_bytes = blob.download_as_bytes()
        df = pd.read_csv(io.StringIO(content_bytes.decode('utf-8')))
        analysis_prompt = f"Summarize this CSV file's content: name='{blob_name}', rows={df.shape[0]}, columns={df.shape[1]}, sample:\n{df.head(3).to_string()}"
        print("--- [Tool] Calling summarizer LLM for CSV... ---")
        return summarizer_llm.invoke(analysis_prompt).content
    except Exception as e: return f"An error analyzing '{blob_name}': {e}. It might be a malformed CSV."

class PDFReaderSchema(BaseModel):
    blob_name: str = Field(description="The full name of the PDF file (e.g., 'statement.pdf').")
    password: Optional[str] = Field(None, description="The password for the PDF file, if required.")
    bucket_name: Optional[str] = Field(None, description="Optional: The GCS bucket name.")
    
def read_and_summarize_pdf(blob_name: str, password: Optional[str] = None, bucket_name: Optional[str] = None) -> str:
    """Reads and summarizes the text content of a PDF file using Fitz. If the file is password-protected, it will return an error asking for the password."""
    bucket_to_use = bucket_name or CONFIGURED_BUCKET_NAME
    if not blob_name.lower().endswith('.pdf'): return "Error: This tool is only for PDF files."
    doc = None
    try:
        storage_client = storage.Client()
        blob = storage_client.bucket(bucket_to_use).blob(blob_name)
        print(f"--- [Tool] Reading PDF gs://{bucket_to_use}/{blob_name} ---")
        pdf_bytes = blob.download_as_bytes()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.is_encrypted:
            if not password:
                print("--- [Tool] PDF is encrypted, no password provided. Informing agent. ---")
                return "Error: This PDF is password-protected. Please ask the user for the password."
            if doc.authenticate(password) == 0:
                print("--- [Tool] Authentication failed with the provided password. Informing agent. ---")
                return "Error: The provided password was incorrect. Please ask the user for the correct password."
        print("--- [Tool] PDF opened successfully. Extracting text. ---")
        all_text = "".join(page.get_text() for page in doc)
        if not all_text.strip(): return "The PDF was opened successfully, but no text content was found."
        print("--- [Tool] Calling summarizer LLM for PDF... ---")
        summary_prompt = f"Concisely summarize the key information from the following document:\n\n{all_text[:4000]}"
        return summarizer_llm.invoke(summary_prompt).content
    except Exception as e: return f"An unexpected error occurred while processing the PDF '{blob_name}': {e}"
    finally:
        if doc: doc.close()


all_tools = [
    list_files_in_bucket,
    analyze_gcs_csv,
    read_generic_text_file,
    read_and_summarize_pdf,
    assess_loan_viability,
    save_report
]
llm_with_tools = llm.bind_tools(all_tools)
tool_executor = ToolNode(all_tools)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

def agent_node(state: AgentState):
    """Calls the LLM to decide the next step."""
    print("--- [Graph] Calling Agent (LLM) ---")
    messages_with_system_prompt = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages_with_system_prompt)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Determines the primary path after the agent thinks."""
    if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].tool_calls:
        return "call_tools"
    else:
        return "end_turn"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_executor)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"call_tools": "tools", "end_turn": END})
workflow.add_edge("tools", "agent")
app = workflow.compile()

if __name__ == "__main__":
    print(f"LangGraph GCS Agent is ready. (Watching bucket: {CONFIGURED_BUCKET_NAME})")
    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
            
        conversation_history.append(HumanMessage(content=user_input))
        final_state = app.invoke({"messages": conversation_history})
        agent_response = final_state["messages"][-1]
        conversation_history.append(agent_response)
        
        print(f"\n[Agent]: {agent_response.content}")
        print("\n" + "="*50 + "\n")