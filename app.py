# app.py

import streamlit as st
from openai import OpenAI
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import PyPDF2
import docx2txt
from PIL import Image
import requests
import logging
import concurrent.futures
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import html
import time
import base64

# =========================
# Configuration and Setup
# =========================

# Configure logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Initialize session state variables
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'user_data' not in st.session_state:
    st.session_state['user_data'] = {}
if 'document_summaries' not in st.session_state:
    st.session_state['document_summaries'] = []
if 'image_contexts' not in st.session_state:
    st.session_state['image_contexts'] = []
if 'case_info' not in st.session_state:
    st.session_state['case_info'] = {}
if 'report_generated' not in st.session_state:
    st.session_state['report_generated'] = False
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'uploaded_medical_bills' not in st.session_state:
    st.session_state['uploaded_medical_bills'] = None

# =========================
# Helper Functions
# =========================

def sanitize_text(text):
    """Sanitize text to prevent injection attacks."""
    return html.escape(text)

def get_location_from_zip(zip_code):
    """Fetch city and state based on ZIP code using Zippopotam.us API."""
    try:
        response = requests.get(f"http://api.zippopotam.us/us/{zip_code}")  # Adjust country code as needed
        if response.status_code == 200:
            data = response.json()
            return {
                'city': data['places'][0]['place name'],
                'state': data['places'][0]['state abbreviation']
            }
        else:
            return None
    except Exception as e:
        logging.error(f"Error fetching location for ZIP code {zip_code}: {e}")
        return None

def extract_text_from_document(file):
    """Extract text from uploaded documents."""
    content = ''
    try:
        if file.type == 'application/pdf':
            # For PDFs
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text
        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            # For DOCX
            content = docx2txt.process(file)
        elif file.type == 'text/plain':
            content = str(file.read(), 'utf-8')
        else:
            # For other types, perform OCR with GPT-40-mini
            content = ocr_document(file)
    except Exception as e:
        logging.error(f"Error extracting text from {file.name}: {e}")
    return content

def ocr_document(file):
    """Perform OCR on documents using GPT-40-mini."""
    try:
        # Read the image file as bytes
        image_bytes = file.read()
        # Encode image to base64
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        # Prepare system prompt
        system_prompt = "You are an OCR assistant that extracts text from images."

        # Prepare user message with base64 image
        user_message = f"Extract the text from the following image encoded in base64:\n\n{encoded_image}"

        # Create messages structure
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please extract the text from the following image."
                    },
                    {
                        "type": "image_base64",
                        "image_base64": {
                            "data": encoded_image
                        }
                    }
                ]
            }
        ]

        # Make OpenAI API call
        response = client.chat.completions.create(
            model="gpt-40-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={
                "type": "text"
            }
        )
        # Extract text from response
        extracted_text = response['choices'][0]['content'][0]['text']
        return extracted_text
    except Exception as e:
        logging.error(f"Error during OCR with OpenAI: {e}")
        return "Error during OCR."

def summarize_text(text):
    """Summarize text using OpenAI API."""
    try:
        system_prompt = "You are an AI assistant that summarizes documents."

        user_message = f"Please summarize the following text:\n\n{text}"

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message
                    }
                ]
            }
        ]

        response = client.chat.completions.create(
            model="gpt-40-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={
                "type": "text"
            }
        )
        summary = response['choices'][0]['content'][0]['text'].strip()
        return summary
    except Exception as e:
        logging.error(f"Error in summarize_text: {e}")
        return "An error occurred while summarizing the document."

def analyze_image(image_file, user_data):
    """Analyze image using OpenAI's gpt-40-mini Vision."""
    try:
        # Read the image file as bytes
        image_bytes = image_file.read()
        # Encode image to base64
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        # Prepare system prompt
        system_prompt = "You are a legal assistant AI with expertise in analyzing images related to personal injury and malpractice cases."

        # Prepare user message with base64 image
        user_message_text = f"""
Analyze the following image in the context of a {user_data.get('case_type', 'personal injury')} case that occurred on {user_data.get('incident_date', 'N/A')} at {user_data.get('incident_location', 'N/A')}. Extract any details, abnormalities, or evidence that are relevant to the case, especially those that support or argue against the user's claims.
"""

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message_text
                    },
                    {
                        "type": "image_base64",
                        "image_base64": {
                            "data": encoded_image
                        }
                    }
                ]
            }
        ]

        # Make OpenAI API call
        response = client.chat.completions.create(
            model="gpt-40-mini",
            messages=messages,
            temperature=0.45,
            max_tokens=1500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={
                "type": "text"
            }
        )
        assistant_message = response['choices'][0]['content'][0]['text'].strip()
        return assistant_message
    except Exception as e:
        logging.error(f"Error analyzing image {image_file.name}: {e}")
        return f"Error analyzing image: {str(e)}"

def process_documents(documents):
    """Process uploaded documents concurrently."""
    summaries = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_document, file): file for file in documents}
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                summary = future.result()
                summaries.append({'filename': file.name, 'summary': summary})
            except Exception as e:
                logging.error(f"Error processing document {file.name}: {e}")
    return summaries

def process_single_document(file):
    """Process a single document: extract text and summarize."""
    text = extract_text_from_document(file)
    if text:
        summary = summarize_text(text)
    else:
        summary = "No text extracted from the document."
    return summary

def process_images(images, user_data):
    """Process uploaded images concurrently."""
    contexts = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_image = {executor.submit(analyze_image, image, user_data): image for image in images}
        for future in concurrent.futures.as_completed(future_to_image):
            image = future_to_image[future]
            try:
                context = future.result()
                contexts.append({'filename': image.name, 'context': context})
            except Exception as e:
                logging.error(f"Error processing image {image.name}: {e}")
                contexts.append({'filename': image.name, 'context': "Error processing image."})
    return contexts

def legal_research(user_data, ser_api_key, ser_api_engine_id, ser_api_params={}):
    """
    Perform legal research using Google Custom Search API.

    Parameters:
    - user_data: Dictionary containing user information.
    - ser_api_key: API key for Google Custom Search.
    - ser_api_engine_id: Custom Search Engine ID.
    - ser_api_params: Additional parameters for the search.

    Returns:
    - Dictionary containing summaries and links.
    """
    try:
        case_type = user_data.get('case_type', 'personal injury')
        location = user_data.get('incident_location', '')

        # Extract medical terms from medical bills if available
        medical_terms = extract_medical_terms(st.session_state.get('uploaded_medical_bills', []))

        # Construct search queries
        search_queries = [
            f"{case_type} laws in {location}",
            f"Relevant statutes for {case_type} cases in {location}",
            f"{' '.join(medical_terms)} treatments in {location}"
        ]

        summaries = []
        links = []

        for query in search_queries:
            params = {
                "key": ser_api_key,
                "cx": ser_api_engine_id,
                "q": query,
                "num": 5
            }
            # Update with any additional search parameters
            params.update(ser_api_params)
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params
            )
            data = response.json()
            if 'items' in data:
                for item in data['items']:
                    summaries.append(item.get('snippet', ''))
                    links.append(item.get('link', ''))
            else:
                logging.warning(f"No items found for query: {query}")

        # Compile findings
        compiled_summary = "\n".join(summaries)
        compiled_links = "\n".join(links)

        return {
            "legal_research_summary": compiled_summary,
            "legal_research_links": compiled_links
        }
    except Exception as e:
        logging.error(f"Error during legal research: {e}")
        return {
            "legal_research_summary": "An error occurred during legal research.",
            "legal_research_links": ""
        }

def extract_medical_terms(uploaded_files):
    """Extract medical terms from uploaded medical bills or related documents."""
    terms = set()
    for file in uploaded_files:
        text = extract_text_from_document(file)
        # Simple example: extract capitalized terms (could be enhanced with NLP)
        extracted = [word for word in text.split() if word.istitle()]
        terms.update(extracted)
    return list(terms)

def case_law_retrieval(user_data, serp_api_key, serp_api_params={}):
    """
    Retrieve relevant case law using SERP API for Google Scholar.

    Parameters:
    - user_data: Dictionary containing user information.
    - serp_api_key: API key for SERP API.
    - serp_api_params: Additional parameters for the search.

    Returns:
    - Dictionary containing case law and potential payout estimate.
    """
    try:
        case_type = user_data.get('case_type', 'personal injury')
        location = user_data.get('incident_location', '')

        # Construct search query
        query = f"case precedents for {case_type} in {location}"

        # Base SERP API parameters
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": serp_api_key,
            "num": 5
        }
        # Update with any additional search parameters
        params.update(serp_api_params)

        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()

        if 'scholar_results' in data:
            results = data['scholar_results']
            cases = []
            for result in results:
                case_name = result.get('title', 'N/A')
                summary = result.get('snippet', 'No summary available.')
                outcome = extract_case_outcome(summary)
                date = extract_case_date(result.get('publication_info', {}).get('date', 'N/A'))
                cases.append({
                    "case_name": case_name,
                    "summary": summary,
                    "outcome": outcome,
                    "date": date
                })
        else:
            cases = []
            logging.warning("No scholar_results found in SERP API response.")

        # Analyze outcomes to estimate potential compensation ranges
        potential_payout = analyze_potential_payout(cases)

        return {
            "case_law": cases,
            "potential_payout_estimate": potential_payout
        }
    except Exception as e:
        logging.error(f"Error during case law retrieval: {e}")
        return {
            "case_law": [],
            "potential_payout_estimate": 0
        }

def extract_case_outcome(summary):
    """Extract outcome from case summary using simple keyword matching."""
    if "won" in summary.lower():
        return "Won"
    elif "lost" in summary.lower():
        return "Lost"
    else:
        return "Unknown"

def extract_case_date(publication_date):
    """Extract and format the date from publication info."""
    try:
        # Attempt to parse the date in various formats
        for fmt in ("%Y-%m-%d", "%B %Y", "%Y"):
            try:
                date_obj = datetime.strptime(publication_date, fmt)
                return date_obj.strftime("%B %Y")
            except ValueError:
                continue
        return publication_date
    except:
        return publication_date

def analyze_potential_payout(cases):
    """Analyze case outcomes to estimate potential compensation ranges."""
    payouts = []
    for case in cases:
        if case['outcome'] == "Won":
            # Placeholder: Assign arbitrary values or use NLP to extract amounts
            payouts.append(50000)  # Example value
    if payouts:
        average_payout = sum(payouts) / len(payouts)
        return average_payout
    else:
        return 0

def generate_case_info(user_data, document_summaries, image_contexts, ser_api_key, ser_api_engine_id, serp_api_params={}):
    """Generate case information using OpenAI API and additional AI agents."""
    # Retrieve legal research data
    legal_research_data = legal_research(user_data, ser_api_key, ser_api_engine_id, serp_api_params)
    
    # Retrieve case law data
    case_law_data = case_law_retrieval(user_data, serp_api_key, serp_api_params)
    
    # Construct prompt for OpenAI API
    prompt = f"""
Based on the following user data, document summaries, image analyses, legal research, and case law, provide a JSON containing:
- 'case_summary': A comprehensive summary of the case.
- 'best_arguments': The strongest arguments tailored to the user's case.
- 'relevant_laws': Specific laws applicable to the case.
- 'medical_literature': Relevant medical literature related to the case.
- 'case_law': Relevant case precedents and their summaries.
- 'potential_payout': The estimated monetary value the user might be awarded.
- 'likelihood_of_winning': A percentage likelihood of winning the case.

User Data:
{json.dumps(user_data, indent=2)}

Document Summaries:
{json.dumps(document_summaries, indent=2)}

Image Analyses:
{json.dumps(image_contexts, indent=2)}

Legal Research:
{json.dumps(legal_research_data, indent=2)}

Case Law:
{json.dumps(case_law_data, indent=2)}
    """

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant that compiles comprehensive case information based on provided data."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-40-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=16383,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={
                "type": "text"
            }
        )
        output_text = response['choices'][0]['content'][0]['text'].strip()
        case_info = json.loads(output_text)
    except Exception as e:
        logging.error(f"Error generating case info: {e}")
        case_info = {
            "case_summary": "An error occurred while generating the case summary.",
            "best_arguments": "",
            "relevant_laws": "",
            "medical_literature": "",
            "case_law": [],
            "potential_payout": 0,
            "likelihood_of_winning": 0
        }
    return case_info

def add_to_mailchimp(user_data, case_info):
    """Add user to MailChimp list."""
    try:
        MAILCHIMP_API_KEY = os.getenv("MAILCHIMP_API_KEY")
        MAILCHIMP_LIST_ID = os.getenv("MAILCHIMP_LIST_ID")
        MAILCHIMP_DC = os.getenv("MAILCHIMP_DC")

        url = f"https://{MAILCHIMP_DC}.api.mailchimp.com/3.0/lists/{MAILCHIMP_LIST_ID}/members"
        data = {
            "email_address": user_data.get('email', ''),
            "status": "subscribed",
            "merge_fields": {
                "FNAME": user_data.get('first_name', ''),
                "LNAME": user_data.get('last_name', ''),
                "CASEVAL": str(case_info.get('potential_payout', '')),
                "LIKELIHOOD": str(case_info.get('likelihood_of_winning', ''))
            }
        }
        auth = ('anystring', MAILCHIMP_API_KEY)
        response = requests.post(url, auth=auth, json=data)
        if response.status_code in [200, 201]:
            return True
        else:
            logging.error(f"MailChimp subscription failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logging.error(f"Exception during MailChimp subscription: {e}")
        return False

def generate_pdf_report(case_info, document_summaries, image_contexts):
    """Generate a PDF report using reportlab."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 50, "Case Analysis Report")

    # Case Summary
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 80, "Case Summary:")
    c.setFont("Helvetica", 12)
    text = c.beginText(50, height - 100)
    for line in case_info.get('case_summary', '').split('\n'):
        text.textLine(line)
    c.drawText(text)

    # Best Arguments
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 200, "Best Arguments:")
    c.setFont("Helvetica", 12)
    text = c.beginText(50, height - 220)
    for line in case_info.get('best_arguments', '').split('\n'):
        text.textLine(line)
    c.drawText(text)

    # Relevant Laws
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 320, "Relevant Laws:")
    c.setFont("Helvetica", 12)
    text = c.beginText(50, height - 340)
    for line in case_info.get('relevant_laws', '').split('\n'):
        text.textLine(line)
    c.drawText(text)
    
    # Medical Literature
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 440, "Medical Literature:")
    c.setFont("Helvetica", 12)
    text = c.beginText(50, height - 460)
    for line in case_info.get('medical_literature', '').split('\n'):
        text.textLine(line)
    c.drawText(text)

    # Case Law
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 560, "Case Law:")
    y_position = height - 580
    for case in case_info.get('case_law', []):
        if y_position < 100:
            c.showPage()
            y_position = height - 50
        c.setFont("Helvetica-Bold", 12)
        c.drawString(60, y_position, f"Case Name: {case['case_name']}")
        y_position -= 20
        c.setFont("Helvetica", 10)
        c.drawString(70, y_position, f"Summary: {case['summary']}")
        y_position -= 15
        c.drawString(70, y_position, f"Outcome: {case['outcome']}")
        y_position -= 15
        c.drawString(70, y_position, f"Date: {case['date']}")
        y_position -= 25

    # Document Summaries
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "Document Summaries:")
    y_position -= 20
    for doc in document_summaries:
        if y_position < 100:
            c.showPage()
            y_position = height - 50
        c.setFont("Helvetica-Bold", 12)
        c.drawString(60, y_position, f"{doc['filename']}:")
        y_position -= 20
        c.setFont("Helvetica", 10)
        for line in doc['summary'].split('\n'):
            c.drawString(70, y_position, line)
            y_position -= 15
        y_position -= 10

    # Image Analyses
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "Image Analyses:")
    y_position -= 20
    for img in image_contexts:
        if y_position < 100:
            c.showPage()
            y_position = height - 50
        c.setFont("Helvetica-Bold", 12)
        c.drawString(60, y_position, f"{img['filename']}:")
        y_position -= 20
        c.setFont("Helvetica", 10)
        for line in img['context'].split('\n'):
            c.drawString(70, y_position, line)
            y_position -= 15
        y_position -= 10

    # Potential Payout and Likelihood
    if y_position < 150:
        c.showPage()
        y_position = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "Potential Payout and Likelihood:")
    y_position -= 20
    c.setFont("Helvetica", 12)
    c.drawString(60, y_position, f"Estimated Potential Payout: ${case_info.get('potential_payout', 0)}")
    y_position -= 20
    c.drawString(60, y_position, f"Likelihood of Winning: {case_info.get('likelihood_of_winning', 0)}%")

    c.save()
    buffer.seek(0)
    return buffer

def generate_markdown_report(case_info, document_summaries, image_contexts):
    """Generate a markdown report."""
    report_content = f"# Case Analysis Report\n\n"

    report_content += f"## Case Summary\n{sanitize_text(case_info.get('case_summary', ''))}\n\n"

    report_content += f"## Best Arguments\n{sanitize_text(case_info.get('best_arguments', ''))}\n\n"

    report_content += f"## Relevant Laws\n{sanitize_text(case_info.get('relevant_laws', ''))}\n\n"
    
    report_content += f"## Medical Literature\n{sanitize_text(case_info.get('medical_literature', ''))}\n\n"

    report_content += f"## Case Law\n"
    for case in case_info.get('case_law', []):
        report_content += f"### {sanitize_text(case['case_name'])}\n"
        report_content += f"**Summary:** {sanitize_text(case['summary'])}\n\n"
        report_content += f"**Outcome:** {sanitize_text(case['outcome'])}\n\n"
        report_content += f"**Date:** {sanitize_text(case['date'])}\n\n"

    report_content += f"## Document Summaries\n"
    for doc in document_summaries:
        report_content += f"### {sanitize_text(doc['filename'])}\n{sanitize_text(doc['summary'])}\n\n"

    report_content += f"## Image Analyses\n"
    for img in image_contexts:
        report_content += f"### {sanitize_text(img['filename'])}\n{sanitize_text(img['context'])}\n\n"

    report_content += f"## Potential Payout and Likelihood\n"
    report_content += f"**Estimated Potential Payout:** ${case_info.get('potential_payout', 0)}\n\n"
    report_content += f"**Likelihood of Winning:** {case_info.get('likelihood_of_winning', 0)}%\n\n"

    return report_content

def initialize_chat_interface(case_info, document_summaries):
    """Initialize the chat interface with system prompt."""
    system_prompt = f"""
You are a legal assistant AI that has analyzed the following case information, documents, legal research, and case law:

Case Summary:
{sanitize_text(case_info.get('case_summary', ''))}

Best Arguments:
{sanitize_text(case_info.get('best_arguments', ''))}

Relevant Laws:
{sanitize_text(case_info.get('relevant_laws', ''))}

Medical Literature:
{sanitize_text(case_info.get('medical_literature', ''))}

Case Law:
{json.dumps(case_info.get('case_law', []), indent=2)}

Document Summaries:
{json.dumps(document_summaries, indent=2)}

Use this information to answer the user's questions accurately and helpfully.
    """
    st.session_state['chat_history'] = [
        {
            'role': 'system',
            'content': [
                {
                    'type': 'text',
                    'text': system_prompt
                }
            ]
        }
    ]

# =========================
# Main Application
# =========================

def main():
    st.set_page_config(page_title="Legal Assistant", layout="wide")
    st.title("Legal Assistant for Personal Injury/Malpractice Cases")

    # Display progress bar
    progress = st.progress((st.session_state.step - 1) / 5)

    # Step 1: Personal Information
    if st.session_state.step == 1:
        st.header("Step 1: Personal Information")
        with st.form("personal_info_form"):
            first_name = st.text_input("First Name *", value=st.session_state['user_data'].get('first_name', ''))
            last_name = st.text_input("Last Name *", value=st.session_state['user_data'].get('last_name', ''))
            email = st.text_input("Email Address *", value=st.session_state['user_data'].get('email', ''))
            phone = st.text_input("Phone Number *", value=st.session_state['user_data'].get('phone', ''))
            submitted = st.form_submit_button("Next")
            
            if submitted:
                errors = []
                if not first_name:
                    errors.append("First name is required.")
                if not last_name:
                    errors.append("Last name is required.")
                if not email or "@" not in email:
                    errors.append("A valid email address is required.")
                if not phone:
                    errors.append("Phone number is required.")
                
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    st.session_state['user_data'].update({
                        'first_name': first_name,
                        'last_name': last_name,
                        'email': email,
                        'phone': phone
                    })
                    st.session_state.step = 2
                    st.success("Personal information saved successfully!")

    # Step 2: Case Details
    elif st.session_state.step == 2:
        st.header("Step 2: Case Details")
        with st.form("case_details_form"):
            case_type = st.selectbox(
                "Type of Case *",
                ["Personal Injury", "Medical Malpractice", "Car Accident", "Other"],
                index=["Personal Injury", "Medical Malpractice", "Car Accident", "Other"].index(
                    st.session_state['user_data'].get('case_type', "Personal Injury")
                )
            )
            incident_date = st.date_input(
                "Date of Incident *",
                value=datetime.strptime(
                    st.session_state['user_data'].get('incident_date', datetime.today().strftime("%Y-%m-%d")),
                    "%Y-%m-%d"
                )
            )
            zip_code = st.text_input(
                "ZIP/Postal Code *",
                value=st.session_state['user_data'].get('zip_code', ''),
                help="Enter your ZIP or Postal Code."
            )
            # Optional: Auto-populate City and State based on ZIP code
            if zip_code:
                location = get_location_from_zip(zip_code)
                if location:
                    incident_city = location.get('city', '')
                    incident_state = location.get('state', '')
                    st.session_state['user_data']['incident_city'] = incident_city
                    st.session_state['user_data']['incident_state'] = incident_state
                    st.success(f"Location auto-filled: {incident_city}, {incident_state}")
                else:
                    incident_city = st.text_input("City *", value=st.session_state['user_data'].get('incident_city', ''))
                    incident_state = st.text_input("State/Province *", value=st.session_state['user_data'].get('incident_state', ''))
            else:
                incident_city = st.text_input("City *", value=st.session_state['user_data'].get('incident_city', ''))
                incident_state = st.text_input("State/Province *", value=st.session_state['user_data'].get('incident_state', ''))
            incident_country = st.text_input("Country *", value=st.session_state['user_data'].get('incident_country', ''))
            incident_description = st.text_area(
                "Description of Incident *",
                value=st.session_state['user_data'].get('incident_description', ''),
                help="Provide a detailed description of the incident, including what happened, when, and where."
            )
            damages_incurred = st.multiselect(
                "Damages Incurred *",
                ["Physical Injuries", "Emotional Distress", "Property Damage", "Financial Losses"],
                default=st.session_state['user_data'].get('damages_incurred', []),
                help="Select all types of damages you have incurred."
            )
            # Replace Medical Bills Amount with Estimated Range or Upload Option
            medical_bills_option = st.selectbox(
                "Medical Bills *",
                ["Provide an Estimated Range", "Upload Medical Bills"],
                index=0  # Default selection
            )
            if medical_bills_option == "Provide an Estimated Range":
                medical_bills_min = st.number_input(
                    "Minimum Medical Bills Amount ($)",
                    min_value=0,
                    value=st.session_state['user_data'].get('medical_bills_min', 0)
                )
                medical_bills_max = st.number_input(
                    "Maximum Medical Bills Amount ($)",
                    min_value=0,
                    value=st.session_state['user_data'].get('medical_bills_max', 0)
                )
            else:
                uploaded_medical_bills = st.file_uploader(
                    "Upload Medical Bills",
                    type=['pdf', 'jpg', 'png'],
                    accept_multiple_files=False
                )
                if uploaded_medical_bills:
                    st.session_state['uploaded_medical_bills'] = uploaded_medical_bills
            medical_treatment = st.text_area(
                "Medical Treatment Received",
                value=st.session_state['user_data'].get('medical_treatment', ''),
                help="Describe the medical treatments you have received as a result of the incident."
            )
            # Add Prompts or Examples for Best Argument
            best_argument = st.text_area(
                "What do you think is your best argument? *",
                value=st.session_state['user_data'].get('best_argument', ''),
                help="Describe the strongest point you believe supports your case. For example, 'The defendant was negligent in maintaining the property, leading to my injuries.'"
            )
            additional_comments = st.text_area(
                "Additional Comments",
                value=st.session_state['user_data'].get('additional_comments', ''),
                help="Provide any additional information that might be relevant to your case."
            )
            submitted = st.form_submit_button("Next")
            
            if submitted:
                errors = []
                if not incident_description:
                    errors.append("Description of incident is required.")
                if not damages_incurred:
                    errors.append("At least one type of damage must be selected.")
                if not zip_code:
                    errors.append("ZIP/Postal Code is required.")
                if not incident_city:
                    errors.append("City is required.")
                if not incident_state:
                    errors.append("State/Province is required.")
                if not incident_country:
                    errors.append("Country is required.")
                if not best_argument:
                    errors.append("Best argument is required.")
                if medical_bills_option == "Provide an Estimated Range":
                    if medical_bills_min > medical_bills_max:
                        errors.append("Minimum medical bills amount cannot exceed maximum amount.")
                else:
                    if not st.session_state.get('uploaded_medical_bills', None):
                        errors.append("Please upload your medical bills or provide an estimated range.")
                
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    user_data_update = {
                        'case_type': case_type,
                        'incident_date': incident_date.strftime("%Y-%m-%d"),
                        'zip_code': zip_code,
                        'incident_country': incident_country,
                        'incident_description': incident_description,
                        'damages_incurred': damages_incurred,
                        'medical_treatment': medical_treatment,
                        'best_argument': best_argument,
                        'additional_comments': additional_comments
                    }
                    if zip_code and location:
                        user_data_update.update({
                            'incident_city': incident_city,
                            'incident_state': incident_state,
                            'incident_location': f"{incident_city}, {incident_state}, {incident_country}",
                        })
                    if medical_bills_option == "Provide an Estimated Range":
                        user_data_update.update({
                            'medical_bills_min': medical_bills_min,
                            'medical_bills_max': medical_bills_max,
                            'medical_bills_option': medical_bills_option
                        })
                    else:
                        user_data_update.update({
                            'medical_bills_option': medical_bills_option
                        })
                    st.session_state['user_data'].update(user_data_update)
                    st.session_state.step = 3
                    st.success("Case details saved successfully!")

    # Step 3: Uploads
    elif st.session_state.step == 3:
        st.header("Step 3: Upload Documents and Images")
        with st.form("uploads_form"):
            uploaded_documents = st.file_uploader(
                "Upload Documents",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                help="Upload relevant documents such as reports, receipts, or any legal documents."
            )
            uploaded_images = st.file_uploader(
                "Upload Images",
                type=['jpg', 'jpeg', 'png', 'gif'],
                accept_multiple_files=True,
                help="Upload images related to your case, such as photos of injuries or accident scenes."
            )
            submitted = st.form_submit_button("Next")
            
            if submitted:
                if not uploaded_documents and not uploaded_images and not st.session_state.get('uploaded_medical_bills', None):
                    st.error("Please upload at least one document, image, or medical bill.")
                else:
                    st.session_state['uploaded_documents'] = uploaded_documents
                    st.session_state['uploaded_images'] = uploaded_images
                    st.session_state.step = 4
                    st.success("Files uploaded successfully!")

        # Navigation Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous"):
                st.session_state.step = 2

    # Step 4: Consent
    elif st.session_state.step == 4:
        st.header("Step 4: Consent and Acknowledgements")
        with st.form("consent_form"):
            st.markdown("**Terms and Conditions**: [Read our Terms and Conditions](https://your-terms-link.com)")
            st.markdown("**Privacy Policy**: [Read our Privacy Policy](https://your-privacy-link.com)")
            disclaimer_agreed = st.checkbox(
                "I agree to the terms and conditions.",
                value=st.session_state['user_data'].get('disclaimer_agreed', False)
            )
            opt_in_contact = st.checkbox(
                "I agree to be contacted by a lawyer regarding my case.",
                value=st.session_state['user_data'].get('opt_in_contact', False)
            )
            submitted = st.form_submit_button("Next")
            
            if submitted:
                if not disclaimer_agreed:
                    st.error("You must agree to the terms and conditions to proceed.")
                else:
                    st.session_state['user_data'].update({
                        'disclaimer_agreed': disclaimer_agreed,
                        'opt_in_contact': opt_in_contact
                    })
                    st.session_state.step = 5
                    st.success("Consent information saved successfully!")

        # Navigation Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous"):
                st.session_state.step = 3

    # Step 5: Review and Submit
    elif st.session_state.step == 5:
        st.header("Step 5: Review and Submit")
        st.subheader("Please review the information you have provided:")
        st.json(st.session_state['user_data'])
        
        if st.button("Generate AI Report"):
            if not st.session_state['report_generated']:
                with st.spinner("Processing your case..."):
                    try:
                        # Process documents and images concurrently
                        document_summaries = process_documents(st.session_state.get('uploaded_documents', []))
                        image_contexts = process_images(st.session_state.get('uploaded_images', []), st.session_state['user_data'])
                        st.session_state['document_summaries'] = document_summaries
                        st.session_state['image_contexts'] = image_contexts

                        # Retrieve SERP API credentials
                        serp_api_key = os.getenv("SERP_API_KEY")
                        serp_api_engine_id = os.getenv("SERP_API_ENGINE_ID")
                        serp_api_params = {}  # Add any additional parameters if needed

                        # Generate case info with AI agents
                        case_info = generate_case_info(
                            st.session_state['user_data'],
                            document_summaries,
                            image_contexts,
                            serp_api_key,
                            serp_api_engine_id,
                            serp_api_params
                        )
                        st.session_state['case_info'] = case_info

                        # MailChimp integration
                        if st.session_state['user_data'].get('opt_in_contact', False):
                            success = add_to_mailchimp(st.session_state['user_data'], case_info)
                            if success:
                                st.success("You have been subscribed to our mailing list.")
                            else:
                                st.error("There was an error subscribing you to our mailing list.")

                        # Initialize chat interface
                        initialize_chat_interface(case_info, document_summaries)

                        # Generate PDF report
                        pdf_buffer = generate_pdf_report(case_info, document_summaries, image_contexts)
                        st.session_state['pdf_report'] = pdf_buffer

                        # Generate Markdown report if needed
                        markdown_report = generate_markdown_report(case_info, document_summaries, image_contexts)
                        st.session_state['markdown_report'] = markdown_report

                        st.session_state['report_generated'] = True
                        st.success("Your AI report has been generated!")
                    except Exception as e:
                        logging.error(f"Error during report generation: {e}")
                        st.error("An error occurred while generating your report. Please try again later.")

        # Navigation Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous"):
                st.session_state.step = 4

        # Display results page
        if st.session_state['report_generated']:
            st.header("Case Analysis Summary")

            st.subheader("Case Overview")
            st.write(st.session_state['case_info'].get('case_summary', ''))

            st.subheader("Best Arguments")
            st.write(st.session_state['case_info'].get('best_arguments', ''))

            st.subheader("Relevant Laws")
            st.write(st.session_state['case_info'].get('relevant_laws', ''))
            
            st.subheader("Medical Literature")
            st.write(st.session_state['case_info'].get('medical_literature', ''))

            st.subheader("Case Law")
            for case in st.session_state['case_info'].get('case_law', []):
                st.write(f"**{case['case_name']}**")
                st.write(f"**Summary:** {case['summary']}")
                st.write(f"**Outcome:** {case['outcome']}")
                st.write(f"**Date:** {case['date']}\n")

            st.subheader("Document Summaries")
            for doc_summary in st.session_state['document_summaries']:
                st.write(f"**{doc_summary['filename']}**")
                st.write(doc_summary['summary'])

            st.subheader("Image Analyses")
            for image_context in st.session_state['image_contexts']:
                st.write(f"**{image_context['filename']}**")
                st.write(image_context['context'])

            st.subheader("Potential Payout and Likelihood")
            st.write(f"**Estimated Potential Payout:** ${st.session_state['case_info'].get('potential_payout', 0)}")
            st.write(f"**Likelihood of Winning:** {st.session_state['case_info'].get('likelihood_of_winning', 0)}%")

            # Downloadable PDF report
            if 'pdf_report' in st.session_state:
                st.download_button(
                    label="Download Full PDF Report",
                    data=st.session_state['pdf_report'],
                    file_name="Case_Analysis_Report.pdf",
                    mime="application/pdf"
                )

            # Downloadable Markdown report (optional)
            if 'markdown_report' in st.session_state:
                st.download_button(
                    label="Download Full Markdown Report",
                    data=st.session_state['markdown_report'],
                    file_name="Case_Analysis_Report.md",
                    mime="text/markdown"
                )

            # Chat Interface
            st.header("Chat with Your Case")
            if 'chat_history' not in st.session_state:
                initialize_chat_interface(st.session_state['case_info'], st.session_state['document_summaries'])

            # Display chat messages
            for message in st.session_state['chat_history']:
                if message['role'] == 'user':
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Assistant:** {message['content']}")

            # User input
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_input("Type your message here:", key="chat_input")
                submitted = st.form_submit_button("Send")
                
                if submitted and user_input:
                    # Append user message
                    st.session_state['chat_history'].append({'role': 'user', 'content': user_input})

                    # Prepare messages for OpenAI API
                    messages = st.session_state['chat_history']

                    # API call
                    try:
                        with st.spinner("Generating response..."):
                            response = client.chat.completions.create(
                                model="gpt-40-mini",
                                messages=messages,
                                temperature=0.45,
                                max_tokens=16383,
                                top_p=1,
                                frequency_penalty=0,
                                presence_penalty=0,
                                response_format={
                                    "type": "text"
                                }
                            )
                            assistant_message = response['choices'][0]['content'][0]['text'].strip()
                            st.session_state['chat_history'].append({'role': 'assistant', 'content': assistant_message})
                            st.success("Assistant has responded!")
                    except Exception as e:
                        logging.error(f"Error in chat interface: {e}")
                        st.error("An error occurred while communicating with the assistant. Please try again later.")

            # Reset Chat Button
            if st.button("Reset Chat"):
                initialize_chat_interface(st.session_state['case_info'], st.session_state['document_summaries'])
                st.success("Chat history has been reset.")

if __name__ == '__main__':
    main()
