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
import base64
from pydantic import BaseModel

# =========================
# Configuration and Setup
# =========================

# Configure logging
logging.basicConfig(
    filename='app_debug.log',  # Separate log file for debugging
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Set to DEBUG to capture all levels of logs
)

logging.debug("Application startup initiated.")

# Load environment variables
load_dotenv()
logging.debug("Environment variables loaded.")

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY not found in environment variables.")
    st.error("Server configuration error: OpenAI API key is missing.")
client = OpenAI(api_key=OPENAI_API_KEY)
logging.debug("OpenAI client initialized.")

# Initialize session state variables
if 'step' not in st.session_state:
    st.session_state.step = 1
    logging.debug("Session state initialized: step set to 1.")
if 'user_data' not in st.session_state:
    st.session_state['user_data'] = {}
    logging.debug("Session state initialized: user_data created.")
if 'document_summaries' not in st.session_state:
    st.session_state['document_summaries'] = []
    logging.debug("Session state initialized: document_summaries created.")
if 'image_contexts' not in st.session_state:
    st.session_state['image_contexts'] = []
    logging.debug("Session state initialized: image_contexts created.")
if 'case_info' not in st.session_state:
    st.session_state['case_info'] = {}
    logging.debug("Session state initialized: case_info created.")
if 'report_generated' not in st.session_state:
    st.session_state['report_generated'] = False
    logging.debug("Session state initialized: report_generated set to False.")
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
    logging.debug("Session state initialized: chat_history created.")
if 'uploaded_medical_bills' not in st.session_state:
    st.session_state['uploaded_medical_bills'] = None
    logging.debug("Session state initialized: uploaded_medical_bills set to None.")

# =========================
# Helper Functions
# =========================

def sanitize_text(text):
    """Sanitize text to prevent injection attacks."""
    sanitized = html.escape(text)
    logging.debug(f"Sanitized text: {sanitized}")
    return sanitized

def get_location_from_zip(zip_code):
    """Fetch city and state based on ZIP code using Zippopotam.us API."""
    try:
        logging.debug(f"Fetching location for ZIP code: {zip_code}")
        response = requests.get(f"http://api.zippopotam.us/us/{zip_code}")  # Adjust country code as needed
        if response.status_code == 200:
            data = response.json()
            location = {
                'city': data['places'][0]['place name'],
                'state': data['places'][0]['state abbreviation']
            }
            logging.debug(f"Location fetched: {location}")
            return location
        else:
            logging.warning(f"Failed to fetch location for ZIP code {zip_code}: Status Code {response.status_code}")
            st.write(f"Warning: Unable to auto-populate location for ZIP code {zip_code}. Please enter manually.")
            return None
    except Exception as e:
        logging.error(f"Exception occurred while fetching location for ZIP code {zip_code}: {e}")
        st.write("Error: An exception occurred while fetching location. Please enter manually.")
        return None

def extract_text_from_document(file):
    """Extract text from uploaded documents."""
    content = ''
    try:
        logging.debug(f"Extracting text from document: {file.name}, Type: {file.type}")
        if file.type == 'application/pdf':
            # For PDFs
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:
                    content += text
                logging.debug(f"Extracted text from page {page_num} of {file.name}")
        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            # For DOCX
            content = docx2txt.process(file)
            logging.debug(f"Extracted text from DOCX: {file.name}")
        elif file.type == 'text/plain':
            content = str(file.read(), 'utf-8')
            logging.debug(f"Extracted text from TXT: {file.name}")
        else:
            # For other types, perform OCR with OpenAI's GPT-4
            content = ocr_document(file)
    except Exception as e:
        logging.error(f"Error extracting text from {file.name}: {e}")
        content = f"Error extracting text from {file.name}: {e}"
    return content

def ocr_document(file):
    """Perform OCR on documents using OpenAI's GPT-4."""
    try:
        logging.debug(f"Performing OCR on document: {file.name}")
        # Read the image file as bytes
        image_bytes = file.read()
        # Encode image to base64
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        # Prepare system prompt
        system_prompt = "You are an OCR assistant that extracts text from images."
        logging.debug("System prompt for OCR prepared.")
        # Prepare user message with base64 image
        user_message = f"Extract the text from the following image encoded in base64:\n\n{encoded_image}"
        logging.debug("User message for OCR prepared.")
        # Create messages structure
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
        # Make OpenAI API call
        response = client.chat.completions.create(
            model="gpt-4",  # Corrected model name
            messages=messages,
            temperature=0.0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        logging.debug(f"OCR API response: {response}")
        # Extract text from response
        extracted_text = response['choices'][0]['message']['content']
        logging.debug(f"Extracted text from OCR: {extracted_text[:200]}...")  # Log first 200 chars
        return extracted_text
    except Exception as e:
        logging.error(f"Error during OCR with OpenAI for {file.name}: {e}")
        return f"Error during OCR: {e}"

def summarize_text(text):
    """Summarize text using OpenAI API."""
    try:
        logging.debug("Summarizing text using OpenAI API.")
        system_prompt = "You are an AI assistant that summarizes documents."
        logging.debug("System prompt for summarization prepared.")
        user_message = f"Please summarize the following text:\n\n{text}"
        logging.debug("User message for summarization prepared.")
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
        response = client.chat.completions.create(
            model="gpt-4",  # Corrected model name
            messages=messages,
            temperature=0.5,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        logging.debug(f"Summarization API response: {response}")
        summary = response['choices'][0]['message']['content'].strip()
        logging.debug(f"Generated summary: {summary[:200]}...")  # Log first 200 chars
        return summary
    except Exception as e:
        logging.error(f"Error in summarize_text: {e}")
        return f"An error occurred while summarizing the document: {e}"

def analyze_image(image_file, user_data):
    """Analyze image using OpenAI's GPT-4 Vision."""
    try:
        logging.debug(f"Analyzing image: {image_file.name}")
        # Read the image file as bytes
        image_bytes = image_file.read()
        # Encode image to base64
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        # Prepare system prompt
        system_prompt = "You are a legal assistant AI with expertise in analyzing images related to personal injury and malpractice cases."
        logging.debug("System prompt for image analysis prepared.")
        # Prepare user message with base64 image
        user_message_text = f"""
Analyze the following image in the context of a {user_data.get('case_type', 'personal injury')} case that occurred on {user_data.get('incident_date', 'N/A')} at {user_data.get('incident_location', 'N/A')}. Extract any details, abnormalities, or evidence that are relevant to the case, especially those that support or argue against the user's claims.
"""
        logging.debug("User message for image analysis prepared.")
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message_text
            },
            {
                "role": "user",
                "content": f"![Image](data:image/png;base64,{encoded_image})"
            }
        ]
        # Make OpenAI API call
        response = client.chat.completions.create(
            model="gpt-4",  # Corrected model name
            messages=messages,
            temperature=0.45,
            max_tokens=1500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        logging.debug(f"Image analysis API response: {response}")
        assistant_message = response['choices'][0]['message']['content'].strip()
        logging.debug(f"Generated image analysis: {assistant_message[:200]}...")  # Log first 200 chars
        return assistant_message
    except Exception as e:
        logging.error(f"Error analyzing image {image_file.name}: {e}")
        return f"Error analyzing image: {e}"

def process_documents(documents):
    """Process uploaded documents concurrently."""
    summaries = []
    if not documents:
        logging.debug("No documents to process.")
        return summaries
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_document, file): file for file in documents}
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                summary = future.result()
                summaries.append({'filename': file.name, 'summary': summary})
                logging.debug(f"Processed document: {file.name}")
            except Exception as e:
                logging.error(f"Error processing document {file.name}: {e}")
    return summaries

def process_single_document(file):
    """Process a single document: extract text and summarize."""
    logging.debug(f"Processing single document: {file.name}")
    text = extract_text_from_document(file)
    if "Error" in text:
        summary = text  # If extraction failed, return the error message
        logging.warning(f"Extraction failed for {file.name}: {text}")
    else:
        summary = summarize_text(text)
    logging.debug(f"Summary for {file.name}: {summary[:200]}...")  # Log first 200 chars
    return summary

def process_images(images, user_data):
    """Process uploaded images concurrently."""
    contexts = []
    if not images:
        logging.debug("No images to process.")
        return contexts
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_image = {executor.submit(analyze_image, image, user_data): image for image in images}
        for future in concurrent.futures.as_completed(future_to_image):
            image = future_to_image[future]
            try:
                context = future.result()
                contexts.append({'filename': image.name, 'context': context})
                logging.debug(f"Processed image: {image.name}")
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
        logging.debug("Starting legal research using Google Custom Search API.")
        case_type = user_data.get('case_type', 'personal injury')
        location = user_data.get('incident_location', '')

        # Extract medical terms from medical bills if available
        medical_terms = extract_medical_terms(st.session_state.get('uploaded_medical_bills', []))
        logging.debug(f"Extracted medical terms: {medical_terms}")

        # Construct search queries
        search_queries = [
            f"{case_type} laws in {location}",
            f"Relevant statutes for {case_type} cases in {location}",
            f"{' '.join(medical_terms)} treatments in {location}"
        ]
        logging.debug(f"Search queries: {search_queries}")

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
            logging.debug(f"Performing search with query: {query}")
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params
            )
            logging.debug(f"Google Custom Search API response: {response.status_code}")
            data = response.json()
            if 'items' in data:
                for item in data['items']:
                    summaries.append(item.get('snippet', ''))
                    links.append(item.get('link', ''))
            else:
                logging.warning(f"No items found for query: {query}")
                st.write(f"Warning: No search results found for '{query}'.")

        # Compile findings
        compiled_summary = "\n".join(summaries)
        compiled_links = "\n".join(links)

        logging.debug("Legal research completed successfully.")
        return {
            "legal_research_summary": compiled_summary,
            "legal_research_links": compiled_links
        }
    except Exception as e:
        logging.error(f"Error during legal research: {e}")
        st.write("Error: An exception occurred during legal research.")
        return {
            "legal_research_summary": "An error occurred during legal research.",
            "legal_research_links": ""
        }

def extract_medical_terms(uploaded_files):
    """Extract medical terms from uploaded medical bills or related documents."""
    terms = set()
    for file in uploaded_files:
        logging.debug(f"Extracting medical terms from file: {file.name}")
        text = extract_text_from_document(file)
        # Simple example: extract capitalized terms (could be enhanced with NLP)
        extracted = [word for word in text.split() if word.istitle()]
        terms.update(extracted)
    logging.debug(f"Extracted medical terms: {terms}")
    return list(terms)

def case_law_retrieval(user_data, ser_api_key, serp_api_params={}):
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
        logging.debug("Starting case law retrieval using SERP API.")
        case_type = user_data.get('case_type', 'personal injury')
        location = user_data.get('incident_location', '')

        # Construct search query
        query = f"case precedents for {case_type} in {location}"
        logging.debug(f"Case law search query: {query}")

        # Base SERP API parameters
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": ser_api_key,
            "num": 5
        }
        # Update with any additional search parameters
        params.update(serp_api_params)

        logging.debug(f"Performing SERP API search with params: {params}")
        response = requests.get("https://serpapi.com/search", params=params)
        logging.debug(f"SERP API response: {response.status_code}")
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
                logging.debug(f"Retrieved case: {case_name}, Outcome: {outcome}, Date: {date}")
        else:
            cases = []
            logging.warning("No scholar_results found in SERP API response.")
            st.write("Warning: No case law results found.")
            return {
                "case_law": cases,
                "potential_payout_estimate": 0
            }

        # Analyze outcomes to estimate potential compensation ranges
        potential_payout = analyze_potential_payout(cases)

        logging.debug("Case law retrieval completed successfully.")
        return {
            "case_law": cases,
            "potential_payout_estimate": potential_payout
        }
    except Exception as e:
        logging.error(f"Error during case law retrieval: {e}")
        st.write("Error: An exception occurred during case law retrieval.")
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
        logging.debug(f"Extracting date from publication info: {publication_date}")
        # Attempt to parse the date in various formats
        for fmt in ("%Y-%m-%d", "%B %Y", "%Y"):
            try:
                date_obj = datetime.strptime(publication_date, fmt)
                formatted_date = date_obj.strftime("%B %Y")
                logging.debug(f"Parsed date: {formatted_date}")
                return formatted_date
            except ValueError:
                continue
        logging.debug(f"Unable to parse date: {publication_date}")
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
        logging.debug(f"Estimated potential payout based on cases: {average_payout}")
        return average_payout
    else:
        logging.debug("No successful cases found to estimate potential payout.")
        return 0

def generate_case_info(user_data, document_summaries, image_contexts, ser_api_key, ser_api_engine_id, serp_api_params={}):
    """Generate case information using OpenAI API and additional AI agents."""
    try:
        logging.debug("Generating case information using OpenAI API.")
        # Retrieve legal research data
        logging.debug("Generating legal research data.")
        legal_research_data = legal_research(user_data, ser_api_key, ser_api_engine_id, serp_api_params)

        # Retrieve case law data
        logging.debug("Generating case law data.")
        case_law_data = case_law_retrieval(user_data, ser_api_key, serp_api_params)

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

        logging.debug("Prompt for case information generated.")
        logging.debug(f"Prompt Content: {prompt[:500]}...")  # Log first 500 chars

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that compiles comprehensive case information based on provided data."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Make OpenAI API call
        response = client.chat.completions.create(
            model="gpt-4",  # Corrected model name
            messages=messages,
            temperature=0.5,
            max_tokens=16383,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        logging.debug(f"OpenAI API response for case info: {response}")
        output_text = response['choices'][0]['message']['content']
        logging.debug(f"Case info JSON received: {output_text[:500]}...")  # Log first 500 chars
        case_info = json.loads(output_text)
        logging.debug("Case information generated successfully.")
    except json.JSONDecodeError as je:
        logging.error(f"JSON decode error: {je}")
        st.write("Error: Received invalid JSON from OpenAI API.")
        case_info = {
            "case_summary": "An error occurred while generating the case summary.",
            "best_arguments": "",
            "relevant_laws": "",
            "medical_literature": "",
            "case_law": [],
            "potential_payout": 0,
            "likelihood_of_winning": 0
        }
    except Exception as e:
        logging.error(f"Error generating case info: {e}")
        st.write("Error: An exception occurred while generating case information.")
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
        logging.debug("Attempting to subscribe user to MailChimp.")
        MAILCHIMP_API_KEY = os.getenv("MAILCHIMP_API_KEY")
        MAILCHIMP_LIST_ID = os.getenv("MAILCHIMP_LIST_ID")
        MAILCHIMP_DC = os.getenv("MAILCHIMP_DC")

        if not all([MAILCHIMP_API_KEY, MAILCHIMP_LIST_ID, MAILCHIMP_DC]):
            logging.error("MailChimp API credentials are not fully set.")
            st.write("Error: MailChimp configuration is incomplete.")
            return False

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
        logging.debug(f"MailChimp subscription payload: {json.dumps(data)}")
        response = requests.post(url, auth=auth, json=data)
        logging.debug(f"MailChimp API response: {response.status_code} - {response.text}")
        if response.status_code in [200, 201]:
            logging.debug("User subscribed to MailChimp successfully.")
            return True
        else:
            logging.error(f"MailChimp subscription failed: {response.status_code} - {response.text}")
            st.write("Error: Failed to subscribe to the mailing list.")
            return False
    except Exception as e:
        logging.error(f"Exception during MailChimp subscription: {e}")
        st.write("Error: An exception occurred while subscribing to the mailing list.")
        return False

def generate_pdf_report(case_info, document_summaries, image_contexts):
    """Generate a PDF report using reportlab."""
    logging.debug("Generating PDF report.")
    buffer = io.BytesIO()
    try:
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(width / 2, height - 50, "Case Analysis Report")
        logging.debug("PDF Title added.")

        # Case Summary
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 80, "Case Summary:")
        c.setFont("Helvetica", 12)
        text = c.beginText(50, height - 100)
        for line in case_info.get('case_summary', '').split('\n'):
            text.textLine(line)
        c.drawText(text)
        logging.debug("Case Summary added to PDF.")

        # Best Arguments
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 200, "Best Arguments:")
        c.setFont("Helvetica", 12)
        text = c.beginText(50, height - 220)
        for line in case_info.get('best_arguments', '').split('\n'):
            text.textLine(line)
        c.drawText(text)
        logging.debug("Best Arguments added to PDF.")

        # Relevant Laws
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 320, "Relevant Laws:")
        c.setFont("Helvetica", 12)
        text = c.beginText(50, height - 340)
        for line in case_info.get('relevant_laws', '').split('\n'):
            text.textLine(line)
        c.drawText(text)
        logging.debug("Relevant Laws added to PDF.")
        
        # Medical Literature
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 440, "Medical Literature:")
        c.setFont("Helvetica", 12)
        text = c.beginText(50, height - 460)
        for line in case_info.get('medical_literature', '').split('\n'):
            text.textLine(line)
        c.drawText(text)
        logging.debug("Medical Literature added to PDF.")

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
            logging.debug(f"Added case law entry to PDF: {case['case_name']}")

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
            logging.debug(f"Added document summary to PDF: {doc['filename']}")

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
            logging.debug(f"Added image analysis to PDF: {img['filename']}")

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
        logging.debug("Potential Payout and Likelihood added to PDF.")

        c.save()
        buffer.seek(0)
        logging.debug("PDF report generated successfully.")
        return buffer
    except Exception as e:
        logging.error(f"Error generating PDF report: {e}")
        st.write("Error: An exception occurred while generating the PDF report.")
        return io.BytesIO()  # Return empty buffer in case of error

def generate_markdown_report(case_info, document_summaries, image_contexts):
    """Generate a markdown report."""
    try:
        logging.debug("Generating Markdown report.")
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

        logging.debug("Markdown report generated successfully.")
        return report_content
    except Exception as e:
        logging.error(f"Error generating Markdown report: {e}")
        st.write("Error: An exception occurred while generating the Markdown report.")
        return "# Case Analysis Report\n\nAn error occurred while generating the report."

def initialize_chat_interface(case_info, document_summaries):
    """Initialize the chat interface with system prompt."""
    try:
        logging.debug("Initializing chat interface.")
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
                'content': system_prompt
            }
        ]
        logging.debug("Chat interface initialized with system prompt.")
    except Exception as e:
        logging.error(f"Error initializing chat interface: {e}")

# =========================
# Main Application
# =========================

def main():
    logging.debug("Starting main application.")
    st.set_page_config(page_title="Legal Assistant", layout="wide")
    st.title("Legal Assistant for Personal Injury/Malpractice Cases")

    # Display progress bar
    progress = st.progress((st.session_state.step - 1) / 5)
    logging.debug(f"Current step: {st.session_state.step}")

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
                logging.debug("Personal Information form submitted.")
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
                        logging.warning(f"Personal Information error: {error}")
                else:
                    st.session_state['user_data'].update({
                        'first_name': first_name,
                        'last_name': last_name,
                        'email': email,
                        'phone': phone
                    })
                    st.session_state.step = 2
                    st.success("Personal information saved successfully!")
                    logging.info("Personal information saved and moved to Step 2.")

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
            # Auto-Populate City and State based on ZIP code
            if zip_code:
                location = get_location_from_zip(zip_code)
                if location:
                    incident_city = location.get('city', '')
                    incident_state = location.get('state', '')
                    st.session_state['user_data']['incident_city'] = incident_city
                    st.session_state['user_data']['incident_state'] = incident_state
                    st.session_state['user_data']['incident_location'] = f"{incident_city}, {incident_state}, {st.session_state['user_data'].get('incident_country', '')}"
                    st.success(f"Location auto-filled: {incident_city}, {incident_state}")
                    # Display auto-filled City and State as disabled inputs
                    st.text_input("City *", value=incident_city, disabled=True)
                    st.text_input("State/Province *", value=incident_state, disabled=True)
                else:
                    # Allow manual entry if auto-population fails
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
                [
                    "Physical Injuries", 
                    "Emotional Distress", 
                    "Property Damage", 
                    "Financial Losses",
                    "Lost Wages",
                    "Medical Expenses",
                    "Pain and Suffering",
                    "Other"
                ],
                default=st.session_state['user_data'].get('damages_incurred', []),
                help="Select all types of damages you have incurred."
            )
            # Handle 'Other' option
            if "Other" in damages_incurred:
                other_damages = st.text_input("Please specify other damages:")
            else:
                other_damages = st.text_input("Please specify other damages:", disabled=True)
            
            # Unified Medical Bills Section
            st.subheader("Medical Bills")
            uploaded_medical_bills = st.file_uploader(
                "Upload Medical Bills (PDF, JPG, PNG) *",
                type=['pdf', 'jpg', 'png'],
                accept_multiple_files=False,
                help="Upload your medical bills. If you do not have the documents, please provide estimated amounts below."
            )
            if uploaded_medical_bills:
                st.session_state['uploaded_medical_bills'] = uploaded_medical_bills
                st.success("Medical bills uploaded successfully!")
                logging.debug(f"Uploaded medical bills: {uploaded_medical_bills.name}")
            else:
                medical_bills_notes = st.text_area(
                    "If you do not have medical bills to upload, please provide estimated amounts or notes:",
                    value=st.session_state['user_data'].get('medical_bills_notes', ''),
                    help="Provide an estimated range or any notes regarding your medical bills."
                )
                logging.debug("No medical bills uploaded; user provided notes.")

            medical_treatment = st.text_area(
                "Medical Treatment Received",
                value=st.session_state['user_data'].get('medical_treatment', ''),
                help="Describe the medical treatments you have received as a result of the incident."
            )
            # Enhanced Best Argument Input with Guidance
            best_argument = st.text_area(
                "What do you think is your best argument? *",
                value=st.session_state['user_data'].get('best_argument', ''),
                help="Provide your strongest argument supporting your case.\n\nFor example: 'The defendant failed to maintain the property, directly resulting in my injuries.'"
            )
            # Structured Additional Comments
            st.subheader("Additional Information")
            additional_comments = st.text_area(
                "Please provide any additional relevant information:",
                value=st.session_state['user_data'].get('additional_comments', ''),
                help="Please answer the following prompts:\n- Have there been any witnesses?\n- Are there any prior incidents related to this case?\n- Any other details you find pertinent."
            )
            submitted = st.form_submit_button("Next")
            
            if submitted:
                logging.debug("Case Details form submitted.")
                errors = []
                if not incident_description:
                    errors.append("Description of incident is required.")
                if not damages_incurred:
                    errors.append("At least one type of damage must be selected.")
                if not zip_code:
                    errors.append("ZIP/Postal Code is required.")
                if not incident_city and (zip_code and not location):
                    errors.append("City is required.")
                if not incident_state and (zip_code and not location):
                    errors.append("State/Province is required.")
                if not incident_country:
                    errors.append("Country is required.")
                if not best_argument:
                    errors.append("Best argument is required.")
                if not uploaded_medical_bills and not st.session_state['user_data'].get('medical_bills_notes', ''):
                    errors.append("Please upload your medical bills or provide estimated amounts/notes.")
                
                if "Other" in damages_incurred and not other_damages:
                    errors.append("Please specify your other damages.")
                
                if errors:
                    for error in errors:
                        st.error(error)
                        logging.warning(f"Case Details error: {error}")
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
                    if "Other" in damages_incurred:
                        user_data_update['other_damages'] = other_damages
                    if zip_code and location:
                        user_data_update.update({
                            'incident_city': incident_city,
                            'incident_state': incident_state,
                            'incident_location': f"{incident_city}, {incident_state}, {incident_country}",
                        })
                    if uploaded_medical_bills:
                        user_data_update.update({
                            'medical_bills_option': "Uploaded",
                            'medical_bills_notes': ""
                        })
                    else:
                        user_data_update.update({
                            'medical_bills_option': "Estimated",
                            'medical_bills_notes': medical_bills_notes
                        })
                    st.session_state['user_data'].update(user_data_update)
                    st.session_state.step = 3
                    st.success("Case details saved successfully!")
                    logging.info("Case details saved and moved to Step 3.")

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
                logging.debug("Uploads form submitted.")
                if not uploaded_documents and not uploaded_images and not st.session_state.get('uploaded_medical_bills', None):
                    st.error("Please upload at least one document, image, or medical bill.")
                    logging.warning("No uploads provided.")
                else:
                    st.session_state['uploaded_documents'] = uploaded_documents
                    st.session_state['uploaded_images'] = uploaded_images
                    st.session_state.step = 4
                    st.success("Files uploaded successfully!")
                    logging.info("Files uploaded and moved to Step 4.")

        # Navigation Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous"):
                st.session_state.step = 2
                logging.debug("Navigated back to Step 2.")
                # Force rerun to reflect state change immediately
                st.experimental_rerun()

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
                logging.debug("Consent form submitted.")
                if not disclaimer_agreed:
                    st.error("You must agree to the terms and conditions to proceed.")
                    logging.warning("User did not agree to terms and conditions.")
                else:
                    st.session_state['user_data'].update({
                        'disclaimer_agreed': disclaimer_agreed,
                        'opt_in_contact': opt_in_contact
                    })
                    st.session_state.step = 5
                    st.success("Consent information saved successfully!")
                    logging.info("Consent information saved and moved to Step 5.")

        # Navigation Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous"):
                st.session_state.step = 3
                logging.debug("Navigated back to Step 3.")
                st.experimental_rerun()

    # Step 5: Review and Submit
    elif st.session_state.step == 5:
        st.header("Step 5: Review and Submit")
        st.subheader("Please review the information you have provided:")
        st.json(st.session_state['user_data'])
        
        if st.button("Generate AI Report"):
            if not st.session_state['report_generated']:
                with st.spinner("Processing your case..."):
                    try:
                        logging.debug("Starting report generation process.")
                        # Process documents and images concurrently
                        document_summaries = process_documents(st.session_state.get('uploaded_documents', []))
                        image_contexts = process_images(st.session_state.get('uploaded_images', []), st.session_state['user_data'])
                        st.session_state['document_summaries'] = document_summaries
                        st.session_state['image_contexts'] = image_contexts
                        logging.debug(f"Document summaries: {document_summaries}")
                        logging.debug(f"Image contexts: {image_contexts}")

                        # Retrieve SERP API credentials
                        serp_api_key = os.getenv("SERP_API_KEY")
                        serp_api_engine_id = os.getenv("SERP_API_ENGINE_ID")
                        serp_api_params = {}  # Add any additional parameters if needed

                        if not serp_api_key or not serp_api_engine_id:
                            logging.error("SERP API credentials are not fully set.")
                            st.error("Legal research cannot be performed due to missing SERP API credentials.")
                            return

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
                        logging.debug(f"Generated case info: {case_info}")

                        # MailChimp integration
                        if st.session_state['user_data'].get('opt_in_contact', False):
                            success = add_to_mailchimp(st.session_state['user_data'], case_info)
                            if success:
                                st.success("You have been subscribed to our mailing list.")
                                logging.info("User subscribed to MailChimp.")
                            else:
                                st.error("There was an error subscribing you to our mailing list.")
                                logging.error("MailChimp subscription failed.")

                        # Initialize chat interface
                        initialize_chat_interface(case_info, document_summaries)

                        # Generate PDF report
                        pdf_buffer = generate_pdf_report(case_info, document_summaries, image_contexts)
                        st.session_state['pdf_report'] = pdf_buffer
                        logging.debug("PDF report generated and stored in session state.")

                        # Generate Markdown report if needed
                        markdown_report = generate_markdown_report(case_info, document_summaries, image_contexts)
                        st.session_state['markdown_report'] = markdown_report
                        logging.debug("Markdown report generated and stored in session state.")

                        st.session_state['report_generated'] = True
                        st.success("Your AI report has been generated!")
                        logging.info("AI report generation completed successfully.")
                    except Exception as e:
                        logging.error(f"Error during report generation: {e}")
                        st.error("An error occurred while generating your report. Please try again later.")

        # Navigation Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous"):
                st.session_state.step = 4
                logging.debug("Navigated back to Step 4.")
                st.experimental_rerun()

        # Display results page
        if st.session_state['report_generated']:
            st.header("Case Analysis Summary")

            st.subheader("Case Overview")
            st.write(st.session_state['case_info'].get('case_summary', ''))
            logging.debug("Displayed Case Overview.")

            st.subheader("Best Arguments")
            st.write(st.session_state['case_info'].get('best_arguments', ''))
            logging.debug("Displayed Best Arguments.")

            st.subheader("Relevant Laws")
            st.write(st.session_state['case_info'].get('relevant_laws', ''))
            logging.debug("Displayed Relevant Laws.")
            
            st.subheader("Medical Literature")
            st.write(st.session_state['case_info'].get('medical_literature', ''))
            logging.debug("Displayed Medical Literature.")

            st.subheader("Case Law")
            for case in st.session_state['case_info'].get('case_law', []):
                st.write(f"**{case['case_name']}**")
                st.write(f"**Summary:** {case['summary']}")
                st.write(f"**Outcome:** {case['outcome']}")
                st.write(f"**Date:** {case['date']}\n")
                logging.debug(f"Displayed case law: {case['case_name']}")

            st.subheader("Document Summaries")
            for doc_summary in st.session_state['document_summaries']:
                st.write(f"**{doc_summary['filename']}**")
                st.write(doc_summary['summary'])
                logging.debug(f"Displayed document summary: {doc_summary['filename']}")

            st.subheader("Image Analyses")
            for image_context in st.session_state['image_contexts']:
                st.write(f"**{image_context['filename']}**")
                st.write(image_context['context'])
                logging.debug(f"Displayed image analysis: {image_context['filename']}")

            st.subheader("Potential Payout and Likelihood")
            st.write(f"**Estimated Potential Payout:** ${st.session_state['case_info'].get('potential_payout', 0)}")
            st.write(f"**Likelihood of Winning:** {st.session_state['case_info'].get('likelihood_of_winning', 0)}%")
            logging.debug("Displayed Potential Payout and Likelihood.")

            # Downloadable PDF report
            if 'pdf_report' in st.session_state:
                st.download_button(
                    label="Download Full PDF Report",
                    data=st.session_state['pdf_report'],
                    file_name="Case_Analysis_Report.pdf",
                    mime="application/pdf"
                )
                logging.debug("PDF download button displayed.")

            # Downloadable Markdown report (optional)
            if 'markdown_report' in st.session_state:
                st.download_button(
                    label="Download Full Markdown Report",
                    data=st.session_state['markdown_report'],
                    file_name="Case_Analysis_Report.md",
                    mime="text/markdown"
                )
                logging.debug("Markdown download button displayed.")

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
                    logging.debug(f"Chat form submitted with user input: {user_input}")
                    # Append user message
                    st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
                    logging.debug("User message appended to chat history.")

                    # Prepare messages for OpenAI API
                    messages = st.session_state['chat_history']

                    # API call
                    try:
                        with st.spinner("Generating response..."):
                            logging.debug("Sending chat messages to OpenAI API.")
                            response = client.chat.completions.create(
                                model="gpt-4",  # Corrected model name
                                messages=messages,
                                temperature=0.45,
                                max_tokens=1500,
                                top_p=1,
                                frequency_penalty=0,
                                presence_penalty=0
                            )
                            logging.debug(f"Chat API response: {response}")
                            assistant_message = response['choices'][0]['message']['content'].strip()
                            st.session_state['chat_history'].append({'role': 'assistant', 'content': assistant_message})
                            st.success("Assistant has responded!")
                            logging.debug(f"Assistant response appended: {assistant_message[:200]}...")
                    except Exception as e:
                        logging.error(f"Error in chat interface: {e}")
                        st.error("An error occurred while communicating with the assistant. Please try again later.")

            # Reset Chat Button
            if st.button("Reset Chat"):
                initialize_chat_interface(st.session_state['case_info'], st.session_state['document_summaries'])
                st.success("Chat history has been reset.")
                logging.info("Chat history has been reset.")

if __name__ == '__main__':
    main()
