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
import concurrent.futures
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import html
import base64

# =========================
# Configuration and Setup
# =========================

# Load environment variables
load_dotenv()
st.write("🔍 **Environment Variables Loaded**")

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ **Error:** OpenAI API key is missing. Please set `OPENAI_API_KEY` in your environment variables.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)
st.write("✅ **OpenAI Client Initialized**")

# Initialize session state variables
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.write("🟢 **Session State Initialized:** step = 1")
if 'user_data' not in st.session_state:
    st.session_state['user_data'] = {}
    st.write("🟢 **Session State Initialized:** user_data = {}")
if 'document_summaries' not in st.session_state:
    st.session_state['document_summaries'] = []
    st.write("🟢 **Session State Initialized:** document_summaries = []")
if 'image_contexts' not in st.session_state:
    st.session_state['image_contexts'] = []
    st.write("🟢 **Session State Initialized:** image_contexts = []")
if 'case_info' not in st.session_state:
    st.session_state['case_info'] = {}
    st.write("🟢 **Session State Initialized:** case_info = {}")
if 'report_generated' not in st.session_state:
    st.session_state['report_generated'] = False
    st.write("🟢 **Session State Initialized:** report_generated = False")
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
    st.write("🟢 **Session State Initialized:** chat_history = []")
if 'uploaded_medical_bills' not in st.session_state:
    st.session_state['uploaded_medical_bills'] = None
    st.write("🟢 **Session State Initialized:** uploaded_medical_bills = None")

# =========================
# Helper Functions
# =========================

def sanitize_text(text):
    """Sanitize text to prevent injection attacks."""
    sanitized = html.escape(text)
    st.write(f"🧹 **Sanitized Text:** {sanitized}")
    return sanitized

def get_location_from_zip(zip_code):
    """Fetch city and state based on ZIP code using Zippopotam.us API."""
    try:
        st.write(f"🔍 **Fetching location for ZIP code:** {zip_code}")
        response = requests.get(f"http://api.zippopotam.us/us/{zip_code}")  # Adjust country code as needed
        if response.status_code == 200:
            data = response.json()
            location = {
                'city': data['places'][0]['place name'],
                'state': data['places'][0]['state abbreviation']
            }
            st.write(f"📍 **Location Fetched:** {location}")
            return location
        else:
            st.warning(f"⚠️ **Failed to fetch location for ZIP code {zip_code}:** Status Code {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ **Exception occurred while fetching location for ZIP code {zip_code}:** {e}")
        return None

def extract_text_from_document(file):
    """Extract text from uploaded documents."""
    content = ''
    try:
        st.write(f"📄 **Extracting text from document:** {file.name} (Type: {file.type})")
        if file.type == 'application/pdf':
            # For PDFs
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:
                    content += text
                st.write(f"📑 **Extracted text from page {page_num} of {file.name}**")
        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            # For DOCX
            content = docx2txt.process(file)
            st.write(f"📄 **Extracted text from DOCX:** {file.name}")
        elif file.type == 'text/plain':
            content = str(file.read(), 'utf-8')
            st.write(f"📄 **Extracted text from TXT:** {file.name}")
        else:
            # For other types, perform OCR with OpenAI's GPT-4
            content = ocr_document(file)
    except Exception as e:
        st.error(f"❌ **Error extracting text from {file.name}:** {e}")
        content = f"Error extracting text from {file.name}: {e}"
    st.write(f"📝 **Extracted Content:** {content[:200]}..." if len(content) > 200 else f"📝 **Extracted Content:** {content}")
    return content

def ocr_document(file):
    """Perform OCR on documents using OpenAI's GPT-4."""
    try:
        st.write(f"🖼️ **Performing OCR on document:** {file.name}")
        # Read the image file as bytes
        image_bytes = file.read()
        # Encode image to base64
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        # Prepare system prompt
        system_prompt = "You are an OCR assistant that extracts text from images."
        st.write("📝 **System Prompt for OCR Prepared**")
        # Prepare user message with base64 image
        user_message = f"Extract the text from the following image encoded in base64:\n\n{encoded_image}"
        st.write("📝 **User Message for OCR Prepared**")
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
        st.write(f"📡 **OCR API Response:** {response}")
        # Extract text from response
        extracted_text = response['choices'][0]['message']['content']
        st.write(f"📝 **Extracted Text from OCR:** {extracted_text[:200]}..." if len(extracted_text) > 200 else f"📝 **Extracted Text from OCR:** {extracted_text}")
        return extracted_text
    except Exception as e:
        st.error(f"❌ **Error during OCR with OpenAI for {file.name}:** {e}")
        return f"Error during OCR: {e}"

def summarize_text(text):
    """Summarize text using OpenAI API."""
    try:
        st.write("📄 **Summarizing Text Using OpenAI API**")
        system_prompt = "You are an AI assistant that summarizes documents."
        st.write("📝 **System Prompt for Summarization Prepared**")
        user_message = f"Please summarize the following text:\n\n{text}"
        st.write("📝 **User Message for Summarization Prepared**")
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
        st.write(f"📡 **Summarization API Response:** {response}")
        summary = response['choices'][0]['message']['content'].strip()
        st.write(f"📝 **Generated Summary:** {summary[:200]}..." if len(summary) > 200 else f"📝 **Generated Summary:** {summary}")
        return summary
    except Exception as e:
        st.error(f"❌ **Error in summarize_text:** {e}")
        return f"An error occurred while summarizing the document: {e}"

def analyze_image(image_file, user_data):
    """Analyze image using OpenAI's GPT-4 Vision."""
    try:
        st.write(f"🖼️ **Analyzing Image:** {image_file.name}")
        # Read the image file as bytes
        image_bytes = image_file.read()
        # Encode image to base64
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        # Prepare system prompt
        system_prompt = "You are a legal assistant AI with expertise in analyzing images related to personal injury and malpractice cases."
        st.write("📝 **System Prompt for Image Analysis Prepared**")
        # Prepare user message with base64 image
        user_message_text = f"""
Analyze the following image in the context of a {user_data.get('case_type', 'personal injury')} case that occurred on {user_data.get('incident_date', 'N/A')} at {user_data.get('incident_location', 'N/A')}. Extract any details, abnormalities, or evidence that are relevant to the case, especially those that support or argue against the user's claims.
"""
        st.write("📝 **User Message for Image Analysis Prepared**")
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
        st.write(f"📡 **Image Analysis API Response:** {response}")
        assistant_message = response['choices'][0]['message']['content'].strip()
        st.write(f"📝 **Generated Image Analysis:** {assistant_message[:200]}..." if len(assistant_message) > 200 else f"📝 **Generated Image Analysis:** {assistant_message}")
        return assistant_message
    except Exception as e:
        st.error(f"❌ **Error analyzing image {image_file.name}:** {e}")
        return f"Error analyzing image: {e}"

def process_documents(documents):
    """Process uploaded documents concurrently."""
    summaries = []
    if not documents:
        st.write("📄 **No documents to process.**")
        return summaries
    st.write(f"📂 **Processing {len(documents)} document(s).**")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_document, file): file for file in documents}
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                summary = future.result()
                summaries.append({'filename': file.name, 'summary': summary})
                st.write(f"✅ **Processed Document:** {file.name}")
            except Exception as e:
                st.error(f"❌ **Error processing document {file.name}:** {e}")
    st.write(f"📄 **Completed Processing Documents.**")
    return summaries

def process_single_document(file):
    """Process a single document: extract text and summarize."""
    st.write(f"🔍 **Processing Single Document:** {file.name}")
    text = extract_text_from_document(file)
    if "Error" in text:
        summary = text  # If extraction failed, return the error message
        st.warning(f"⚠️ **Extraction failed for {file.name}:** {text}")
    else:
        summary = summarize_text(text)
    st.write(f"📝 **Summary for {file.name}:** {summary[:200]}..." if len(summary) > 200 else f"📝 **Summary for {file.name}:** {summary}")
    return summary

def process_images(images, user_data):
    """Process uploaded images concurrently."""
    contexts = []
    if not images:
        st.write("🖼️ **No images to process.**")
        return contexts
    st.write(f"📂 **Processing {len(images)} image(s).**")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_image = {executor.submit(analyze_image, image, user_data): image for image in images}
        for future in concurrent.futures.as_completed(future_to_image):
            image = future_to_image[future]
            try:
                context = future.result()
                contexts.append({'filename': image.name, 'context': context})
                st.write(f"✅ **Processed Image:** {image.name}")
            except Exception as e:
                st.error(f"❌ **Error processing image {image.name}:** {e}")
                contexts.append({'filename': image.name, 'context': "Error processing image."})
    st.write(f"🖼️ **Completed Processing Images.**")
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
        st.write("🔍 **Starting Legal Research Using Google Custom Search API**")
        case_type = user_data.get('case_type', 'personal injury')
        location = user_data.get('incident_location', '')

        # Extract medical terms from medical bills if available
        medical_terms = extract_medical_terms(st.session_state.get('uploaded_medical_bills', []))
        st.write(f"🩺 **Extracted Medical Terms:** {medical_terms}")

        # Construct search queries
        search_queries = [
            f"{case_type} laws in {location}",
            f"Relevant statutes for {case_type} cases in {location}",
            f"{' '.join(medical_terms)} treatments in {location}"
        ]
        st.write(f"🔑 **Search Queries:** {search_queries}")

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
            st.write(f"🔍 **Performing Search with Query:** {query}")
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params
            )
            st.write(f"📡 **Google Custom Search API Response Status:** {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if 'items' in data:
                    for item in data['items']:
                        summaries.append(item.get('snippet', ''))
                        links.append(item.get('link', ''))
                else:
                    st.warning(f"⚠️ **No items found for query:** {query}")
            else:
                st.error(f"❌ **Search failed for query '{query}':** Status Code {response.status_code}")
        
        # Compile findings
        compiled_summary = "\n".join(summaries)
        compiled_links = "\n".join(links)

        st.write("📑 **Legal Research Summary:**")
        st.write(compiled_summary[:500] + "..." if len(compiled_summary) > 500 else compiled_summary)
        st.write("🔗 **Legal Research Links:**")
        for link in links:
            st.write(f"- {link}")

        st.write("✅ **Legal Research Completed Successfully**")
        return {
            "legal_research_summary": compiled_summary,
            "legal_research_links": compiled_links
        }
    except Exception as e:
        st.error(f"❌ **Error during Legal Research:** {e}")
        return {
            "legal_research_summary": "An error occurred during legal research.",
            "legal_research_links": ""
        }

def extract_medical_terms(uploaded_files):
    """Extract medical terms from uploaded medical bills or related documents."""
    terms = set()
    for file in uploaded_files:
        st.write(f"🩺 **Extracting Medical Terms from File:** {file.name}")
        text = extract_text_from_document(file)
        # Simple example: extract capitalized terms (could be enhanced with NLP)
        extracted = [word for word in text.split() if word.istitle()]
        terms.update(extracted)
    st.write(f"🩺 **Extracted Medical Terms:** {list(terms)}")
    return list(terms)

def case_law_retrieval(user_data, ser_api_key, serp_api_params={}):
    """
    Retrieve relevant case law using SERP API for Google Scholar.

    Parameters:
    - user_data: Dictionary containing user information.
    - ser_api_key: API key for SERP API.
    - serp_api_params: Additional parameters for the search.

    Returns:
    - Dictionary containing case law and potential payout estimate.
    """
    try:
        st.write("🔍 **Starting Case Law Retrieval Using SERP API**")
        case_type = user_data.get('case_type', 'personal injury')
        location = user_data.get('incident_location', '')

        # Construct search query
        query = f"case precedents for {case_type} in {location}"
        st.write(f"🔑 **Case Law Search Query:** {query}")

        # Base SERP API parameters
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": ser_api_key,
            "num": 5
        }
        # Update with any additional search parameters
        params.update(serp_api_params)

        st.write(f"🔍 **Performing SERP API Search with Params:** {params}")
        response = requests.get("https://serpapi.com/search", params=params)
        st.write(f"📡 **SERP API Response Status:** {response.status_code}")
        if response.status_code == 200:
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
                    st.write(f"📚 **Retrieved Case:** {case_name}, **Outcome:** {outcome}, **Date:** {date}")
            else:
                st.warning("⚠️ **No scholar_results found in SERP API response.**")
                return {
                    "case_law": [],
                    "potential_payout_estimate": 0
                }

            # Analyze outcomes to estimate potential compensation ranges
            potential_payout = analyze_potential_payout(cases)

            st.write(f"💰 **Estimated Potential Payout:** ${potential_payout}")
            st.write("✅ **Case Law Retrieval Completed Successfully**")
            return {
                "case_law": cases,
                "potential_payout_estimate": potential_payout
            }
        else:
            st.error(f"❌ **SERP API Search Failed:** Status Code {response.status_code}")
            return {
                "case_law": [],
                "potential_payout_estimate": 0
            }
    except Exception as e:
        st.error(f"❌ **Error during Case Law Retrieval:** {e}")
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
        st.write(f"📅 **Extracting Date from Publication Info:** {publication_date}")
        # Attempt to parse the date in various formats
        for fmt in ("%Y-%m-%d", "%B %Y", "%Y"):
            try:
                date_obj = datetime.strptime(publication_date, fmt)
                formatted_date = date_obj.strftime("%B %Y")
                st.write(f"📆 **Parsed Date:** {formatted_date}")
                return formatted_date
            except ValueError:
                continue
        st.write(f"📆 **Unable to Parse Date:** {publication_date}")
        return publication_date
    except:
        st.write(f"📆 **Exception occurred while parsing date:** {publication_date}")
        return publication_date

def analyze_potential_payout(cases):
    """Analyze case outcomes to estimate potential compensation ranges."""
    st.write("💰 **Analyzing Potential Payout Based on Case Outcomes**")
    payouts = []
    for case in cases:
        if case['outcome'] == "Won":
            # Placeholder: Assign arbitrary values or use NLP to extract amounts
            payouts.append(50000)  # Example value
            st.write(f"💵 **Added Payout for Won Case:** {case['case_name']} - $50,000")
    if payouts:
        average_payout = sum(payouts) / len(payouts)
        st.write(f"📈 **Average Potential Payout:** ${average_payout}")
        return average_payout
    else:
        st.write("📉 **No Successful Cases Found to Estimate Potential Payout.**")
        return 0

def generate_case_info(user_data, document_summaries, image_contexts, ser_api_key, ser_api_engine_id, serp_api_params={}):
    """Generate case information using OpenAI API and additional AI agents."""
    try:
        st.write("📝 **Generating Case Information Using OpenAI API**")
        # Retrieve legal research data
        st.write("🔍 **Generating Legal Research Data**")
        legal_research_data = legal_research(user_data, ser_api_key, ser_api_engine_id, serp_api_params)

        # Retrieve case law data
        st.write("📚 **Generating Case Law Data**")
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

        st.write("📑 **Prompt for Case Information Generated**")
        st.write(f"📜 **Prompt Content:** {prompt[:500]}..." if len(prompt) > 500 else f"📜 **Prompt Content:** {prompt}")

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
        st.write(f"📡 **OpenAI API Response for Case Info:** {response}")
        output_text = response['choices'][0]['message']['content']
        st.write(f"📝 **Case Info JSON Received:** {output_text[:500]}..." if len(output_text) > 500 else f"📝 **Case Info JSON Received:** {output_text}")
        case_info = json.loads(output_text)
        st.write("✅ **Case Information Generated Successfully**")
    except json.JSONDecodeError as je:
        st.error(f"❌ **JSON Decode Error:** {je}")
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
        st.error(f"❌ **Error Generating Case Info:** {e}")
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
        st.write("📧 **Attempting to Subscribe User to MailChimp**")
        MAILCHIMP_API_KEY = os.getenv("MAILCHIMP_API_KEY")
        MAILCHIMP_LIST_ID = os.getenv("MAILCHIMP_LIST_ID")
        MAILCHIMP_DC = os.getenv("MAILCHIMP_DC")

        if not all([MAILCHIMP_API_KEY, MAILCHIMP_LIST_ID, MAILCHIMP_DC]):
            st.error("❌ **MailChimp API credentials are not fully set.**")
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
        st.write(f"📦 **MailChimp Subscription Payload:** {json.dumps(data)}")
        response = requests.post(url, auth=auth, json=data)
        st.write(f"📡 **MailChimp API Response Status:** {response.status_code}")
        st.write(f"📡 **MailChimp API Response Text:** {response.text}")
        if response.status_code in [200, 201]:
            st.write("✅ **User Subscribed to MailChimp Successfully**")
            return True
        else:
            st.error(f"❌ **MailChimp Subscription Failed:** {response.status_code} - {response.text}")
            return False
    except Exception as e:
        st.error(f"❌ **Exception During MailChimp Subscription:** {e}")
        return False

def generate_pdf_report(case_info, document_summaries, image_contexts):
    """Generate a PDF report using reportlab."""
    st.write("🖨️ **Generating PDF Report**")
    buffer = io.BytesIO()
    try:
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(width / 2, height - 50, "Case Analysis Report")
        st.write("📑 **PDF Title Added**")

        # Case Summary
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 80, "Case Summary:")
        c.setFont("Helvetica", 12)
        text = c.beginText(50, height - 100)
        for line in case_info.get('case_summary', '').split('\n'):
            text.textLine(line)
        c.drawText(text)
        st.write("📝 **Case Summary Added to PDF**")

        # Best Arguments
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 200, "Best Arguments:")
        c.setFont("Helvetica", 12)
        text = c.beginText(50, height - 220)
        for line in case_info.get('best_arguments', '').split('\n'):
            text.textLine(line)
        c.drawText(text)
        st.write("📝 **Best Arguments Added to PDF**")

        # Relevant Laws
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 320, "Relevant Laws:")
        c.setFont("Helvetica", 12)
        text = c.beginText(50, height - 340)
        for line in case_info.get('relevant_laws', '').split('\n'):
            text.textLine(line)
        c.drawText(text)
        st.write("📝 **Relevant Laws Added to PDF**")
        
        # Medical Literature
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 440, "Medical Literature:")
        c.setFont("Helvetica", 12)
        text = c.beginText(50, height - 460)
        for line in case_info.get('medical_literature', '').split('\n'):
            text.textLine(line)
        c.drawText(text)
        st.write("📝 **Medical Literature Added to PDF**")

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
            st.write(f"📚 **Case Law Added to PDF:** {case['case_name']}")

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
            st.write(f"📄 **Document Summary Added to PDF:** {doc['filename']}")

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
            st.write(f"🖼️ **Image Analysis Added to PDF:** {img['filename']}")

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
        st.write("💰 **Potential Payout and Likelihood Added to PDF**")

        c.save()
        buffer.seek(0)
        st.write("📄 **PDF Report Generated Successfully**")
        return buffer
    except Exception as e:
        st.error(f"❌ **Error Generating PDF Report:** {e}")
        return io.BytesIO()  # Return empty buffer in case of error

def generate_markdown_report(case_info, document_summaries, image_contexts):
    """Generate a markdown report."""
    try:
        st.write("📝 **Generating Markdown Report**")
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
            st.write(f"📚 **Case Law Added to Markdown:** {case['case_name']}")

        report_content += f"## Document Summaries\n"
        for doc in document_summaries:
            report_content += f"### {sanitize_text(doc['filename'])}\n{sanitize_text(doc['summary'])}\n\n"
            st.write(f"📄 **Document Summary Added to Markdown:** {doc['filename']}")

        report_content += f"## Image Analyses\n"
        for img in image_contexts:
            report_content += f"### {sanitize_text(img['filename'])}\n{sanitize_text(img['context'])}\n\n"
            st.write(f"🖼️ **Image Analysis Added to Markdown:** {img['filename']}")

        report_content += f"## Potential Payout and Likelihood\n"
        report_content += f"**Estimated Potential Payout:** ${case_info.get('potential_payout', 0)}\n\n"
        report_content += f"**Likelihood of Winning:** {case_info.get('likelihood_of_winning', 0)}%\n\n"

        st.write("✅ **Markdown Report Generated Successfully**")
        return report_content
    except Exception as e:
        st.error(f"❌ **Error Generating Markdown Report:** {e}")
        return "# Case Analysis Report\n\nAn error occurred while generating the report."

def initialize_chat_interface(case_info, document_summaries):
    """Initialize the chat interface with system prompt."""
    try:
        st.write("💬 **Initializing Chat Interface**")
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
        st.write("✅ **Chat Interface Initialized with System Prompt**")
    except Exception as e:
        st.error(f"❌ **Error Initializing Chat Interface:** {e}")

# =========================
# Main Application
# =========================

def main():
    st.set_page_config(page_title="Legal Assistant", layout="wide")
    st.title("⚖️ **Legal Assistant for Personal Injury/Malpractice Cases**")

    # Display progress bar
    progress = st.progress((st.session_state.step - 1) / 5)
    st.write(f"📈 **Current Step:** {st.session_state.step}/5")
    
    # Step 1: Personal Information
    if st.session_state.step == 1:
        st.header("🔵 Step 1: Personal Information")
        with st.form("personal_info_form"):
            first_name = st.text_input("First Name *", value=st.session_state['user_data'].get('first_name', ''))
            last_name = st.text_input("Last Name *", value=st.session_state['user_data'].get('last_name', ''))
            email = st.text_input("Email Address *", value=st.session_state['user_data'].get('email', ''))
            phone = st.text_input("Phone Number *", value=st.session_state['user_data'].get('phone', ''))
            submitted = st.form_submit_button("Next")
            
            if submitted:
                st.write("🚀 **Submitting Personal Information Form**")
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
                        st.error(f"❌ {error}")
                else:
                    st.session_state['user_data'].update({
                        'first_name': first_name,
                        'last_name': last_name,
                        'email': email,
                        'phone': phone
                    })
                    st.session_state.step = 2
                    st.success("✅ **Personal information saved successfully!**")
                    st.write("➡️ **Moving to Step 2: Case Details**")

    # Step 2: Case Details
    elif st.session_state.step == 2:
        st.header("🟢 Step 2: Case Details")
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
                    st.success(f"📍 **Location Auto-Filled:** {incident_city}, {incident_state}")
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
            st.subheader("💉 Medical Bills")
            uploaded_medical_bills = st.file_uploader(
                "Upload Medical Bills (PDF, JPG, PNG) *",
                type=['pdf', 'jpg', 'png'],
                accept_multiple_files=False,
                help="Upload your medical bills. If you do not have the documents, please provide estimated amounts below."
            )
            if uploaded_medical_bills:
                st.session_state['uploaded_medical_bills'] = uploaded_medical_bills
                st.success("✅ **Medical bills uploaded successfully!**")
                st.write(f"📁 **Uploaded Medical Bills:** {uploaded_medical_bills.name}")
            else:
                medical_bills_notes = st.text_area(
                    "If you do not have medical bills to upload, please provide estimated amounts or notes:",
                    value=st.session_state['user_data'].get('medical_bills_notes', ''),
                    help="Provide an estimated range or any notes regarding your medical bills."
                )
                st.write("📝 **User Provided Medical Bills Notes**")
    
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
            st.subheader("🗒️ Additional Information")
            additional_comments = st.text_area(
                "Please provide any additional relevant information:",
                value=st.session_state['user_data'].get('additional_comments', ''),
                help="Please answer the following prompts:\n- Have there been any witnesses?\n- Are there any prior incidents related to this case?\n- Any other details you find pertinent."
            )
            submitted = st.form_submit_button("Next")
            
            if submitted:
                st.write("🚀 **Submitting Case Details Form**")
                errors = []
                if not incident_description:
                    errors.append("❌ Description of incident is required.")
                if not damages_incurred:
                    errors.append("❌ At least one type of damage must be selected.")
                if not zip_code:
                    errors.append("❌ ZIP/Postal Code is required.")
                if not incident_city and (zip_code and not location):
                    errors.append("❌ City is required.")
                if not incident_state and (zip_code and not location):
                    errors.append("❌ State/Province is required.")
                if not incident_country:
                    errors.append("❌ Country is required.")
                if not best_argument:
                    errors.append("❌ Best argument is required.")
                if not uploaded_medical_bills and not st.session_state['user_data'].get('medical_bills_notes', ''):
                    errors.append("❌ Please upload your medical bills or provide estimated amounts/notes.")
                
                if "Other" in damages_incurred and not other_damages:
                    errors.append("❌ Please specify your other damages.")
                
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
                    st.success("✅ **Case details saved successfully!**")
                    st.write("➡️ **Moving to Step 3: Upload Documents and Images**")

    # Step 3: Uploads
    elif st.session_state.step == 3:
        st.header("🟡 Step 3: Upload Documents and Images")
        with st.form("uploads_form"):
            uploaded_documents = st.file_uploader(
                "📄 Upload Documents",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                help="Upload relevant documents such as reports, receipts, or any legal documents."
            )
            uploaded_images = st.file_uploader(
                "🖼️ Upload Images",
                type=['jpg', 'jpeg', 'png', 'gif'],
                accept_multiple_files=True,
                help="Upload images related to your case, such as photos of injuries or accident scenes."
            )
            submitted = st.form_submit_button("Next")
            
            if submitted:
                st.write("🚀 **Submitting Uploads Form**")
                if not uploaded_documents and not uploaded_images and not st.session_state.get('uploaded_medical_bills', None):
                    st.error("❌ **Please upload at least one document, image, or medical bill.**")
                else:
                    st.session_state['uploaded_documents'] = uploaded_documents
                    st.session_state['uploaded_images'] = uploaded_images
                    st.session_state.step = 4
                    st.success("✅ **Files uploaded successfully!**")
                    st.write("➡️ **Moving to Step 4: Consent and Acknowledgements**")

        # Navigation Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Previous"):
                st.session_state.step = 2
                st.experimental_rerun()

    # Step 4: Consent
    elif st.session_state.step == 4:
        st.header("🟠 Step 4: Consent and Acknowledgements")
        with st.form("consent_form"):
            st.markdown("**📜 Terms and Conditions**: [Read our Terms and Conditions](https://your-terms-link.com)")
            st.markdown("**🔒 Privacy Policy**: [Read our Privacy Policy](https://your-privacy-link.com)")
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
                st.write("🚀 **Submitting Consent Form**")
                if not disclaimer_agreed:
                    st.error("❌ **You must agree to the terms and conditions to proceed.**")
                else:
                    st.session_state['user_data'].update({
                        'disclaimer_agreed': disclaimer_agreed,
                        'opt_in_contact': opt_in_contact
                    })
                    st.session_state.step = 5
                    st.success("✅ **Consent information saved successfully!**")
                    st.write("➡️ **Moving to Step 5: Review and Submit**")

        # Navigation Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Previous"):
                st.session_state.step = 3
                st.experimental_rerun()

    # Step 5: Review and Submit
    elif st.session_state.step == 5:
        st.header("🟣 Step 5: Review and Submit")
        st.subheader("📄 **Please review the information you have provided:**")
        st.json(st.session_state['user_data'])
        
        if st.button("🚀 Generate AI Report"):
            if not st.session_state['report_generated']:
                with st.spinner("🔄 **Processing your case...**"):
                    try:
                        st.write("🗂️ **Processing Documents and Images**")
                        # Process documents and images concurrently
                        document_summaries = process_documents(st.session_state.get('uploaded_documents', []))
                        image_contexts = process_images(st.session_state.get('uploaded_images', []), st.session_state['user_data'])
                        st.session_state['document_summaries'] = document_summaries
                        st.session_state['image_contexts'] = image_contexts

                        st.write("🔍 **Retrieving SERP API Credentials**")
                        # Retrieve SERP API credentials
                        serp_api_key = os.getenv("SERP_API_KEY")
                        serp_api_engine_id = os.getenv("SERP_API_ENGINE_ID")
                        serp_api_params = {}  # Add any additional parameters if needed

                        if not serp_api_key or not serp_api_engine_id:
                            st.error("❌ **SERP API credentials are missing. Legal research cannot be performed.**")
                            st.stop()

                        st.write("🧠 **Generating Case Information with AI Agents**")
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
                            st.write("📧 **Opt-in for MailChimp Subscription Detected**")
                            success = add_to_mailchimp(st.session_state['user_data'], case_info)
                            if success:
                                st.success("✅ **You have been subscribed to our mailing list.**")
                            else:
                                st.error("❌ **There was an error subscribing you to our mailing list.**")

                        # Initialize chat interface
                        initialize_chat_interface(case_info, document_summaries)

                        # Generate PDF report
                        st.write("🖨️ **Generating PDF Report**")
                        pdf_buffer = generate_pdf_report(case_info, document_summaries, image_contexts)
                        st.session_state['pdf_report'] = pdf_buffer

                        # Generate Markdown report if needed
                        st.write("📝 **Generating Markdown Report**")
                        markdown_report = generate_markdown_report(case_info, document_summaries, image_contexts)
                        st.session_state['markdown_report'] = markdown_report

                        st.session_state['report_generated'] = True
                        st.success("✅ **Your AI report has been generated!**")
                    except Exception as e:
                        st.error(f"❌ **An error occurred while generating your report:** {e}")

        # Navigation Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Previous"):
                st.session_state.step = 4
                st.experimental_rerun()

        # Display results page
        if st.session_state['report_generated']:
            st.header("📑 Case Analysis Summary")

            st.subheader("📄 Case Overview")
            st.write(st.session_state['case_info'].get('case_summary', ''))
            st.write("---")

            st.subheader("💪 Best Arguments")
            st.write(st.session_state['case_info'].get('best_arguments', ''))
            st.write("---")

            st.subheader("📜 Relevant Laws")
            st.write(st.session_state['case_info'].get('relevant_laws', ''))
            st.write("---")
            
            st.subheader("🩺 Medical Literature")
            st.write(st.session_state['case_info'].get('medical_literature', ''))
            st.write("---")

            st.subheader("📚 Case Law")
            for case in st.session_state['case_info'].get('case_law', []):
                st.write(f"**{case['case_name']}**")
                st.write(f"**Summary:** {case['summary']}")
                st.write(f"**Outcome:** {case['outcome']}")
                st.write(f"**Date:** {case['date']}\n")
            st.write("---")

            st.subheader("📄 Document Summaries")
            for doc_summary in st.session_state['document_summaries']:
                st.write(f"**{doc_summary['filename']}**")
                st.write(doc_summary['summary'])
            st.write("---")

            st.subheader("🖼️ Image Analyses")
            for image_context in st.session_state['image_contexts']:
                st.write(f"**{image_context['filename']}**")
                st.write(image_context['context'])
            st.write("---")

            st.subheader("💰 Potential Payout and Likelihood")
            st.write(f"**Estimated Potential Payout:** ${st.session_state['case_info'].get('potential_payout', 0)}")
            st.write(f"**Likelihood of Winning:** {st.session_state['case_info'].get('likelihood_of_winning', 0)}%")
            st.write("---")

            # Downloadable PDF report
            if 'pdf_report' in st.session_state:
                st.download_button(
                    label="📥 Download Full PDF Report",
                    data=st.session_state['pdf_report'],
                    file_name="Case_Analysis_Report.pdf",
                    mime="application/pdf"
                )
                st.write("✅ **PDF Report Available for Download**")

            # Downloadable Markdown report (optional)
            if 'markdown_report' in st.session_state:
                st.download_button(
                    label="📥 Download Full Markdown Report",
                    data=st.session_state['markdown_report'],
                    file_name="Case_Analysis_Report.md",
                    mime="text/markdown"
                )
                st.write("✅ **Markdown Report Available for Download**")

            # Chat Interface
            st.header("💬 Chat with Your Case")
            if 'chat_history' not in st.session_state or not st.session_state['chat_history']:
                initialize_chat_interface(st.session_state['case_info'], st.session_state['document_summaries'])

            # Display chat messages
            for message in st.session_state['chat_history']:
                if message['role'] == 'user':
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Assistant:** {message['content']}")

            # User input
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_input("🗨️ Type your message here:", key="chat_input")
                submitted = st.form_submit_button("Send")
                
                if submitted and user_input:
                    st.write("🚀 **Sending Message to Assistant**")
                    # Append user message
                    st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
                    st.write(f"**You:** {user_input}")

                    # Prepare messages for OpenAI API
                    messages = st.session_state['chat_history']

                    # API call
                    try:
                        with st.spinner("🔄 **Generating Response...**"):
                            response = client.chat.completions.create(
                                model="gpt-4",  # Corrected model name
                                messages=messages,
                                temperature=0.45,
                                max_tokens=1500,
                                top_p=1,
                                frequency_penalty=0,
                                presence_penalty=0
                            )
                            assistant_message = response['choices'][0]['message']['content'].strip()
                            st.session_state['chat_history'].append({'role': 'assistant', 'content': assistant_message})
                            st.write(f"**Assistant:** {assistant_message}")
                    except Exception as e:
                        st.error(f"❌ **Error Communicating with Assistant:** {e}")

            # Reset Chat Button
            if st.button("🔄 Reset Chat"):
                initialize_chat_interface(st.session_state['case_info'], st.session_state['document_summaries'])
                st.success("✅ **Chat history has been reset.**")

if __name__ == '__main__':
    main()
