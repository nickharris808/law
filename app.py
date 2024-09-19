# app.py

import streamlit as st
import openai
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import PyPDF2
import docx2txt
import pytesseract
from PIL import Image
import requests
import logging
import concurrent.futures
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import html

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

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

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

# =========================
# Helper Functions
# =========================

def sanitize_text(text):
    """Sanitize text to prevent injection attacks."""
    return html.escape(text)

def upload_image_and_get_url(image_file):
    """Upload image to Imgur and return the public URL."""
    IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
    api_url = "https://api.imgur.com/3/image"

    try:
        response = requests.post(
            api_url,
            headers=headers,
            data={'image': image_file.read()}
        )
        data = response.json()
        if data['success']:
            return data['data']['link']
        else:
            logging.error(f"Imgur upload failed: {data['data']['error']}")
            return None
    except Exception as e:
        logging.error(f"Exception during image upload: {e}")
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
            # For other types, perform OCR
            content = ocr_document(file)
    except Exception as e:
        logging.error(f"Error extracting text from {file.name}: {e}")
    return content

def ocr_document(file):
    """Perform OCR on documents."""
    try:
        temp_filename = 'temp_file'
        with open(temp_filename, 'wb') as out_file:
            out_file.write(file.getbuffer())
        text = pytesseract.image_to_string(Image.open(temp_filename))
        os.remove(temp_filename)
        return text
    except Exception as e:
        logging.error(f"OCR failed for {file.name}: {e}")
        return ''

def summarize_text(text):
    """Summarize text using OpenAI API."""
    prompt = f"Summarize the following text:\n\n{text}"
    try:
        response = openai.Completion.create(
            engine='gpt-40-mini',  # Using the latest model
            prompt=prompt,
            max_tokens=500,
            temperature=0.5,
        )
        summary = response['choices'][0]['text'].strip()
        return summary
    except openai.error.RateLimitError:
        logging.warning("Rate limit reached. Retrying after delay...")
        time.sleep(5)  # Wait before retrying
        return summarize_text(text)  # Recursive retry
    except Exception as e:
        logging.error(f"Error in summarize_text: {e}")
        return "An error occurred while summarizing the document."

def analyze_image(image_file, user_data):
    """Analyze image using gpt-40-mini Vision."""
    image_url = upload_image_and_get_url(image_file)

    if not image_url:
        return "Failed to retrieve image URL."

    messages = [
        {
            "role": "system",
            "content": "You are a legal assistant AI with expertise in analyzing images related to personal injury and malpractice cases."
        },
        {
            "role": "user",
            "content": f"""
            Analyze the following image in the context of a {user_data.get('case_type', 'personal injury')} case that occurred on {user_data.get('incident_date', 'N/A')} at {user_data.get('incident_location', 'N/A')}. Extract any details, abnormalities, or evidence that are relevant to the case, especially those that support or argue against the user's claims.
            
            Image URL: {image_url}
            """
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-40-mini",
            messages=messages,
            temperature=0.45,
            max_tokens=1500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        assistant_message = response['choices'][0]['message']['content'].strip()
        return assistant_message
    except openai.error.RateLimitError:
        logging.warning("Rate limit reached during image analysis. Retrying after delay...")
        time.sleep(5)
        return analyze_image(image_file, user_data)
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

def generate_case_info(user_data, document_summaries, image_contexts):
    """Generate case information using OpenAI API."""
    prompt = f"""
    Based on the following user data, document summaries, and image analyses, provide a JSON containing:
    - 'case_summary': A comprehensive summary of the case.
    - 'best_arguments': The strongest arguments tailored to the user's case.
    - 'relevant_laws': Specific laws applicable to the case.
    - 'potential_payout': The estimated monetary value the user might be awarded.
    - 'likelihood_of_winning': A percentage likelihood of winning the case.

    User Data:
    {json.dumps(user_data, indent=2)}

    Document Summaries:
    {json.dumps(document_summaries, indent=2)}

    Image Analyses:
    {json.dumps(image_contexts, indent=2)}
    """
    try:
        response = openai.Completion.create(
            engine='gpt-40-mini',
            prompt=prompt,
            max_tokens=1500,
            temperature=0.5,
        )
        output_text = response['choices'][0]['text'].strip()
        case_info = json.loads(output_text)
    except openai.error.RateLimitError:
        logging.warning("Rate limit reached during case info generation. Retrying after delay...")
        time.sleep(5)
        return generate_case_info(user_data, document_summaries, image_contexts)
    except json.JSONDecodeError:
        logging.error("Failed to parse case info JSON.")
        case_info = {
            "case_summary": "Could not generate case summary.",
            "best_arguments": "",
            "relevant_laws": "",
            "potential_payout": 0,
            "likelihood_of_winning": 0
        }
    except Exception as e:
        logging.error(f"Error generating case info: {e}")
        case_info = {
            "case_summary": "An error occurred while generating the case summary.",
            "best_arguments": "",
            "relevant_laws": "",
            "potential_payout": 0,
            "likelihood_of_winning": 0
        }
    return case_info

def add_to_mailchimp(user_data, case_info):
    """Add user to MailChimp list."""
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
    try:
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

    # Document Summaries
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 440, "Document Summaries:")
    y_position = height - 460
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

def initialize_chat_interface(case_info, document_summaries):
    """Initialize the chat interface with system prompt."""
    system_prompt = f"""
    You are a legal assistant AI that has analyzed the following case information and documents:

    Case Summary:
    {sanitize_text(case_info.get('case_summary', ''))}

    Document Summaries:
    {json.dumps(document_summaries, indent=2)}

    Use this information to answer the user's questions accurately and helpfully.
    """
    st.session_state['chat_history'] = [{'role': 'system', 'content': system_prompt}]

def generate_markdown_report(case_info, document_summaries, image_contexts):
    """Generate a markdown report."""
    report_content = f"# Case Analysis Report\n\n"

    report_content += f"## Case Summary\n{sanitize_text(case_info.get('case_summary', ''))}\n\n"

    report_content += f"## Best Arguments\n{sanitize_text(case_info.get('best_arguments', ''))}\n\n"

    report_content += f"## Relevant Laws\n{sanitize_text(case_info.get('relevant_laws', ''))}\n\n"

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
            first_name = st.text_input("First Name", value=st.session_state['user_data'].get('first_name', ''))
            last_name = st.text_input("Last Name", value=st.session_state['user_data'].get('last_name', ''))
            email = st.text_input("Email Address", value=st.session_state['user_data'].get('email', ''))
            phone = st.text_input("Phone Number", value=st.session_state['user_data'].get('phone', ''))
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
            case_type = st.selectbox("Type of Case", ["Personal Injury", "Medical Malpractice", "Car Accident", "Other"], 
                                     index=["Personal Injury", "Medical Malpractice", "Car Accident", "Other"].index(st.session_state['user_data'].get('case_type', "Personal Injury")))
            incident_date = st.date_input("Date of Incident", 
                                          value=datetime.strptime(st.session_state['user_data'].get('incident_date', datetime.today().strftime("%Y-%m-%d")), "%Y-%m-%d"))
            incident_city = st.text_input("City", value=st.session_state['user_data'].get('incident_city', ''))
            incident_state = st.text_input("State/Province", value=st.session_state['user_data'].get('incident_state', ''))
            incident_country = st.text_input("Country", value=st.session_state['user_data'].get('incident_country', ''))
            incident_description = st.text_area("Description of Incident", value=st.session_state['user_data'].get('incident_description', ''))
            damages_incurred = st.multiselect("Damages Incurred", ["Physical Injuries", "Emotional Distress", "Property Damage", "Financial Losses"],
                                             default=st.session_state['user_data'].get('damages_incurred', []))
            medical_bills_amount = st.number_input("Medical Bills Amount", min_value=0, value=st.session_state['user_data'].get('medical_bills_amount', 0))
            medical_treatment = st.text_area("Medical Treatment Received", value=st.session_state['user_data'].get('medical_treatment', ''))
            best_argument = st.text_area("What do you think is your best argument?", value=st.session_state['user_data'].get('best_argument', ''))
            additional_comments = st.text_area("Additional Comments", value=st.session_state['user_data'].get('additional_comments', ''))
            submitted = st.form_submit_button("Next")
            
            if submitted:
                errors = []
                if not incident_description:
                    errors.append("Description of incident is required.")
                if not damages_incurred:
                    errors.append("At least one type of damage must be selected.")
                
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    st.session_state['user_data'].update({
                        'case_type': case_type,
                        'incident_date': incident_date.strftime("%Y-%m-%d"),
                        'incident_city': incident_city,
                        'incident_state': incident_state,
                        'incident_country': incident_country,
                        'incident_location': f"{incident_city}, {incident_state}, {incident_country}",
                        'incident_description': incident_description,
                        'damages_incurred': damages_incurred,
                        'medical_bills_amount': medical_bills_amount,
                        'medical_treatment': medical_treatment,
                        'best_argument': best_argument,
                        'additional_comments': additional_comments
                    })
                    st.session_state.step = 3
                    st.success("Case details saved successfully!")

        # Navigation Buttons
        col1, col2 = st.columns(2)
        with col2:
            if st.button("Previous"):
                st.session_state.step = 1

    # Step 3: Uploads
    elif st.session_state.step == 3:
        st.header("Step 3: Upload Documents and Images")
        with st.form("uploads_form"):
            uploaded_documents = st.file_uploader("Upload Documents", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
            uploaded_images = st.file_uploader("Upload Images", type=['jpg', 'jpeg', 'png', 'gif'], accept_multiple_files=True)
            submitted = st.form_submit_button("Next")
            
            if submitted:
                if not uploaded_documents and not uploaded_images:
                    st.error("Please upload at least one document or image.")
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
            disclaimer_agreed = st.checkbox("I agree to the terms and conditions.", value=st.session_state['user_data'].get('disclaimer_agreed', False))
            opt_in_contact = st.checkbox("I agree to be contacted by a lawyer regarding my case.", value=st.session_state['user_data'].get('opt_in_contact', False))
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

                        # Generate case info
                        case_info = generate_case_info(st.session_state['user_data'], document_summaries, image_contexts)
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
                        response = openai.ChatCompletion.create(
                            model="gpt-40-mini",
                            messages=messages,
                            temperature=0.45,
                            max_tokens=500,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0,
                        )
                        assistant_message = response['choices'][0]['message']['content'].strip()
                        st.session_state['chat_history'].append({'role': 'assistant', 'content': assistant_message})
                        st.success("Assistant has responded!")
                except openai.error.RateLimitError:
                    logging.warning("Rate limit reached during chat. Retrying after delay...")
                    time.sleep(5)
                    st.error("Rate limit reached. Please try again later.")
                except Exception as e:
                    logging.error(f"Error in chat interface: {e}")
                    st.error("An error occurred while communicating with the assistant. Please try again later.")

        # Reset Chat Button
        if st.button("Reset Chat"):
            initialize_chat_interface(st.session_state['case_info'], st.session_state['document_summaries'])
            st.success("Chat history has been reset.")

if __name__ == '__main__':
    main()
