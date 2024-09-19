# app.py

import streamlit as st
import openai
import os
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state variables
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

# Function to extract text from uploaded documents
async def extract_text_from_document(file):
    content = ''
    if file.type == 'application/pdf':
        # For PDFs
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                content += text
    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        # For DOCX
        import docx2txt
        content = docx2txt.process(file)
    elif file.type == 'text/plain':
        content = str(file.read(), 'utf-8')
    else:
        content = await ocr_document(file)
    return content

# Function to perform OCR on documents
async def ocr_document(file):
    import pytesseract
    from PIL import Image
    temp_filename = 'temp_file'
    with open(temp_filename, 'wb') as out_file:
        out_file.write(file.getbuffer())
    text = pytesseract.image_to_string(Image.open(temp_filename))
    os.remove(temp_filename)
    return text

# Function to summarize text using OpenAI API
async def summarize_text(text):
    prompt = f"Summarize the following text:\n\n{text}"
    response = await openai.Completion.acreate(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=500,
        temperature=0.5,
    )
    summary = response['choices'][0]['text'].strip()
    return summary

# Updated Function to analyze images using GPT-4 Vision
async def analyze_image(image_file, user_data):
    # IMPORTANT: Ensure that the image is accessible via a public URL.
    # You might need to upload the image to a hosting service and obtain the URL.
    # For demonstration purposes, we'll assume that you have a way to get the image URL.
    
    # Placeholder for image URL retrieval
    # You need to implement the logic to upload the image and get the URL
    # For example, using an image hosting service API
    image_url = await upload_image_and_get_url(image_file)
    
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
            Analyze the following image in the context of a {user_data['case_type']} that occurred on {user_data['incident_date']} at {user_data['incident_location']}. Extract any details, abnormalities, or evidence that are relevant to the case, especially those that support or argue against the user's claims.
            
            Image URL: {image_url}
            """
        }
    ]

    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=messages,
            temperature=0.45,
            max_tokens=1500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        assistant_message = response['choices'][0]['message']['content'].strip()
        return assistant_message
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Placeholder function to upload image and get URL
async def upload_image_and_get_url(image_file):
    # Implement your image uploading logic here
    # This could be uploading to AWS S3, Google Cloud Storage, Imgur, etc.
    # Return the publicly accessible URL of the uploaded image
    # For demonstration, we'll return None
    return None  # Replace this with actual URL after uploading

# Function to process documents concurrently
async def process_documents(documents):
    summaries = []
    tasks = []
    for file in documents:
        tasks.append(process_single_document(file))
    summaries = await asyncio.gather(*tasks)
    return summaries

async def process_single_document(file):
    text = await extract_text_from_document(file)
    summary = await summarize_text(text)
    return {'filename': file.name, 'summary': summary}

# Function to process images concurrently
async def process_images(images, user_data):
    contexts = []
    tasks = []
    for image_file in images:
        tasks.append(analyze_image(image_file, user_data))
    contexts = await asyncio.gather(*tasks)
    return [{'filename': image_file.name, 'context': context} for image_file, context in zip(images, contexts)]

# Function to generate case summary and other information
async def generate_case_info(user_data, document_summaries, image_contexts):
    prompt = f"""
    Based on the following user data, document summaries, and image analyses, provide a JSON containing:
    - 'case_summary': A comprehensive summary of the case.
    - 'best_arguments': The strongest arguments tailored to the user's case.
    - 'relevant_laws': Specific laws applicable to the case.
    - 'potential_payout': The estimated monetary value the user might be awarded.
    - 'likelihood_of_winning': A percentage likelihood of winning the case.

    User Data:
    {json.dumps(user_data)}

    Document Summaries:
    {json.dumps(document_summaries)}

    Image Analyses:
    {json.dumps(image_contexts)}
    """
    response = await openai.Completion.acreate(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=1500,
        temperature=0.5,
    )
    output_text = response['choices'][0]['text'].strip()
    try:
        case_info = json.loads(output_text)
    except json.JSONDecodeError:
        case_info = {
            "case_summary": "Could not generate case summary.",
            "best_arguments": "",
            "relevant_laws": "",
            "potential_payout": 0,
            "likelihood_of_winning": 0
        }
    return case_info

# Function to add user to MailChimp list
def add_to_mailchimp(user_data, case_info):
    import requests
    MAILCHIMP_API_KEY = os.getenv("MAILCHIMP_API_KEY")
    MAILCHIMP_LIST_ID = os.getenv("MAILCHIMP_LIST_ID")
    MAILCHIMP_DC = os.getenv("MAILCHIMP_DC")

    url = f"https://{MAILCHIMP_DC}.api.mailchimp.com/3.0/lists/{MAILCHIMP_LIST_ID}/members"
    data = {
        "email_address": user_data['email'],
        "status": "subscribed",
        "merge_fields": {
            "FNAME": user_data['first_name'],
            "LNAME": user_data['last_name'],
            "CASEVAL": str(case_info['potential_payout']),
            "LIKELIHOOD": str(case_info['likelihood_of_winning'])
        }
    }
    auth = ('anystring', MAILCHIMP_API_KEY)
    response = requests.post(url, auth=auth, json=data)
    return response.status_code == 200

# Function to initialize chat interface
def initialize_chat_interface(case_info, document_summaries):
    system_prompt = f"""
    You are a legal assistant AI that has analyzed the following case information and documents:

    Case Summary:
    {case_info['case_summary']}

    Document Summaries:
    {json.dumps(document_summaries, indent=2)}

    Use this information to answer the user's questions accurately and helpfully.
    """
    st.session_state['chat_history'] = [{'role': 'system', 'content': system_prompt}]

# Main function
def main():
    st.title("Legal Assistant for Personal Injury/Malpractice Cases")

    # Multi-step form using tabs
    tabs = st.tabs(["Personal Information", "Case Details", "Uploads", "Consent", "Review & Submit"])

    # Step 1: Personal Information
    with tabs[0]:
        st.header("Step 1: Personal Information")
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        email = st.text_input("Email Address")
        phone = st.text_input("Phone Number")

        st.session_state['user_data'].update({
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'phone': phone
        })

    # Step 2: Case Details
    with tabs[1]:
        st.header("Step 2: Case Details")
        case_type = st.selectbox("Type of Case", ["Personal Injury", "Medical Malpractice", "Car Accident", "Other"])
        incident_date = st.date_input("Date of Incident")
        incident_city = st.text_input("City")
        incident_state = st.text_input("State/Province")
        incident_country = st.text_input("Country")
        incident_description = st.text_area("Description of Incident")
        damages_incurred = st.multiselect("Damages Incurred", ["Physical Injuries", "Emotional Distress", "Property Damage", "Financial Losses"])
        medical_bills_amount = st.number_input("Medical Bills Amount", min_value=0)
        medical_treatment = st.text_area("Medical Treatment Received")
        best_argument = st.text_area("What do you think is your best argument?")
        additional_comments = st.text_area("Additional Comments")

        st.session_state['user_data'].update({
            'case_type': case_type,
            'incident_date': incident_date.strftime("%Y-%m-%d"),
            'incident_location': f"{incident_city}, {incident_state}, {incident_country}",
            'incident_description': incident_description,
            'damages_incurred': damages_incurred,
            'medical_bills_amount': medical_bills_amount,
            'medical_treatment': medical_treatment,
            'best_argument': best_argument,
            'additional_comments': additional_comments
        })

    # Step 3: Uploads
    with tabs[2]:
        st.header("Step 3: Upload Documents and Images")
        uploaded_documents = st.file_uploader("Upload Documents", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        uploaded_images = st.file_uploader("Upload Images", type=['jpg', 'jpeg', 'png', 'gif'], accept_multiple_files=True)
        st.session_state['uploaded_documents'] = uploaded_documents
        st.session_state['uploaded_images'] = uploaded_images

    # Step 4: Consent
    with tabs[3]:
        st.header("Step 4: Consent and Acknowledgements")
        disclaimer_agreed = st.checkbox("I agree to the terms and conditions.", value=True)
        opt_in_contact = st.checkbox("I agree to be contacted by a lawyer regarding my case.", value=True)
        st.session_state['user_data'].update({
            'disclaimer_agreed': disclaimer_agreed,
            'opt_in_contact': opt_in_contact
        })

    # Step 5: Review and Submit
    with tabs[4]:
        st.header("Step 5: Review and Submit")
        st.write("Please review the information you have provided:")
        st.json(st.session_state['user_data'])
        if st.button("Generate AI Report"):
            if not st.session_state['report_generated']:
                with st.spinner("Processing your case..."):
                    # Process documents and images concurrently
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    document_summaries = loop.run_until_complete(process_documents(st.session_state['uploaded_documents']))
                    image_contexts = loop.run_until_complete(process_images(st.session_state['uploaded_images'], st.session_state['user_data']))
                    st.session_state['document_summaries'] = document_summaries
                    st.session_state['image_contexts'] = image_contexts

                    # Generate case info
                    case_info = loop.run_until_complete(generate_case_info(st.session_state['user_data'], document_summaries, image_contexts))
                    st.session_state['case_info'] = case_info

                    # MailChimp integration
                    if st.session_state['user_data']['opt_in_contact']:
                        success = add_to_mailchimp(st.session_state['user_data'], case_info)
                        if success:
                            st.success("You have been subscribed to our mailing list.")
                        else:
                            st.error("There was an error subscribing you to our mailing list.")

                    # Initialize chat interface
                    initialize_chat_interface(case_info, st.session_state['document_summaries'])

                    st.session_state['report_generated'] = True
                    st.success("Your AI report has been generated!")

    # Display results page
    if st.session_state['report_generated']:
        st.header("Case Analysis Summary")

        st.subheader("Case Overview")
        st.write(st.session_state['case_info']['case_summary'])

        st.subheader("Best Arguments")
        st.write(st.session_state['case_info']['best_arguments'])

        st.subheader("Relevant Laws")
        st.write(st.session_state['case_info']['relevant_laws'])

        st.subheader("Document Summaries")
        for doc_summary in st.session_state['document_summaries']:
            st.write(f"**{doc_summary['filename']}**")
            st.write(doc_summary['summary'])

        st.subheader("Image Analyses")
        for image_context in st.session_state['image_contexts']:
            st.write(f"**{image_context['filename']}**")
            st.write(image_context['context'])

        st.subheader("Potential Payout and Likelihood")
        st.write(f"**Estimated Potential Payout:** ${st.session_state['case_info']['potential_payout']}")
        st.write(f"**Likelihood of Winning:** {st.session_state['case_info']['likelihood_of_winning']}%")

        # Downloadable report
        if st.button("Download Full Report"):
            report_content = f"""
            Case Analysis Report

            Case Summary:
            {st.session_state['case_info']['case_summary']}

            Best Arguments:
            {st.session_state['case_info']['best_arguments']}

            Relevant Laws:
            {st.session_state['case_info']['relevant_laws']}

            Document Summaries:
            """
            for doc_summary in st.session_state['document_summaries']:
                report_content += f"\n\n{doc_summary['filename']}:\n{doc_summary['summary']}"

            report_content += f"\n\nPotential Payout: ${st.session_state['case_info']['potential_payout']}"
            report_content += f"\nLikelihood of Winning: {st.session_state['case_info']['likelihood_of_winning']}%"

            st.download_button(
                label="Download Report",
                data=report_content,
                file_name="Case_Analysis_Report.txt",
                mime="text/plain"
            )

        # Chat Interface
        st.header("Chat with Your Case")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        # Display chat messages
        for message in st.session_state['chat_history'][1:]:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")

        # User input
        user_input = st.text_input("Type your message here:")
        if user_input:
            st.session_state['chat_history'].append({'role': 'user', 'content': user_input})

            # Prepare messages for OpenAI API
            messages = st.session_state['chat_history']

            # API call
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.45,
                    max_tokens=500,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )

                assistant_message = response['choices'][0]['message']['content'].strip()

                # Append assistant message to history
                st.session_state['chat_history'].append({'role': 'assistant', 'content': assistant_message})

                # Display assistant message
                st.markdown(f"**Assistant:** {assistant_message}")
            except Exception as e:
                st.error(f"Error communicating with OpenAI: {str(e)}")

if __name__ == '__main__':
    main()
