import chromadb
from chromadb.config import Settings
from chromadb import Client
from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd
import numpy as np
import streamlit as st
import speech_recognition as sr
from textblob import TextBlob
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import torch
import faiss
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import zipfile


SPREADSHEET_ID = "1CsBub3Jlwyo7WHMQty6SDnBShIZMjl5XTVSoOKrxZhc"
RANGE_NAME = 'Sheet1!A1:B1'
SERVICE_ACCOUNT_FILE = r"C:\Users\bhagy\AI\credentials.json"


csv_file_path = r"C:\Users\bhagy\OneDrive\Desktop\INFOSYS PROJECT\900_products_dataset.csv"


class CustomEmbeddingFunction:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings

# Initialize components
sentiment_pipeline = pipeline("sentiment-analysis")
chroma_client = Client(Settings(persist_directory="chromadb_storage"))
embedding_fn = CustomEmbeddingFunction()
collection_name = "crm_data"

try:
    collection = chroma_client.get_collection(collection_name)
except Exception:
    collection = chroma_client.create_collection(collection_name)

def get_google_sheets_service():
    credentials = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build('sheets', 'v4', credentials=credentials)

def update_google_sheet(response, sentiment):
    """
    Writes the AI response and sentiment to Google Sheets.
    """
    try:
        service = get_google_sheets_service()
        values = [[str(response), str(sentiment)]]
        body = {'values': values}
        result = service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=RANGE_NAME,
            valueInputOption="RAW",
            body=body
        ).execute()
        st.success("Response and sentiment written to Google Sheets!")
    except Exception as e:
        st.error(f"Failed to update Google Sheets: {e}")



def analyze_sentiment_combined(text):
  
    textblob_polarity = TextBlob(text).sentiment.polarity

    huggingface_result = sentiment_pipeline(text)[0]
    huggingface_label = huggingface_result['label']
    huggingface_score = huggingface_result['score']
    print("huggingface_score:", huggingface_score)
    textblob_normalized_score = (textblob_polarity + 1) / 2
    print("textblob_normalized_score:", textblob_normalized_score)
    combined_score = (textblob_normalized_score + huggingface_score) / 2
    print("combined_score:", combined_score)
    # Determine final sentiment
    if combined_score > 0.6:
        return "Positive", combined_score
    elif combined_score < 0.4:
        return "Negative", combined_score
    else:
        return "Neutral", combined_score


def generate_response(prompt):
    analysis = TextBlob(prompt) 
    sentiment = analysis.sentiment.polarity 
    if sentiment > 0: 
        return "Positive", sentiment
    elif sentiment < 0:
        return "Negative", sentiment
    else: 
        return "Neutral", sentiment



def load_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        if data is not None:
            st.session_state.crm_data = data  
            print("CRM data loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

data = load_csv(csv_file_path)

    
def process_crm_data(data):
    try:
        chunks = [str(row) for row in data.to_dict(orient="records")]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        embeddings = [embedding_fn(chunk) for chunk in chunks]
        
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=ids
        )
        print(f"Processed and stored {len(chunks)} CRM records.")
        print("CRM data processed and stored successfully!")
    except Exception as e:
        st.error(f"Error processing CRM data: {e}")

product_keywords = ['phone', 'smartphone', 'mobile', 'tablet', 'laptop', 'cell phone', 'headphones', 'smartwatch','vivo','xiaomi','sony','Apple','Oppo','Realme','Asus','Nokia','Lenovo','Samsung','Google','Motorola','OnePlus','Huawei',]


def query_crm_data_with_context(prompt, top_k=3):

    try:
        prompt_embedding = embedding_fn(prompt)
        collection = chroma_client.get_collection("crm_data")
        results = collection.query(
            query_embeddings=[prompt_embedding],
            n_results=top_k
        )
        matched_keywords = [kw for kw in product_keywords if kw in prompt.lower()]

        if not matched_keywords:
            return ["No relevant recommendations found as no product names were mentioned in the query."]
        relevant_docs = []
        for doc in results["documents"][0]:
            if any(kw in doc.lower() for kw in matched_keywords):
                relevant_docs.append(doc)
        return relevant_docs if relevant_docs else ["No relevant recommendations found for the mentioned products."]
    except Exception as e:
        st.error(f"Error querying CRM data: {e}")
        return ["Error in querying recommendations."]



sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
faiss_index = faiss.IndexFlatL2(384)

def load_objection_responses(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        objection_response_pairs = dict(zip(df['Objection'], df['Response']))
        return objection_response_pairs
    except Exception as e:
        print(f"Error loading objections CSV: {e}")
        return {}

objection_response_pairs = load_objection_responses(r"C:\Users\bhagy\OneDrive\Desktop\INFOSYS PROJECT\objections_responses.csv")
objections = list(objection_response_pairs.keys())
objection_embeddings = sentence_model.encode(objections)
faiss_index.add(np.array(objection_embeddings, dtype="float32"))

def find_closest_objection(query):
    query_embedding = sentence_model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding, dtype="float32"), 1)
    closest_index = indices[0][0]
    closest_objection = objections[closest_index]
    response = objection_response_pairs[closest_objection]
    if distances[0][0] > 0.6:
        return "No objection found", "No Response"
    return closest_objection, response

def handle_objection_and_recommendation(prompt):
    closest_objection, objection_response = find_closest_objection(prompt)
    recommendations = query_crm_data_with_context(prompt)

    return closest_objection, objection_response, recommendations


if "is_listening" not in st.session_state:
    st.session_state.is_listening = False

if "sentiment_history" not in st.session_state:
    st.session_state.sentiment_history = []

if "crm_data" not in st.session_state:
    st.session_state.crm_data = load_csv(csv_file_path)
else:
    print("CRM data already loaded from session state.")

if st.session_state.crm_data is not None:
    process_crm_data(st.session_state.crm_data)
else:
    st.error("Failed to load CRM data.")

if "crm_history" not in st.session_state:
    st.session_state["crm_history"] = []

if "app_feedback" not in st.session_state:
    st.session_state["app_feedback"] = []


def add_to_sentiment_history(text, sentiment_label, sentiment_score, closest_objection, response):
    st.session_state.sentiment_history.append({
        "Text": text,
        "Sentiment": sentiment_label,
        "Score": sentiment_score,
    })

def show_help():
    
    st.title("Help Section - AI-Powered Assistant for Live Sales Calls")

    st.header("1. Introduction to the AI Assistant")
    st.write("""
        - **What It Does**: The assistant analyzes live sales calls in real-time. It detects sentiment shifts, provides product recommendations, and suggests dynamic question handling techniques.
        - **Key Features**:
            - Real-time speech-to-text conversion and sentiment analysis.
            - Product recommendations based on customer context.
            - Dynamic question prompt generator.
            - Objection handling suggestions.
    """)


    st.header("2. Getting Started")
    st.write("""
        - **How to Start a Call**: To start a sales call, Click on Start Listening. Once connected, initiate the call, and the assistant will begin analyzing.
        - **What to Expect**: During the call, the assistant will provide real-time feedback, such as sentiment scores, product recommendations, and objection handling tips.
    """)

    st.header("3. Using the Assistant During Sales Calls")
    st.write("""
        - **Speech-to-Text Instructions**: Speak clearly into your microphone for the assistant to accurately capture and analyze your speech.
        - **Real-time Feedback**: The assistant will display real-time feedback on the sentiment of the conversation, suggest responses for objections, and provide product recommendations.
    """)


    st.header("4. Understanding the Interface")
    st.write("""
        - **Tabs Navigation**: The interface has different tabs:
            - **Call Summary**: After the call, review the summary, which highlights conversation key points.
            - **Sentiment Analysis**: See how the sentiment changed throughout the conversation.
            - **Product Recommendations**: View the recommended products based on customer intent and conversation context.
    """)


    st.header("5. FAQs and Troubleshooting")
    st.write("""
        - **Sentiment Detection Accuracy**: If the assistant's sentiment analysis isn't accurate, ensure you speak clearly and avoid background noise.
        - **Speech Recognition Issues**: Rephrase unclear statements and ensure the microphone is working well.
        - **Context Handling**: If the assistant misses some context, remind it of the product or the customer’s intent.
    """)


    st.header("6. Support and Contact Information")
    st.write("""
        - **Live Chat Support**: Chat with us in real-time by clicking the support icon in the bottom right.
        - **Email and Phone Support**: You can also reach us at support@aisupport.com or call us at +1-800-555-1234.
        - **Feedback**: Please provide feedback to help us improve the assistant.
    """)

    st.header("7. Advanced Features")
    st.write("""
        - **Integration with CRM and Google Sheets**: Sync with CRM systems and Google Sheets to enhance product recommendations.
        - **Customization Options**: Customize the assistant’s tone, product categories, and question prompts through the settings tab.
    """)

    st.header("8. Privacy and Security")
    st.write("""
        - **Data Privacy**: All conversations are anonymized for analysis purposes. We ensure compliance with privacy regulations.
        - **Security Protocols**: All data is encrypted and stored securely.
    """)


    st.header("9. Updates and New Features")
    st.write("""
        - **Changelog**: We release regular updates to improve performance. Please refer to the changelog for new features and improvements.
        - **How to Update**: If an update is available, follow the instructions in the settings tab to install the latest version.
    """)


def process_real_time_audio():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    st.write("Adjusting microphone for ambient noise... Please wait.")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    st.write("Listening for audio... Speak into the microphone.")
    while True:
        try:
            with microphone as source:
                audio = recognizer.listen(source, timeout=15, phrase_time_limit=20)

            st.write("Transcribing audio...")
            transcribed_text = recognizer.recognize_google(audio)
            st.write(f"You said: {transcribed_text}")

            if 'stop' in transcribed_text.lower():
                st.warning("Stopping the speech recognition process.")
                break

            st.markdown("### **Sentiment Analysis**")
            sentiment_label, sentiment_score = analyze_sentiment_combined(transcribed_text)
            st.write(f"Sentiment: {sentiment_label}")
            st.write(f"Sentiment Score: {sentiment_score}")

            closest_objection = None
            response = None

            add_to_sentiment_history(transcribed_text, sentiment_label, sentiment_score, closest_objection, response)
            st.markdown("### **Recommendations**")
            recommendations = query_crm_data_with_context(transcribed_text)
            for i, rec in enumerate(recommendations, start=1):
                if isinstance(rec, dict) and 'Product' in rec and 'Recommendations' in rec:
                    st.markdown(f"- **{rec['Product']}**: {rec['Recommendations']}")
                else:
                    st.markdown(f"- {rec}")

            st.markdown("### **Objection Handling**")
            closest_objection, response = find_closest_objection(transcribed_text)
            st.write(f"Objection: {closest_objection}")
            st.write(f" Response: {response}")

            update_google_sheet(f"Recommendations: {recommendations}", "N/A")

        except sr.UnknownValueError:
            st.warning("Could not understand the audio.")
        except Exception as e:
            st.error(f"Error: {e}")
            break

def generate_sentiment_pie_chart(sentiment_history):
    if not sentiment_history:
        st.warning("No sentiment history available to generate a pie chart.")
        return


    sentiment_counts = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }

    for entry in sentiment_history:
        sentiment_counts[entry["Sentiment"]] += 1

   
    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()
    colors = ['#6dcf6d', '#f76c6c', '#6c8df7']  

   
    fig, ax = plt.subplots()
    plt.figure(figsize=(6,6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,textprops={'fontsize':12, 'color':'white'})
    fig.patch.set_facecolor('none')
    ax.axis('equal')  
    st.markdown("### *Sentiment Distribution*")
    st.pyplot(fig)

def generate_post_call_summary(sentiment_history, recommendations=[]): 
    
    if not sentiment_history:
        st.warning("No sentiment history available to summarize.")
        return  
    df = pd.DataFrame(sentiment_history)
    st.write(df)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    combined_text = " ".join([item["Text"] for item in sentiment_history])

    summary = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
    scores = [item["Score"] for item in sentiment_history]
    average_sentiment_score = sum(scores) / len(scores)

    if average_sentiment_score > 0.05:
        overall_sentiment = "Positive"
    elif average_sentiment_score < -0.05:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral" 

    st.markdown("## Summary of the Call")
    st.write(summary)

    st.markdown("### **Overall Sentiment for the Call**")
    st.write(f"Overall Sentiment: {overall_sentiment}")
    st.write(f"Average Sentiment Score: {average_sentiment_score:.2f}")
    sentiment_scores = df["Score"].values

    col1,col2=st.columns(2)
    with col1:
        colors = ['green' if entry["Sentiment"] == "Positive" else 'red' if entry["Sentiment"] == "Negative" else 'blue' for entry in sentiment_history]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sentiment_scores)), sentiment_scores, color=colors)
        plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Neutral')
        st.markdown("### **Sentiment Trend Bar Chart**")
        plt.title("Sentiment Trend Throughout the Call")
        plt.xlabel("Segment")
        plt.ylabel("Sentiment Score")
        plt.legend(["Neutral"])
        plt.grid(axis='y', linestyle='--', linewidth=0.7)
        st.pyplot(plt)

    with col2:
        generate_sentiment_pie_chart(sentiment_history)

    st.markdown("### **Future Insights**")
    
    
    if overall_sentiment == "Negative":
        st.write("Consider addressing customer pain points more directly. More empathy might improve the sentiment.")
    elif overall_sentiment == "Positive":
        st.write("Great engagement! Continue the positive experience by offering more personalized recommendations.")
    else:
        st.write("The call was neutral. Identifying specific customer concerns can help drive a more positive outcome.")

   
    if recommendations:
        st.write("### **Product Recommendations**")
        for rec in recommendations:
            st.write(f"- {rec}")

    if sentiment_history:
        st.write("### **Sentiment Breakdown by Segment**")
        for idx, entry in enumerate(sentiment_history, 1):
            st.write(f"Segment {idx}: Sentiment = {entry['Sentiment']}, Score = {entry['Score']:.2f}")

# Main
def main():
    
    st.set_page_config(page_title="AI-Powered Sales Assistant", layout="wide")
    st.title("🤖 AI-Powered Sales Assistant")
    st.markdown(
        "An intelligent assistant to analyze speech, handle objections, and recommend products in real-time."
    )

    # Tabs for navigation
    tabs = st.tabs(["🎙️ Real-Time Audio", "📊 Text Search ", "📋 Visualization","🕘 Query History","❓Help","💬 Feedback"])

    
    with tabs[0]:
        st.header("🎙️ Real-Time Audio Analysis")
        st.write(
            "Use this feature to analyze live speech, perform sentiment analysis, and get product recommendations."
        )

        if st.button("Start Listening"):
            process_real_time_audio()

    
    with tabs[1]:
        st.header("📊 Search")
        st.write(
            "Retrieve the most relevant product recommendations based on your input query."
        )
        query = st.text_input("Enter your query:")
        recommendations=[]
        if st.button("Submit Query"):
            if query:
                
                result = query_crm_data_with_context(query)  
                st.success(f"Query submitted: {query}")
                
            if result:
                recommendations = result
                st.markdown("### Recommendations")
                for i, rec in enumerate(recommendations, start=1):
                    st.markdown(f"- {rec}")
            else:
                st.error("Please enter a query!")

            st.session_state["crm_history"].append({"Query": query, "Result": recommendations})
    
    with tabs[2]:
        st.header("📊 Dashboard")
        st.write("Visualize the sentiment analysis results.")
        generate_post_call_summary(st.session_state.sentiment_history)

    with tabs[3]:
        st.subheader("🕘 Query History")
        if "crm_history" in st.session_state and st.session_state["crm_history"]:
            st.subheader("Query History")
            st.dataframe(st.session_state["crm_history"])

    with tabs[4]:
        # st.subheader("❓Help")
        show_help()
    
    with tabs[5]:
        st.subheader("💬 App Feedback")
        
        feedback = st.text_area("We would love to hear your feedback on the app! Please share your thoughts:")

        if st.button("Submit Feedback") and feedback:
            
            st.session_state["app_feedback"].append(feedback)
            st.success("Thank you for your feedback!")
        
        # Display previous feedback
        if st.session_state["app_feedback"]:
            st.write("### Previous Feedback:")
            for idx, feedback_entry in enumerate(st.session_state["app_feedback"], 1):
                st.markdown(f"{idx}. {feedback_entry}")
        else:
            st.warning("No feedback submitted yet.")
        
        feedback = st.radio("Was this helpful?", ["Yes", "No"])
        st.button("Sumbit")

    file_path = csv_file_path  
    data = load_csv(file_path)

    
if __name__ == "__main__":
    main()