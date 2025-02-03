## RealTime AI-Powered Sales Assistant for Enhanced Conversation
## 🚀 Overview
The AI-Powered Sales Assistant is an intelligent, real-time speech analysis tool designed to enhance sales conversations. It uses advanced NLP and sentiment analysis techniques to provide insights, handle objections, and recommend products dynamically during sales calls. The system integrates with Google Sheets and CRM data for enhanced sales performance tracking and post-call analysis.

## 🔥 Features

- **🎙 Real-time Speech Analysis:** Convert speech to text using SpeechRecognition.
- **🧠 Sentiment Detection:** Analyze emotions in the conversation using NLP.
- **🎯 Objection Handling:** Identify and respond to customer objections dynamically.
- **📊 Product Recommendations:** Fetch relevant product suggestions based on CRM data.
- **📈 Dashboard & Visualization:** Generate insights with sentiment distribution pie charts and trend analysis.
- **📂 Google Sheets Integration:** Store and track sales conversations.
- **📝 Post-Call Summary:** Generate an overview of call performance and future recommendations.

## 🛠️ Installation

Follow these steps to set up the project locally:

1. **🔀 Clone the Repository:**
   ```bash
   git clone https://github.com/Bhagyabonam/ai_sales_assistant.git
   ```
2. **📥 Install Required System Dependencies:**
   **Version Notes:**
   - ✅ Python 3.10: Recommended version (best stability)
   - ❌ Python <3.10: Not Supported
     
   First, ensure you have the correct Python version:
   ```bash
   # Check Python version
   python --version  # Windows
   python3 --version  # macOS/Linux
   ```
   **Windows:**
   ```bash
   # Create virtual environment with Python
   python -m new_env myenv
   
   # Activate virtual environment
   new_env\Scripts\activate
   
   # Upgrade pip
   python -m pip install --upgrade pip

   # Verify Python version
   python --version  # Should show Python 3.10.x or 3.12.x
   ```

    **macOS/Linux:**
     ```bash
     # Create virtual environment with Python 3.10
     python3.10 -m venv venv
     
     # Activate virtual environment
     source venv/bin/activate
     
     # Upgrade pip
     pip3 install --upgrade pip
     
     # Verify Python version
     python3 --version  # Should show Python 3.10.x
     ```
     If you have multiple Python versions installed, you can also use these alternative paths:
     ```bash
     # Windows alternative
     C:\Python310\python -m new_env venv
  
     # macOS alternative (Intel)
     /usr/local/opt/python@3.10/bin/python3.10 -m new_env venv
  
     # macOS alternative (Apple Silicon)
     /opt/homebrew/opt/python@3.10/bin/python3.10 -m new_env venv
     ```
3. **📦Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **🔑 Configure Hugging Face:**

   a. Create a Hugging Face account at https://huggingface.co/
   
   b. Generate an access token:
      - Go to https://huggingface.co/settings/tokens
      - Click "New token"
      - Name your token and select "read" role
      - Copy the generated token
   
   c. Login using the CLI:

   **Windows:**
   ```bash
   huggingface-cli login
   # Enter your token when prompted
   ```

   **macOS/Linux:**
   ```bash
   huggingface-cli login
   # Enter your token when prompted
   ```
   d. Set up environment variable:
   
   **Windows:**
   ```bash
   setx HUGGING_FACE_TOKEN "your_token_here"
   ```

   **macOS/Linux:**
   ```bash
   echo "export HUGGING_FACE_TOKEN='your_token_here'" >> ~/.zshrc
   # OR for bash
   echo "export HUGGING_FACE_TOKEN='your_token_here'" >> ~/.bashrc
   ```

## 🔹Usage Guide
▶️ Run the Streamlit application using the following command:
     
 streamlit run main.py

 Once the application using the following command:
  
 - 🎙 Real-Time Audio Analysis
    
    Click the Start Listening button to begin speech recognition.
    
    The system transcribes audio, performs sentiment analysis, and recommends products dynamically.
    
    Say "stop" to exit the real-time analysis.
    
  - 📊 Dashboard & Visualization
    
    View Sentiment Distribution via a pie chart.
    
    Analyze Sentiment Trends with bar charts.
    
    Get Post-Call Insights for improved sales strategies.
    
  - 🕘 Query CRM Data
    
    Input a query to fetch relevant product recommendations from the CRM database.
  
  - ❓ Help Section
  
    Explains application functionalities and provides troubleshooting steps.
    
    Contact support for further assistance.
    
  - 💬 Feedback
      
    Submit feedback and access help within the app interface.
       
  
  
