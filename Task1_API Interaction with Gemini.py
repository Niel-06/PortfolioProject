# To run this code you need to install the following dependencies:
# pip install google-genai
# pip install pip-system-certs (if running within corporate laptop)
#
# This script handles interactions with an AI model and includes logging to capture key information such as usage and error handling.
# It connects to the Gemini model using the free version.
# Other models may require different methods or SDKs for API integration, which should be considered if switching models.
# The API key is secured by storing it as an environment variable. An alternative approach would be to use a configuration file.


import os #Module to interact with operating system, getting files, directories and creating files.
import logging #Module to used for logging 
from google import genai #Module for google AI
from google.genai import types  #Module for google AI
from google.genai.errors import APIError #Module for google AI 
from datetime import datetime #Module to manage the datetime 
from httpx import ConnectError #Module to work within HTTP since we are using API

# --- Configuration ---

# 1. Setup Logging - this will be the file name for the log
LOG_FILE = "gemini_api_log.txt"

# 2. Define Model - based on the model name of Gemini
MODEL_NAME = "gemini-2.0-flash" 

# --- Main Logic ---

def run_gemini_chat():

#this section will check if the log file exist within the same directory of this project and will thrown an error if there's some issue on the access within the folder
    if not os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "a") as f:
                f.write(f"Log file created : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"Log file '{LOG_FILE}' created successfully.")
        except IOError as logfile_error:
            print(f"Error creating log file {LOG_FILE}: {logfile_error}")
            exit(1)

#this will setup the config for the log
    logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Chat Session Start: {timestamp}")

#variable to set for the usage log and the last response log    
    last_response = None # Variable to hold the last response for exit logging
    total_input_tokens = 0
    total_output_tokens = 0

#initialize the client that will connect to the Gemini API using API key
#This also enclosed within the try function to catch any error during runtime
    try:
        # 1. Client Initialization
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY")
        )
        
        # 2. Create the Chat Session
        chat_user = client.chats.create(
            model=MODEL_NAME
        ) 
        
        # 3. Main Chat Loop
        print("Welcome to the Gemini Chatbot! Type 'exit' to end the conversation.")

        #Identify if the user inputted on the chat and if they type exit this will close the connection
        while True:
            user_input = input(">: ")
            if user_input.lower() == "exit":
                print("Exiting Chatbot. Thanks!")
                break
            
            # Skip empty input
            if not user_input.strip():
                continue

            try:
                #this will provide the response based on the inputted text of the user
                response = chat_user.send_message(user_input)
                
                print(f"Gemini: {response.text}")
                
                # Update usage tracking for the current turn
                if response.usage_metadata:
                    usage = response.usage_metadata
                    total_input_tokens += usage.prompt_token_count
                    total_output_tokens += usage.candidates_token_count
                    
                last_response = response # Keep track of the last response
                
            except Exception as chatError:
                print(f"An error Occured: {chatError}")
                logging.error(f"Chat turn error: {chatError}")
                print("Please try again.")
                
        # 4. Log Total Session Usage upon exiting the loop
        if last_response and total_input_tokens > 0:
            log_message = (
                f"Model: {MODEL_NAME}, Turns: {len(chat_user.get_history()) // 2}\n"
                f"Token Usage - Total Input: {total_input_tokens}, "
                f"Total Output: {total_output_tokens}, "
                f"Total Session: {total_input_tokens + total_output_tokens}"
            )
            
            # Log Success and Total Token Usage
            logging.info(f"Chat Session END SUCCESS: {log_message}")
            logging.info(f"Last Response: {last_response.text[:100]}...")

        #If there's no message provided by user below warning message will shoo    
        elif total_input_tokens == 0:
             logging.warning("Chat Session ended without sending any messages.")

    # --- Error Handling Layer ---
    
    # Catch specific connection errors (like your SSL/proxy issue)
    except ConnectError as e:
        error_type = "CONNECTION ERROR"
        error_message = f"Failed to connect to API endpoint. Check network, proxy, and SSL configuration. Error: {e}"
        print(f"\n{error_type}: {error_message}")
        logging.error(f"{error_type}: {error_message}")

    # Catch general API errors (e.g., 400 Bad Request, 403 Permission Denied)
    except APIError as e:
        error_type = "GEMINI API ERROR"
        error_message = f"An API-specific error occurred. Status: {getattr(e, 'status_code', 'N/A')}. Details: {e}"
        print(f"\n{error_type}: {error_message}")
        logging.error(f"{error_type}: {error_message}")

    # Catch any other unexpected Python errors
    except Exception as e:
        error_type = "UNEXPECTED ERROR"
        error_message = f"An unhandled exception occurred: {type(e).__name__}: {e}"
        print(f"\n{error_type}: {error_message}")
        logging.critical(f"{error_type}: {error_message}")

    finally:
        logging.info(f"Chat Session End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

# Run the function
if __name__ == "__main__":
    run_gemini_chat()