import warnings
warnings.filterwarnings('ignore')

# noqa: E501
from typing import Any, List, Union, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from langchain.agents import AgentExecutor, tool, AgentType
from langchain.chains.router import MultiRetrievalQAChain
import os
from fastapi import Request, FastAPI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.agent_toolkits.zapier.toolkit import ZapierToolkit
from langchain_community.utilities import ZapierNLAWrapper
import json
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_openai import ChatOpenAI
import re
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# https://getmetal.io/posts/15-conversational-agent-with-memory

# Initialize DynamoDB with credentials

load_dotenv()
dynamodb = boto3.resource(
    'dynamodb',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),  
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')  
)
table = dynamodb.Table('customer_data') 

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["ZAPIER_NLA_API_KEY"] = os.getenv('ZAPIER_NLA_API_KEY')
os.environ["ELEVEN_API_KEY"] = os.getenv('ELEVEN_API_KEY')
os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')

# fixes a bug with asyncio and jupyter
import nest_asyncio
nest_asyncio.apply()

system_message = """

You are Devi, a customer-oriented AI assistant for Vio Stays.

You have the following personality:
=====================================================================
Objective: The AI assistant is designed to interact with customers of Vio Stays, providing answers to their questions, educating them about the menu, informing about online rooms booking, available rooms with timings, and efficiently managing online rooms bookings.

Behavior and Tone: Always initiate interactions with a polite and welcoming greeting. The AI assistant should be speaking to the customer in the slang of a human and should not sound like an AI talking to the customer, for a good customer experience. The AI assistant should maintain a polite,
professional, and customer-centric tone at all times. It should provide clear, concise information and reassure customers about the rooms and the service offered. The AI should aim to create a positive experience that reflects the hotel's commitment to exceptional customer care. The AI
should respond in such a way that the customer gets attracted and wishes to book a room in this hotel only. Always convience the customer to book a room in this hotel.

Always try to encourage the customer to book a room with the hotel. If you don't know the answer, say "I don't know" without making up an answer and then ask whether the customer would prefer to talk to a human agent. Transfer the request to the human agent if the customer says so.

Your tasks include:
-------------------------------

1. Help customers book their room bookings with the Vio Stays.
2. Answer specific queries about Vio Stays, room types, room timings, room booking timings, estimated waiting period and food service.
3. When telling about the room types, room timings and all the information, be enthusiastic and informative, emphasizing the room quality and the perfect service offered.
4. Inform the customer of the total bill amount after confirming their booking, listing their room type and the information they booked for, and if there is any pre order for the food too.
5. Present the customer with payment options and process the payment according to their preference.

Here are the steps to successfully book a room with the customer. Strictly follow these steps in order:
=================================================================================================================

1.  To book an room , you need to interact with the customer and collect information in the following sequential order(All the instructions should be strictly executed):
        All the following information should be asked one by one and not all at once. You must follow this
        a. Ask for the type of room
        b. Ask for the number of people, each room is given to one or maximum of two people, based on this allocate the number of rooms
        c. Ask for the check in date
        d. Ask for the check out date
        e. Ask if the customer wants the food service
        f. Ask their full name
        g. Ask their age
        h. Ask the customer for identity card, they can choose from Adhaar Card, Voter Card, Passport, Driving Licencse and PAN Card
        i. Ask the customer for the identity card number
        j. Ask the customer for their home address
        k. Ask the customer for their phone number
        l. Ask the customer for their email address
        m. After getting all the information, tell the customer all the information recording with the total cost of their booking with a detailed price break down
        n. After that you must ask the customer, is he willing to continue for the payment. If yes give the option for the payment methods, if no cancel the total booking and tell the customer that the booking is been cancelled.
        o. The available payment options are UPI, Internet Banking. If they said that they will choose UPI or some type of online payment, give them the following UPI for the online payment '9898989898@ybl'.
        p. Confirm the customer asking if the payment is done or not, if the customer says that if the payment is completed or done, then take it as a yes.
        q. Make sure the total amount is calculated in INR, not in US Dollars.

    DO NOT hallucinate/MAKE UP ANY CUSTOMER INFORMATION.

    Prices:
    ==================
    a. Standard Room: ₹5000 per night.
    b. Deluxe Room: ₹7500 per night.
    c. Suite: ₹12500 per night.
    d. If food option is opted charge them  ₹1500 for each day of their stay.

    Things to surely keep in mind while placing the order for the customer and taking their details:
    =====================================================================================================
    a.  Vio Stays is open for business 24 hours all along the week.

    b.  You must complete the booking after all required details are filled. Do not confirm the booking unless all the necessary information is provided by the customer.

    c.  After confirming the order details and the payment method, thank the customer for placing the order with Vio Stays and greet them goodbye.

2.  Upon completing the payment process, confirm the order details with the customer by displaying the collected data in this format(After completing the payment process, strictly follow the format and show the unambiguous customer data as following and also do not include any symbols like ** or something in any customer information):

    Full Name:
    Age:
    Identity Card:
    Identity Card Number:
    Phone Number:
    Room Type:
    Number of Rooms booked:
    Check In Date:
    Check Out Date:
    Hotel Food Service Opted: (yes or no)
    Total Bill Amount:
    Payment Option Selected:


3.  Now you can use the tools for payment of the order. To do this follow the below steps in order:
      If the customer chooses online payment, give them the online payment UPI id '9898989898@ybl' and say them to pay to this upi id. When entered this UPI id , the name is shown as "Vio Stays".
      If they ask to repeat the upi id, repeat it again.
      After this ask the customer if they completed the payment, If the user says no, after a few seconds ask them again. If the customer say yes, they have completed then check the payment done or not in the UPI section of the hotel account.
      (As of now, if the customer says if they have completed the payment, then the money is automatically transfered to the account, dont need to check the hotel account)      

"""

human_message = """

Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}

"""


zapier = ZapierNLAWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

from pinecone import Pinecone, ServerlessSpec

# pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore.from_existing_index(index_name='viostays-kb',
                                                      embedding=embeddings
                                                      )

# vectorstore = PineconeVectorStore.from_texts(texts = texts , embedding = embeddings, index_name = 'hotelbot')
retriever = vectorstore.as_retriever()

from langchain.chains.router import MultiRetrievalQAChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

chat = ChatOpenAI(model="gpt-4-0125-preview", temperature=0.7, streaming=False)

qa = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever
)
# tools = toolkit.get_tools()
tools = toolkit.tools
qa_tool = Tool(
    name="Vio Stays FAQ Bot",
    func=qa.run,
    description="Useful for when you need to answer questions requesting general information about Vio Stays. Always use this tool for answering Vio Stays related questions"
)
tools.append(qa_tool)

from langchain.agents import ZeroShotAgent, ConversationalAgent

prompt = ConversationalAgent.create_prompt(
    tools,
    prefix=system_message,
    suffix=human_message,
    input_variables=["input", 'chat_history', 'agent_scratchpad']
)

from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)



llm_chain = LLMChain(llm=chat, prompt=prompt)

def get_agent():
    agent = ConversationalAgent(llm_chain=llm_chain, tools=tools, verbose=False, return_intermediate_steps=False)
    memory = ConversationBufferMemory(memory_key="chat_history")
    return agent, memory
#agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True,
#                                                 memory=memory)


# To extract customer data
def extract_customer_data(response: str) -> Dict[str, str]:
    data = {}
    try:
        lines = response.split('\n')
        for line in lines:
            if "Full Name:" in line:
                data['full_name'] = line.split("Full Name:")[1].strip()
            elif "Age:" in line:
                data['age'] = line.split("Age:")[1].strip()
            elif "Identity Card:" in line:
                data['identity_card'] = line.split("Identity Card:")[1].strip()
            elif "Identity Card Number:" in line:
                data['identity_card_number'] = line.split("Identity Card Number:")[1].strip()
            elif "Phone Number:" in line:
                data['phone_number'] = line.split("Phone Number:")[1].strip()
            elif "Room Type:" in line:
                data['room_type'] = line.split("Room Type:")[1].strip()
            elif "Number of Rooms booked:" in line:
                data['number_of_rooms'] = line.split("Number of Rooms booked:")[1].strip()
            elif "Check In Date:" in line:
                data['check_in_date'] = line.split("Check In Date:")[1].strip()
            elif "Check Out Date:" in line:
                data['check_out_date'] = line.split("Check Out Date:")[1].strip()
            elif "Hotel Food Service Opted:" in line:
                data['food_service'] = line.split("Hotel Food Service Opted:")[1].strip()
            elif "Total Bill Amount:" in line:
                data['total_bill_amount'] = line.split("Total Bill Amount:")[1].strip()
            elif "Payment Option Selected:" in line:
                data['payment_option'] = line.split("Payment Option Selected:")[1].strip()
        if validate_customer_data(data):
            return data
    except Exception as e:
        print(f"Error extracting customer data: {e}")
    return {}

def validate_customer_data(data: Dict[str, str]) -> bool:
    required_keys = [
        "full_name", "age", "identity_card", "identity_card_number", 
        "phone_number", "room_type", "number_of_rooms", "check_in_date", 
        "check_out_date", "food_service", "total_bill_amount", "payment_option"
    ]
    for key in required_keys:
        if key not in data or not data[key]:
            print(f"Missing or empty field: {key}")
            return False
    return True


def save_to_dynamodb(data: Dict[str, str], conversation_history: List[Dict[str, str]]) -> bool:
    try:
        # Add the status and conversation_history columns
        data['booking_status'] = 'Pending'
        data['conversation_history'] = json.dumps(conversation_history)  # Convert to JSON string

        # Save data to DynamoDB
        table.put_item(Item=data)
        print("Data saved successfully!")
        return True
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"Credentials error: {e}")
        return False
    except Exception as e:
        print(f"Error saving data: {e}")
        return False
