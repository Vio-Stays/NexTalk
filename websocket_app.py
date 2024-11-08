from fastapi import WebSocket, FastAPI, WebSocketDisconnect, Request
from bot import get_agent , tools,  extract_customer_data, save_to_dynamodb
from langchain.agents import Tool, AgentExecutor, AgentType
import logging
#from twilio.twiml.voice_response import VoiceResponse
#import speech_recognition as sr
import numpy as np
#from voice_actions import int2float, validate

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Class defining socket events"""

    def __init__(self):
        """init method, keeping track of connections"""
        self.active_connections = []
        self.active_agents = []
        self.conversation_data = {}

    async def connect(self, websocket: WebSocket):
        """connect event"""
        await websocket.accept()
        self.active_connections.append(websocket)
        agent, memory = get_agent()
        agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False,
                                                         handle_parsing_errors=True,
                                                         memory=memory)
        self.active_agents.append(agent_chain)
        self.conversation_data[websocket] = []
        #self.active_connections.append(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Direct Message"""
        await websocket.send_text(message)

    async def disconnect(self, websocket: WebSocket):
        """disconnect event"""
        #self.active_connections.remove(websocket)
        my_index = self.active_connections.index(websocket)
        websocket_agent = self.active_agents[my_index]
        self.active_connections.remove(websocket)
        self.active_agents.remove(websocket_agent)
        del self.conversation_data[websocket]
        await websocket.close()

app = FastAPI(
    title="LangChain Websocket Server",
    version="1.0",
    description="Spin up a simple websocket server using FastAPI",
)

manager = ConnectionManager()

@app.get("/status")
async def root():
    return {"status": "healthy"}

@app.websocket("/webhooks")
async def websocket_endpoint(websocket: WebSocket):
    #print(websocket.query_params)
    await manager.connect(websocket)
    my_index = manager.active_connections.index(websocket)
    chat_agent = manager.active_agents[my_index]
    conversation = manager.conversation_data[websocket]
    
    try:
        while True:
            data = await websocket.receive_text()
            if data is None:
                continue
            
            # Store the incoming message in the conversation list
            conversation.append({"type": "customer", "message": data})
            
            response = chat_agent({'input': data})
            response_text = response['output']
            
            # Store the outgoing message in the conversation list
            conversation.append({"type": "agent", "message": response_text})
            
            print('===========')
            print(response_text)
            
            # Check if conversation is complete based on specific criteria in the response
            if "Payment Option Selected:" in response_text:
                customer_data = extract_customer_data(response_text)
                print('Printing customer data: \n')
                print(customer_data)
                
                # Save data to DynamoDB
                save_to_dynamodb(customer_data, conversation)
                
            await manager.send_personal_message(response['output'], websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        #await manager.send_personal_message("Bye!!!",websocket)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
