import streamlit as st
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, TypedDict, Annotated
import pytz
from urllib.parse import parse_qs
from pytz import UTC
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from langchain_core.tools import tool
import os
import json
from langgraph.graph import END, START

# Timezone setup
IST = pytz.timezone('Asia/Kolkata')

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "tool_outputs" not in st.session_state:
    st.session_state.tool_outputs = []

# Custom CSS for better display
st.markdown("""
<style>
.event-card {
    border-left: 4px solid #4285F4;
    padding: 1rem;
    margin: 0.5rem 0;
    background-color: #f8f9fa;
    border-radius: 0.25rem;
}
.time-badge {
    background-color: #E8F0FE;
    color: #1967D2;
    padding: 0.25rem 0.5rem;
    border-radius: 1rem;
    font-size: 0.85rem;
    display: inline-block;
    margin-right: 0.5rem;
}
.assistant-message {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Google Calendar setup
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service():
    """Authenticate and return the Google Calendar service instance."""
    # Use Streamlit secrets for service account credentials
    service_account_info = {
        "type": st.secrets["google"]["type"],
        "project_id": st.secrets["google"]["project_id"],
        "private_key_id": st.secrets["google"]["private_key_id"],
        "private_key": st.secrets["google"]["private_key"],
        "client_email": st.secrets["google"]["client_email"],
        "client_id": st.secrets["google"]["client_id"],
        "auth_uri": st.secrets["google"]["auth_uri"],
        "token_uri": st.secrets["google"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["google"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["google"]["client_x509_cert_url"]
    }
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(service_account_info, SCOPES)
    return build('calendar', 'v3', credentials=credentials)

calendar_service = get_calendar_service()

# Define tools
@tool()
def today(tool_input: dict = None) -> dict:
    """Returns today's date and day in UTC."""
    now = datetime.now(UTC)
    return {
        "role": "system",
        "content": f"Today's date is {now.strftime('%Y-%m-%d')}, and it is a {now.strftime('%A')} (UTC)."
    }

@tool()
def list_events_tool(query: str = None) -> list:
    """
    Fetches events based on time range parameters.
    Format: "start_time=ISO_UTC_TIME&end_time=ISO_UTC_TIME"
    Returns list of events or message if none found.
    """
    # Default time range (now to 7 days later)
    now = datetime.now(UTC)
    time_min = now.isoformat()
    time_max = (now + timedelta(days=7)).isoformat()
    
    # Parse query parameters if provided
    if query and '=' in query:
        try:
            params = {k: v[0] for k, v in parse_qs(query).items()}
            time_min = params.get('start_time', time_min)
            time_max = params.get('end_time', time_max)
        except:
            return "Invalid query format. Use: start_time=ISO_TIME&end_time=ISO_TIME"
    
    # Fetch events
    try:
        events_result = calendar_service.events().list(
            calendarId=st.secrets["google"]["calendar_id"],
            timeMin=time_min,
            timeMax=time_max,
            maxResults=20,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        # Format and deduplicate events
        unique_events = []
        seen = set()
        for event in events:
            event_key = (
                event['start'].get('dateTime', event['start'].get('date')),
                event['summary']
            )
            if event_key not in seen:
                seen.add(event_key)
                unique_events.append({
                    'start': event['start'].get('dateTime', event['start'].get('date')),
                    'summary': event['summary'],
                    'end': event['end'].get('dateTime', event['end'].get('date'))
                })
        
        if not unique_events:
            return f"No events found between {time_min} and {time_max}"
        
        return unique_events
    
    except Exception as e:
        return f"Error fetching events: {str(e)}"

@tool()
def find_available_slots_tool(min_duration: int = 30) -> list:
    """
    Finds available time slots of at least min_duration minutes.
    Returns list of slots in UTC.
    """
    now = datetime.now(UTC)
    events = calendar_service.events().list(
        calendarId=st.secrets["google"]["calendar_id"],
        timeMin=now.isoformat(),
        maxResults=20,
        singleEvents=True,
        orderBy='startTime'
    ).execute().get('items', [])
    
    available_slots = []
    last_end = now
    
    for event in events:
        start = datetime.fromisoformat(event['start']['dateTime']).astimezone(UTC)
        end = datetime.fromisoformat(event['end']['dateTime']).astimezone(UTC)
        
        if start > last_end + timedelta(minutes=min_duration):
            available_slots.append({
                'start': last_end.isoformat(),
                'end': start.isoformat(),
                'duration_minutes': int((start - last_end).total_seconds() / 60)
            })
        last_end = max(last_end, end)
    
    # Add remaining time until end of day
    end_of_day = now.replace(hour=23, minute=59, second=59)
    if end_of_day > last_end + timedelta(minutes=min_duration):
        available_slots.append({
            'start': last_end.isoformat(),
            'end': end_of_day.isoformat(),
            'duration_minutes': int((end_of_day - last_end).total_seconds() / 60)
        })
    
    return available_slots

@tool()
def create_event_tool(summary: str, start_time: str, end_time: str) -> dict:
    """
    Creates a calendar event in UTC.
    Args:
        summary: Event title
        start_time: ISO format in UTC
        end_time: ISO format in UTC
    """
    try:
        # Validate times
        start = datetime.fromisoformat(start_time).astimezone(UTC)
        end = datetime.fromisoformat(end_time).astimezone(UTC)
        
        if start >= end:
            return {"error": "End time must be after start time"}
        if start < datetime.now(UTC):
            return {"error": "Cannot create events in the past"}
        
        event = {
            'summary': summary,
            'start': {'dateTime': start.isoformat(), 'timeZone': 'UTC'},
            'end': {'dateTime': end.isoformat(), 'timeZone': 'UTC'}
        }
        
        created_event = calendar_service.events().insert(
            calendarId=st.secrets["google"]["calendar_id"],
            body=event
        ).execute()
        
        return {
            "status": "success",
            "link": created_event.get("htmlLink"),
            "event_id": created_event.get("id")
        }
    
    except Exception as e:
        return {"error": str(e)}

# LangGraph setup
class State(TypedDict):
    messages: Annotated[List[SystemMessage | HumanMessage], add_messages]
    today_info: dict  # Injected by the `today` tool

# Initialize LLM and tools
llm = init_chat_model("groq:qwen-qwq-32b")
tools = [today, list_events_tool, find_available_slots_tool, create_event_tool]
llm_with_tools = llm.bind_tools(tools)

# Helper functions
def provide_today_info(state: State):
    """Inject today's date (and weekday) into the state."""
    state["today_info"] = today({})
    return state

def inject_additional_system_message(state: State):
    """Add the scheduling‐role system message into the conversation."""
    sys_msg = SystemMessage(
        content="""Your role is to schedule events based on user instructions. Follow these rules:
        1. Firstly check the todays date by using the above node called provide_today_info.
        2. Parse user instructions to extract the date, time range, and description.
        3. Assume the user's specified time is in their local timezone. For this workflow, use 'Asia/Kolkata' (UTC+5:30) as the default timezone.
        4. Convert the extracted date and time into ISO 8601 UTC format.
        5. ALWAYS CALL THE TOOLS IN ISO 8601 UTC FORMAT ONLY.Now to call tools in the following order:
        - `list_events_tool`: To check for existing events on the specified date.
        - `find_available_slots_tool`: To find gaps between events (optional if the user provides a specific time).
        - `create_event_tool`: To schedule the meeting, using the validated date and time.
        6. Use the current date provided by the system to calculate dates like tomorrow or next week etc.
        7. Provide the final output in precise and correct manner."""
    )
    state["messages"].append(sys_msg)
    return state

def call_llm_model(state: State):
    """Invoke the model with the accumulated messages (including tools)."""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build the LangGraph
graph_builder = StateGraph(State)
graph_builder.add_node("inject_today_info", provide_today_info)
graph_builder.add_node("inject_additional_message", inject_additional_system_message)
graph_builder.add_node("process_input", call_llm_model)

# Tool‐execution node
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Edges
graph_builder.add_edge(START, "inject_today_info")
graph_builder.add_edge("inject_today_info", "inject_additional_message")
graph_builder.add_edge("inject_additional_message", "process_input")

graph_builder.add_conditional_edges(
    "process_input",
    tools_condition  # routes to tools node when needed, else to END
)
graph_builder.add_edge("tools", "process_input")

graph = graph_builder.compile()

# Function to parse and display events from response
def display_events(response_text: str):
    """Enhanced event display with custom formatting"""
    if "Here are your scheduled events" not in response_text:
        return st.markdown(response_text)
    
    # Split response into parts
    parts = response_text.split("\n\n")
    header = parts[0]
    events_text = "\n\n".join(parts[1:-1])  # Skip the last part which is usually a note
    
    # Display header
    st.markdown(f"### {header}")
    
    # Parse and display each event
    events = [e for e in events_text.split("\n") if e.strip()]
    for i in range(0, len(events), 2):
        if i+1 >= len(events):
            break
            
        time_line = events[i].strip()
        summary_line = events[i+1].strip()
        
        # Extract time and summary
        time_part = time_line.split("**")[1] if "**" in time_line else time_line
        summary = summary_line.split("**")[1] if "**" in summary_line else summary_line
        
        # Display as card
        st.markdown(f"""
        <div class="event-card">
            <div><span class="time-badge">{time_part}</span></div>
            <h4>{summary}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # Display any notes
    if len(parts) > 2:
        st.info(parts[-1])

# Streamlit app layout
st.title("📅 Google Calendar AI Assistant")
st.markdown("""
Welcome to your smart calendar assistant! I can:
- Show your upcoming events
- Find available time slots
- Schedule new meetings
- Answer questions about your schedule
""")

# Sidebar for conversation history
st.sidebar.title("Conversation History")
for msg in st.session_state.conversation_history:
    if msg["role"] == "user":
        st.sidebar.markdown(f"**You**: {msg['content']}")
    else:
        st.sidebar.markdown(f"**Assistant**: {msg['content']}")

# Main chat interface
user_input = st.chat_input("Ask about your calendar (e.g. 'What's on my schedule tomorrow?')")

# Quick action buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Today's Events"):
        user_input = "What events do I have today?"
with col2:
    if st.button("Tomorrow's Events"):
        user_input = "What events do I have tomorrow?"
with col3:
    if st.button("Find Available Slots"):
        user_input = "Show me available slots tomorrow between 9am and 5pm"

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.conversation_history = []
    st.session_state.tool_outputs = []
    st.rerun()

# Instructions
with st.expander("💡 How to use this assistant"):
    st.markdown("""
    **Example commands:**
    - "What's on my calendar today?"
    - "Schedule a meeting with Alex tomorrow at 2pm for 1 hour"
    - "Find 30-minute slots available tomorrow"
    - "Do I have any conflicts on Friday?"
    
    **Features:**
    - Displays events in your local time (Asia/Kolkata)
    - Shows event durations clearly
    - Identifies scheduling conflicts
    - Provides direct calendar links for new events
    """)
    
if user_input:
    # Add user message to history
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Process the message with LangGraph
    try:
        # Convert conversation history to LangChain messages
        messages = []
        for msg in st.session_state.conversation_history[:-1]:  # Exclude the current message
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(SystemMessage(content=msg["content"]))
        
        # Add the new user message
        messages.append(HumanMessage(content=user_input))
        
        # Run the graph
        state = {"messages": messages}
        result = graph.invoke(state)
        
        # Extract the response
        last_message = result["messages"][-1].content
        tool_outputs = []
        
        # Check for tool outputs in the result
        if hasattr(result["messages"][-1], 'tool_calls'):
            for tool_call in result["messages"][-1].tool_calls:
                tool_outputs.append({
                    "tool_name": tool_call['name'],
                    "input": tool_call['args'],
                    "output": None  # You might want to capture actual tool outputs
                })
        
        # Update session state
        st.session_state.conversation_history.append({"role": "assistant", "content": last_message})
        st.session_state.tool_outputs = tool_outputs
        
        # Display assistant response
        with st.chat_message("assistant"):
            if "Here are your scheduled events" in last_message:
                display_events(last_message)
            else:
                st.write(last_message)
        
        # Show tool outputs if available
        if st.session_state.tool_outputs:
            with st.expander("🔧 Tool Execution Details"):
                for tool in st.session_state.tool_outputs:
                    st.json(tool)
                    
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        st.session_state.conversation_history.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.error(error_msg)
