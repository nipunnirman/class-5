"""
Simple Requirements Gathering Agent using LangGraph's create_agent
Non-class based implementation following the react_agent.ipynb pattern
"""

import os
import re
import json
from typing import Literal
from datetime import datetime

import requests
from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


# Get Convex base URL from environment or use default
CONVEX_BASE = os.getenv("CONVEX_BASE", "https://standing-fish-574.convex.site")


class FlightAvailabilityArgs(BaseModel):
    """Arguments for flight availability tool"""

    from_iata: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Origin airport IATA code, e.g., NRT",
    )
    to_iata: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Destination airport IATA code, e.g., ICN",
    )
    date: str = Field(..., description="Flight date in YYYY-MM-DD")
    passengers: int = Field(
        ..., ge=1, description="Total passengers (adults + children)"
    )
    cabin: Literal["economy", "premium", "business"] = "economy"
    non_stop: bool = False

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date is in YYYY-MM-DD format"""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}") from exc
        return v


@tool("flight_availability", args_schema=FlightAvailabilityArgs)
def flight_availability(
    from_iata: str,
    to_iata: str,
    date: str,
    passengers: int,  # noqa: ARG001
    cabin: str,  # noqa: ARG001
    non_stop: bool,  # noqa: ARG001
) -> dict:
    """
    Check if flights are available on a given date between two airports.
    Returns available options sorted by price.

    NOTE: Convex ignores passengers/cabin/non_stop, but we include them to match the spec.
    """
    params = {"origin": from_iata, "destination": to_iata, "date": date}

    try:
        r = requests.get(f"{CONVEX_BASE}/flights/search", params=params, timeout=20)
        r.raise_for_status()
        flights = r.json().get("flights", [])
    except requests.RequestException as e:
        return {"available": False, "error": str(e), "options": []}

    options = []
    for f in flights[:5]:  # Limit to 5 options
        depart_iso = f"{f['flightDate']}T{f['departureTime']}:00"
        arrive_iso = f"{f['flightDate']}T{f['arrivalTime']}:00"
        options.append(
            {
                "carrier": f.get("airline", "Unknown"),
                "flight_number": f.get("flightNumber", "N/A"),
                "depart_iso": depart_iso,
                "arrive_iso": arrive_iso,
                "stops": 0,  # Dataset doesn't expose stops
                "price_usd": f.get("price", 0),
            }
        )

    return {"available": len(options) > 0, "options": options}


class GenerateRequirementsArgs(BaseModel):
    """Arguments for generate_requirements tool"""

    traveler_adults: int = Field(..., ge=1, description="Number of adults")
    traveler_children: int = Field(..., ge=0, description="Number of children")
    trip_type: str = Field(..., description="Trip type: 'one_way' or 'round_trip'")
    origin_city: str = Field(..., description="Origin city name")
    origin_airport_iata: str = Field(
        ..., min_length=3, max_length=3, description="Origin airport IATA code"
    )
    destination_city: str = Field(..., description="Destination city name")
    destination_airport_iata: str = Field(
        ..., min_length=3, max_length=3, description="Destination airport IATA code"
    )
    depart_date: str = Field(..., description="Departure date in YYYY-MM-DD format")
    return_date: str = Field(
        None, description="Return date in YYYY-MM-DD format (null for one-way)"
    )
    cabin_class: str = Field(
        ..., description="Cabin class: economy, premium, or business"
    )
    non_stop: bool = Field(..., description="Non-stop preference")
    max_layovers: int = Field(..., ge=0, description="Maximum number of layovers")
    date_flex_days: int = Field(..., ge=0, description="Date flexibility in days")
    interests: list = Field(..., description="List of interests")
    total_currency: str = Field(..., description="Currency code (e.g., USD)")
    total_amount: int = Field(..., ge=0, description="Total budget amount")
    flights_amount: int = Field(..., ge=0, description="Flight budget amount")
    hotels_amount: int = Field(..., ge=0, description="Hotel budget amount")
    hotel_stars: str = Field("", description="Hotel star rating preference")
    hotel_area: str = Field("", description="Hotel area preference")
    hotel_room_type: str = Field("", description="Hotel room type preference")
    outbound_query: dict = Field(None, description="Outbound flight query details")
    outbound_result: dict = Field(None, description="Outbound flight search results")
    return_query: dict = Field(None, description="Return flight query details")
    return_result: dict = Field(None, description="Return flight search results")
    accept_outbound_top_option: bool = Field(
        ..., description="Whether user accepted outbound flight"
    )
    notes: str = Field("", description="Additional notes from user")
    missing_info: list = Field(
        default_factory=list, description="List of missing information"
    )


@tool("generate_requirements", args_schema=GenerateRequirementsArgs)
def generate_requirements(
    traveler_adults: int,
    traveler_children: int,
    trip_type: str,
    origin_city: str,
    origin_airport_iata: str,
    destination_city: str,
    destination_airport_iata: str,
    depart_date: str,
    return_date: str = None,
    cabin_class: str = "economy",
    non_stop: bool = False,
    max_layovers: int = 1,
    date_flex_days: int = 0,
    interests: list = None,
    total_currency: str = "USD",
    total_amount: int = 0,
    flights_amount: int = 0,
    hotels_amount: int = 0,
    hotel_stars: str = "",
    hotel_area: str = "",
    hotel_room_type: str = "",
    outbound_query: dict = None,
    outbound_result: dict = None,
    return_query: dict = None,
    return_result: dict = None,
    accept_outbound_top_option: bool = False,
    notes: str = "",
    missing_info: list = None,
) -> dict:
    """
    Generate the final structured requirements JSON response.
    Call this when you have gathered all required information and user has confirmed flight options.
    """
    if interests is None:
        interests = []
    if missing_info is None:
        missing_info = []

    requirements = {
        "requirements": {
            "traveler": {"adults": traveler_adults, "children": traveler_children},
            "trip": {
                "type": trip_type,
                "origin": {"city": origin_city, "airport_iata": origin_airport_iata},
                "destination": {
                    "city": destination_city,
                    "airport_iata": destination_airport_iata,
                },
                "depart_date": depart_date,
                "return_date": return_date,
            },
            "preferences": {
                "cabin_class": cabin_class,
                "non_stop": non_stop,
                "max_layovers": max_layovers,
                "date_flex_days": date_flex_days,
                "interests": interests,
            },
            "budget": {
                "total_currency": total_currency,
                "total_amount": total_amount,
                "flights_amount": flights_amount,
                "hotels_amount": hotels_amount,
            },
            "hotel_prefs": {
                "stars": hotel_stars,
                "area": hotel_area,
                "room_type": hotel_room_type,
            },
            "flight_check": {
                "outbound_query": outbound_query,
                "outbound_result": outbound_result,
                "return_query": return_query,
                "return_result": return_result,
            },
            "user_confirmations": {
                "accept_outbound_top_option": accept_outbound_top_option,
                "notes": notes,
            },
            "missing_info": missing_info,
        }
    }

    return requirements


# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize checkpointer for memory
checkpointer = InMemorySaver()

# Create the agent with tools and checkpointer
agent = create_agent(
    model=model,
    tools=[flight_availability, generate_requirements],
    checkpointer=checkpointer,
)

# System prompt for requirements gathering
SYSTEM_PROMPT = """You are a **Requirements-Gathering Agent** for a travel assistant.

Your job is to interview the user and capture every required field to plan a trip later.

You **do not** create an itinerary. You only gather and validate inputs, then **verify flight availability** for the given dates.

Follow these rules:

1. **Collect & confirm** the following fields (ask clarifying questions if missing/ambiguous):
    - Traveler profile: number of adults/children; citizenship (optional); special needs (optional).
    - Trip basics: origin city/airport, destination city/airport, trip type (one-way/round-trip), departure date, return date (if round-trip).
    - Preferences: cabin class (economy/premium/business), non-stop preference, max layovers (0/1/2+), date flexibility (± days).
    - Budget: **total** budget, **flight** budget, **hotel** budget (rough figures are fine).
    - Interests/scenery: e.g., nature, beaches, food, culture, shopping, nightlife (select 2–5).
    - Hotel prefs (optional): star range, area vibe (central/quiet/near beach), room type (single/double/family).
2. **Validate**: date formats (ISO YYYY-MM-DD), that departure ≤ return (if round-trip), origin ≠ destination, traveler counts ≥ 1.
3. **When dates & airports/cities are known**, call the **flight_availability** tool once per direction (outbound and return if applicable).
4. **If available options exist**, read back a **concise top option** (carrier, times, price) and ask for **confirmation** ("Do you want to proceed with this flight as your preferred option?").
5. **When you have gathered all required information and user has confirmed flight options**, call the **generate_requirements** tool to create the final structured response.

Remember to ASK questions to gather missing information before making tool calls."""

# Create prompt template
prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{input}")]
)


def invoke_agent(user_input: str, thread_id: str = "default"):
    """
    Invoke the agent with user input and return response.

    Args:
        user_input: The user's message
        thread_id: Thread ID for conversation memory

    Returns:
        The agent's response as a string
    """
    messages = prompt.invoke({"input": user_input}).to_messages()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}  # type: ignore

    response = agent.invoke({"messages": messages}, config)

    # Extract the last assistant message
    if response["messages"]:
        last_message = response["messages"][-1]
        return (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )
    return "No response generated"


def stream_agent(user_input: str, thread_id: str = "default"):
    """
    Stream the agent's response for real-time display.

    Args:
        user_input: The user's message
        thread_id: Thread ID for conversation memory

    Yields:
        Stream chunks from the agent
    """
    messages = prompt.invoke({"input": user_input}).to_messages()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}  # type: ignore

    for chunk in agent.stream({"messages": messages}, config):
        yield chunk


def extract_json(text: str):
    """Extract JSON object from text if present."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# Example usage function
def run_conversation():
    """Run an interactive conversation with the agent."""
    print("=" * 60)
    print("Requirements Gathering Agent")
    print("=" * 60)
    print("I'll help you gather travel requirements. Type 'quit' to exit.\n")

    thread_id = "session_1"

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        print("\nAgent: ", end="", flush=True)
        response = invoke_agent(user_input, thread_id)
        print(response)

        # Check if JSON requirements are in the response
        json_data = extract_json(response)
        if json_data and "requirements" in json_data:
            print("\n" + "=" * 60)
            print("FINAL REQUIREMENTS JSON:")
            print("=" * 60)
            print(json.dumps(json_data, indent=2))
            break


if __name__ == "__main__":
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY environment variable")
    else:
        run_conversation()
