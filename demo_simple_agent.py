#!/usr/bin/env python3
"""
Demo script for the simple requirements gathering agent.
Shows how to use the agent in an interview-style conversation.
"""

import os
import sys
import json

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simple_requirements_agent import (
    invoke_agent,
    stream_agent,
    extract_json,
    generate_requirements,
)


def demo_interview():
    """Run an interactive interview-style conversation."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    print("=" * 70)
    print("TRAVEL REQUIREMENTS GATHERING AGENT")
    print("=" * 70)
    print("I'll interview you to gather your travel requirements.")
    print("I'll ask questions, validate your inputs, and check flight availability.")
    print("Type 'quit' to exit at any time.\n")

    thread_id = "interview_session"

    # Start the conversation with a greeting
    print("Agent: ", end="", flush=True)
    response = invoke_agent(
        "Hello! I need to gather some travel requirements. Please tell me about your trip.",
        thread_id,
    )
    print(response)

    # Continue the conversation
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

        # Check if we have final JSON
        json_data = extract_json(response)
        if json_data and "requirements" in json_data:
            print("\n" + "=" * 70)
            print("REQUIREMENTS GATHERED SUCCESSFULLY!")
            print("=" * 70)
            print(json.dumps(json_data, indent=2))
            print("\n✅ All requirements have been collected and validated.")
            break


def demo_streaming():
    """Demo streaming responses for better UX."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    print("=" * 70)
    print("STREAMING DEMO")
    print("=" * 70)

    thread_id = "stream_demo"

    # Example inputs for demo
    test_inputs = [
        "I need to plan a trip from San Francisco to New York",
        "Departing December 15, 2025, returning December 20. 2 adults, economy class.",
        "SFO and JFK airports. Total budget $2000, flights $1200, hotels $800. Interested in museums and theater.",
        "Yes, I confirm the flight options. Please proceed.",
    ]

    for user_input in test_inputs:
        print(f"\nYou: {user_input}")
        print("\nAgent (streaming): ", end="", flush=True)

        # Stream the response
        full_response = ""
        for chunk in stream_agent(user_input, thread_id):
            if "model" in chunk and "messages" in chunk["model"]:
                for message in chunk["model"]["messages"]:
                    if hasattr(message, "content") and message.content:
                        print(message.content, end="", flush=True)
                        full_response += message.content

        print()  # New line after streaming

        # Check for final JSON
        json_data = extract_json(full_response)
        if json_data and "requirements" in json_data:
            print("\n" + "=" * 70)
            print("FINAL REQUIREMENTS:")
            print("=" * 70)
            print(json.dumps(json_data, indent=2))
            break


def demo_quick_test():
    """Quick test with comprehensive input."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    print("=" * 70)
    print("QUICK TEST - One-shot comprehensive input")
    print("=" * 70)

    thread_id = "quick_test"

    # Comprehensive input
    user_input = (
        "I want to fly from Tokyo (NRT) to Seoul (ICN) on 2025-11-15. "
        "One-way trip, 1 adult, economy class. Non-stop preferred but not required. "
        "Total budget USD 1200, flights $700, hotels $500. "
        "Interested in food and culture. "
        "3-4 star hotels in central area, double room."
    )

    print(f"You: {user_input}\n")
    print("Agent: ", end="", flush=True)
    response = invoke_agent(user_input, thread_id)
    print(response)

    # If agent needs confirmation about flights
    if "confirm" in response.lower() or "proceed" in response.lower():
        print("\nYou: Yes, please proceed with the top flight option.\n")
        print("Agent: ", end="", flush=True)
        response = invoke_agent(
            "Yes, please proceed with the top flight option.", thread_id
        )
        print(response)

    # Extract final JSON
    json_data = extract_json(response)
    if json_data and "requirements" in json_data:
        print("\n" + "=" * 70)
        print("REQUIREMENTS JSON:")
        print("=" * 70)
        print(json.dumps(json_data, indent=2))


def demo_tool_usage():
    """Demo the generate_requirements tool directly."""

    print("=" * 70)
    print("GENERATE REQUIREMENTS TOOL DEMO")
    print("=" * 70)

    # Example of calling the tool directly
    result = generate_requirements(
        traveler_adults=2,
        traveler_children=0,
        trip_type="round_trip",
        origin_city="San Francisco",
        origin_airport_iata="SFO",
        destination_city="New York",
        destination_airport_iata="JFK",
        depart_date="2025-12-15",
        return_date="2025-12-20",
        cabin_class="economy",
        non_stop=False,
        max_layovers=1,
        date_flex_days=1,
        interests=["museums", "theater", "food"],
        total_currency="USD",
        total_amount=2000,
        flights_amount=1200,
        hotels_amount=800,
        hotel_stars="3-4",
        hotel_area="central",
        hotel_room_type="double",
        outbound_query={"from_iata": "SFO", "to_iata": "JFK", "date": "2025-12-15"},
        outbound_result={
            "available": True,
            "options": [{"carrier": "DemoAir", "price_usd": 600}],
        },
        return_query={"from_iata": "JFK", "to_iata": "SFO", "date": "2025-12-20"},
        return_result={
            "available": True,
            "options": [{"carrier": "DemoAir", "price_usd": 600}],
        },
        accept_outbound_top_option=True,
        notes="User confirmed both flights",
        missing_info=[],
    )

    print("Generated Requirements:")
    print(json.dumps(result, indent=2))


def main():
    """Main entry point - choose demo mode."""

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "stream":
            demo_streaming()
        elif mode == "quick":
            demo_quick_test()
        elif mode == "tool":
            demo_tool_usage()
        else:
            print("Usage: python demo_simple_agent.py [stream|quick|tool]")
            print("  No args: Interactive interview mode")
            print("  stream:  Demo streaming responses")
            print("  quick:   Quick test with comprehensive input")
            print("  tool:    Demo generate_requirements tool directly")
    else:
        # Default: interactive interview
        demo_interview()


if __name__ == "__main__":
    main()
