#!/usr/bin/env python3
"""
Demo CLI for the Requirements Gathering Agent
Run with: python demo_single_agent.py
"""

import os
import sys
import json
from src.requirements_agent import RequirementsGatheringAgent


def main():
    """Run an interactive demo of the requirements gathering agent."""

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY environment variable")
        sys.exit(1)

    print("=" * 60)
    print("Requirements Gathering Agent Demo")
    print("=" * 60)
    print("This agent will help you gather travel requirements.")
    print("It will ask questions, validate inputs, and check flight availability.")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    # Initialize the agent
    agent = RequirementsGatheringAgent()

    # Use a fixed thread ID for the conversation
    thread_id = "demo_session"

    # Conversation loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for exit commands
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        try:
            # Get agent response
            print("\nAgent: ", end="", flush=True)
            response = agent.invoke(user_input, thread_id)
            print(response)

            # Try to extract and validate JSON if present
            json_data = agent.extract_first_json(response)
            if json_data and "requirements" in json_data:
                print("\n" + "=" * 60)
                print("REQUIREMENTS JSON DETECTED:")
                print("=" * 60)
                print(json.dumps(json_data, indent=2))

                # Validate the requirements
                if agent.validate_requirements(json_data):
                    print("\n✅ Requirements validated successfully!")
                else:
                    print(
                        "\n⚠️  Requirements validation failed - some fields may be missing or invalid"
                    )

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except (ValueError, RuntimeError) as e:
            print(f"\nError: {e}")
            print("Please try again or type 'quit' to exit.")


def demo_streaming():
    """Demo streaming responses (optional alternative mode)."""

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY environment variable")
        sys.exit(1)

    print("=" * 60)
    print("Requirements Agent - Streaming Mode Demo")
    print("=" * 60)

    agent = RequirementsGatheringAgent()
    thread_id = "stream_demo"

    # Example conversation
    test_messages = [
        "I want to fly from Tokyo (NRT) to Seoul (ICN) on 2025-11-15. 1 adult, economy. Non-stop is not required. Budget total USD 1200, flights 700, hotels 500. Interests food and culture.",
        "Yes, please proceed with the top outbound flight option. One-way trip only.",
    ]

    for msg in test_messages:
        print(f"\nYou: {msg}")
        print("\nAgent (streaming): ")

        # Stream the response
        for chunk in agent.stream(msg, thread_id):
            # Print model responses as they come
            if "model" in chunk and "messages" in chunk["model"]:
                for message in chunk["model"]["messages"]:
                    if hasattr(message, "content") and message.content:
                        print(message.content, end="", flush=True)
        print()  # New line after streaming


if __name__ == "__main__":
    # Run the main interactive demo by default
    # Uncomment the next line to run streaming demo instead
    # demo_streaming()
    main()
