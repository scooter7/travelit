from textwrap import dedent
import streamlit as st
from amadeus import Client, ResponseError
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
from agno.models.openai import OpenAIChat

# Set up the Streamlit app
st.title("AI Travel Planner ✈️")
st.caption("Plan your next adventure with AI Travel Planner by researching and planning a personalized itinerary on autopilot using GPT-4o and Amadeus for real-time travel data.")

# Retrieve API keys and Amadeus credentials from Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]
serp_api_key = st.secrets["serpapi"]["api_key"]
amadeus_client_id = st.secrets["amadeus"]["client_id"]
amadeus_client_secret = st.secrets["amadeus"]["client_secret"]

# Initialize Amadeus client
amadeus = Client(
    client_id=amadeus_client_id,
    client_secret=amadeus_client_secret
)

# Initialize the Researcher agent for travel research
researcher = Agent(
    name="Researcher",
    role="Searches for travel destinations, activities, and accommodations based on user preferences",
    model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
    description=dedent(
        """\
        You are a world-class travel researcher. Given a travel destination and the number of days the user wants to travel for,
        generate a list of search terms for finding relevant travel activities and accommodations.
        Then search the web for each term, analyze the results, and return the 10 most relevant results.
        """
    ),
    instructions=[
        "Given a travel destination and the number of days the user wants to travel for, first generate a list of 3 search terms related to that destination and the number of days.",
        "For each search term, `search_google` and analyze the results. "
        "From the results of all searches, return the 10 most relevant results to the user's preferences.",
        "Remember: the quality of the results is important.",
    ],
    tools=[SerpApiTools(api_key=serp_api_key)],
    add_datetime_to_instructions=True,
)

# Initialize the Planner agent for itinerary planning
planner = Agent(
    name="Planner",
    role="Generates a draft itinerary based on user preferences and research results",
    model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
    description=dedent(
        """\
        You are a senior travel planner. Given a travel destination, the number of days the user wants to travel for, and a list of research results,
        your goal is to generate a draft itinerary that meets the user's needs and preferences.
        """
    ),
    instructions=[
        "Given a travel destination, the number of days the user wants to travel for, and a list of research results, generate a draft itinerary that includes suggested activities and accommodations.",
        "Ensure the itinerary is well-structured, informative, and engaging.",
        "Ensure you provide a nuanced and balanced itinerary, quoting facts where possible.",
        "Remember: the quality of the itinerary is important.",
        "Focus on clarity, coherence, and overall quality.",
        "Never make up facts or plagiarize. Always provide proper attribution.",
    ],
    add_datetime_to_instructions=True,
)

# Input fields for the user's destination and travel dates
destination = st.text_input("Where do you want to go?")
origin = st.text_input("Enter your departure airport IATA code (e.g., JFK)")
departure_date = st.date_input("Select your departure date")
num_days = st.number_input("How many days do you want to travel for?", min_value=1, max_value=30, value=7)

if st.button("Generate Itinerary & Offers"):
    with st.spinner("Processing..."):
        # Get the itinerary using the Planner agent
        itinerary_response = planner.run(f"{destination} for {num_days} days", stream=False)
        st.subheader("Draft Itinerary")
        st.write(itinerary_response.content)
        
        # Fetch real-time flight offers from Amadeus
        try:
            flight_offers = amadeus.shopping.flight_offers_search.get(
                originLocationCode=origin.upper(),
                destinationLocationCode=destination[:3].upper(),  # This is a placeholder: ensure you provide the correct IATA code for the destination
                departureDate=departure_date.strftime("%Y-%m-%d"),
                adults=1
            )
            st.subheader("Flight Offers")
            st.write(flight_offers.data)
        except ResponseError as error:
            st.error(f"Error fetching flight offers: {error}")
        
        # Fetch hotel offers from Amadeus (example using a city code, adjust as necessary)
        try:
            hotel_offers = amadeus.shopping.hotel_offers_search.get(
                cityCode=destination[:3].upper()  # Again, ensure you provide a valid city code
            )
            st.subheader("Hotel Offers")
            st.write(hotel_offers.data)
        except ResponseError as error:
            st.error(f"Error fetching hotel offers: {error}")
