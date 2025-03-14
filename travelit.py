import streamlit as st
import pandas as pd
from datetime import date
from textwrap import dedent
from amadeus import Client, ResponseError

# AGNO imports (adjust to your actual package structure & versions)
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
from agno.models.openai import OpenAIChat

# ---------------------------
# Streamlit App Configuration
# ---------------------------
st.title("AI Travel Planner ✈️")
st.caption(
    "Plan your next adventure with AI Travel Planner by researching and planning "
    "a personalized itinerary on autopilot using GPT-4o and Amadeus for real-time travel data."
)

# Retrieve API keys and Amadeus credentials from Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]
serp_api_key = st.secrets["serpapi"]["api_key"]
amadeus_client_id = st.secrets["amadeus"]["client_id"]
amadeus_client_secret = st.secrets["amadeus"]["client_secret"]

# ------------------
# Amadeus Client Init
# ------------------
amadeus = Client(
    client_id=amadeus_client_id,
    client_secret=amadeus_client_secret
)

# -----------------------------
# Define Parsing Helper Functions
# -----------------------------
def parse_flight_offers(flight_offers_data):
    """
    Convert flight offers JSON data into a user-friendly DataFrame.
    """
    flight_rows = []
    for offer in flight_offers_data:
        # Price info is usually in offer["price"]["total"]
        total_price = offer.get("price", {}).get("total", "N/A")
        
        # 'itineraries' holds the flight segments info
        itineraries = offer.get("itineraries", [])
        for itinerary in itineraries:
            segments = itinerary.get("segments", [])
            for segment in segments:
                # Extract relevant segment information
                carrier_code = segment.get("carrierCode", "N/A")
                departure = segment.get("departure", {})
                arrival = segment.get("arrival", {})

                departure_airport = departure.get("iataCode", "N/A")
                departure_time = departure.get("at", "N/A")
                arrival_airport = arrival.get("iataCode", "N/A")
                arrival_time = arrival.get("at", "N/A")

                flight_rows.append({
                    "Airline": carrier_code,
                    "Departure Airport": departure_airport,
                    "Departure Time": departure_time,
                    "Arrival Airport": arrival_airport,
                    "Arrival Time": arrival_time,
                    "Total Price": total_price
                })

    return pd.DataFrame(flight_rows)


def parse_hotel_offers(hotel_offers_data):
    """
    Convert hotel offers JSON data into a user-friendly DataFrame.
    """
    hotel_rows = []
    for offer in hotel_offers_data:
        # Example: top-level fields might include 'hotel' and 'offers'
        hotel_info = offer.get("hotel", {})
        hotel_name = hotel_info.get("name", "N/A")
        hotel_city = hotel_info.get("cityCode", "N/A")

        # 'offers' may be a list of different room/rate options
        offers_list = offer.get("offers", [])
        for room_offer in offers_list:
            room_type = room_offer.get("room", {}).get("type", "N/A")
            room_description = room_offer.get("room", {}).get("description", {}).get("text", "N/A")
            price = room_offer.get("price", {}).get("total", "N/A")

            hotel_rows.append({
                "Hotel Name": hotel_name,
                "City Code": hotel_city,
                "Room Type": room_type,
                "Room Description": room_description,
                "Price (Total)": price
            })

    return pd.DataFrame(hotel_rows)

# -----------------------------
# AGNO Agents (Researcher & Planner)
# -----------------------------
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

# -----------------------------
# Streamlit UI Inputs
# -----------------------------
destination = st.text_input("Where do you want to go?")
origin = st.text_input("Enter your departure airport IATA code (e.g., JFK)")
departure_date = st.date_input("Select your departure date", date.today())
num_days = st.number_input("How many days do you want to travel for?", min_value=1, max_value=30, value=7)

# -----------------------------
# Main Button / Action
# -----------------------------
if st.button("Generate Itinerary & Offers"):
    with st.spinner("Processing..."):
        # 1) Generate a draft itinerary using Planner
        itinerary_response = planner.run(f"{destination} for {num_days} days", stream=False)
        st.subheader("Draft Itinerary")
        st.write(itinerary_response.content)

        # 2) Fetch real-time flight offers from Amadeus
        try:
            flight_offers = amadeus.shopping.flight_offers_search.get(
                originLocationCode=origin.upper(),
                # NOTE: In production, you should map the city name to a valid IATA code (e.g. "SFO")
                destinationLocationCode=destination[:3].upper(),
                departureDate=departure_date.strftime("%Y-%m-%d"),
                adults=1
            )
            flight_df = parse_flight_offers(flight_offers.data)
            st.subheader("Flight Offers (Parsed)")
            st.dataframe(flight_df)
        except ResponseError as error:
            st.error(f"Error fetching flight offers: {error}")

        # 3) Fetch hotel offers from Amadeus
        try:
            hotel_offers = amadeus.shopping.hotel_offers_search.get(
                # NOTE: You must provide a valid city code, not just the first 3 letters of a city name
                cityCode=destination[:3].upper()
            )
            hotel_df = parse_hotel_offers(hotel_offers.data)
            st.subheader("Hotel Offers (Parsed)")
            st.dataframe(hotel_df)
        except ResponseError as error:
            st.error(f"Error fetching hotel offers: {error}")
