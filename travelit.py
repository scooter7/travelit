import streamlit as st
import pandas as pd
from datetime import date
from textwrap import dedent
import re
import logging
from amadeus import Client, ResponseError

# AGNO imports (adjust if needed)
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
from agno.models.openai import OpenAIChat

# -----------------------------------------------------------------------------
# Configure Logging for Amadeus (Debug)
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG)

# -----------------------------------------------------------------------------
# 1. Streamlit & Amadeus Configuration
# -----------------------------------------------------------------------------
st.title("AI Travel Planner ✈️")
st.caption(
    "Plan your next adventure with AI Travel Planner by researching and planning "
    "a personalized itinerary on autopilot using GPT-4o and Amadeus for real-time travel data."
)

# Retrieve credentials from Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]
serp_api_key = st.secrets["serpapi"]["api_key"]
amadeus_client_id = st.secrets["amadeus"]["client_id"]
amadeus_client_secret = st.secrets["amadeus"]["client_secret"]

# Initialize Amadeus client with debug logging
amadeus = Client(
    client_id=amadeus_client_id,
    client_secret=amadeus_client_secret,
    log_level='debug'
)

# -----------------------------------------------------------------------------
# 2. Helper Functions (Sanitization, IATA code lookup, Parsing, Booking)
# -----------------------------------------------------------------------------
def sanitize_place_name(place_name: str) -> str:
    """
    Remove potentially problematic characters (quotes, braces, angle brackets)
    and strip leading/trailing whitespace.
    """
    place_name = place_name.strip()
    place_name = re.sub(r'[\"\'{}<>]', '', place_name)
    return place_name

def get_iata_code(amadeus_client: Client, place_name: str, sub_type: str = "CITY,AIRPORT") -> str:
    """
    Safely transform a user-input place name into an IATA code using Amadeus.
    """
    # 1. Sanitize user input
    place_name_clean = sanitize_place_name(place_name)
    if not place_name_clean:
        return None

    # 2. Attempt the location API call
    try:
        response = amadeus_client.reference_data.locations.get(
            keyword=place_name_clean,
            subType=sub_type
        )
        if response.data and len(response.data) > 0:
            return response.data[0].get("iataCode", None)
        else:
            return None
    except ResponseError as e:
        print(f"Location lookup error: {e}")
        return None
    except Exception as ex:
        print(f"Unexpected error in get_iata_code: {ex}")
        return None

def parse_flight_offers(flight_offers_data):
    flight_rows = []
    for offer in flight_offers_data:
        total_price = offer.get("price", {}).get("total", "N/A")
        itineraries = offer.get("itineraries", [])
        for itinerary in itineraries:
            segments = itinerary.get("segments", [])
            for segment in segments:
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
    rows = []
    offer_map = {}
    idx_counter = 0

    for hotel_offer in hotel_offers_data:
        hotel_info = hotel_offer.get("hotel", {})
        hotel_name = hotel_info.get("name", "N/A")
        city_code = hotel_info.get("cityCode", "N/A")

        for offer in hotel_offer.get("offers", []):
            offer_id = offer.get("id", "N/A")
            room_type = offer.get("room", {}).get("type", "N/A")
            room_desc = offer.get("room", {}).get("description", {}).get("text", "")
            price_total = offer.get("price", {}).get("total", "N/A")
            check_in_date = offer.get("checkInDate", "N/A")
            check_out_date = offer.get("checkOutDate", "N/A")

            rows.append({
                "Index": idx_counter,
                "Hotel Name": hotel_name,
                "City Code": city_code,
                "Offer ID": offer_id,
                "Room Type": room_type,
                "Room Description": room_desc,
                "Check-In": check_in_date,
                "Check-Out": check_out_date,
                "Total Price": price_total,
            })
            offer_map[idx_counter] = (offer_id, offer)
            idx_counter += 1

    df = pd.DataFrame(rows)
    return df, offer_map

def book_hotel_offer(offer_id, traveler_info, payment_info):
    payload = {
        "data": {
            "offerId": offer_id,
            "guests": [traveler_info],
            "payments": [payment_info],
        }
    }
    try:
        response = amadeus.booking.hotel_bookings.post(payload)
        return response.data
    except ResponseError as e:
        return {"error": str(e)}

# -----------------------------------------------------------------------------
# 3. AGNO Agents (Researcher & Planner)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 4. GPT Itinerary & Flight Search Section
# -----------------------------------------------------------------------------
st.subheader("GPT-4o Itinerary & Flight Offers")
destination_input = st.text_input("Destination (City Name)", "San Francisco")
origin_input = st.text_input("Departure (City Name or Airport)", "New York")
departure_date = st.date_input("Departure Date", date.today())
num_days = st.number_input("Trip Length (Days)", min_value=1, max_value=30, value=5)

if st.button("Generate Itinerary & Flight Offers"):
    with st.spinner("Processing..."):
        # A) Generate itinerary
        itinerary_response = planner.run(f"{destination_input} for {num_days} days", stream=False)
        st.markdown("### Draft Itinerary")
        st.write(itinerary_response.content)

        # B) Convert user input to IATA codes (with sanitization)
        origin_code = get_iata_code(amadeus, origin_input, sub_type="CITY,AIRPORT")
        if not origin_code:
            st.error(f"Could not find a valid IATA code for origin: {origin_input}")
            st.stop()

        destination_code = get_iata_code(amadeus, destination_input, sub_type="CITY,AIRPORT")
        if not destination_code:
            st.error(f"Could not find a valid IATA code for destination: {destination_input}")
            st.stop()

        # C) Flight Offers
        try:
            flight_offers = amadeus.shopping.flight_offers_search.get(
                originLocationCode=origin_code.upper(),
                destinationLocationCode=destination_code.upper(),
                departureDate=departure_date.strftime("%Y-%m-%d"),
                adults=1
            )
            if flight_offers.data:
                flight_df = parse_flight_offers(flight_offers.data)
                st.markdown("### Flight Offers (Parsed)")
                st.dataframe(flight_df)
            else:
                st.warning("No flight offers found. Try different parameters or a valid city/airport code.")
        except ResponseError as e:
            st.error(f"Error fetching flight offers: {e}")
            if hasattr(e, "response"):
                st.write("Full error response:")
                st.json(e.response)

# -----------------------------------------------------------------------------
# 5. Hotel Search (City-Based Only)
# -----------------------------------------------------------------------------
st.subheader("Hotel Search (City-Based)")

hotel_input = st.text_input("Hotel City (e.g. 'San Francisco' or 'SFO')", "San Francisco")
check_in = st.date_input("Check-In Date", date.today())
check_out = st.date_input("Check-Out Date", date.today())
adults = st.number_input("Number of Adults", min_value=1, value=2)
room_quantity = st.number_input("Number of Rooms", min_value=1, value=1)

if st.button("Search Hotels by City"):
    with st.spinner("Searching for hotels by city..."):
        # Convert user city name to a city code
        hotel_city_code = get_iata_code(amadeus, hotel_input, sub_type="CITY")
        if not hotel_city_code:
            st.error(f"Could not find a valid IATA code for hotel city: {hotel_input}")
            st.stop()

        try:
            search_response = amadeus.shopping.hotel_offers_search.get(
                cityCode=hotel_city_code.upper(),
                checkInDate=check_in.strftime("%Y-%m-%d"),
                checkOutDate=check_out.strftime("%Y-%m-%d"),
                adults=adults,
                roomQuantity=room_quantity,
                radius="5",
                radiusUnit="KM",
                paymentPolicy="NONE",
                includeClosed="false",
                bestRateOnly="true",
                view="FULL",
                sort="PRICE"
            )
            if search_response.data:
                df, offer_map = parse_hotel_offers(search_response.data)
                st.dataframe(df)
                # Store in session state for booking
                st.session_state["hotel_offers_df"] = df
                st.session_state["hotel_offers_map"] = offer_map
            else:
                st.warning("No hotel offers found. Try adjusting your search parameters.")
        except ResponseError as e:
            st.error(f"Hotel search error: {e}")
            if hasattr(e, "response"):
                st.write("Full error response:")
                st.json(e.response)

# -----------------------------------------------------------------------------
# 6. Hotel Booking Section
# -----------------------------------------------------------------------------
st.subheader("Hotel Booking")
with st.expander("Book a Hotel Offer"):
    if "hotel_offers_df" not in st.session_state:
        st.info("Please search for hotels by city first.")
    else:
        df = st.session_state["hotel_offers_df"]
        offer_map = st.session_state["hotel_offers_map"]

        # Let the user pick an offer by index
        offer_indices = df["Index"].tolist()
        selected_index = st.selectbox("Select Offer Index to Book", offer_indices)

        # Collect traveler & payment info
        st.markdown("#### Traveler Info")
        title = st.selectbox("Title", ["MR", "MRS", "MS"])
        first_name = st.text_input("First Name", "John")
        last_name = st.text_input("Last Name", "Doe")
        phone = st.text_input("Phone", "+33679278416")
        email = st.text_input("Email", "test@test.com")

        st.markdown("#### Payment Info (Demo)")
        card_vendor = st.selectbox("Card Vendor", ["VI", "MC", "AX"])  # Visa, Mastercard, Amex
        card_number = st.text_input("Card Number", "4111111111111111")
        expiry_date = st.text_input("Expiry Date (YYYY-MM)", "2025-08")

        if st.button("Book Selected Hotel Offer"):
            traveler_info = {
                "id": 1,
                "name": {
                    "title": title,
                    "firstName": first_name,
                    "lastName": last_name
                },
                "contact": {
                    "phone": phone,
                    "email": email
                }
            }
            payment_info = {
                "id": 1,
                "method": "creditCard",
                "card": {
                    "vendorCode": card_vendor,
                    "cardNumber": card_number,
                    "expiryDate": expiry_date
                }
            }

            offer_id, full_offer = offer_map[selected_index]
            with st.spinner("Attempting to book..."):
                booking_response = book_hotel_offer(offer_id, traveler_info, payment_info)
                st.markdown("### Booking Response")
                st.json(booking_response)
