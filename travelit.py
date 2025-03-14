import streamlit as st
import pandas as pd
from amadeus import Client, ResponseError

# --------------------------
# Amadeus Client Initialization
# --------------------------
amadeus = Client(
    client_id=st.secrets["amadeus"]["client_id"],
    client_secret=st.secrets["amadeus"]["client_secret"]
)

# --------------------------
# Helper Functions
# --------------------------
def parse_hotel_offers(hotel_offers_data):
    """
    Convert hotel offers JSON data into a user-friendly DataFrame.
    Returns a DataFrame and a mapping of row index -> (offerId, entire offer dict).
    """
    rows = []
    offer_map = {}  # store a reference to each offer by index

    index_counter = 0
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
                "Index": index_counter,
                "Hotel Name": hotel_name,
                "City Code": city_code,
                "Offer ID": offer_id,
                "Room Type": room_type,
                "Room Description": room_desc,
                "Check-In": check_in_date,
                "Check-Out": check_out_date,
                "Total Price": price_total,
            })
            # Keep track of the entire offer data
            offer_map[index_counter] = (offer_id, offer)
            index_counter += 1

    df = pd.DataFrame(rows)
    return df, offer_map


def book_hotel_offer(offer_id, traveler_info, payment_info):
    """
    Attempt to book a specific hotel offer using the Amadeus Booking API.
    :param offer_id: The offer ID from the search response
    :param traveler_info: A dict containing traveler info (name, contact, etc.)
    :param payment_info: A dict containing payment details
    :return: Booking confirmation or error message
    """
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

# --------------------------
# Streamlit UI
# --------------------------
st.title("Hotel Search & Booking (Amadeus Example)")

st.write(
    "Use this demo to search for hotel offers and then (theoretically) book one. "
    "You need a valid Amadeus account with booking enabled for the booking call to succeed."
)

# 1. Collect Hotel Search Parameters
with st.expander("Hotel Search"):
    city_code = st.text_input("City Code (e.g. 'PAR' for Paris)", "NYC")
    check_in = st.date_input("Check-In Date")
    check_out = st.date_input("Check-Out Date")
    adults = st.number_input("Number of Adults", min_value=1, value=2)
    room_quantity = st.number_input("Number of Rooms", min_value=1, value=1)

    if st.button("Search Hotels"):
        with st.spinner("Searching..."):
            try:
                search_response = amadeus.shopping.hotel_offers_search.get(
                    cityCode=city_code.upper(),
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

                if not search_response.data:
                    st.warning("No hotel offers found. Try different parameters.")
                else:
                    df, offer_map = parse_hotel_offers(search_response.data)
                    st.dataframe(df)
                    st.session_state["hotel_offers_df"] = df
                    st.session_state["hotel_offers_map"] = offer_map

            except ResponseError as e:
                st.error(f"Hotel search error: {e}")

# 2. Hotel Booking Flow
with st.expander("Book a Hotel Offer"):
    # Only proceed if we have a stored DataFrame of offers
    if "hotel_offers_df" not in st.session_state:
        st.info("Search for hotels first.")
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

        st.markdown("#### Payment Info (Test Environment Example)")
        card_vendor = st.selectbox("Card Vendor", ["VI", "MC", "AX"])  # Visa, Mastercard, Amex
        card_number = st.text_input("Card Number", "4111111111111111")
        expiry_date = st.text_input("Expiry Date (YYYY-MM)", "2025-08")

        if st.button("Book Selected Offer"):
            # Prepare traveler and payment dicts
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
            with st.spinner("Booking..."):
                booking_response = book_hotel_offer(offer_id, traveler_info, payment_info)
                st.write("**Booking Response:**")
                st.json(booking_response)
