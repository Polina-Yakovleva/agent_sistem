"""Инструменты субагентов.

Flight Agent      — flight.py
Booking Agent     — booking.py
Compliance Agent  — compliance.py
"""

from app.tools.booking import (
    BOOKING_TOOLS,
    add_passenger,
    cancel_reservation,
    reserve_ticket,
)
from app.tools.compliance import (
    COMPLIANCE_TOOLS,
    check_visa_requirements,
    get_carrier_policy,
)
from app.tools.external import (
    EXTERNAL_TOOLS,
    get_weather,
    search_nearby_hotels,
)
from app.tools.flight import (
    FLIGHT_TOOLS,
    get_flight_details,
    get_flights,
    search_flights,
)

__all__ = [
    "get_flights",
    "get_flight_details",
    "search_flights",
    "FLIGHT_TOOLS",
    "add_passenger",
    "reserve_ticket",
    "cancel_reservation",
    "BOOKING_TOOLS",
    "check_visa_requirements",
    "get_carrier_policy",
    "COMPLIANCE_TOOLS",
    "search_nearby_hotels",
    "get_weather",
    "EXTERNAL_TOOLS",
]
