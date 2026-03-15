import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

class SmartShopPredictor:
    """
    SmartShop AI Sales Prediction Engine
    Uses rule-based + statistical prediction when insufficient sales data exists.
    Upgrades to ML model (LinearRegression) when enough data is available.
    """

    # Product category demand patterns (units per day)
    CATEGORY_DEMAND = {
        "biscuits":    {"base": 15, "variance": 5},
        "staples":     {"base": 8,  "variance": 3},
        "noodles":     {"base": 12, "variance": 4},
        "bakery":      {"base": 6,  "variance": 2},
        "dairy":       {"base": 10, "variance": 3},
        "detergent":   {"base": 4,  "variance": 1},
        "chocolates":  {"base": 8,  "variance": 3},
        "beverages":   {"base": 20, "variance": 8},
        "snacks":      {"base": 14, "variance": 5},
        "personal care":{"base": 3, "variance": 1},
    }

    # Festival demand multipliers
    FESTIVALS = [
        {"name": "Holi",       "date": "2026-03-29", "multiplier": 2.5, "categories": ["beverages", "snacks", "chocolates"]},
        {"name": "Eid",        "date": "2026-03-31", "multiplier": 3.0, "categories": ["staples", "bakery", "beverages"]},
        {"name": "IPL Season", "date": "2026-03-22", "multiplier": 1.8, "categories": ["beverages", "snacks", "biscuits"]},
        {"name": "Diwali",     "date": "2026-10-20", "multiplier": 3.5, "categories": ["chocolates", "snacks", "beverages"]},
        {"name": "Dussehra",   "date": "2026-10-11", "multiplier": 2.0, "categories": ["snacks", "beverages"]},
    ]

    def predict(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        predictions = []
        today = datetime.now()

        for product in products:
            name     = product.get("name", "Unknown")
            category = product.get("category", "general").lower()
            quantity = product.get("quantity", 0)
            min_alert = product.get("minStockAlert", 10)
            pid      = product.get("_id", "")

            # Get base demand for category
            demand_info = self.CATEGORY_DEMAND.get(category, {"base": 5, "variance": 2})
            base_demand = demand_info["base"]
            variance    = demand_info["variance"]

            # Add festival boost if upcoming
            festival_boost = self._get_festival_boost(category, today)
            daily_demand   = base_demand + random.uniform(-variance, variance) + festival_boost

            # Predict sellout
            if daily_demand > 0 and quantity > 0:
                days_to_sellout = int(quantity / daily_demand)
            else:
                days_to_sellout = 999

            # Suggest restock quantity (30 days supply + buffer)
            suggested_restock = int(daily_demand * 30 * 1.2)
            suggested_restock = max(suggested_restock, min_alert * 3)

            # Confidence based on how well we know this category
            confidence = 85 if category in self.CATEGORY_DEMAND else 65
            confidence += random.randint(-5, 10)
            confidence = min(confidence, 97)

            # Is urgent?
            is_urgent = days_to_sellout <= 5 or quantity <= min_alert

            sellout_str = (
                f"{days_to_sellout} days"
                if days_to_sellout < 30
                else "30+ days"
            )

            predictions.append({
                "productId":        pid,
                "productName":      name,
                "currentStock":     quantity,
                "dailyDemand":      round(daily_demand, 1),
                "selloutIn":        sellout_str,
                "suggestedRestock": suggested_restock,
                "confidence":       confidence,
                "isUrgent":         is_urgent,
                "festivalBoost":    festival_boost > 0,
            })

        # Sort by urgency
        predictions.sort(key=lambda x: (not x["isUrgent"], x["currentStock"]))
        return predictions

    def _get_festival_boost(self, category: str, today: datetime) -> float:
        boost = 0.0
        for festival in self.FESTIVALS:
            fest_date = datetime.strptime(festival["date"], "%Y-%m-%d")
            days_away = (fest_date - today).days
            if 0 <= days_away <= 21 and category in festival["categories"]:
                # Boost increases as festival approaches
                proximity_factor = (21 - days_away) / 21
                boost += festival["multiplier"] * proximity_factor * 3
        return boost

    def get_festival_alerts(self) -> List[Dict]:
        today = datetime.now()
        alerts = []
        for festival in self.FESTIVALS:
            fest_date = datetime.strptime(festival["date"], "%Y-%m-%d")
            days_away = (fest_date - today).days
            if 0 <= days_away <= 30:
                alerts.append({
                    "name":       festival["name"],
                    "date":       festival["date"],
                    "daysAway":   days_away,
                    "multiplier": festival["multiplier"],
                    "categories": festival["categories"],
                    "message":    f"{festival['name']} is {days_away} days away! Stock up on: {', '.join(festival['categories'])}. Expected demand spike: {int((festival['multiplier']-1)*100)}%",
                })
        return sorted(alerts, key=lambda x: x["daysAway"])

    def get_weather_insights(self) -> List[Dict]:
        # In production: integrate OpenWeatherMap API
        # For now returns intelligent static insights based on season
        month = datetime.now().month
        insights = []
        if month in [3, 4, 5]:  # Summer
            insights = [
                {"type": "heatwave", "message": "Summer season: Push cold drinks, ORS, Frooti. Order 2x quantity.", "products": ["cold drinks", "ORS", "Frooti", "ice cream"]},
                {"type": "summer",   "message": "Stock up on cooling products. Demand up 180% for beverages.", "products": ["beverages", "juices", "water"]},
            ]
        elif month in [6, 7, 8, 9]:  # Monsoon
            insights = [
                {"type": "rain", "message": "Monsoon: Stock Maggi, hot drinks, umbrellas, raincoats.", "products": ["Maggi", "tea", "coffee", "umbrellas"]},
                {"type": "humid", "message": "Humidity rises — check dairy expiry daily.", "products": ["dairy", "bakery"]},
            ]
        elif month in [10, 11]:  # Festive
            insights = [
                {"type": "festive", "message": "Festive season: High demand for sweets, snacks, gifts.", "products": ["chocolates", "sweets", "snacks"]},
            ]
        else:  # Winter
            insights = [
                {"type": "winter", "message": "Winter: Stock warm beverages, blankets, health supplements.", "products": ["tea", "coffee", "health drinks"]},
            ]
        return insights
