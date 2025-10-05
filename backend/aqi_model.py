import requests
import pandas as pd
from dateutil import parser
from datetime import datetime, timedelta
import math
import io
import base64

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available - charts will be disabled")

# EPA PM2.5 breakpoint table (µg/m3)
PM25_BREAKPOINTS = [
    (0.0,   12.0,   0,   50),   # Good
    (12.1,  35.4,  51,  100),   # Moderate
    (35.5,  55.4, 101, 150),    # Unhealthy for Sensitive Groups
    (55.5, 150.4, 151, 200),    # Unhealthy
    (150.5,250.4, 201, 300),    # Very Unhealthy
    (250.5,350.4, 301, 400),
    (350.5,500.4, 401, 500),
]

def geocode_city(city_name, email=None):
    """Get latitude and longitude for a city name"""
    params = {"q": city_name, "format": "json", "limit": 1}
    if email:
        params["email"] = email
    resp = requests.get("https://nominatim.openstreetmap.org/search", 
                       params=params, 
                       headers={"User-Agent":"aqi-frontend/1.0"})
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"No geocoding result for: {city_name}")
    lat = float(results[0]["lat"])
    lon = float(results[0]["lon"])
    display_name = results[0].get("display_name", city_name)
    return lat, lon, display_name

def aqi_from_concentration(c, breakpoints=PM25_BREAKPOINTS):
    """Convert PM2.5 concentration to AQI"""
    if c is None or (isinstance(c, float) and math.isnan(c)):
        return None
    if c < 0:
        c = 0.0
    for (C_lo, C_hi, I_lo, I_hi) in breakpoints:
        if C_lo <= c <= C_hi:
            aqi = ((I_hi - I_lo)/(C_hi - C_lo)) * (c - C_lo) + I_lo
            return int(round(aqi))
    C_lo, C_hi, I_lo, I_hi = breakpoints[-1]
    aqi = ((I_hi - I_lo)/(C_hi - C_lo)) * (min(c, C_hi) - C_lo) + I_lo
    return int(round(aqi))

def fetch_open_meteo_airquality(lat, lon, timezone="auto", hours=48):
    """Fetch air quality data from Open-Meteo API"""
    base = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide",
        "timezone": timezone
    }
    resp = requests.get(base, params=params, headers={"User-Agent":"aqi-frontend/1.0"})
    resp.raise_for_status()
    j = resp.json()
    if "hourly" not in j:
        raise RuntimeError("No hourly data in Open-Meteo response")
    
    hourly = j["hourly"]
    times = [parser.isoparse(t) for t in hourly["time"]]
    df = pd.DataFrame({"time": times})
    
    for key in ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide"]:
        if key in hourly:
            df[key] = hourly[key]
        else:
            df[key] = pd.NA
    
    now = datetime.now(times[0].tzinfo) if len(times)>0 else datetime.utcnow()
    df = df[df["time"] >= now]
    df = df.sort_values("time").head(hours).reset_index(drop=True)
    return df

def aqi_category(aqi):
    """Get AQI category from numeric value"""
    if aqi is None:
        return None
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    if aqi <= 500:
        return "Hazardous"
    return "Hazardous+"

def get_aqi_color(aqi):
    """Get color code for AQI value"""
    if aqi is None:
        return "#999999"
    if aqi <= 50:
        return "#00E400"  # Green
    if aqi <= 100:
        return "#FFFF00"  # Yellow
    if aqi <= 150:
        return "#FF7E00"  # Orange
    if aqi <= 200:
        return "#FF0000"  # Red
    if aqi <= 300:
        return "#8F3F97"  # Purple
    return "#7E0023"      # Maroon

def forecast_aqi(city_name, hours=48):
    """Main function to get AQI forecast for a city"""
    try:
        # Get coordinates
        lat, lon, display_name = geocode_city(city_name)
        
        # Fetch air quality data
        df = fetch_open_meteo_airquality(lat, lon, hours=hours)
        
        # Calculate AQI
        df["pm2_5_aqi"] = df["pm2_5"].apply(lambda x: aqi_from_concentration(float(x)) if pd.notna(x) else None)
        df["pm2_5_aqi_category"] = df["pm2_5_aqi"].apply(aqi_category)
        df["aqi_color"] = df["pm2_5_aqi"].apply(get_aqi_color)
        
        # Format time for display
        df["time_str"] = df["time"].dt.strftime("%Y-%m-%d %H:%M")
        
        # Create chart (optional)
        chart_data = None
        try:
            chart_data = create_aqi_chart(df, display_name)
        except Exception as e:
            print(f"Chart generation failed: {e}")
        
        # Prepare response - convert to native Python types
        current_aqi = None
        current_category = None
        if len(df) > 0:
            aqi_val = df.iloc[0]["pm2_5_aqi"]
            current_aqi = int(aqi_val) if pd.notna(aqi_val) else None
            cat_val = df.iloc[0]["pm2_5_aqi_category"]
            current_category = str(cat_val) if pd.notna(cat_val) else None
        
        forecast_data = []
        for _, row in df.iterrows():
            # Convert pandas/numpy types to native Python types for JSON serialization
            aqi_val = row["pm2_5_aqi"]
            pm25_val = row["pm2_5"]
            pm10_val = row["pm10"]
            
            forecast_data.append({
                "time": str(row["time_str"]),
                "aqi": int(aqi_val) if pd.notna(aqi_val) else None,
                "category": str(row["pm2_5_aqi_category"]) if pd.notna(row["pm2_5_aqi_category"]) else None,
                "color": str(row["aqi_color"]),
                "pm2_5": float(pm25_val) if pd.notna(pm25_val) else None,
                "pm10": float(pm10_val) if pd.notna(pm10_val) else None
            })
        
        return {
            "success": True,
            "city": str(display_name),
            "current_aqi": int(current_aqi) if current_aqi is not None else None,
            "current_category": str(current_category) if current_category is not None else None,
            "current_color": str(get_aqi_color(current_aqi)),
            "forecast": forecast_data,
            "chart": chart_data
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def create_aqi_chart(df, city_name):
    """Create AQI chart and return as base64 string"""
    if not MATPLOTLIB_AVAILABLE:
        return None
        
    try:
        plt.figure(figsize=(12, 6))
        
        # Simple line plot
        valid_data = df.dropna(subset=['pm2_5_aqi'])
        if len(valid_data) > 0:
            plt.plot(valid_data["time"], valid_data["pm2_5_aqi"], 
                    color='#667eea', linewidth=3, marker='o', markersize=4)
        
        # Add AQI level lines
        aqi_levels = [50, 100, 150, 200, 300]
        level_colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97']
        
        for level, color in zip(aqi_levels, level_colors):
            plt.axhline(y=level, color=color, linestyle='--', alpha=0.5, linewidth=1)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel("Time")
        plt.ylabel("AQI (PM2.5)")
        plt.title(f"48-Hour AQI Forecast for {city_name}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{chart_base64}"
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None

def forecast_aqi_by_coords(lat, lon, hours=48):
    """Get AQI forecast for specific coordinates"""
    try:
        # Use coordinates directly
        display_name = f"Location ({lat:.4f}, {lon:.4f})"
        
        # Try to get a better name using reverse geocoding
        try:
            resp = requests.get("https://nominatim.openstreetmap.org/reverse", 
                               params={"lat": lat, "lon": lon, "format": "json"},
                               headers={"User-Agent":"aqi-frontend/1.0"})
            if resp.status_code == 200:
                geo_data = resp.json()
                if geo_data.get('display_name'):
                    display_name = geo_data['display_name']
        except:
            pass  # Use coordinate-based name if reverse geocoding fails
        
        # Fetch air quality data
        df = fetch_open_meteo_airquality(lat, lon, hours=hours)
        
        # Calculate AQI
        df["pm2_5_aqi"] = df["pm2_5"].apply(lambda x: aqi_from_concentration(float(x)) if pd.notna(x) else None)
        df["pm2_5_aqi_category"] = df["pm2_5_aqi"].apply(aqi_category)
        df["aqi_color"] = df["pm2_5_aqi"].apply(get_aqi_color)
        
        # Format time for display
        df["time_str"] = df["time"].dt.strftime("%Y-%m-%d %H:%M")
        
        # Create chart (optional)
        chart_data = None
        try:
            chart_data = create_aqi_chart(df, display_name)
        except Exception as e:
            print(f"Chart generation failed: {e}")
        
        # Prepare response - convert to native Python types
        current_aqi = None
        current_category = None
        if len(df) > 0:
            aqi_val = df.iloc[0]["pm2_5_aqi"]
            current_aqi = int(aqi_val) if pd.notna(aqi_val) else None
            cat_val = df.iloc[0]["pm2_5_aqi_category"]
            current_category = str(cat_val) if pd.notna(cat_val) else None
        
        forecast_data = []
        for _, row in df.iterrows():
            # Convert pandas/numpy types to native Python types for JSON serialization
            aqi_val = row["pm2_5_aqi"]
            pm25_val = row["pm2_5"]
            pm10_val = row["pm10"]
            
            forecast_data.append({
                "time": str(row["time_str"]),
                "aqi": int(aqi_val) if pd.notna(aqi_val) else None,
                "category": str(row["pm2_5_aqi_category"]) if pd.notna(row["pm2_5_aqi_category"]) else None,
                "color": str(row["aqi_color"]),
                "pm2_5": float(pm25_val) if pd.notna(pm25_val) else None,
                "pm10": float(pm10_val) if pd.notna(pm10_val) else None
            })
        
        return {
            "success": True,
            "city": str(display_name),
            "coordinates": {"lat": float(lat), "lon": float(lon)},
            "current_aqi": int(current_aqi) if current_aqi is not None else None,
            "current_category": str(current_category) if current_category is not None else None,
            "current_color": str(get_aqi_color(current_aqi)),
            "forecast": forecast_data,
            "chart": chart_data
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }