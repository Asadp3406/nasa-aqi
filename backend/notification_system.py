"""
Smart Notification and Alert System
Provides real-time notifications for air quality changes and health alerts
"""

import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
from dataclasses import dataclass
import sqlite3
import os

@dataclass
class NotificationPreferences:
    """User notification preferences"""
    email: Optional[str] = None
    phone: Optional[str] = None
    push_enabled: bool = True
    email_enabled: bool = False
    sms_enabled: bool = False
    threshold_aqi: int = 100
    health_conditions: List[str] = None
    location_alerts: bool = True
    forecast_alerts: bool = True
    emergency_only: bool = False

class NotificationSystem:
    """
    Advanced notification system for air quality alerts
    """
    
    def __init__(self):
        self.db_path = "notifications.db"
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for user preferences and alert history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                email TEXT,
                phone TEXT,
                preferences TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                alert_type TEXT,
                message TEXT,
                aqi_value INTEGER,
                location TEXT,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                delivery_status TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subscription_zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                zone_name TEXT,
                latitude REAL,
                longitude REAL,
                radius_km REAL DEFAULT 10,
                active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_user(self, user_id: str, preferences: NotificationPreferences) -> Dict:
        """Register user with notification preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            preferences_json = json.dumps({
                "email": preferences.email,
                "phone": preferences.phone,
                "push_enabled": preferences.push_enabled,
                "email_enabled": preferences.email_enabled,
                "sms_enabled": preferences.sms_enabled,
                "threshold_aqi": preferences.threshold_aqi,
                "health_conditions": preferences.health_conditions or [],
                "location_alerts": preferences.location_alerts,
                "forecast_alerts": preferences.forecast_alerts,
                "emergency_only": preferences.emergency_only
            })
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences 
                (user_id, email, phone, preferences, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, preferences.email, preferences.phone, preferences_json))
            
            conn.commit()
            return {"success": True, "message": "User registered successfully"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            conn.close()
    
    def add_location_zone(self, user_id: str, zone_name: str, lat: float, lon: float, radius_km: float = 10) -> Dict:
        """Add a location zone for monitoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO subscription_zones 
                (user_id, zone_name, latitude, longitude, radius_km)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, zone_name, lat, lon, radius_km))
            
            conn.commit()
            return {"success": True, "message": f"Zone '{zone_name}' added successfully"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            conn.close()
    
    def check_alert_conditions(self, aqi_data: Dict, user_id: str = None) -> List[Dict]:
        """Check if alert conditions are met and generate notifications"""
        alerts = []
        
        # Safely extract AQI value and convert to int
        current_aqi = aqi_data.get("current_aqi", 0)
        if current_aqi is None:
            current_aqi = 0
        else:
            current_aqi = int(float(current_aqi))  # Handle numpy types
            
        city = str(aqi_data.get("city", "Unknown Location"))
        forecast = aqi_data.get("forecast", [])
        
        # Get user preferences
        if user_id:
            preferences = self._get_user_preferences(user_id)
        else:
            # Default preferences for anonymous users
            preferences = NotificationPreferences()
        
        # Check current AQI threshold
        if current_aqi >= preferences.threshold_aqi:
            alerts.append(self._create_threshold_alert(current_aqi, city, preferences))
        
        # Check for rapid deterioration
        deterioration_alert = self._check_rapid_deterioration(forecast, preferences)
        if deterioration_alert:
            alerts.append(deterioration_alert)
        
        # Check for health-specific alerts
        health_alerts = self._check_health_specific_alerts(aqi_data, preferences)
        alerts.extend(health_alerts)
        
        # Check for emergency conditions
        emergency_alert = self._check_emergency_conditions(current_aqi, city)
        if emergency_alert:
            alerts.append(emergency_alert)
        
        # Check forecast alerts
        if preferences.forecast_alerts:
            forecast_alerts = self._check_forecast_alerts(forecast, preferences)
            alerts.extend(forecast_alerts)
        
        return alerts
    
    def _get_user_preferences(self, user_id: str) -> NotificationPreferences:
        """Get user preferences from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT preferences FROM user_preferences WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            prefs_dict = json.loads(result[0])
            return NotificationPreferences(**prefs_dict)
        else:
            return NotificationPreferences()
    
    def _create_threshold_alert(self, aqi: int, city: str, preferences: NotificationPreferences) -> Dict:
        """Create threshold exceeded alert"""
        severity = self._get_alert_severity(aqi)
        
        return {
            "type": "threshold_exceeded",
            "severity": severity,
            "title": f"Air Quality Alert - {city}",
            "message": f"AQI has reached {aqi}, exceeding your threshold of {preferences.threshold_aqi}",
            "aqi": aqi,
            "location": city,
            "recommendations": self._get_recommendations_for_aqi(aqi),
            "timestamp": datetime.now().isoformat(),
            "priority": "high" if aqi > 150 else "medium"
        }
    
    def _check_rapid_deterioration(self, forecast: List[Dict], preferences: NotificationPreferences) -> Optional[Dict]:
        """Check for rapid air quality deterioration"""
        if len(forecast) < 6:
            return None
        
        # Check next 6 hours for rapid increase - safely convert AQI values
        aqi_values = []
        for item in forecast[:6]:
            aqi = item.get("aqi", 0)
            if aqi is None:
                aqi = 0
            aqi_values.append(int(float(aqi)))
        
        # Calculate rate of change
        if len(aqi_values) >= 3:
            recent_change = aqi_values[2] - aqi_values[0]  # Change over 2 hours
            
            if recent_change > 50:  # Rapid deterioration
                return {
                    "type": "rapid_deterioration",
                    "severity": "high",
                    "title": "Rapid Air Quality Deterioration",
                    "message": f"Air quality is rapidly worsening. AQI expected to increase by {recent_change} points in the next 2 hours.",
                    "forecast_change": recent_change,
                    "recommendations": [
                        "Consider moving indoor activities earlier than planned",
                        "Close windows and doors",
                        "Prepare air purifiers if available"
                    ],
                    "timestamp": datetime.now().isoformat(),
                    "priority": "high"
                }
        
        return None
    
    def _check_health_specific_alerts(self, aqi_data: Dict, preferences: NotificationPreferences) -> List[Dict]:
        """Check for health condition specific alerts"""
        alerts = []
        
        if not preferences.health_conditions:
            return alerts
        
        current_aqi = aqi_data.get("current_aqi", 0)
        health_assessment = aqi_data.get("health_assessment", {})
        
        for condition in preferences.health_conditions:
            threshold = self._get_health_condition_threshold(condition)
            
            if current_aqi >= threshold:
                alerts.append({
                    "type": "health_specific",
                    "severity": "high",
                    "title": f"Health Alert - {condition}",
                    "message": f"Current air quality (AQI {current_aqi}) may affect your {condition}. Take precautions.",
                    "health_condition": condition,
                    "aqi": current_aqi,
                    "specific_recommendations": self._get_health_specific_recommendations(condition, current_aqi),
                    "timestamp": datetime.now().isoformat(),
                    "priority": "high"
                })
        
        return alerts
    
    def _check_emergency_conditions(self, aqi: int, city: str) -> Optional[Dict]:
        """Check for emergency air quality conditions"""
        if aqi >= 300:  # Hazardous level
            return {
                "type": "emergency",
                "severity": "critical",
                "title": "AIR QUALITY EMERGENCY",
                "message": f"HAZARDOUS air quality in {city} (AQI {aqi}). Immediate action required!",
                "aqi": aqi,
                "location": city,
                "emergency_actions": [
                    "Stay indoors immediately",
                    "Seal all windows and doors", 
                    "Use high-efficiency air purifiers",
                    "Avoid all outdoor activities",
                    "Seek medical attention if experiencing symptoms"
                ],
                "timestamp": datetime.now().isoformat(),
                "priority": "critical"
            }
        
        return None
    
    def _check_forecast_alerts(self, forecast: List[Dict], preferences: NotificationPreferences) -> List[Dict]:
        """Check forecast for upcoming poor air quality periods"""
        alerts = []
        
        # Check next 24 hours
        for i, item in enumerate(forecast[:24]):
            aqi = item.get("aqi", 0)
            if aqi is None:
                aqi = 0
            else:
                aqi = int(float(aqi))  # Handle numpy types
            time = str(item.get("time", ""))
            
            if aqi >= preferences.threshold_aqi and i >= 2:  # At least 2 hours ahead
                alerts.append({
                    "type": "forecast_alert",
                    "severity": "medium",
                    "title": "Upcoming Air Quality Alert",
                    "message": f"Poor air quality expected at {time} (AQI {aqi})",
                    "forecast_time": time,
                    "forecast_aqi": aqi,
                    "hours_ahead": i,
                    "recommendations": [
                        "Plan indoor activities for this time period",
                        "Reschedule outdoor exercise",
                        "Prepare air purification systems"
                    ],
                    "timestamp": datetime.now().isoformat(),
                    "priority": "medium"
                })
                break  # Only send one forecast alert
        
        return alerts
    
    def send_notifications(self, alerts: List[Dict], user_id: str = None) -> Dict:
        """Send notifications through configured channels"""
        if not alerts:
            return {"success": True, "sent": 0}
        
        sent_count = 0
        failed_count = 0
        
        for alert in alerts:
            try:
                # Send push notification (simulated)
                self._send_push_notification(alert, user_id)
                
                # Send email if configured
                if user_id:
                    preferences = self._get_user_preferences(user_id)
                    if preferences.email_enabled and preferences.email:
                        self._send_email_notification(alert, preferences.email)
                
                # Log alert
                self._log_alert(alert, user_id)
                sent_count += 1
                
            except Exception as e:
                print(f"Failed to send notification: {e}")
                failed_count += 1
        
        return {
            "success": True,
            "sent": sent_count,
            "failed": failed_count,
            "total_alerts": len(alerts)
        }
    
    def _send_push_notification(self, alert: Dict, user_id: str = None):
        """Send push notification (simulated - would integrate with FCM/APNS)"""
        # This would integrate with Firebase Cloud Messaging or Apple Push Notifications
        print(f"PUSH NOTIFICATION: {alert['title']} - {alert['message']}")
    
    def _send_email_notification(self, alert: Dict, email: str):
        """Send email notification (simulated)"""
        # This would integrate with SendGrid, AWS SES, or SMTP
        print(f"EMAIL to {email}: {alert['title']} - {alert['message']}")
    
    def _log_alert(self, alert: Dict, user_id: str = None):
        """Log alert to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alert_history 
            (user_id, alert_type, message, aqi_value, location, delivery_status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id or "anonymous",
            alert["type"],
            alert["message"],
            alert.get("aqi", 0),
            alert.get("location", ""),
            "sent"
        ))
        
        conn.commit()
        conn.close()
    
    def _get_alert_severity(self, aqi: int) -> str:
        """Get alert severity based on AQI"""
        if aqi >= 300:
            return "critical"
        elif aqi >= 200:
            return "high"
        elif aqi >= 150:
            return "medium"
        else:
            return "low"
    
    def _get_recommendations_for_aqi(self, aqi: int) -> List[str]:
        """Get recommendations based on AQI level"""
        if aqi <= 50:
            return ["Great day for outdoor activities"]
        elif aqi <= 100:
            return ["Generally safe for outdoor activities", "Sensitive individuals should monitor symptoms"]
        elif aqi <= 150:
            return ["Limit prolonged outdoor activities", "Sensitive groups should stay indoors"]
        elif aqi <= 200:
            return ["Avoid outdoor activities", "Stay indoors with air purification"]
        elif aqi <= 300:
            return ["Emergency conditions - stay indoors", "Seek medical attention for symptoms"]
        else:
            return ["HAZARDOUS - Emergency conditions", "Stay indoors immediately", "Seek medical attention"]
    
    def _get_health_condition_threshold(self, condition: str) -> int:
        """Get AQI threshold for specific health conditions"""
        thresholds = {
            "asthma": 75,
            "copd": 75,
            "heart_disease": 100,
            "pregnancy": 100,
            "elderly": 100,
            "children": 75,
            "respiratory_conditions": 75,
            "cardiovascular_disease": 100
        }
        return thresholds.get(condition.lower(), 100)
    
    def _get_health_specific_recommendations(self, condition: str, aqi: int) -> List[str]:
        """Get health-specific recommendations"""
        base_recommendations = {
            "asthma": [
                "Keep rescue inhaler nearby",
                "Avoid outdoor exercise",
                "Stay in air-conditioned spaces",
                "Consider wearing N95 mask if going outside"
            ],
            "copd": [
                "Use prescribed medications as directed",
                "Stay indoors with air purification",
                "Monitor oxygen levels if available",
                "Contact healthcare provider if symptoms worsen"
            ],
            "heart_disease": [
                "Avoid physical exertion outdoors",
                "Monitor for chest pain or shortness of breath",
                "Stay in climate-controlled environments",
                "Contact doctor if experiencing symptoms"
            ],
            "pregnancy": [
                "Limit outdoor exposure",
                "Use air purifiers indoors",
                "Wear protective masks when necessary",
                "Consult healthcare provider about precautions"
            ]
        }
        
        return base_recommendations.get(condition.lower(), [
            "Limit outdoor activities",
            "Stay indoors when possible",
            "Use air purification systems",
            "Monitor health symptoms"
        ])
    
    def get_alert_history(self, user_id: str, days: int = 7) -> List[Dict]:
        """Get user's alert history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT alert_type, message, aqi_value, location, sent_at
            FROM alert_history 
            WHERE user_id = ? AND sent_at >= datetime('now', '-{} days')
            ORDER BY sent_at DESC
        '''.format(days), (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "type": row[0],
                "message": row[1],
                "aqi": row[2],
                "location": row[3],
                "timestamp": row[4]
            }
            for row in results
        ]