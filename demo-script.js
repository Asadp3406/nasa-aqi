// Demo version for GitHub Pages (works without backend)
// This version simulates API responses for demonstration

// Demo data for different cities
const demoData = {
    'new delhi': {
        success: true,
        city: 'New Delhi, Delhi, India',
        current_aqi: 164,
        current_category: 'Unhealthy',
        current_color: '#FF0000',
        coordinates: { lat: 28.6139, lon: 77.2090 },
        forecast: generateDemoForecast(164),
        tempo_integration: generateDemoTEMPO(),
        ml_enhancement: generateDemoML(),
        health_assessment: generateDemoHealth('high')
    },
    'london': {
        success: true,
        city: 'London, England, UK',
        current_aqi: 42,
        current_category: 'Good',
        current_color: '#00E400',
        coordinates: { lat: 51.5074, lon: -0.1278 },
        forecast: generateDemoForecast(42),
        tempo_integration: generateDemoTEMPO(),
        ml_enhancement: generateDemoML(),
        health_assessment: generateDemoHealth('low')
    },
    'beijing': {
        success: true,
        city: 'Beijing, China',
        current_aqi: 187,
        current_category: 'Unhealthy',
        current_color: '#FF0000',
        coordinates: { lat: 39.9042, lon: 116.4074 },
        forecast: generateDemoForecast(187),
        tempo_integration: generateDemoTEMPO(),
        ml_enhancement: generateDemoML(),
        health_assessment: generateDemoHealth('high')
    }
};

// Override API calls for demo
const originalFetch = window.fetch;
window.fetch = function(url, options) {
    if (url.includes('/api/forecast')) {
        return new Promise((resolve) => {
            setTimeout(() => {
                const requestData = JSON.parse(options.body);
                const city = requestData.city?.toLowerCase() || '';
                
                let responseData = demoData['london']; // Default
                
                if (city.includes('delhi') || city.includes('new delhi')) {
                    responseData = demoData['new delhi'];
                } else if (city.includes('beijing')) {
                    responseData = demoData['beijing'];
                } else if (city.includes('mumbai')) {
                    responseData = { ...demoData['new delhi'], city: 'Mumbai, India', current_aqi: 89 };
                }
                
                resolve({
                    ok: true,
                    status: 200,
                    json: () => Promise.resolve(responseData)
                });
            }, 1000); // Simulate network delay
        });
    }
    
    return originalFetch.apply(this, arguments);
};

function generateDemoForecast(baseAqi) {
    const forecast = [];
    for (let i = 0; i < 24; i++) {
        const variation = Math.random() * 20 - 10;
        const aqi = Math.max(10, Math.min(300, baseAqi + variation));
        forecast.push({
            time: new Date(Date.now() + i * 3600000).toISOString().slice(0, 16),
            aqi: Math.round(aqi),
            category: getAQICategory(aqi),
            color: getAQIColor(aqi),
            pm2_5: aqi * 0.4,
            pm10: aqi * 0.6
        });
    }
    return forecast;
}

function generateDemoTEMPO() {
    return {
        satellite_data: {
            no2_column: 2.5e15,
            so2_column: 1.2e15,
            hcho_column: 8e14,
            ozone_column: 3.2e18,
            aerosol_optical_depth: 0.15,
            cloud_fraction: 0.3,
            quality_flag: 0
        },
        enhanced_accuracy: '+28.5%'
    };
}

function generateDemoML() {
    return {
        ml_enhancement: true,
        confidence_score: 0.92,
        prediction_accuracy: '+25%',
        ensemble_models: ['random_forest', 'gradient_boost', 'neural_network']
    };
}

function generateDemoHealth(level) {
    const healthData = {
        low: {
            risk_level: 'low',
            health_recommendations: [
                'Great day for outdoor activities',
                'Perfect time for exercise outside',
                'Air quality is excellent for all groups'
            ],
            vulnerable_groups: []
        },
        moderate: {
            risk_level: 'moderate',
            health_recommendations: [
                'Generally safe for outdoor activities',
                'Sensitive individuals should monitor symptoms'
            ],
            vulnerable_groups: ['People with respiratory conditions']
        },
        high: {
            risk_level: 'high',
            health_recommendations: [
                'Limit prolonged outdoor activities',
                'Sensitive groups should stay indoors',
                'Wear N95 masks when going outside'
            ],
            vulnerable_groups: ['Children', 'Elderly', 'People with asthma or COPD']
        }
    };
    
    return healthData[level] || healthData.moderate;
}

function getAQICategory(aqi) {
    if (aqi <= 50) return 'Good';
    if (aqi <= 100) return 'Moderate';
    if (aqi <= 150) return 'Unhealthy for Sensitive Groups';
    if (aqi <= 200) return 'Unhealthy';
    if (aqi <= 300) return 'Very Unhealthy';
    return 'Hazardous';
}

function getAQIColor(aqi) {
    if (aqi <= 50) return '#00E400';
    if (aqi <= 100) return '#FFFF00';
    if (aqi <= 150) return '#FF7E00';
    if (aqi <= 200) return '#FF0000';
    if (aqi <= 300) return '#8F3F97';
    return '#7E0023';
}

// Add demo banner
document.addEventListener('DOMContentLoaded', function() {
    const header = document.querySelector('.header-content');
    if (header) {
        const demoBanner = document.createElement('div');
        demoBanner.innerHTML = `
            <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 15px; margin-top: 20px;">
                <p style="margin: 0; color: #38bdf8; font-weight: 600;">
                    🚀 Live Demo Mode - Simulated NASA TEMPO data for demonstration
                </p>
                <p style="margin: 5px 0 0 0; color: #cbd5e1; font-size: 0.9rem;">
                    For full functionality with real-time data, run the backend server locally
                </p>
            </div>
        `;
        header.appendChild(demoBanner);
    }
});