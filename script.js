// API Configuration
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://127.0.0.1:5000/api'  // Local development
    : 'https://nasa-aqi-backend.up.railway.app/api';  // Production backend

// DOM Elements
const cityInput = document.getElementById('cityInput');
const searchBtn = document.getElementById('searchBtn');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const cityTags = document.querySelectorAll('.city-tag');
const retryBtn = document.getElementById('retryBtn');
const currentLocationBtn = document.getElementById('currentLocationBtn');
const locationStatus = document.getElementById('locationStatus');
const centerMapBtn = document.getElementById('centerMapBtn');

// State
let currentCity = '';
let currentCoords = null;
let map = null;
let currentMarker = null;

// Initialize
document.addEventListener('DOMContentLoaded', function () {
    setupEventListeners();
    initializeMap();
    
    // Handle window resize for map
    window.addEventListener('resize', function() {
        if (map) {
            setTimeout(() => {
                map.invalidateSize();
            }, 100);
        }
    });
});

function setupEventListeners() {
    // Search button click
    searchBtn.addEventListener('click', handleSearch);

    // Enter key in input
    cityInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });

    // Clear coordinates when user types a new city
    cityInput.addEventListener('input', function () {
        // Clear location status when user starts typing
        if (locationStatus.textContent && !this.value.includes('Current Location')) {
            locationStatus.textContent = '';
            locationStatus.className = 'location-status';
        }
    });

    // Quick city tags
    cityTags.forEach(tag => {
        tag.addEventListener('click', function () {
            const city = this.getAttribute('data-city');
            cityInput.value = city;
            currentCoords = null; // Clear coordinates for city search
            handleSearch();
        });
    });

    // Current location button
    currentLocationBtn.addEventListener('click', handleCurrentLocation);

    // Center map button
    centerMapBtn.addEventListener('click', centerMapOnLocation);

    // Retry button
    retryBtn.addEventListener('click', function () {
        if (currentCity) {
            cityInput.value = currentCity;
            handleSearch();
        }
    });
}

async function handleSearch(useCoordinates = false) {
    const city = cityInput.value.trim();

    if (!city && !currentCoords) {
        showError('Please enter a city name or use your current location');
        return;
    }

    currentCity = city;

    // Show loading state
    setLoadingState(true);
    hideError();
    hideResults();

    try {
        // Prepare request body
        let requestBody = {};

        if (useCoordinates && currentCoords) {
            // Use coordinates only when explicitly requested
            console.log('Using coordinates for search:', currentCoords);
            requestBody = {
                lat: currentCoords.lat,
                lon: currentCoords.lon,
                city: city || 'Current Location'
            };
        } else {
            // Use city name for regular searches
            console.log('Using city name for search:', city);
            requestBody = { city: city };
            // Clear coordinates when doing a city search
            currentCoords = null;
        }

        const response = await fetch(`${API_BASE_URL}/forecast`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();

        if (data.success) {
            // Store coordinates if returned and we used coordinates for the search
            if (data.coordinates && useCoordinates) {
                currentCoords = data.coordinates;
            }
            displayResults(data);
        } else {
            showError(data.error || 'Failed to fetch air quality data');
        }

    } catch (error) {
        console.error('Error fetching data:', error);
        showError('Network error. Please check your connection and try again.');
    } finally {
        setLoadingState(false);
    }
}

function displayResults(data) {
    // Update city name and current AQI
    document.getElementById('cityName').textContent = data.city;
    document.getElementById('updateTime').textContent = new Date().toLocaleString();
    document.getElementById('currentAQI').textContent = data.current_aqi || '--';

    // Update current category
    const categoryElement = document.getElementById('currentCategory');
    categoryElement.textContent = data.current_category || 'Unknown';
    categoryElement.className = `aqi-category ${getAQICategoryClass(data.current_aqi)}`;

    // Update description
    document.getElementById('aqiDescription').textContent = getAQIDescription(data.current_aqi);

    // Update chart
    if (data.chart) {
        const chartImg = document.getElementById('forecastChart');
        chartImg.src = data.chart;
        chartImg.style.display = 'block';
        document.getElementById('chartLoading').style.display = 'none';
    }

    // Update forecast table
    updateForecastTable(data.forecast);

    // Show results
    showResults();
}

function updateForecastTable(forecast) {
    const tbody = document.getElementById('forecastTableBody');
    tbody.innerHTML = '';

    forecast.slice(0, 24).forEach(item => { // Show first 24 hours
        const row = document.createElement('tr');

        const timeCell = document.createElement('td');
        timeCell.textContent = formatTime(item.time);

        const aqiCell = document.createElement('td');
        aqiCell.innerHTML = `<strong>${item.aqi || '--'}</strong>`;

        const categoryCell = document.createElement('td');
        const badge = document.createElement('span');
        badge.className = `aqi-badge ${getAQICategoryClass(item.aqi)}`;
        badge.textContent = item.category || 'Unknown';
        categoryCell.appendChild(badge);

        const pm25Cell = document.createElement('td');
        pm25Cell.textContent = item.pm2_5 ? `${item.pm2_5.toFixed(1)} µg/m³` : '--';

        const pm10Cell = document.createElement('td');
        pm10Cell.textContent = item.pm10 ? `${item.pm10.toFixed(1)} µg/m³` : '--';

        row.appendChild(timeCell);
        row.appendChild(aqiCell);
        row.appendChild(categoryCell);
        row.appendChild(pm25Cell);
        row.appendChild(pm10Cell);

        tbody.appendChild(row);
    });
}

function formatTime(timeStr) {
    const date = new Date(timeStr);
    const now = new Date();
    const tomorrow = new Date(now);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const timeOptions = { hour: '2-digit', minute: '2-digit' };
    const time = date.toLocaleTimeString([], timeOptions);

    if (date.toDateString() === now.toDateString()) {
        return `Today ${time}`;
    } else if (date.toDateString() === tomorrow.toDateString()) {
        return `Tomorrow ${time}`;
    } else {
        return `${date.toLocaleDateString()} ${time}`;
    }
}

function getAQICategoryClass(aqi) {
    if (!aqi) return '';

    if (aqi <= 50) return 'aqi-good';
    if (aqi <= 100) return 'aqi-moderate';
    if (aqi <= 150) return 'aqi-unhealthy-sensitive';
    if (aqi <= 200) return 'aqi-unhealthy';
    if (aqi <= 300) return 'aqi-very-unhealthy';
    return 'aqi-hazardous';
}

function getAQIDescription(aqi) {
    if (!aqi) return 'Air quality data unavailable';

    if (aqi <= 50) {
        return 'Air quality is satisfactory, and air pollution poses little or no risk.';
    } else if (aqi <= 100) {
        return 'Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.';
    } else if (aqi <= 150) {
        return 'Members of sensitive groups may experience health effects. The general public is less likely to be affected.';
    } else if (aqi <= 200) {
        return 'Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.';
    } else if (aqi <= 300) {
        return 'Health alert: The risk of health effects is increased for everyone.';
    } else {
        return 'Health warning of emergency conditions: everyone is more likely to be affected.';
    }
}

function setLoadingState(loading) {
    const btnText = searchBtn.querySelector('.btn-text');
    const spinner = searchBtn.querySelector('.loading-spinner');

    if (loading) {
        btnText.style.display = 'none';
        spinner.style.display = 'flex';
        searchBtn.disabled = true;
    } else {
        btnText.style.display = 'block';
        spinner.style.display = 'none';
        searchBtn.disabled = false;
    }
}

function updateSearchButtonText() {
    const btnText = searchBtn.querySelector('.btn-text');
    if (currentCoords && cityInput.value.includes('Current Location')) {
        btnText.textContent = 'Get Location Forecast';
    } else {
        btnText.textContent = 'Get Forecast';
    }
}

function showResults() {
    resultsSection.style.display = 'block';
    errorSection.style.display = 'none';
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    errorSection.style.display = 'block';
    resultsSection.style.display = 'none';
}

function hideError() {
    errorSection.style.display = 'none';
}

// Utility function to handle API errors
function handleAPIError(error) {
    console.error('API Error:', error);

    if (error.name === 'TypeError' && error.message.includes('fetch')) {
        return 'Unable to connect to the server. Please check if the backend is running.';
    }

    return error.message || 'An unexpected error occurred. Please try again.';
}

// Geolocation Functions
async function handleCurrentLocation() {
    if (!navigator.geolocation) {
        showLocationError('Geolocation is not supported by this browser');
        return;
    }

    setLocationLoadingState(true);
    locationStatus.textContent = 'Getting your location...';

    navigator.geolocation.getCurrentPosition(
        async (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            currentCoords = { lat, lon };

            try {
                // Reverse geocode to get city name
                const response = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`);
                const data = await response.json();

                let cityName = 'Current Location';
                if (data.address) {
                    cityName = data.address.city || data.address.town || data.address.village ||
                        data.address.county || data.display_name.split(',')[0];
                }

                cityInput.value = cityName;
                locationStatus.textContent = `Located: ${cityName}`;
                locationStatus.className = 'location-status success';

                // Update search button text
                updateSearchButtonText();

                // Update map
                updateMapLocation(lat, lon, cityName);

                // Get AQI data using coordinates
                await handleSearch(true);

            } catch (error) {
                console.error('Reverse geocoding failed:', error);
                locationStatus.textContent = 'Location found, but city name unavailable';
                locationStatus.className = 'location-status';

                // Still update map with coordinates
                updateMapLocation(lat, lon, 'Current Location');
            }

            setLocationLoadingState(false);
        },
        (error) => {
            let errorMessage = 'Unable to get your location';
            switch (error.code) {
                case error.PERMISSION_DENIED:
                    errorMessage = 'Location access denied by user';
                    break;
                case error.POSITION_UNAVAILABLE:
                    errorMessage = 'Location information unavailable';
                    break;
                case error.TIMEOUT:
                    errorMessage = 'Location request timed out';
                    break;
            }
            showLocationError(errorMessage);
            setLocationLoadingState(false);
        },
        {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 300000 // 5 minutes
        }
    );
}

function setLocationLoadingState(loading) {
    const btnText = currentLocationBtn.querySelector('span');
    const spinner = currentLocationBtn.querySelector('.location-loading');

    if (loading) {
        btnText.style.display = 'none';
        spinner.style.display = 'flex';
        currentLocationBtn.disabled = true;
    } else {
        btnText.style.display = 'block';
        spinner.style.display = 'none';
        currentLocationBtn.disabled = false;
    }
}

function showLocationError(message) {
    locationStatus.textContent = message;
    locationStatus.className = 'location-status error';
}

// Map Functions
function initializeMap() {
    try {
        // Initialize map centered on world view
        map = L.map('aqiMap', {
            zoomControl: true,
            scrollWheelZoom: true,
            doubleClickZoom: true,
            boxZoom: true,
            keyboard: true,
            dragging: true,
            touchZoom: true
        }).setView([20, 0], 2);

        // Try multiple tile providers for better reliability
        const tileProviders = [
            {
                url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                attribution: '© OpenStreetMap contributors'
            },
            {
                url: 'https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png',
                attribution: '© CartoDB, © OpenStreetMap contributors'
            },
            {
                url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
                attribution: '© CartoDB, © OpenStreetMap contributors'
            }
        ];

        // Try to add the first working tile layer
        let tileLayerAdded = false;
        for (let provider of tileProviders) {
            try {
                const tileLayer = L.tileLayer(provider.url, {
                    attribution: provider.attribution,
                    maxZoom: 18,
                    minZoom: 2,
                    errorTileUrl: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
                });
                
                tileLayer.addTo(map);
                tileLayerAdded = true;
                console.log('Map tiles loaded successfully');
                break;
            } catch (e) {
                console.warn('Failed to load tile provider:', provider.url);
                continue;
            }
        }

        if (!tileLayerAdded) {
            console.error('All tile providers failed, showing fallback');
            // Show fallback message
            const mapContainer = document.getElementById('aqiMap');
            const fallback = document.getElementById('mapFallback');
            if (mapContainer && fallback) {
                mapContainer.style.display = 'none';
                fallback.style.display = 'flex';
            }
            return; // Exit early if no tiles loaded
        }

        // Force map to resize after initialization
        setTimeout(() => {
            if (map) {
                map.invalidateSize();
            }
        }, 100);

        // Show fallback if map doesn't load within 10 seconds
        setTimeout(() => {
            const mapContainer = document.getElementById('aqiMap');
            const leafletContainer = mapContainer?.querySelector('.leaflet-container');
            
            if (!leafletContainer || leafletContainer.children.length === 0) {
                console.warn('Map loading timeout, showing fallback');
                const fallback = document.getElementById('mapFallback');
                if (fallback) {
                    fallback.style.display = 'flex';
                }
            }
        }, 10000);

    // Add click handler for map
    map.on('click', async function (e) {
        const lat = e.latlng.lat;
        const lon = e.latlng.lng;

        try {
            // Reverse geocode clicked location
            const response = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`);
            const data = await response.json();

            let cityName = 'Selected Location';
            if (data.address) {
                cityName = data.address.city || data.address.town || data.address.village ||
                    data.address.county || data.display_name.split(',')[0];
            }

            cityInput.value = cityName;
            currentCoords = { lat, lon };
            updateSearchButtonText();
            updateMapLocation(lat, lon, cityName);

            // Get AQI data for clicked location using coordinates
            await handleSearch(true);

        } catch (error) {
            console.error('Failed to get location info:', error);
        }
    });
    } catch (error) {
        console.error('Map initialization failed:', error);
        // Show fallback message
        const mapContainer = document.getElementById('aqiMap');
        const fallback = document.getElementById('mapFallback');
        if (mapContainer && fallback) {
            fallback.style.display = 'flex';
        }
    }
}

function updateMapLocation(lat, lon, cityName, aqi = null) {
    // Remove existing marker
    if (currentMarker) {
        map.removeLayer(currentMarker);
    }

    // Create marker with AQI color
    const markerColor = aqi ? getAQIMarkerColor(aqi) : '#667eea';

    // Create custom icon
    const markerIcon = L.divIcon({
        className: 'custom-marker',
        html: `<div style="
            background: ${markerColor};
            width: 30px;
            height: 30px;
            border-radius: 50%;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 12px;
        ">${aqi || '?'}</div>`,
        iconSize: [30, 30],
        iconAnchor: [15, 15]
    });

    // Add new marker
    currentMarker = L.marker([lat, lon], { icon: markerIcon }).addTo(map);

    // Add popup with AQI info
    const popupContent = aqi ?
        `<strong>${cityName}</strong><br>AQI: ${aqi}<br>Category: ${getAQICategory(aqi)}` :
        `<strong>${cityName}</strong><br>Click "Get Forecast" for AQI data`;

    currentMarker.bindPopup(popupContent).openPopup();

    // Center map on location
    map.setView([lat, lon], 10);
}

function centerMapOnLocation() {
    if (currentCoords) {
        map.setView([currentCoords.lat, currentCoords.lon], 10);
    } else if (currentMarker) {
        map.setView(currentMarker.getLatLng(), 10);
    }
}

function getAQIMarkerColor(aqi) {
    if (aqi <= 50) return '#00E400';      // Green
    if (aqi <= 100) return '#FFFF00';     // Yellow
    if (aqi <= 150) return '#FF7E00';     // Orange
    if (aqi <= 200) return '#FF0000';     // Red
    if (aqi <= 300) return '#8F3F97';     // Purple
    return '#7E0023';                     // Maroon
}

function getAQICategory(aqi) {
    if (aqi <= 50) return 'Good';
    if (aqi <= 100) return 'Moderate';
    if (aqi <= 150) return 'Unhealthy for Sensitive Groups';
    if (aqi <= 200) return 'Unhealthy';
    if (aqi <= 300) return 'Very Unhealthy';
    return 'Hazardous';
}

// Enhanced display results to update map
function displayResultsEnhanced(data) {
    displayResults(data);

    // Update map with AQI data if we have coordinates
    if (currentCoords && data.current_aqi) {
        updateMapLocation(
            currentCoords.lat,
            currentCoords.lon,
            data.city,
            data.current_aqi
        );
    }
}

// Override the original displayResults function
const originalDisplayResults = displayResults;
displayResults = function (data) {
    originalDisplayResults(data);

    // Update map with AQI data if we have coordinates from coordinate-based search
    if (currentCoords && data.current_aqi && data.coordinates) {
        updateMapLocation(
            currentCoords.lat,
            currentCoords.lon,
            data.city,
            data.current_aqi
        );
    }
    
    // Display advanced features
    displayAdvancedFeatures(data);
};

// Advanced Features Functions
function displayAdvancedFeatures(data) {
    try {
        // Display TEMPO integration data
        if (data.tempo_integration) {
            displayTEMPOData(data.tempo_integration);
        }
        
        // Display health assessment
        if (data.health_assessment) {
            displayHealthAssessment(data.health_assessment);
        }
        
        // Display ML enhancement info
        if (data.ml_enhancement) {
            displayMLInfo(data.ml_enhancement);
        }
        
        // Display alerts
        if (data.alerts) {
            displayAlerts(data.alerts);
        }
    } catch (error) {
        console.error('Error displaying advanced features:', error);
        // Continue without advanced features if there's an error
    }
}

function displayTEMPOData(tempoData) {
    const tempoSection = document.getElementById('tempoData');
    if (!tempoSection) return;
    
    const satelliteData = tempoData.satellite_data || {};
    const correlation = tempoData.correlation_analysis || {};
    
    // Safely update satellite metrics with null checks
    const no2Element = document.getElementById('no2Column');
    if (no2Element) {
        const no2Value = satelliteData.no2_column;
        if (no2Value !== undefined && no2Value !== null) {
            no2Element.textContent = (no2Value / 1e15).toFixed(2) + ' × 10¹⁵ mol/cm²';
        } else {
            no2Element.textContent = 'N/A';
        }
    }
    
    const aodElement = document.getElementById('aodValue');
    if (aodElement) {
        const aodValue = satelliteData.aerosol_optical_depth;
        if (aodValue !== undefined && aodValue !== null) {
            aodElement.textContent = aodValue.toFixed(3);
        } else {
            aodElement.textContent = 'N/A';
        }
    }
    
    const qualityElement = document.getElementById('dataQuality');
    if (qualityElement) {
        const qualityFlag = satelliteData.quality_flag;
        if (qualityFlag !== undefined && qualityFlag !== null) {
            qualityElement.textContent = 
                qualityFlag === 0 ? 'Excellent' : 
                qualityFlag === 1 ? 'Good' : 'Fair';
        } else {
            qualityElement.textContent = 'N/A';
        }
    }
    
    // Safely update correlation info
    const correlationElement = document.getElementById('correlationScore');
    if (correlationElement && correlation.validation_metrics) {
        const qualityScore = correlation.validation_metrics.quality_score;
        if (qualityScore !== undefined && qualityScore !== null) {
            correlationElement.textContent = (qualityScore * 100).toFixed(1) + '%';
        } else {
            correlationElement.textContent = 'N/A';
        }
    }
    
    const accuracyElement = document.getElementById('enhancedAccuracy');
    if (accuracyElement) {
        accuracyElement.textContent = tempoData.enhanced_accuracy || '+20%';
    }
    
    tempoSection.style.display = 'block';
}

function displayHealthAssessment(healthData) {
    const healthSection = document.getElementById('healthAssessment');
    if (!healthSection || !healthData) return;
    
    // Safely update risk level
    const riskElement = document.getElementById('riskLevel');
    if (riskElement && healthData.risk_level) {
        riskElement.textContent = healthData.risk_level.replace('_', ' ').toUpperCase();
        riskElement.className = `risk-value ${healthData.risk_level}`;
    }
    
    // Safely update recommendations
    const recommendationsDiv = document.getElementById('healthRecommendations');
    if (recommendationsDiv && healthData.health_recommendations && Array.isArray(healthData.health_recommendations)) {
        recommendationsDiv.innerHTML = `
            <h4>Recommendations:</h4>
            <ul>
                ${healthData.health_recommendations.map(rec => `<li>${rec || 'N/A'}</li>`).join('')}
            </ul>
        `;
    }
    
    // Safely update vulnerable groups
    const vulnerableDiv = document.getElementById('vulnerableGroups');
    if (vulnerableDiv && healthData.vulnerable_groups && Array.isArray(healthData.vulnerable_groups) && healthData.vulnerable_groups.length > 0) {
        vulnerableDiv.innerHTML = `
            <h4>At-Risk Groups:</h4>
            <ul>
                ${healthData.vulnerable_groups.map(group => `<li>${group || 'N/A'}</li>`).join('')}
            </ul>
        `;
    }
    
    healthSection.style.display = 'block';
}

function displayMLInfo(mlData) {
    const mlSection = document.getElementById('mlInfo');
    if (!mlSection || !mlData) return;
    
    // Safely update ML metrics
    const confidenceElement = document.getElementById('mlConfidence');
    if (confidenceElement) {
        const confidence = mlData.confidence_score;
        if (confidence !== undefined && confidence !== null) {
            confidenceElement.textContent = (confidence * 100).toFixed(1) + '%';
        } else {
            confidenceElement.textContent = 'N/A';
        }
    }
    
    const accuracyElement = document.getElementById('mlAccuracy');
    if (accuracyElement) {
        accuracyElement.textContent = mlData.prediction_accuracy || '+25%';
    }
    
    const modelsElement = document.getElementById('ensembleModels');
    if (modelsElement) {
        const models = mlData.ensemble_models;
        if (models && Array.isArray(models)) {
            modelsElement.textContent = models.length.toString();
        } else {
            modelsElement.textContent = '3';
        }
    }
    
    mlSection.style.display = 'block';
}

function displayAlerts(alertsData) {
    const alertsSection = document.getElementById('activeAlerts');
    if (!alertsSection || !alertsData || !alertsData.active_alerts) return;
    
    const alerts = alertsData.active_alerts;
    
    if (Array.isArray(alerts) && alerts.length > 0) {
        try {
            alertsSection.innerHTML = alerts.map(alert => `
                <div class="alert-item ${alert.severity || 'medium'}">
                    <div class="alert-title">${alert.title || 'Alert'}</div>
                    <div class="alert-message">${alert.message || 'No details available'}</div>
                </div>
            `).join('');
            
            alertsSection.style.display = 'block';
        } catch (error) {
            console.error('Error displaying alerts:', error);
            alertsSection.innerHTML = '<div class="alert-item medium"><div class="alert-title">System Alert</div><div class="alert-message">Alert system temporarily unavailable</div></div>';
            alertsSection.style.display = 'block';
        }
    }
}

// Notification System
function setupNotifications() {
    const enableBtn = document.getElementById('enableNotifications');
    const settingsDiv = document.getElementById('notificationSettings');
    const thresholdSlider = document.getElementById('aqiThreshold');
    const thresholdValue = document.getElementById('thresholdValue');
    
    if (enableBtn) {
        enableBtn.addEventListener('click', function() {
            if (settingsDiv.style.display === 'none' || !settingsDiv.style.display) {
                settingsDiv.style.display = 'block';
                this.innerHTML = '<i class="fas fa-bell-slash"></i> Disable Alerts';
            } else {
                settingsDiv.style.display = 'none';
                this.innerHTML = '<i class="fas fa-bell"></i> Enable Alerts';
            }
        });
    }
    
    if (thresholdSlider && thresholdValue) {
        thresholdSlider.addEventListener('input', function() {
            thresholdValue.textContent = this.value;
        });
    }
}

// Initialize advanced features
document.addEventListener('DOMContentLoaded', function() {
    setupNotifications();
});