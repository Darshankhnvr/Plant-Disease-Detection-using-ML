/**
 * Weather Auto-fill functionality
 * Automatically fills temperature and humidity fields using geolocation
 */

class WeatherAutoFill {
    constructor() {
        this.apiKey = 'your_openweather_api_key'; // Replace with actual API key
        this.baseUrl = 'https://api.openweathermap.org/data/2.5/weather';
    }

    /**
     * Get user's current location and fetch weather data
     */
    async autoFillWeatherData(temperatureFieldId, humidityFieldId, buttonId = null) {
        const button = buttonId ? document.getElementById(buttonId) : null;
        
        try {
            if (button) {
                button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Getting location...';
                button.disabled = true;
            }

            // Get user's location
            const position = await this.getCurrentPosition();
            const { latitude, longitude } = position.coords;

            if (button) {
                button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Fetching weather...';
            }

            // Fetch weather data
            const weatherData = await this.fetchWeatherData(latitude, longitude);
            
            // Fill the form fields
            this.fillWeatherFields(temperatureFieldId, humidityFieldId, weatherData);

            if (button) {
                button.innerHTML = '<i class="fas fa-check"></i> Weather Updated';
                button.classList.add('success');
                setTimeout(() => {
                    button.innerHTML = '<i class="fas fa-cloud-sun"></i> Auto-fill Weather';
                    button.classList.remove('success');
                    button.disabled = false;
                }, 2000);
            }

        } catch (error) {
            console.error('Weather auto-fill error:', error);
            
            if (button) {
                button.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error';
                button.classList.add('error');
                setTimeout(() => {
                    button.innerHTML = '<i class="fas fa-cloud-sun"></i> Auto-fill Weather';
                    button.classList.remove('error');
                    button.disabled = false;
                }, 2000);
            }

            // Show user-friendly error message
            this.showErrorMessage(error.message);
        }
    }

    /**
     * Get user's current position using Geolocation API
     */
    getCurrentPosition() {
        return new Promise((resolve, reject) => {
            if (!navigator.geolocation) {
                reject(new Error('Geolocation is not supported by this browser'));
                return;
            }

            navigator.geolocation.getCurrentPosition(
                resolve,
                (error) => {
                    switch (error.code) {
                        case error.PERMISSION_DENIED:
                            reject(new Error('Location access denied by user'));
                            break;
                        case error.POSITION_UNAVAILABLE:
                            reject(new Error('Location information unavailable'));
                            break;
                        case error.TIMEOUT:
                            reject(new Error('Location request timed out'));
                            break;
                        default:
                            reject(new Error('Unknown location error'));
                            break;
                    }
                },
                {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 300000 // 5 minutes
                }
            );
        });
    }

    /**
     * Fetch weather data from OpenWeatherMap API
     */
    async fetchWeatherData(latitude, longitude) {
        // If no API key, use mock data
        if (this.apiKey === 'your_openweather_api_key') {
            return this.getMockWeatherData();
        }

        const url = `${this.baseUrl}?lat=${latitude}&lon=${longitude}&appid=${this.apiKey}&units=metric`;
        
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`Weather API error: ${response.status}`);
        }

        const data = await response.json();
        
        return {
            temperature: Math.round(data.main.temp * 10) / 10,
            humidity: data.main.humidity,
            location: data.name
        };
    }

    /**
     * Generate mock weather data for testing
     */
    getMockWeatherData() {
        const mockTemperatures = [18, 22, 25, 28, 30, 24, 26];
        const mockHumidities = [65, 70, 75, 80, 85, 72, 78];
        
        return {
            temperature: mockTemperatures[Math.floor(Math.random() * mockTemperatures.length)],
            humidity: mockHumidities[Math.floor(Math.random() * mockHumidities.length)],
            location: 'Mock Location'
        };
    }

    /**
     * Fill weather data into form fields
     */
    fillWeatherFields(temperatureFieldId, humidityFieldId, weatherData) {
        const tempField = document.getElementById(temperatureFieldId);
        const humidityField = document.getElementById(humidityFieldId);

        if (tempField) {
            tempField.value = weatherData.temperature;
            tempField.classList.add('auto-filled');
            setTimeout(() => tempField.classList.remove('auto-filled'), 2000);
        }

        if (humidityField) {
            humidityField.value = weatherData.humidity;
            humidityField.classList.add('auto-filled');
            setTimeout(() => humidityField.classList.remove('auto-filled'), 2000);
        }

        // Show success message
        this.showSuccessMessage(`Weather data updated for ${weatherData.location}`);
    }

    /**
     * Show success message
     */
    showSuccessMessage(message) {
        this.showMessage(message, 'success');
    }

    /**
     * Show error message
     */
    showErrorMessage(message) {
        this.showMessage(message, 'error');
    }

    /**
     * Show message to user
     */
    showMessage(message, type = 'info') {
        // Create message element
        const messageEl = document.createElement('div');
        messageEl.className = `weather-message weather-message-${type}`;
        messageEl.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;

        // Add to page
        document.body.appendChild(messageEl);

        // Animate in
        setTimeout(() => messageEl.classList.add('show'), 100);

        // Remove after 3 seconds
        setTimeout(() => {
            messageEl.classList.remove('show');
            setTimeout(() => document.body.removeChild(messageEl), 300);
        }, 3000);
    }
}

// Global instance
const weatherAutoFill = new WeatherAutoFill();

// Helper function for easy use
function autoFillWeather(temperatureFieldId, humidityFieldId, buttonId = null) {
    weatherAutoFill.autoFillWeatherData(temperatureFieldId, humidityFieldId, buttonId);
}