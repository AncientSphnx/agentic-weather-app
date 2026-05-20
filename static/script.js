// Weather Prediction App JavaScript
class WeatherPredictionApp {
    constructor() {
        this.model = null;
        this.initializeApp();
    }

    async initializeApp() {
        this.setupEventListeners();
        this.setupSliders();
        await this.loadModel();
        this.setDefaultDate();
    }

    setupEventListeners() {
        const form = document.getElementById('predictionForm');
        form.addEventListener('submit', (e) => this.handlePrediction(e));
    }

    setupSliders() {
        // Humidity slider sync
        const humidity = document.getElementById('humidity');
        const humiditySlider = document.getElementById('humiditySlider');
        
        humidity.addEventListener('input', () => {
            humiditySlider.value = humidity.value;
        });
        
        humiditySlider.addEventListener('input', () => {
            humidity.value = humiditySlider.value;
        });

        // Wind speed slider sync
        const windSpeed = document.getElementById('windSpeed');
        const windSpeedSlider = document.getElementById('windSpeedSlider');
        
        windSpeed.addEventListener('input', () => {
            windSpeedSlider.value = windSpeed.value;
        });
        
        windSpeedSlider.addEventListener('input', () => {
            windSpeed.value = windSpeedSlider.value;
        });

        // Pressure slider sync
        const pressure = document.getElementById('pressure');
        const pressureSlider = document.getElementById('pressureSlider');
        
        pressure.addEventListener('input', () => {
            pressureSlider.value = pressure.value;
        });
        
        pressureSlider.addEventListener('input', () => {
            pressure.value = pressureSlider.value;
        });
    }

    setDefaultDate() {
        const dateInput = document.getElementById('date');
        const today = new Date();
        dateInput.value = today.toISOString().split('T')[0];
    }

    async loadModel() {
        // Check if backend is available
        try {
            const response = await fetch('/api/model-info');
            if (response.ok) {
                const modelInfo = await response.json();
                console.log('Backend model loaded:', modelInfo);
                this.backendAvailable = true;
                this.updateModelStats(modelInfo);
            } else {
                throw new Error('Backend not available');
            }
        } catch (error) {
            console.log('Backend not available, using fallback model');
            this.backendAvailable = false;
            this.model = new SimpleLinearRegression();
        }
    }

    updateModelStats(modelInfo) {
        // Update accuracy
        const accuracyEl = document.getElementById('modelAccuracy');
        const metricsEl = document.getElementById('modelMetrics');
        if (accuracyEl && modelInfo.metrics) {
            accuracyEl.textContent = `${modelInfo.metrics.accuracy_percent}%`;
            metricsEl.textContent = `R² Score: ${modelInfo.metrics.r2_score.toFixed(3)}`;
        }

        // Update data info
        const samplesEl = document.getElementById('dataSamples');
        const rangeEl = document.getElementById('dataRange');
        if (samplesEl && modelInfo.dataset_info) {
            samplesEl.textContent = modelInfo.dataset_info.total_samples.toLocaleString();
            rangeEl.textContent = modelInfo.dataset_info.date_range;
        }

        // Update algorithm info
        const algorithmEl = document.getElementById('algorithmName');
        const detailsEl = document.getElementById('algorithmDetails');
        if (algorithmEl && modelInfo.algorithm) {
            algorithmEl.textContent = modelInfo.algorithm;
            
            // Create detailed description
            let details = '';
            if (modelInfo.model_params) {
                const params = Object.entries(modelInfo.model_params)
                    .map(([key, value]) => `${key}: ${value}`)
                    .join(', ');
                details = params || 'Optimized for accuracy';
            } else {
                details = 'Linear model with regularization';
            }
            detailsEl.textContent = details;
        }
    }

    validateInputs(data) {
        const errors = [];
        
        if (data.day_of_year < 1 || data.day_of_year > 365) {
            errors.push('Day of year must be between 1 and 365');
        }
        
        if (data.humidity < 0 || data.humidity > 100) {
            errors.push('Humidity must be between 0 and 100');
        }
        
        if (data.wind_speed < 0 || data.wind_speed > 50) {
            errors.push('Wind speed must be between 0 and 50 km/h');
        }
        
        if (data.pressure < 900 || data.pressure > 1050) {
            errors.push('Pressure must be between 900 and 1050 hPa');
        }
        
        return errors;
    }

    async handlePrediction(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const date = formData.get('date');
        const dayOfYear = this.getDayOfYear(new Date(date));
        
        const inputData = {
            date: date,
            day_of_year: dayOfYear,
            humidity: parseFloat(formData.get('humidity')),
            wind_speed: parseFloat(formData.get('windSpeed')),
            pressure: parseFloat(formData.get('pressure'))
        };

        // Validate inputs
        const errors = this.validateInputs(inputData);
        if (errors.length > 0) {
            this.showError(errors.join(', '));
            return;
        }

        // Show loading state
        this.showLoading();
        
        if (this.backendAvailable) {
            // Use real backend prediction
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(inputData)
                });
                
                if (response.ok) {
                    const result = await response.json();
                    this.displayResult(result.temperature, result.weather_condition, inputData);
                } else {
                    const error = await response.json();
                    throw new Error(error.error || 'Prediction failed');
                }
            } catch (error) {
                console.error('Backend prediction failed:', error);
                this.showError('Backend prediction failed. Using fallback model.');
                // Fallback to local model
                const prediction = this.predictTemperature(inputData);
                this.displayResult(prediction, this.getWeatherCondition(prediction), inputData);
            }
        } else {
            // Use local fallback model
            setTimeout(() => {
                const prediction = this.predictTemperature(inputData);
                this.displayResult(prediction, this.getWeatherCondition(prediction), inputData);
            }, 1500);
        }
    }

    getDayOfYear(date) {
        const start = new Date(date.getFullYear(), 0, 0);
        const diff = date - start;
        const oneDay = 1000 * 60 * 60 * 24;
        return Math.floor(diff / oneDay);
    }

    predictTemperature(inputData) {
        // Simple linear regression formula based on the Python model
        // This is a simplified version for demonstration
        const { day_of_year, humidity, wind_speed, pressure } = inputData;
        
        // Coefficients (these would normally come from the trained model)
        const temp = (day_of_year * 0.05) + 
                    (humidity * -0.1) + 
                    (wind_speed * -0.2) + 
                    (pressure * 0.01) + 
                    15;
        
        // Add some seasonal variation
        const seasonalFactor = Math.sin((day_of_year - 80) * 2 * Math.PI / 365) * 10;
        
        return Math.round((temp + seasonalFactor) * 10) / 10;
    }

    displayResult(temperature, weatherCondition, inputData) {
        const resultContainer = document.getElementById('result');
        const tempValue = document.getElementById('tempValue');
        const weatherConditionEl = document.getElementById('weatherCondition');
        
        tempValue.textContent = temperature.toFixed(1);
        weatherConditionEl.textContent = weatherCondition;
        
        resultContainer.classList.remove('hidden');
        
        // Add animation
        resultContainer.style.animation = 'none';
        setTimeout(() => {
            resultContainer.style.animation = 'slideIn 0.5s ease';
        }, 10);
    }

    getWeatherCondition(temperature) {
        if (temperature < 0) return '❄️ Freezing Cold';
        if (temperature < 10) return '🧥 Cold';
        if (temperature < 20) return '🌤️ Cool';
        if (temperature < 30) return '☀️ Warm';
        if (temperature < 35) return '🔥 Hot';
        return '🌋 Very Hot';
    }

    showLoading() {
        const resultContainer = document.getElementById('result');
        const tempValue = document.getElementById('tempValue');
        const weatherCondition = document.getElementById('weatherCondition');
        
        tempValue.innerHTML = '<div class="loading"></div>';
        weatherCondition.textContent = 'Calculating...';
        resultContainer.classList.remove('hidden');
    }

    showError(message) {
        // Create error toast
        const toast = document.createElement('div');
        toast.className = 'error-toast';
        toast.innerHTML = `
            <i class="fas fa-exclamation-circle"></i>
            <span>${message}</span>
        `;
        
        // Style the toast
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #f44336;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(244, 67, 54, 0.3);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            z-index: 1000;
            animation: slideInRight 0.3s ease;
        `;
        
        document.body.appendChild(toast);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

// Simple Linear Regression Class (simplified for demo)
class SimpleLinearRegression {
    constructor() {
        // Simplified coefficients for demonstration
        this.coefficients = {
            day_of_year: 0.05,
            humidity: -0.1,
            wind_speed: -0.2,
            pressure: 0.01,
            intercept: 15
        };
    }

    predict(features) {
        const { day_of_year, humidity, wind_speed, pressure } = features;
        return (
            day_of_year * this.coefficients.day_of_year +
            humidity * this.coefficients.humidity +
            wind_speed * this.coefficients.wind_speed +
            pressure * this.coefficients.pressure +
            this.coefficients.intercept
        );
    }
}

// Add custom animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .error-toast i {
        font-size: 1.2rem;
    }
`;
document.head.appendChild(style);

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new WeatherPredictionApp();
});

// Add some interactive effects
document.addEventListener('DOMContentLoaded', () => {
    // Add hover effect to cards
    const cards = document.querySelectorAll('.prediction-card, .stat-card, .chart-container');
    cards.forEach(card => {
        card.addEventListener('mouseenter', (e) => {
            e.currentTarget.style.transform = 'translateY(-5px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', (e) => {
            e.currentTarget.style.transform = 'translateY(0) scale(1)';
        });
    });

    // Add ripple effect to buttons
    const buttons = document.querySelectorAll('.predict-btn');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.5);
                left: ${x}px;
                top: ${y}px;
                pointer-events: none;
                transform: scale(0);
                animation: ripple 0.6s ease-out;
            `;
            
            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });
    });
});

// Add ripple animation
const rippleStyle = document.createElement('style');
rippleStyle.textContent = `
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;
document.head.appendChild(rippleStyle);
