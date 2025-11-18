// Language Management
let currentLang = 'en';

// Language data
const languageData = {
    en: {
        title: 'AgriPest - AI-Powered Pest Detection for Farmers',
        home: 'Home',
        identify: 'Identify',
        gallery: 'Gallery',
        about: 'About',
        startIdentifying: 'Start Identifying →',
        learnMore: 'Learn More',
        openCamera: 'Open Camera',
        uploadPhoto: 'Upload Photo',
        analyze: 'Analyze Insect',
        remove: 'Remove',
        analyzing: 'Analyzing...',
        resultsTitle: 'Identification Results',
        commonName: 'Common Name',
        scientificName: 'Scientific Name',
        family: 'Family:',
        habitat: 'Habitat:',
        harmful: 'Harmful:',
        recommendation: 'Recommendation:',
        description: 'Description',
        noDescription: 'No description available.'
    },
    hi: {
        title: 'AgriPest - किसानों के लिए AI-संचालित कीट पहचान',
        home: 'होम',
        identify: 'पहचानें',
        gallery: 'गैलरी',
        about: 'के बारे में',
        startIdentifying: 'पहचान शुरू करें →',
        learnMore: 'और जानें',
        openCamera: 'कैमरा खोलें',
        uploadPhoto: 'फोटो अपलोड करें',
        analyze: 'कीट का विश्लेषण करें',
        remove: 'हटाएं',
        analyzing: 'विश्लेषण कर रहे हैं...',
        resultsTitle: 'पहचान परिणाम',
        commonName: 'सामान्य नाम',
        scientificName: 'वैज्ञानिक नाम',
        family: 'परिवार:',
        habitat: 'आवास:',
        harmful: 'हानिकारक:',
        recommendation: 'सुझाव:',
        description: 'विवरण',
        noDescription: 'कोई विवरण उपलब्ध नहीं है।'
    }
};

// Sample insect data for demonstration
const sampleInsects = {
    aphid: {
        en: {
            name: 'Aphid',
            scientific: 'Aphidoidea',
            family: 'Aphididae',
            habitat: 'Crops, gardens, and plants',
            harmful: 'Yes - Harmful',
            recommendation: 'Use insecticidal soap or neem oil',
            description: 'Aphids are small sap-sucking insects that can cause significant damage to crops. They reproduce quickly and can spread plant viruses.'
        },
        hi: {
            name: 'एफिड',
            scientific: 'Aphidoidea',
            family: 'एफिडिडे',
            habitat: 'फसलें, बगीचे और पौधे',
            harmful: 'हाँ - हानिकारक',
            recommendation: 'कीटनाशक साबुन या नीम का तेल उपयोग करें',
            description: 'एफिड छोटे रस-चूसने वाले कीट हैं जो फसलों को महत्वपूर्ण नुकसान पहुंचा सकते हैं। वे तेजी से प्रजनन करते हैं और पौधों के वायरस फैला सकते हैं।'
        }
    },
    ladybug: {
        en: {
            name: 'Ladybug',
            scientific: 'Coccinellidae',
            family: 'Coccinellidae',
            habitat: 'Gardens and agricultural fields',
            harmful: 'No - Beneficial',
            recommendation: 'Keep - Helps control aphids',
            description: 'Ladybugs are beneficial insects that prey on aphids and other harmful pests. They are natural pest controllers and should be protected.'
        },
        hi: {
            name: 'लेडीबग',
            scientific: 'Coccinellidae',
            family: 'कोक्सिनेलिडे',
            habitat: 'बगीचे और कृषि क्षेत्र',
            harmful: 'नहीं - लाभकारी',
            recommendation: 'रखें - एफिड को नियंत्रित करने में मदद करता है',
            description: 'लेडीबग लाभकारी कीट हैं जो एफिड और अन्य हानिकारक कीटों पर शिकार करते हैं। वे प्राकृतिक कीट नियंत्रक हैं और उनकी सुरक्षा की जानी चाहिए।'
        }
    },
    caterpillar: {
        en: {
            name: 'Caterpillar',
            scientific: 'Lepidoptera larvae',
            family: 'Various',
            habitat: 'Plants and crops',
            harmful: 'Yes - Harmful',
            recommendation: 'Remove manually or use Bt (Bacillus thuringiensis)',
            description: 'Caterpillars are the larval stage of butterflies and moths. Many species feed on plant leaves and can cause significant crop damage.'
        },
        hi: {
            name: 'कैटरपिलर',
            scientific: 'लेपिडोप्टेरा लार्वा',
            family: 'विविध',
            habitat: 'पौधे और फसलें',
            harmful: 'हाँ - हानिकारक',
            recommendation: 'मैन्युअल रूप से हटाएं या Bt (Bacillus thuringiensis) उपयोग करें',
            description: 'कैटरपिलर तितलियों और पतंगों का लार्वा चरण है। कई प्रजातियां पौधे की पत्तियों पर भोजन करती हैं और महत्वपूर्ण फसल नुकसान पहुंचा सकती हैं।'
        }
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeLanguage();
    setupEventListeners();
    setupSmoothScroll();
});

// Language switching function
function switchLanguage(lang) {
    currentLang = lang;
    
    // Update active button
    document.querySelectorAll('.lang-btn').forEach(btn => {
        if (btn.dataset.lang === lang) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // Update all elements with data-en and data-hi attributes
    document.querySelectorAll('[data-en][data-hi]').forEach(element => {
        if (lang === 'hi') {
            element.textContent = element.getAttribute('data-hi');
        } else {
            element.textContent = element.getAttribute('data-en');
        }
    });
    
    // Update title
    document.title = languageData[lang].title;
    
    // Update result section if visible
    updateResultsLanguage();
}

// Initialize language
function initializeLanguage() {
    switchLanguage('en'); // English is active by default
}

// Update results section language
function updateResultsLanguage() {
    const resultsSection = document.getElementById('results');
    if (resultsSection && resultsSection.style.display !== 'none') {
        // Update labels
        const labels = resultsSection.querySelectorAll('.label');
        labels.forEach(label => {
            const key = label.textContent.replace(':', '').toLowerCase();
            if (languageData[currentLang][key]) {
                label.textContent = languageData[currentLang][key];
            }
        });
    }
}

// Setup event listeners
function setupEventListeners() {
    // Language switcher buttons
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            switchLanguage(btn.dataset.lang);
        });
    });
    
    // Start Identifying buttons
    document.querySelectorAll('.start-btn, .hero-cta').forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('identify').scrollIntoView({ behavior: 'smooth' });
        });
    });
    
    // Learn More button
    const learnMoreBtn = document.querySelector('.btn-secondary');
    if (learnMoreBtn && learnMoreBtn.textContent.includes('Learn More')) {
        learnMoreBtn.addEventListener('click', () => {
            document.querySelector('.problem-section').scrollIntoView({ behavior: 'smooth' });
        });
    }
    
    // Camera button
    const cameraBtn = document.getElementById('cameraBtn');
    if (cameraBtn) {
        cameraBtn.addEventListener('click', () => {
            openCamera();
        });
    }
    
    // Upload photo button
    const uploadPhotoBtn = document.getElementById('uploadPhotoBtn');
    const imageInput = document.getElementById('imageInput');
    
    if (uploadPhotoBtn) {
        uploadPhotoBtn.addEventListener('click', () => {
            imageInput.click();
        });
    }
    
    // File input change
    if (imageInput) {
        imageInput.addEventListener('change', (e) => {
            handleFileSelect(e.target.files[0]);
        });
    }
    
    // Drag and drop
    const uploadCard = document.querySelector('.upload-card');
    if (uploadCard) {
        uploadCard.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadCard.style.borderColor = '#50C878';
            uploadCard.style.background = '#F0FDF4';
        });
        
        uploadCard.addEventListener('dragleave', () => {
            uploadCard.style.borderColor = '#E0E0E0';
            uploadCard.style.background = '#FFFFFF';
        });
        
        uploadCard.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadCard.style.borderColor = '#E0E0E0';
            uploadCard.style.background = '#FFFFFF';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleFileSelect(file);
            }
        });
    }
    
    // Analyze button
    const analyzeBtn = document.querySelector('.analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', () => {
            analyzeInsect();
        });
    }
    
    // Remove button
    const removeBtn = document.querySelector('.remove-btn');
    if (removeBtn) {
        removeBtn.addEventListener('click', () => {
            if (imageInput) imageInput.value = '';
            const previewArea = document.getElementById('previewArea');
            if (previewArea) previewArea.style.display = 'none';
            document.getElementById('results').style.display = 'none';
        });
    }
}

// Open camera
async function openCamera() {
    try {
        // Check if getUserMedia is available
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            // Fallback to file input for older browsers
            const imageInput = document.getElementById('imageInput');
            if (imageInput) {
                imageInput.setAttribute('capture', 'environment');
                imageInput.click();
            }
            return;
        }

        // Request camera access
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment' // Use back camera on mobile
            }
        });

        // Create camera preview modal
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 10000;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        `;

        const video = document.createElement('video');
        video.srcObject = stream;
        video.autoplay = true;
        video.playsInline = true;
        video.style.cssText = `
            width: 100%;
            max-width: 600px;
            height: auto;
            border-radius: 8px;
        `;

        const controls = document.createElement('div');
        controls.style.cssText = `
            display: flex;
            gap: 1rem;
            margin-top: 2rem;
        `;

        const captureBtn = document.createElement('button');
        captureBtn.textContent = currentLang === 'hi' ? 'फोटो लें' : 'Capture Photo';
        captureBtn.className = 'btn-primary';
        captureBtn.style.cssText = `
            padding: 1rem 2rem;
            font-size: 1.1rem;
            border-radius: 50px;
        `;

        const cancelBtn = document.createElement('button');
        cancelBtn.textContent = currentLang === 'hi' ? 'रद्द करें' : 'Cancel';
        cancelBtn.className = 'btn-secondary';
        cancelBtn.style.cssText = `
            padding: 1rem 2rem;
            font-size: 1.1rem;
            border-radius: 50px;
        `;

        controls.appendChild(captureBtn);
        controls.appendChild(cancelBtn);
        modal.appendChild(video);
        modal.appendChild(controls);
        document.body.appendChild(modal);

        // Capture photo
        captureBtn.addEventListener('click', () => {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);

            // Stop camera stream
            stream.getTracks().forEach(track => track.stop());

            // Convert to blob and handle as file
            canvas.toBlob((blob) => {
                const file = new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' });
                handleFileSelect(file);
                document.body.removeChild(modal);
            }, 'image/jpeg', 0.9);
        });

        // Cancel camera
        cancelBtn.addEventListener('click', () => {
            stream.getTracks().forEach(track => track.stop());
            document.body.removeChild(modal);
        });

        // Also close on click outside video area
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                stream.getTracks().forEach(track => track.stop());
                document.body.removeChild(modal);
            }
        });

    } catch (error) {
        console.error('Error accessing camera:', error);
        
        // Fallback to file input if camera access fails
        const imageInput = document.getElementById('imageInput');
        if (imageInput) {
            imageInput.setAttribute('capture', 'environment');
            imageInput.click();
        } else {
            showError(
                currentLang === 'hi' 
                    ? 'कैमरा तक पहुंचने में त्रुटि। कृपया फ़ाइल अपलोड का उपयोग करें।'
                    : 'Error accessing camera. Please use file upload instead.'
            );
        }
    }
}

// Handle file selection
function handleFileSelect(file) {
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewImage = document.getElementById('previewImage');
            const previewArea = document.getElementById('previewArea');
            
            if (previewImage) {
                previewImage.src = e.target.result;
            }
            
            if (previewArea) {
                previewArea.style.display = 'flex';
            }
            
            // Hide upload buttons
            const uploadButtons = document.querySelector('.upload-buttons');
            if (uploadButtons) {
                uploadButtons.style.display = 'none';
            }
        };
        reader.readAsDataURL(file);
    }
}

// Convert image to base64
function imageToBase64(imageElement) {
    return new Promise((resolve, reject) => {
        // Wait for image to load
        if (!imageElement.complete) {
            imageElement.onload = () => processImage();
            imageElement.onerror = () => reject(new Error('Image failed to load'));
        } else {
            processImage();
        }

        function processImage() {
            try {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                
                // Set canvas dimensions
                const maxWidth = 1024;
                const maxHeight = 1024;
                let width = imageElement.naturalWidth || imageElement.width;
                let height = imageElement.naturalHeight || imageElement.height;
                
                // Resize if too large
                if (width > maxWidth || height > maxHeight) {
                    const ratio = Math.min(maxWidth / width, maxHeight / height);
                    width = width * ratio;
                    height = height * ratio;
                }
                
                canvas.width = width;
                canvas.height = height;
                
                // Draw image to canvas
                ctx.drawImage(imageElement, 0, 0, width, height);
                
                // Convert to base64
                const base64 = canvas.toDataURL('image/jpeg', 0.85);
                // Remove data:image/jpeg;base64, prefix
                const base64Data = base64.split(',')[1];
                
                if (!base64Data) {
                    reject(new Error('Failed to convert image to base64'));
                    return;
                }
                
                resolve(base64Data);
            } catch (error) {
                reject(error);
            }
        }
    });
}

// Helper function to list available models (for debugging)
async function listAvailableModels() {
    try {
        if (!GEMINI_API_KEY || GEMINI_API_KEY === 'YOUR_GEMINI_API_KEY_HERE' || GEMINI_API_KEY.trim() === '') {
            console.warn('API key not configured, cannot list models');
            return [];
        }

        const apiUrl = `https://generativelanguage.googleapis.com/v1/models?key=${GEMINI_API_KEY}`;
        const response = await fetch(apiUrl);
        
        if (response.ok) {
            const data = await response.json();
            const models = data.models || [];
            console.log('Available models:', models.map(m => m.name));
            return models.map(m => m.name);
        } else {
            console.warn('Could not list models:', await response.text());
            return [];
        }
    } catch (error) {
        console.warn('Error listing models:', error);
        return [];
    }
}

// Call Gemini API for insect identification
async function callGeminiAPI(imageBase64) {
    try {
        // Check if API key is configured
        if (!GEMINI_API_KEY || GEMINI_API_KEY === 'YOUR_GEMINI_API_KEY_HERE' || GEMINI_API_KEY.trim() === '') {
            throw new Error('API_KEY_NOT_CONFIGURED');
        }

        // Create prompt based on language
        const prompt = currentLang === 'hi' 
            ? `यह एक कृषि कीट पहचान प्रणाली है। इस छवि में दिखाई देने वाले कीट की पहचान करें और निम्नलिखित जानकारी JSON format में दें:
{
  "name": "कीट का सामान्य नाम (हिंदी और अंग्रेजी दोनों)",
  "scientific": "वैज्ञानिक नाम",
  "family": "परिवार",
  "habitat": "आवास",
  "harmful": "हाँ - हानिकारक या नहीं - लाभकारी",
  "recommendation": "कीट नियंत्रण के लिए सुझाव",
  "description": "विस्तृत विवरण"
}

कृपया केवल JSON response दें, कोई अन्य text नहीं।`
            : `This is an agricultural pest identification system. Identify the insect shown in this image and provide the following information in JSON format:
{
  "name": "Common name of the insect (both English and Hindi if possible)",
  "scientific": "Scientific name",
  "family": "Family",
  "habitat": "Habitat",
  "harmful": "Yes - Harmful or No - Beneficial",
  "recommendation": "Recommendation for pest control",
  "description": "Detailed description"
}

Please provide only the JSON response, no other text.`;

        const requestBody = {
            contents: [{
                parts: [
                    {
                        text: prompt
                    },
                    {
                        inline_data: {
                            mime_type: "image/jpeg",
                            data: imageBase64
                        }
                    }
                ]
            }],
            generationConfig: {
                temperature: 0.4,
                topK: 32,
                topP: 1,
                maxOutputTokens: 2048,
            },
            safetySettings: [
                {
                    category: "HARM_CATEGORY_HARASSMENT",
                    threshold: "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    category: "HARM_CATEGORY_HATE_SPEECH",
                    threshold: "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold: "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    category: "HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold: "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        };

        // List of models to try in order (prioritize free-tier available models)
        // Free tier typically supports: gemini-1.5-flash, gemini-1.5-pro (with limits)
        const modelsToTry = [
                     // More capable model with vision - free tier supported
            'gemini-2.5-flash',            // Specific version that might be available
        ];

        // API versions to try
        const apiVersions = ['v1', 'v1beta'];

        let lastError = null;
        let lastErrorData = null;

        // Try each model with each API version
        for (const model of modelsToTry) {
            for (const version of apiVersions) {
                const apiUrl = `https://generativelanguage.googleapis.com/${version}/models/${model}:generateContent?key=${GEMINI_API_KEY}`;
                
                console.log(`Trying model: ${model} with API version: ${version}`);
                
                try {
                    const response = await fetch(apiUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(requestBody)
                    });
                    
                    if (response.ok) {
                        console.log(`Success with model: ${model} (${version})`);
                        const data = await response.json();
                        console.log('API Response:', data);
                        
                        if (!data.candidates || data.candidates.length === 0) {
                            throw new Error('No response from API. Check if the model is available.');
                        }

                        const candidate = data.candidates[0];
                        
                        if (!candidate.content || !candidate.content.parts || candidate.content.parts.length === 0) {
                            throw new Error('Invalid API response format - no content parts');
                        }

                        const responseText = candidate.content.parts[0].text;
                        
                        if (!responseText) {
                            throw new Error('Empty response from API');
                        }
                        
                        console.log('API Response Text:', responseText);
                        
                        // Extract JSON from response (handle cases where there might be extra text)
                        let jsonMatch = responseText.match(/\{[\s\S]*\}/);
                        if (!jsonMatch) {
                            // Try to parse as is
                            try {
                                const parsed = JSON.parse(responseText);
                                return parsed;
                            } catch (e) {
                                throw new Error('No valid JSON found in API response. Response: ' + responseText.substring(0, 200));
                            }
                        }

                        try {
                            const insectData = JSON.parse(jsonMatch[0]);
                            
                            // Validate required fields
                            if (!insectData.name) {
                                insectData.name = 'Unknown Insect';
                            }
                            if (!insectData.scientific) {
                                insectData.scientific = 'N/A';
                            }
                            if (!insectData.family) {
                                insectData.family = 'N/A';
                            }
                            if (!insectData.habitat) {
                                insectData.habitat = 'N/A';
                            }
                            if (!insectData.harmful) {
                                insectData.harmful = 'Unknown';
                            }
                            if (!insectData.recommendation) {
                                insectData.recommendation = 'N/A';
                            }
                            if (!insectData.description) {
                                insectData.description = 'No description available.';
                            }
                            
                            return insectData;
                        } catch (parseError) {
                            console.error('JSON Parse Error:', parseError);
                            throw new Error('Failed to parse JSON response: ' + parseError.message);
                        }
                    } else {
                        // Not successful, save error and continue
                        const errorText = await response.text();
                        let errorData = {};
                        try {
                            errorData = JSON.parse(errorText);
                        } catch (e) {
                            errorData = { error: { message: errorText } };
                        }
                        
                        lastError = response.status;
                        lastErrorData = errorData;
                        
                        console.log(`Model ${model} (${version}) failed:`, errorData.error?.message || errorText);
                        
                        // If it's an API key issue, throw immediately
                        if (response.status === 400 || response.status === 403) {
                            throw new Error(`API_KEY_INVALID: ${errorData.error?.message || 'Invalid API key or request format'}`);
                        }
                        // For 429 (rate limit) or 404 (not found), continue trying other models
                        // Rate limit might be specific to one model, so try others
                        // Continue to next model/version
                    }
                } catch (fetchError) {
                    // If it's an API key error, rethrow immediately
                    if (fetchError.message.includes('API_KEY_INVALID')) {
                        throw fetchError;
                    }
                    // For other errors (including RATE_LIMIT), continue trying other models
                    lastError = fetchError.message;
                    console.log(`Error with ${model} (${version}):`, fetchError.message);
                }
            }
        }

        // If we get here, all models failed
        console.error('All models failed. Last error:', lastErrorData);
        
        // Check if all failures were due to rate limits
        const allRateLimited = lastError === 429 || (lastErrorData?.error?.message?.includes('quota') || lastErrorData?.error?.message?.includes('rate limit'));
        
        if (lastError === 400 || lastError === 403) {
            throw new Error('API_KEY_INVALID: Invalid API key or request format');
        } else if (allRateLimited) {
            throw new Error('RATE_LIMIT: API rate limit exceeded for all available models. Please try again later or check your API quota.');
        } else {
            const errorMsg = lastErrorData?.error?.message || `All models failed. Last status: ${lastError}`;
            throw new Error(errorMsg + ' Please check your API key and available models.');
        }

    } catch (error) {
        console.error('Gemini API Error:', error);
        throw error;
    }
}

// Call Custom Model API (Your trained EfficientNetB0)
async function callCustomModelAPI(imageBase64) {
    try {
        console.log('Calling custom model API...');
        
        const response = await fetch(CUSTOM_MODEL_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageBase64
            })
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Custom model API error: ${response.status} - ${errorText}`);
        }
        
        const data = await response.json();
        console.log('Custom model response:', data);
        
        return data;
    } catch (error) {
        console.error('Custom model API error:', error);
        throw error;
    }
}

// Combine results from custom model and Gemini
function combineResults(customModelData, geminiData) {
    // Prefer custom model for classification, Gemini for detailed info
    const combined = {
        name: customModelData.name || geminiData.name || 'Unknown',
        scientific: geminiData.scientific || customModelData.scientific || 'N/A',
        family: geminiData.family || customModelData.family || 'N/A',
        habitat: geminiData.habitat || customModelData.habitat || 'N/A',
        harmful: geminiData.harmful || customModelData.harmful || 'Unknown',
        recommendation: geminiData.recommendation || customModelData.recommendation || 'N/A',
        description: geminiData.description || customModelData.description || 'No description available.',
        confidence: customModelData.confidence || null,
        model_type: 'hybrid' // Indicates both models were used
    };
    
    return combined;
}

// Analyze insect using Custom Model + Gemini API
async function analyzeInsect() {
    const previewImage = document.getElementById('previewImage');
    const resultsSection = document.getElementById('results');
    const resultImage = document.getElementById('resultImage');
    const analyzeBtn = document.querySelector('.analyze-btn');
    
    if (!previewImage || !resultsSection || !resultImage) {
        showError(currentLang === 'hi' ? 'छवि नहीं मिली' : 'Image not found');
        return;
    }

    // Check if image is loaded
    if (!previewImage.src || previewImage.src === '') {
        showError(currentLang === 'hi' ? 'कृपया पहले एक छवि अपलोड करें' : 'Please upload an image first');
        return;
    }
    
    // Show loading state
    if (analyzeBtn) {
        const originalText = analyzeBtn.textContent;
        analyzeBtn.textContent = languageData[currentLang].analyzing;
        analyzeBtn.disabled = true;
        analyzeBtn.style.opacity = '0.7';
    }

    try {
        // Convert image to base64
        const imageBase64 = await imageToBase64(previewImage);
        
        let customModelData = null;
        let geminiData = null;
        let finalData = null;
        
        // Step 1: Try custom model first (if enabled)
        if (USE_CUSTOM_MODEL_FIRST) {
            try {
                console.log('Step 1: Using custom trained model...');
                customModelData = await callCustomModelAPI(imageBase64);
                console.log('Custom model prediction:', customModelData);
            } catch (customError) {
                console.warn('Custom model failed, will use Gemini:', customError);
                // Continue to Gemini
            }
        }
        
        // Step 2: Use Gemini for detailed information
        try {
            console.log('Step 2: Getting detailed info from Gemini...');
            geminiData = await callGeminiAPI(imageBase64);
            console.log('Gemini response:', geminiData);
        } catch (geminiError) {
            console.warn('Gemini API failed:', geminiError);
            // If we have custom model data, use it
            if (customModelData) {
                finalData = customModelData;
            } else {
                throw geminiError;
            }
        }
        
        // Step 3: Combine results
        if (customModelData && geminiData) {
            finalData = combineResults(customModelData, geminiData);
            console.log('Combined results from both models:', finalData);
        } else if (customModelData) {
            finalData = customModelData;
            console.log('Using custom model only:', finalData);
        } else if (geminiData) {
            finalData = geminiData;
            console.log('Using Gemini only:', finalData);
        } else {
            throw new Error('Both models failed');
        }
        
        // Update result image
        resultImage.src = previewImage.src;
        
        // Update result information
        const insectNameEl = document.getElementById('insectName');
        const scientificNameEl = document.getElementById('scientificName');
        const familyEl = document.getElementById('family');
        const habitatEl = document.getElementById('habitat');
        const harmfulEl = document.getElementById('harmful');
        const recommendationEl = document.getElementById('recommendation');
        const descriptionEl = document.getElementById('description');
        
        if (insectNameEl) {
            let nameText = finalData.name || 'Unknown';
            if (finalData.confidence) {
                nameText += ` (${(finalData.confidence * 100).toFixed(1)}% confidence)`;
            }
            insectNameEl.textContent = nameText;
        }
        if (scientificNameEl) scientificNameEl.textContent = finalData.scientific || 'N/A';
        if (familyEl) familyEl.textContent = finalData.family || 'N/A';
        if (habitatEl) habitatEl.textContent = finalData.habitat || 'N/A';
        if (harmfulEl) harmfulEl.textContent = finalData.harmful || 'N/A';
        if (recommendationEl) recommendationEl.textContent = finalData.recommendation || 'N/A';
        if (descriptionEl) {
            let descText = finalData.description || 'No description available.';
            if (finalData.model_type === 'hybrid') {
                descText += ' (Enhanced with AI analysis)';
            }
            descriptionEl.textContent = descText;
        }
        
        // Show results section
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        console.error('Analysis error:', error);
        
        // Handle specific errors
        if (error.message === 'API_KEY_NOT_CONFIGURED') {
            showError(
                currentLang === 'hi' 
                    ? 'कृपया config.js में अपना Gemini API Key डालें' 
                    : 'Please add your Gemini API Key in config.js'
            );
            return;
        } else if (error.message.includes('API_KEY_INVALID') || error.message.includes('403') || error.message.includes('400')) {
            showError(
                currentLang === 'hi' 
                    ? 'API Key गलत है या invalid है। कृपया सही API Key डालें और Google AI Studio में verify करें।' 
                    : 'Invalid API Key. Please check your API key and verify it in Google AI Studio.'
            );
            console.error('API Key Error Details:', error.message);
            return;
        } else if (error.message.includes('RATE_LIMIT') || error.message.includes('429')) {
            showError(
                currentLang === 'hi' 
                    ? 'API limit पार हो गई है। कृपया कुछ समय बाद कोशिश करें' 
                    : 'API rate limit exceeded. Please try again later'
            );
            return;
        } else if (error.message.includes('Custom model API error') || error.message.includes('Failed to fetch')) {
            // Custom model backend not available, try Gemini only
            console.log('Custom model unavailable, using Gemini only...');
            try {
                const imageBase64 = await imageToBase64(previewImage);
                const geminiData = await callGeminiAPI(imageBase64);
                // Use Gemini data directly
                const insectNameEl = document.getElementById('insectName');
                const scientificNameEl = document.getElementById('scientificName');
                const familyEl = document.getElementById('family');
                const habitatEl = document.getElementById('habitat');
                const harmfulEl = document.getElementById('harmful');
                const recommendationEl = document.getElementById('recommendation');
                const descriptionEl = document.getElementById('description');
                
                if (insectNameEl) insectNameEl.textContent = geminiData.name || 'Unknown';
                if (scientificNameEl) scientificNameEl.textContent = geminiData.scientific || 'N/A';
                if (familyEl) familyEl.textContent = geminiData.family || 'N/A';
                if (habitatEl) habitatEl.textContent = geminiData.habitat || 'N/A';
                if (harmfulEl) harmfulEl.textContent = geminiData.harmful || 'N/A';
                if (recommendationEl) recommendationEl.textContent = geminiData.recommendation || 'N/A';
                if (descriptionEl) descriptionEl.textContent = geminiData.description || 'No description available.';
                
                resultImage.src = previewImage.src;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth' });
                return;
            } catch (geminiError) {
                // Both failed, use fallback
                showError(
                    currentLang === 'hi' 
                        ? 'Backend unavailable. Fallback data use हो रहा है।' 
                        : 'Backend unavailable. Using fallback data.'
                );
                useFallbackData(previewImage, resultsSection, resultImage);
                return;
            }
        } else if (error.message.includes('Image failed to load') || error.message.includes('Failed to convert')) {
            showError(
                currentLang === 'hi' 
                    ? 'छवि लोड करने में समस्या। कृपया दूसरी छवि try करें।' 
                    : 'Error loading image. Please try a different image.'
            );
            return;
        } else {
            // Show error but also try fallback
            showError(
                currentLang === 'hi' 
                    ? 'Error: ' + error.message + ' - Fallback data use हो रहा है' 
                    : 'Error: ' + error.message + ' - Using fallback data'
            );
            // Fallback to sample data if API fails
            console.log('Using fallback data due to error');
            useFallbackData(previewImage, resultsSection, resultImage);
        }
    } finally {
        // Reset analyze button
        if (analyzeBtn) {
            analyzeBtn.textContent = languageData[currentLang].analyze;
            analyzeBtn.disabled = false;
            analyzeBtn.style.opacity = '1';
        }
    }
}

// Use fallback sample data if API fails
function useFallbackData(previewImage, resultsSection, resultImage) {
    const insectKeys = Object.keys(sampleInsects);
    const randomKey = insectKeys[Math.floor(Math.random() * insectKeys.length)];
    const insect = sampleInsects[randomKey][currentLang];
    
    resultImage.src = previewImage.src;
    
    const insectNameEl = document.getElementById('insectName');
    const scientificNameEl = document.getElementById('scientificName');
    const familyEl = document.getElementById('family');
    const habitatEl = document.getElementById('habitat');
    const harmfulEl = document.getElementById('harmful');
    const recommendationEl = document.getElementById('recommendation');
    const descriptionEl = document.getElementById('description');
    
    if (insectNameEl) insectNameEl.textContent = insect.name;
    if (scientificNameEl) scientificNameEl.textContent = insect.scientific;
    if (familyEl) familyEl.textContent = insect.family;
    if (habitatEl) habitatEl.textContent = insect.habitat;
    if (harmfulEl) harmfulEl.textContent = insect.harmful;
    if (recommendationEl) recommendationEl.textContent = insect.recommendation;
    if (descriptionEl) descriptionEl.textContent = insect.description;
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Show error message
function showError(message) {
    // Create or update error message element
    let errorDiv = document.getElementById('errorMessage');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.id = 'errorMessage';
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #E74C3C;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            z-index: 10000;
            max-width: 400px;
            animation: slideIn 0.3s ease;
        `;
        document.body.appendChild(errorDiv);
    }
    
    errorDiv.textContent = message;
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (errorDiv) {
            errorDiv.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                if (errorDiv && errorDiv.parentNode) {
                    errorDiv.parentNode.removeChild(errorDiv);
                }
            }, 300);
        }
    }, 5000);
}

// Setup smooth scroll for navigation links
function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const target = document.querySelector(targetId);
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Add animation on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe elements for animation
document.addEventListener('DOMContentLoaded', () => {
    const animatedElements = document.querySelectorAll('.step-card, .insect-item, .problem-list li, .solution-list li');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});
