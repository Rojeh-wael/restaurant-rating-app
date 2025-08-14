🍽️ Restaurant Rating Prediction Web App
A comprehensive machine learning web application that analyzes restaurant data and predicts ratings with interactive visualizations and real-time predictions.
🌟 Features
📊 Interactive Analysis Dashboard

Distribution Analysis: Comprehensive visualization of numerical features including ratings, costs, and votes
Categorical Analysis:

Top 10 restaurant locations with highest restaurant density
Most common restaurant types and service categories
Online ordering and table booking availability analysis

Location Intelligence:

Restaurant distribution across different areas
Service availability (online orders & table booking) by location
Average pricing analysis by location

Correlation Analysis:

Heatmap of feature relationships
Cost vs Rating scatter plots
Restaurant type performance analysis

🔮 Advanced Prediction Engine

Dual Model Architecture:

Machine Learning Model: Optimized traditional ML algorithm
Neural Network Model: Deep learning with confidence scoring

Real-time Predictions: Instant rating predictions based on restaurant characteristics
Confidence Analysis: Detailed breakdown of prediction certainty
Interactive Input Form: User-friendly interface for entering restaurant details

🛠️ Technologies Used

Frontend: Streamlit (Interactive web interface)
Data Processing: Pandas, NumPy
Visualization: Plotly (Interactive charts), Matplotlib, Seaborn
Machine Learning: Scikit-learn, TensorFlow/Keras
Model Persistence: Joblib
Deployment: Streamlit Cloud

🚀 Quick Start
Run Locally

Clone the repository:
bashgit clone https://github.com/Rojeh-wael/restaurant-rating-app.git
cd restaurant-rating-app

Install dependencies:
bashpip install -r requirements.txt

Launch the application:
bashstreamlit run app.py

Open your browser to http://localhost:8501

Live Demo
🌐 Try the Live App Here!
📊 Dataset Overview
The application analyzes comprehensive restaurant data including:

Location Data: Geographic distribution and area-wise analysis
Service Features: Online ordering, table booking availability
Restaurant Categories: Types, cuisines, and service modes
Financial Data: Cost for two people, pricing analysis
Performance Metrics: Customer ratings, vote counts
Service Types: Delivery, Dine-out, Cafes, etc.

🤖 Machine Learning Models
Model Architecture

Traditional ML: Best performing model from comparative analysis (Random Forest/XGBoost)
Neural Network: Multi-layer deep learning model with dropout regularization
Feature Engineering: OneHot encoding, ordinal encoding, standard scaling
Prediction Categories:

Poor (0-2.5): Below average restaurants
Good (2.5-4.0): Above average restaurants
Excellent (4.0-5.0): Top-rated restaurants

Model Performance

Accuracy: High prediction accuracy across multiple metrics
Confidence Scoring: Neural network provides prediction certainty
Real-time Processing: Optimized for instant predictions

🎯 Key Insights from Analysis
Location Intelligence

Top Performing Areas: Identification of restaurant hotspots
Service Distribution: Geographic spread of online services
Price Analysis: Cost variations across different locations

Business Intelligence

Service Adoption: Online ordering and table booking trends
Restaurant Categories: Most successful restaurant types
Rating Patterns: Factors influencing customer satisfaction

📁 Project Structure
restaurant-rating-app/
│
├── app.py                 # Main Streamlit application
├── raw.csv               # Restaurant dataset
├── best_model.pkl        # Trained ML model
├── my_model.keras        # Neural network model
├── encoder.pkl           # OneHot encoder for categorical features
├── ord_enc.pkl          # Ordinal encoder for binary features
├── scaler.pkl           # Standard scaler for numerical features
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Git ignore file
🔄 Making Predictions
Input Features
The prediction system uses the following features:

Online Order: Yes/No availability
Table Booking: Yes/No availability
Votes: Number of customer reviews
Location: Restaurant area/location
Restaurant Type: Category (Quick Bites, Casual Dining, etc.)
Cuisines: Type of food served
Cost for 2 Plates: Price range
Service Type: Delivery, Dine-out, Cafes, etc.

Output

ML Model Prediction: Rating category (Poor/Good/Excellent)
Neural Network Prediction: Rating category with confidence scores
Confidence Breakdown: Percentage confidence for each category

🎨 Visualization Features

Interactive Charts: Plotly-powered responsive visualizations
Multi-tab Interface: Organized analysis sections
Real-time Filtering: Dynamic data exploration
Professional Styling: Modern, clean interface design
Mobile Responsive: Works on all device sizes

🔧 Technical Implementation
Data Processing Pipeline

Data Cleaning: Automatic handling of rating formats ("3.8/5" → 3.8)
Feature Encoding: Categorical and binary feature transformation
Scaling: Numerical feature standardization
Model Loading: Cached model loading for performance

Performance Optimizations

Caching: @st.cache_data and @st.cache_resource for faster loading
Memory Management: Efficient data handling for large datasets
Responsive Design: Optimized for various screen sizes

🤝 Contributing
Contributions are welcome! Please feel free to:

Fork the repository
Create a feature branch: git checkout -b feature/amazing-feature
Commit your changes: git commit -m 'Add amazing feature'
Push to the branch: git push origin feature/amazing-feature
Open a Pull Request

📈 Future Enhancements

 Advanced Filtering: More sophisticated data filtering options
 Recommendation System: Restaurant recommendation engine
 Geospatial Analysis: Map-based visualizations
 API Integration: Real-time data updates
 A/B Testing: Model comparison interface
 Export Features: Download predictions and visualizations

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
👨‍💻 Author
Your Name

GitHub: Rojeh-wael
LinkedIn: www.linkedin.com/in/rojeh-wael-45896924a
Email: rojehwael@yahoo.com

🙏 Acknowledgments

Dataset providers and contributors
Streamlit community for excellent documentation
Open source machine learning libraries
Restaurant industry for inspiring this analysis


📞 Support
If you found this project helpful, please give it a ⭐ on GitHub!
For questions or support, please open an issue in the GitHub repository.
Built with Python and Streamlit