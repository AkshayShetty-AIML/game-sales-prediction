# game-sales-prediction
**Overview:**
This project aims to predict video game sales based on features like platform, genre, publisher, and user scores. Using machine learning models, we analyze historical sales data to understand the impact of various factors on game success.

🔹 Key Objectives:
✅ Clean and preprocess game sales data
✅ Perform exploratory data analysis (EDA)
✅ Train regression models to predict game sales
✅ Tune models for better accuracy
✅ Deploy the final model using an API

📊 Dataset
Source: Kaggle - Video Game Sales Dataset
Description: The dataset contains sales data for thousands of video games across multiple platforms.
Columns:
Name – Game title
Platform – Console/platform (PS4, Xbox, PC, etc.)
Year_of_Release – Year the game was released
Genre – Game category (Action, RPG, Sports, etc.)
Publisher – Publishing company
NA_Sales, EU_Sales, JP_Sales, Global_Sales – Sales data in millions

Programming Language: Python
Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost
Model Deployment: Flask/FastAPI, Streamlit
Version Control: GitHub
Cloud Hosting: AWS/Render/Hugging Face Spaces

📌 Project Structure
📂 game-sales-prediction/
│── 📂 data/                     # Dataset folder
│── 📂 notebooks/                # Jupyter Notebooks for analysis
│── 📂 scripts/                  # Python scripts for ML pipeline
│── 📂 models/                   # Saved models
│── 📂 deployment/               # API & web app deployment
│── README.md                    # Project documentation
│── requirements.txt             # Dependencies
│── .gitignore                   # Files to ignore

🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/your-username/game-sales-prediction.git
cd game-sales-prediction

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Data Preprocessing
python scripts/data_preprocessing.py

4️⃣ Train the Model
python scripts/model_training_evaluation.py

5️⃣ Run the API (for Predictions)
cd deployment
python app.py

Access the API at http://127.0.0.1:5000/predict

📊 Exploratory Data Analysis (EDA)
Check missing values & clean data
Visualize sales trends across different platforms and genres
Identify correlations between features
📌 Find detailed analysis in notebooks/data_exploration.ipynb

📝 Future Improvements
🔹 Include more features like user reviews and critic scores
🔹 Try deep learning models for better accuracy
🔹 Deploy on cloud services for scalability

📌 Contributing
Fork the repository
Create a new branch (feature/improvement)
Commit your changes
Push to GitHub & create a PR

📜 License
This project is open-source under the MIT License.

📌 Author: Akshay Shetty

🔥 Star ⭐ the repo if you found it useful! 🚀
