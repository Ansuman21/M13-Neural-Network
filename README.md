# From Headlines to Hits: Predicting News Article Popularity with Deep Learning

## Overview
This project focuses on predicting the popularity level of news articles based on various features using machine learning and neural networks. The task involves building, evaluating, and selecting the best model for predicting whether a news article will gain high or low popularity, as measured by its engagement or viewership.

## Business Case
In the competitive world of digital content, predicting the popularity of news articles helps media outlets, bloggers, and marketers tailor their content strategy. By identifying which articles are likely to become popular, businesses can optimize content delivery, enhance user engagement, and maximize advertising revenue.

## Key Steps in Solving the Problem

### 1. **Data Collection & Preprocessing**
   - **Dataset:** The dataset used for this project includes features such as article text, metadata, and user engagement metrics.
   - **Data Cleaning:** 
     - Checked for missing values, duplicates, and inconsistencies.
     - Normalized data where necessary to ensure consistency.
   - **Exploratory Data Analysis (EDA):**
     - Analyzed feature distributions and relationships to understand the dataset better.
     - Visualized key patterns to guide model selection.

### 2. **Feature Engineering & Selection**
   - **Feature Selection:** Utilized techniques like **Random Forest feature importance** and **Principal Component Analysis (PCA)** to identify the most relevant features.
   - **Text Processing:** Transformed textual data into features suitable for neural network models (e.g., using **TF-IDF** or **word embeddings**).

### 3. **Model Selection & Training**
   Three different models were explored:
   - **Multilayer Perceptron (MLP):** A fully connected feedforward neural network.
   - **RBM with Logistic Regression:** A hybrid approach using a **Restricted Boltzmann Machine (RBM)** for feature extraction and **Logistic Regression** for classification.
   - **Convolutional Neural Network (CNN):** A CNN model typically used for image and sequential data, adapted for text data here.

### 4. **Model Evaluation**
   - Models were evaluated using the following metrics:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1-Score**
   - **Cross-Validation** was applied to assess each model's robustness.

### 5. **Results and Insights**
   - **MLP Model:** The best-performing model with the highest accuracy (~0.59), balanced precision (~0.53), recall (~0.59), and a favorable F1-score (~0.50).
   - **RBM with Logistic Regression:** Achieved high precision (~0.76) but had lower recall and F1-score (~0.43), indicating a trade-off between false positives and false negatives.
   - **CNN Model:** Achieved similar recall (~0.58) to MLP but had much lower precision (~0.35), leading to more false positives.

### 6. **Model Selection**
   After evaluating the models based on performance metrics, **MLP** was selected as the best model for predicting news article popularity. Its consistency across various metrics and higher overall performance made it the preferred choice.

## Recommendations
1. **Content Strategy Optimization:** Use the MLP model to predict popular articles, which can help tailor marketing strategies and content promotion.
2. **Real-time Predictions:** Integrate the model into a news platform to provide real-time popularity predictions for new articles, enhancing the user experience.
3. **Model Improvement:** Explore other advanced models like **Recurrent Neural Networks (RNNs)** or **Transformers** for better performance in handling sequential data like article text.

## Conclusion
This project demonstrates how neural networks can be effectively used to predict the popularity of news articles. By employing techniques such as data preprocessing, exploratory analysis, feature selection, and model evaluation, we were able to build a robust predictive model that outperformed alternative approaches.

## Author
**Ansuman Patnaik**  
MS in Data Science & Analytics, Yeshiva University  
Email: ansu1p89k@gmail.com
