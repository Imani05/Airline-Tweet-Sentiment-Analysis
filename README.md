# Airline Tweet Sentiment Analysis

## Overview
This project focuses on improving sentiment analysis of airline tweets using advanced deep learning techniques, including Bidirectional Long Short-Term Memory (BiLSTM) neural networks, Word2Vec embeddings, and negative sentiment reason analysis. The goal is to provide more accurate and interpretable sentiment classifications that can help airlines better understand and respond to customer feedback.

## Key Features
- **BiLSTM Neural Networks**: Utilizes BiLSTM to capture contextual information from both past and future words in tweets.
- **Word2Vec Embeddings**: Employs Word2Vec to generate semantic word embeddings, enhancing the model's understanding of language.
- **Negative Sentiment Reason Analysis**: Incorporates the reasons behind negative sentiments to improve the interpretability of the sentiment analysis model.
- **High Performance**: Achieves an accuracy of 91.94% and a weighted average F1-score of 91.91%.

## Dataset
The dataset used in this project contains 14,640 tweets related to six major US airlines, labeled with positive, neutral, or negative sentiments. The data includes features like tweet text, airline name, sentiment label, and negative sentiment reason.

## Methodology
1. **Data Preprocessing**: Includes cleaning, tokenization, stopword removal, and feature engineering.
2. **Model Architecture**: The sentiment analysis model is built using a BiLSTM architecture, with Word2Vec embeddings and negative sentiment reasons as additional features.
3. **Training and Evaluation**: The model is trained and evaluated using standard metrics such as accuracy, precision, recall, and F1-score.

## Results
- The proposed model outperforms traditional machine learning approaches, achieving a significant improvement in accuracy and interpretability.
- The model demonstrates strong generalization capabilities on unseen data, with an accuracy of 91.94%.

## Future Work
- Explore the use of transformer models and attention mechanisms to further improve sentiment analysis performance.
- Address class imbalance issues through advanced techniques like data augmentation or cost-sensitive learning.
- Enhance model explainability to provide more actionable insights for airlines.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Airline-Tweet-Sentiment-Analysis.git
   cd Airline-Tweet-Sentiment-Analysis
