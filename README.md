# Movie Recommendation System using MovieLens 20M Dataset

## Overview  
This project implements a **movie recommendation system** using the **MovieLens 20M dataset**. The system applies the **Apriori algorithm** to discover association rules between movies based on user interactions and provides recommendations accordingly. The project includes both a **backend for processing recommendations** and a **Tkinter-based graphical user interface (GUI) for user interaction**.

## Features  
- **Frequent Pattern Mining**: Utilizes the **Apriori algorithm** to identify patterns in user movie preferences.  
- **Association Rule-Based Recommendations**: Suggests movies based on user viewing history.  
- **User Interaction via GUI**: Allows users to select their ID and receive recommendations.  
- **Dataset Processing & Cleaning**: Handles large-scale data from MovieLens 20M.  

## Dataset - MovieLens 20M  
The **MovieLens 20M dataset** contains:  
- **20 million ratings** from **138,000 users** for **27,000 movies**.  
- Additional metadata including **movie genres and titles**.  

The dataset is structured into multiple CSV files:  
- `ratings.csv` – Contains user ratings for movies.  
- `movies.csv` – Contains movie titles and genres.  
- `tags.csv` – Contains user-generated tags for movies.  

## Methodology  
1. **Data Preprocessing**  
   - Load `ratings.csv` to create a **user-movie interaction matrix**.  
   - Convert sparse data into a format suitable for Apriori.  
2. **Frequent Itemset Mining with Apriori**  
   - Identify movies that are commonly watched together.  
   - Generate **association rules** with support and confidence thresholds.  
3. **Movie Recommendations**  
   - Recommend movies based on previously watched films.  
   - Filter recommendations to ensure they are **not already watched**.  
4. **GUI Implementation**  
   - Enable users to **select their ID** and receive **real-time movie recommendations**.  

## Technologies Used  
- **Python** for backend processing.  
- **pandas, NumPy** for data manipulation.  
- **mlxtend** for implementing the Apriori algorithm.  
- **Tkinter** for the user interface.  
- **SciPy (csr_matrix)** for efficient sparse matrix representation.  

## Usage  
1. **Run the Apriori model** to generate frequent itemsets and association rules.  
2. **Launch the GUI** to interact with the recommendation system.  
   - Select a user ID from the provided list.  
   - View movie recommendations based on historical data.  

## Installation  
1. Clone this repository:  
   ```bash
   git clone https://github.com/bengubal/yazilim-gel-1
