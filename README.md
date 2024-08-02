# Movie Recommendation System

This methodology outlines the comprehensive process from data preparation to model training and evaluation, culminating in a functional recommendation system tailored to user emotions.
To handle the large-scale data processing required for this project, we utilized PySpark. We began by importing the necessary libraries, including **PySpark** and **pandas**, and initialized a Spark session.

We used two primary datasets: ratings.csv and movies.csv. One containing movie ratings and another containing movie genres. These datasets were read into Spark DataFrames and then merged to associate ratings with their respective movie genres.

Before training the model, we applied data cleaning steps to ensure the datasets were appropriately formatted. This included dropping any rows with missing values. The data was split to training data(80%) for model training purpose and testing data(20%) for model evaluation purpose.


We employed the **Alternating Least Squares (ALS)** algorithm, a popular method for collaborative filtering, to build our recommendation model. The ALS model was trained separately for each genre to cater to specific user preferences.

For each genre in the dataset, we filtered the data to include only movies of that genre. An ALS model(als_models) was then instantiated and trained using this filtered dataset.
Model Evaluation

The ALS models were evaluated to ensure they provided accurate recommendations. 

For the final recommendation system, we defined a function(recommend_movies) to recommend movies based on user ID and the user's emotion. The function is designed to provide personalized movie recommendations based on a user's ID and their current emotion. Initially, the function checks if there is existing data for the given user ID in the dataset. If user data is available and the specified emotion maps to a known genre using the emotion_genre_map dictionary, the function retrieves the corresponding genre model from the pre-trained ALS models(als_models). It then predicts and displays movie ratings tailored to the user's preferences within that genre. If the user's emotion does not map to any genre, the function defaults to displaying the top 10 highest-rated movies across all genres. In cases where there is no data for the specified user ID, the function still checks the emotion-to-genre mapping and provides the top 10 highest-rated movies within the relevant genre. If no genre is associated with the emotion, it defaults to showing the top 10 movies across all genres. This approach ensures that every user receives meaningful recommendations, even if they are new to the system or their specific preferences are not directly available.
