import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import pyspark
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.sql.functions import col, explode, split, avg
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import udf,row_number
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
from pyspark.sql import Row


def train_als_model(train_data, rank=10, maxIter=5, regParam=0.1):
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=rank,
        maxIter=maxIter,
        regParam=regParam,
        coldStartStrategy="drop"
    )
    return als.fit(train_data)


def predict_rating(user_id, genre, model, movies):
    # Create a DataFrame with single row for prediction
    schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True)
    ])
    data = [(user_id, None)]
    predict_df = spark.createDataFrame(data, schema=schema)

    # Get the top recommendations for the user in the specified genre
    recommendations = model.recommendForUserSubset(predict_df, 10)

    # Extract movie recommendations
    movie_recommendations = recommendations.selectExpr("userId", "explode(recommendations) as rec") \
        .select("userId", col("rec.movieId").alias("movieId"))

    # Join with movies_df to get the movie titles
    recommendations_with_titles = movie_recommendations.join(movies, "movieId") \
        .select("userId", "movieId", "title")

    return recommendations_with_titles

# Initialize Spark session with increased driver memory
spark = SparkSession.builder \
    .appName("MovieLens Recommendation System") \
    .config("spark.driver.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.cores", "2") \
    .config("spark.task.cpus", "2")\
    .config("spark.ui.port", "4041") \
    .config("spark.eventLog.gcMetrics.youngGenerationGarbageCollectors", "G1 Young Generation") \
    .config("spark.eventLog.gcMetrics.oldGenerationGarbageCollectors", "G1 Old Generation") \
    .getOrCreate()

# Set the log level to WARN to reduce the amount of log output
spark.sparkContext.setLogLevel("WARN")

# Load the movies data
movies_file_path = '/Users/ziweijin/Desktop/SCS3252/FinalProject/ml-20m/movies.csv'
movies = pd.read_csv(movies_file_path)

# Load the ratings data
ratings_file_path = '/Users/ziweijin/Desktop/SCS3252/FinalProject/ml-20m/ratings.csv'
ratings = pd.read_csv(ratings_file_path)

# Merge the datasets on movieId
merged_data = pd.merge(ratings, movies, on='movieId').dropna()

# Load your data into a Spark DataFrame
schema = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", FloatType(), True),
    StructField("timestamp", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("genres", StringType(), True)
])

all_data_df = spark.createDataFrame(merged_data, schema=schema)

# Repartition the DataFrame to a higher number of partitions
all_data_df = all_data_df.repartition(200)

# Split train test set
train_df, test_df = all_data_df.randomSplit([0.8, 0.2], seed=42)

# Split unique genres from data
unique_genres = all_data_df.select(explode(split(col("genres"), "\\|")).alias("genre")).distinct().collect()

# Get unique genres
unique_genres = [row.genre for row in unique_genres]

# Create a dictionary to hold DataFrames for each genre
genre_data = {genre: train_df.filter(col("genres").contains(genre)) for genre in unique_genres}
print(genre_data)

# Dictionary to hold the models
als_models = {}

# Train a model for each genre
for genre in unique_genres:
    genre_train_df = genre_data[genre]

    if genre_train_df.count() > 0:

        # Train the ALS model
        model = train_als_model(genre_train_df)

        # Save the model in the dictionary
        als_models[genre] = model

        print(f"Model trained for genre: {genre}")
    else:
        print(f"No data available for genre: {genre}")

# Evaluate each model on the test set
for genre, model in als_models.items():
    # Filter test data for the specific genre
    genre_test_df = test_df.filter(col("genres").contains(genre))

    # Make predictions on the test set
    predictions = model.transform(genre_test_df)

    # Remove NaN values that could result from the ALS cold start strategy
    predictions = predictions.na.drop()

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Root-mean-square error (RMSE) for genre {genre}: {rmse}")

# Load movie data
movies = spark.read.csv('/Users/ziweijin/Desktop/SCS3252/FinalProject/ml-20m/movies.csv', header=True, inferSchema=True)

# Example usage
user_id = 1
genre = "Action"

if genre in als_models:
    model = als_models[genre]
    predictions = predict_rating(user_id, genre, model, movies)
    predictions.show()
else:
    print(f"No model available for genre: {genre}")


# Dictionary mapping emotions to genres
emotion_genre_map = {
    "Suspense": "Crime",
    "Loneliness": "Romance",
    "Fear": "Thriller",
    "Anticipation": "Adventure",
    "Sadness": "Drama",
    "Anger": "War",
    "Curiosity": "Documentary",
    "Surprise": "Fantasy",
    "Confusion": "Mystery",
    "Happiness": "Musical",
    "Excitement": "Animation",
    "Disgust": "Film-Noir",
    "Awe": "IMAX",
    "Nostalgia": "Western",
    "Boredom": "Children",
    "Excitement": "Action",
    "Anticipation": "Sci-Fi"
}

# Define a function to get genre based on emotion
def get_genre(emotion):
    return emotion_genre_map.get(emotion, "Unknown")

# Register the function as a UDF
get_genre_udf = udf(get_genre, StringType())


# Function to get top 10 movies for a specific genre
def get_top_10_for_genre(df, genre):
    # Step 1: Split and explode genres
    df_exploded = df.withColumn("genre", explode(split(col("genres"), "\\|")))

    # Step 2: Filter for the specific genre
    genre_df = df_exploded.filter(col("genre") == genre)

    # Step 3: Calculate average ratings for the specific genre
    avg_ratings = genre_df.groupBy("title").agg(avg("rating").alias("avg_rating"))

    # Step 4: Order by average rating and limit to top 10
    top_10_movies = avg_ratings.orderBy(col("avg_rating").desc()).limit(10)

    return top_10_movies


# Function to recommend movies
def recommend_movies(user_id, emotion):
    user_exists = all_data_df.filter(col("userId") == user_id).count() > 0

    if user_exists:

        if emotion in emotion_genre_map:
            genre = emotion_genre_map[emotion]
            model = als_models.get(genre)

            if model:
                top_movies = predict_rating(user_id, genre, model, movies)
                # predictions.show()
            else:
                print(f"No model available for genre: {genre}, here's the list of top 10 rated movies among all genres")

                # Use a window function to get the top 10 movies for each genre
                top_movies = get_top_10_for_genre(all_data_df, genre)
                # top_movies.show()
        else:
            print(
                f"No genre mapping available for emotion: {emotion}, here's the list of top 10 rated movies among all genres")
            t  # Calculate average rating for each movie
            average_ratings_df = all_data_df.groupBy("title").agg(avg("rating").alias("average_rating"))
            top_movies = average_ratings_df.orderBy("average_rating", ascending=False).limit(10)
    else:
        if emotion in emotion_genre_map:
            genre = emotion_genre_map[emotion]
            print(f"No information for this user: {user_id}, here's the list of top 10 rated movies for your emotion.")
            top_movies = get_top_10_for_genre(all_data_df, genre)
            # top_movies.show()
        else:
            print(
                f"No information for this user: {user_id}, No genre mapping available for emotion: {emotion}, here's the list of top 10 rated movies among all genres")
            # Calculate average rating for each movie
            average_ratings_df = all_data_df.groupBy("title").agg(avg("rating").alias("average_rating"))
            top_movies = average_ratings_df.orderBy("average_rating", ascending=False).limit(10)

            # top_movies.show()

    return top_movies.select("title")



# Example usage
user_id = 1
emotion = "Excitement"
recommended_movies = recommend_movies(user_id, emotion)
recommended_movies.show()

user_id = 333
emotion = "Suspense"
recommended_movies = recommend_movies(user_id, emotion)
recommended_movies.show()

user_id = 123456789
emotion = "Nostalgia"
recommended_movies = recommend_movies(user_id, emotion)
recommended_movies.show()

user_id = 123456789
emotion = "Fearless"
recommended_movies = recommend_movies(user_id, emotion)
recommended_movies.show()

spark.stop()