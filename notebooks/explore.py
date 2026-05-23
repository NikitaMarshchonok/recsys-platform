import pandas as pd

movies = pd.read_csv("data/raw/movies.csv")
users = pd.read_csv("data/raw/users.csv")
ratings = pd.read_csv("data/raw/ratings.csv")

print("=== Фильмы ===")
print(movies.head())
print(f"Форма: {movies.shape}")

print("\n=== Оценки ===")
print(ratings.head())
print(f"Форма: {ratings.shape}")

print("\n=== Статистика оценок ===")
print(ratings["rating"].describe())

print("\n=== Топ жанров ===")
print(movies["genres"].value_counts())

print("\n=== Сколько фильмов оценил каждый пользователь (среднее) ===")
ratings_per_user = ratings.groupby("userId")["movieId"].count()
print(f"Среднее: {ratings_per_user.mean():.1f}")
print(f"Минимум: {ratings_per_user.min()}")
print(f"Максимум: {ratings_per_user.max()}")