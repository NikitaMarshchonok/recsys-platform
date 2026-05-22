import pandas as pd
import numpy as np
import os

np.random.seed(42)

# 500 пользователей, 200 фильмов, 20000 оценок
N_USERS = 500
N_MOVIES = 200
N_RATINGS = 20000

# Фильмы с жанрами
genres = ["Action", "Comedy", "Drama", "Thriller", "Romance", "Sci-Fi"]
movies = pd.DataFrame({
    "movieId": range(1, N_MOVIES + 1),
    "title": [f"Movie_{i}" for i in range(1, N_MOVIES + 1)],
    "genres": np.random.choice(genres, N_MOVIES),
})

# Пользователи
users = pd.DataFrame({
    "userId": range(1, N_USERS + 1),
    "age_group": np.random.choice(["18-25", "26-35", "36-50", "50+"], N_USERS),
})

# Оценки — реалистичные (не случайные)
# Пользователи предпочитают определённые жанры
user_genre_pref = {}
for uid in range(1, N_USERS + 1):
    # каждый пользователь любит 1-2 жанра
    user_genre_pref[uid] = np.random.choice(genres, 2, replace=False)

ratings_data = []
for _ in range(N_RATINGS):
    uid = np.random.randint(1, N_USERS + 1)
    mid = np.random.randint(1, N_MOVIES + 1)
    movie_genre = movies.loc[mid - 1, "genres"]
    
    # если жанр совпадает с предпочтением — оценка выше
    if movie_genre in user_genre_pref[uid]:
        rating = np.clip(np.random.normal(4.0, 0.7), 1, 5)
    else:
        rating = np.clip(np.random.normal(2.5, 1.0), 1, 5)
    
    ratings_data.append({
        "userId": uid,
        "movieId": mid,
        "rating": round(rating, 1),
        "timestamp": np.random.randint(1000000000, 1700000000),
    })

ratings = pd.DataFrame(ratings_data).drop_duplicates(subset=["userId", "movieId"])

# Сохраняем
os.makedirs("data/raw", exist_ok=True)
movies.to_csv("data/raw/movies.csv", index=False)
users.to_csv("data/raw/users.csv", index=False)
ratings.to_csv("data/raw/ratings.csv", index=False)

print(f"Фильмов: {len(movies)}")
print(f"Пользователей: {len(users)}")
print(f"Оценок: {len(ratings)}")
print("Данные сохранены в data/raw/")