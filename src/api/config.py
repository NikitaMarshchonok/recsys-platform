import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    model_path: Path
    model_card_path: Path
    movies_path: Path
    ratings_path: Path
    users_path: Path
    user_features_path: Path
    movie_features_path: Path
    db_host: str
    db_name: str
    db_user: str
    db_password: str
    db_port: int


def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[2]

    return Settings(
        base_dir=base_dir,
        model_path=Path(os.getenv("RECSYS_MODEL_PATH", base_dir / "models" / "als_model")),
        model_card_path=Path(
            os.getenv("RECSYS_MODEL_CARD_PATH", base_dir / "models" / "als_model" / "model_card.json")
        ),
        movies_path=Path(os.getenv("RECSYS_MOVIES_PATH", base_dir / "data" / "raw" / "movies.csv")),
        ratings_path=Path(os.getenv("RECSYS_RATINGS_PATH", base_dir / "data" / "raw" / "ratings.csv")),
        users_path=Path(os.getenv("RECSYS_USERS_PATH", base_dir / "data" / "raw" / "users.csv")),
        user_features_path=Path(
            os.getenv("RECSYS_USER_FEATURES_PATH", base_dir / "data" / "processed" / "user_features.csv")
        ),
        movie_features_path=Path(
            os.getenv("RECSYS_MOVIE_FEATURES_PATH", base_dir / "data" / "processed" / "movie_features.csv")
        ),
        db_host=os.getenv("RECSYS_DB_HOST", "localhost"),
        db_name=os.getenv("RECSYS_DB_NAME", "recsys"),
        db_user=os.getenv("RECSYS_DB_USER", "recsys"),
        db_password=os.getenv("RECSYS_DB_PASSWORD", "recsys"),
        db_port=int(os.getenv("RECSYS_DB_PORT", "5434")),
    )
