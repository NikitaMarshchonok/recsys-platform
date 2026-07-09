from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MOVIES_COLUMNS = ["movieId", "title", "genres"]
RATINGS_COLUMNS = ["userId", "movieId", "rating", "timestamp"]


def read_csv_with_columns(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required MovieLens file not found: {path}")

    data = pd.read_csv(path)
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    return data[required_columns].copy()


def normalize_movies(movies: pd.DataFrame) -> pd.DataFrame:
    normalized = movies.copy()
    normalized["movieId"] = normalized["movieId"].astype(int)
    normalized["title"] = normalized["title"].astype(str)
    normalized["genres"] = (
        normalized["genres"]
        .fillna("Unknown")
        .replace("(no genres listed)", "Unknown")
        .astype(str)
        .str.replace("|", ", ", regex=False)
    )
    return normalized.sort_values("movieId").reset_index(drop=True)


def normalize_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    normalized = ratings.copy()
    normalized["userId"] = normalized["userId"].astype(int)
    normalized["movieId"] = normalized["movieId"].astype(int)
    normalized["rating"] = normalized["rating"].astype(float)
    normalized["timestamp"] = normalized["timestamp"].astype(int)
    return normalized.sort_values(["userId", "movieId"]).reset_index(drop=True)


def build_users(ratings: pd.DataFrame) -> pd.DataFrame:
    users = ratings[["userId"]].drop_duplicates().sort_values("userId")
    return users.reset_index(drop=True)


def import_movielens(source_dir: Path, output_dir: Path) -> dict[str, int]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    movies = normalize_movies(
        read_csv_with_columns(source_dir / "movies.csv", MOVIES_COLUMNS)
    )
    ratings = normalize_ratings(
        read_csv_with_columns(source_dir / "ratings.csv", RATINGS_COLUMNS)
    )
    users = build_users(ratings)

    output_dir.mkdir(parents=True, exist_ok=True)
    movies.to_csv(output_dir / "movies.csv", index=False)
    ratings.to_csv(output_dir / "ratings.csv", index=False)
    users.to_csv(output_dir / "users.csv", index=False)

    return {
        "movies": len(movies),
        "ratings": len(ratings),
        "users": len(users),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import MovieLens CSV files into the RecSys raw data format.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/external/ml-latest-small"),
        help="Directory containing MovieLens movies.csv and ratings.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory for normalized movies, ratings, and users CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = import_movielens(args.source_dir, args.output_dir)

    print(f"Imported movies: {summary['movies']}")
    print(f"Imported ratings: {summary['ratings']}")
    print(f"Imported users: {summary['users']}")
    print(f"Saved raw data to: {args.output_dir}")


if __name__ == "__main__":
    main()
