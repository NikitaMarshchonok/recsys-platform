import pandas as pd
import pytest

from scripts.import_movielens import import_movielens


def test_import_movielens_writes_project_raw_files(tmp_path):
    source_dir = tmp_path / "ml-latest-small"
    output_dir = tmp_path / "raw"
    source_dir.mkdir()

    pd.DataFrame({
        "movieId": [2, 1],
        "title": ["Heat (1995)", "Toy Story (1995)"],
        "genres": ["Action|Crime|Thriller", "Adventure|Animation|Children"],
    }).to_csv(source_dir / "movies.csv", index=False)
    pd.DataFrame({
        "userId": [2, 1, 2],
        "movieId": [1, 2, 2],
        "rating": [4.0, 5.0, 3.5],
        "timestamp": [964982703, 964981247, 964982224],
    }).to_csv(source_dir / "ratings.csv", index=False)

    summary = import_movielens(source_dir, output_dir)

    assert summary == {"movies": 2, "ratings": 3, "users": 2}

    movies = pd.read_csv(output_dir / "movies.csv")
    ratings = pd.read_csv(output_dir / "ratings.csv")
    users = pd.read_csv(output_dir / "users.csv")

    assert movies.to_dict("records") == [
        {
            "movieId": 1,
            "title": "Toy Story (1995)",
            "genres": "Adventure, Animation, Children",
        },
        {
            "movieId": 2,
            "title": "Heat (1995)",
            "genres": "Action, Crime, Thriller",
        },
    ]
    assert ratings["userId"].tolist() == [1, 2, 2]
    assert ratings["movieId"].tolist() == [2, 1, 2]
    assert users.to_dict("records") == [{"userId": 1}, {"userId": 2}]


def test_import_movielens_requires_expected_columns(tmp_path):
    source_dir = tmp_path / "ml-latest-small"
    output_dir = tmp_path / "raw"
    source_dir.mkdir()

    pd.DataFrame({
        "movieId": [1],
        "title": ["Toy Story (1995)"],
    }).to_csv(source_dir / "movies.csv", index=False)
    pd.DataFrame({
        "userId": [1],
        "movieId": [1],
        "rating": [4.0],
        "timestamp": [964982703],
    }).to_csv(source_dir / "ratings.csv", index=False)

    with pytest.raises(ValueError, match="genres"):
        import_movielens(source_dir, output_dir)
