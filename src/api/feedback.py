from collections.abc import Callable
from contextlib import closing
import logging
from typing import Any

from src.api.schemas import (
    RecommendationFeedbackDeleteRequest,
    RecommendationFeedbackRequest,
)


logger = logging.getLogger("recsys.api")


class RecommendationFeedbackStore:
    def __init__(self, connection_factory: Callable[[], Any]):
        self._connection_factory = connection_factory

    @staticmethod
    def _ensure_table(cur):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_feedback (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                feedback VARCHAR(16) NOT NULL,
                source VARCHAR(50) NOT NULL,
                ranking_strategy VARCHAR(32) NOT NULL DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            ALTER TABLE recommendation_feedback
            ADD COLUMN IF NOT EXISTS ranking_strategy VARCHAR(32) NOT NULL DEFAULT 'unknown'
        """)

    def record(self, feedback: RecommendationFeedbackRequest) -> str:
        try:
            with closing(self._connection_factory()) as conn:
                with conn:
                    with conn.cursor() as cur:
                        self._ensure_table(cur)
                        cur.execute(
                            """
                            UPDATE recommendation_feedback
                            SET feedback = %s, created_at = NOW()
                            WHERE user_id = %s
                              AND movie_id = %s
                              AND source = %s
                              AND ranking_strategy = %s
                            """,
                            (
                                feedback.feedback,
                                feedback.user_id,
                                feedback.movie_id,
                                feedback.source,
                                feedback.ranking_strategy,
                            ),
                        )
                        if cur.rowcount == 0:
                            cur.execute(
                                """
                                INSERT INTO recommendation_feedback
                                    (user_id, movie_id, feedback, source, ranking_strategy)
                                VALUES (%s, %s, %s, %s, %s)
                                """,
                                (
                                    feedback.user_id,
                                    feedback.movie_id,
                                    feedback.feedback,
                                    feedback.source,
                                    feedback.ranking_strategy,
                                ),
                            )
            return "postgres"
        except Exception as exc:
            logger.info(
                "recommendation_feedback_log_only user_id=%s movie_id=%s feedback=%s source=%s ranking_strategy=%s error=%s",
                feedback.user_id,
                feedback.movie_id,
                feedback.feedback,
                feedback.source,
                feedback.ranking_strategy,
                exc,
            )
            return "log_only"

    def remove(self, feedback: RecommendationFeedbackDeleteRequest) -> str:
        try:
            with closing(self._connection_factory()) as conn:
                with conn:
                    with conn.cursor() as cur:
                        self._ensure_table(cur)
                        cur.execute(
                            """
                            DELETE FROM recommendation_feedback
                            WHERE user_id = %s
                              AND movie_id = %s
                              AND source = %s
                              AND ranking_strategy = %s
                            """,
                            (
                                feedback.user_id,
                                feedback.movie_id,
                                feedback.source,
                                feedback.ranking_strategy,
                            ),
                        )
            return "postgres"
        except Exception as exc:
            logger.info(
                "recommendation_feedback_remove_log_only user_id=%s movie_id=%s source=%s ranking_strategy=%s error=%s",
                feedback.user_id,
                feedback.movie_id,
                feedback.source,
                feedback.ranking_strategy,
                exc,
            )
            return "log_only"

    def summarize(self) -> dict:
        try:
            with closing(self._connection_factory()) as conn:
                with conn:
                    with conn.cursor() as cur:
                        self._ensure_table(cur)
                        cur.execute("""
                            SELECT
                                COUNT(*),
                                COUNT(*) FILTER (WHERE feedback = 'like'),
                                COUNT(*) FILTER (WHERE feedback = 'dislike')
                            FROM recommendation_feedback
                        """)
                        total_feedback, likes, dislikes = cur.fetchone()
                        cur.execute("""
                            SELECT
                                ranking_strategy,
                                COUNT(*),
                                COUNT(*) FILTER (WHERE feedback = 'like'),
                                COUNT(*) FILTER (WHERE feedback = 'dislike')
                            FROM recommendation_feedback
                            GROUP BY ranking_strategy
                            ORDER BY COUNT(*) DESC, ranking_strategy
                        """)
                        strategy_rows = cur.fetchall()
        except Exception as exc:
            logger.info("recommendation_feedback_summary_unavailable error=%s", exc)
            return {
                "total_feedback": 0,
                "likes": 0,
                "dislikes": 0,
                "like_rate": 0.0,
                "storage": "unavailable",
                "strategies": [],
            }

        total_feedback = int(total_feedback or 0)
        likes = int(likes or 0)
        dislikes = int(dislikes or 0)
        like_rate = round(likes / total_feedback * 100, 1) if total_feedback else 0.0
        strategies = []
        for strategy, strategy_total, strategy_likes, strategy_dislikes in strategy_rows:
            strategy_total = int(strategy_total or 0)
            strategy_likes = int(strategy_likes or 0)
            strategies.append({
                "ranking_strategy": strategy,
                "total_feedback": strategy_total,
                "likes": strategy_likes,
                "dislikes": int(strategy_dislikes or 0),
                "like_rate": (
                    round(strategy_likes / strategy_total * 100, 1)
                    if strategy_total
                    else 0.0
                ),
            })

        return {
            "total_feedback": total_feedback,
            "likes": likes,
            "dislikes": dislikes,
            "like_rate": like_rate,
            "storage": "postgres",
            "strategies": strategies,
        }

    def load_for_user(self, user_id: int, source: str) -> list[dict]:
        try:
            with closing(self._connection_factory()) as conn:
                with conn:
                    with conn.cursor() as cur:
                        self._ensure_table(cur)
                        cur.execute(
                            """
                            SELECT DISTINCT ON (movie_id, ranking_strategy)
                                movie_id,
                                feedback,
                                ranking_strategy
                            FROM recommendation_feedback
                            WHERE user_id = %s AND source = %s
                            ORDER BY
                                movie_id,
                                ranking_strategy,
                                created_at DESC,
                                id DESC
                            """,
                            (user_id, source),
                        )
                        rows = cur.fetchall()
        except Exception as exc:
            logger.info(
                "recommendation_feedback_state_unavailable user_id=%s source=%s error=%s",
                user_id,
                source,
                exc,
            )
            return []

        return [
            {
                "movie_id": int(movie_id),
                "feedback": feedback,
                "ranking_strategy": ranking_strategy,
            }
            for movie_id, feedback, ranking_strategy in rows
        ]
