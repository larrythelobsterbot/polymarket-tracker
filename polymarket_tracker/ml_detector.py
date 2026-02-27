"""
Machine Learning-based Insider Detection for Polymarket.

This module provides an optional ML classifier for detecting suspicious
betting patterns. It uses feature engineering on betting data and trains
a Random Forest or XGBoost classifier.

Features engineered:
- Timing features (time to resolution, hour of day, day of week)
- Size features (bet amount, relative to market volume)
- History features (trader's past behavior, win rate)
- Market features (liquidity, volume, category)
- Probability features (implied odds, betting against consensus)

Requirements:
    pip install scikit-learn xgboost pandas numpy

Usage:
    from polymarket_tracker.ml_detector import MLInsiderDetector

    # Train on labeled data
    detector = MLInsiderDetector()
    detector.train(labeled_bets)

    # Predict on new bets
    predictions = detector.predict(new_bets)
"""

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

from .database import Database, db

logger = logging.getLogger(__name__)

# Try to import ML libraries (optional)
try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML libraries not installed. Install with: pip install scikit-learn numpy")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class BetFeatures:
    """Feature vector for a single bet."""
    bet_id: str

    # Timing features
    hours_to_resolution: float = 0.0
    hour_of_day: int = 0
    day_of_week: int = 0
    is_weekend: int = 0
    is_low_liquidity_hour: int = 0
    hours_since_market_creation: float = 0.0

    # Size features
    bet_amount: float = 0.0
    bet_cost_basis: float = 0.0
    relative_to_market_volume: float = 0.0
    relative_to_avg_bet: float = 0.0
    log_bet_amount: float = 0.0

    # Trader history features
    trader_total_bets: int = 0
    trader_total_volume: float = 0.0
    trader_avg_bet_size: float = 0.0
    trader_win_rate: float = 0.0
    trader_days_active: int = 0
    days_since_last_bet: int = 0
    is_dormant_reactivation: int = 0

    # Market features
    market_volume: float = 0.0
    market_liquidity: float = 0.0
    is_niche_market: int = 0
    market_age_hours: float = 0.0
    is_early_bet: int = 0

    # Probability features
    implied_probability: float = 0.5
    is_betting_underdog: int = 0
    odds_deviation: float = 0.0

    # Outcome features (for training only)
    outcome_known: int = 0
    bet_won: int = 0

    def to_array(self) -> list:
        """Convert to feature array for ML model."""
        return [
            self.hours_to_resolution,
            self.hour_of_day,
            self.day_of_week,
            self.is_weekend,
            self.is_low_liquidity_hour,
            self.hours_since_market_creation,
            self.bet_amount,
            self.bet_cost_basis,
            self.relative_to_market_volume,
            self.relative_to_avg_bet,
            self.log_bet_amount,
            self.trader_total_bets,
            self.trader_total_volume,
            self.trader_avg_bet_size,
            self.trader_win_rate,
            self.trader_days_active,
            self.days_since_last_bet,
            self.is_dormant_reactivation,
            self.market_volume,
            self.market_liquidity,
            self.is_niche_market,
            self.market_age_hours,
            self.is_early_bet,
            self.implied_probability,
            self.is_betting_underdog,
            self.odds_deviation,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        """Get feature names for model interpretation."""
        return [
            "hours_to_resolution",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_low_liquidity_hour",
            "hours_since_market_creation",
            "bet_amount",
            "bet_cost_basis",
            "relative_to_market_volume",
            "relative_to_avg_bet",
            "log_bet_amount",
            "trader_total_bets",
            "trader_total_volume",
            "trader_avg_bet_size",
            "trader_win_rate",
            "trader_days_active",
            "days_since_last_bet",
            "is_dormant_reactivation",
            "market_volume",
            "market_liquidity",
            "is_niche_market",
            "market_age_hours",
            "is_early_bet",
            "implied_probability",
            "is_betting_underdog",
            "odds_deviation",
        ]


class FeatureExtractor:
    """
    Extracts ML features from bet data.

    Handles all feature engineering including:
    - Temporal features
    - Trader history
    - Market context
    """

    def __init__(self, database: Optional[Database] = None):
        self.db = database or db

    def extract_features(self, bet_id: str) -> Optional[BetFeatures]:
        """
        Extract features for a single bet.

        Args:
            bet_id: Bet ID to extract features for.

        Returns:
            BetFeatures or None if bet not found.
        """
        with self.db.get_connection() as conn:
            # Get bet with market info
            cursor = conn.execute(
                """
                SELECT
                    b.*,
                    m.question,
                    m.end_date,
                    m.volume as market_volume,
                    m.liquidity,
                    m.resolved,
                    m.outcome as market_outcome,
                    m.outcome_prices,
                    m.created_at as market_created
                FROM bets b
                JOIN markets m ON b.market_id = m.market_id
                WHERE b.bet_id = ?
                """,
                (bet_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            features = BetFeatures(bet_id=bet_id)

            # Parse timestamps
            bet_timestamp = row["timestamp"]
            if isinstance(bet_timestamp, str):
                bet_timestamp = datetime.fromisoformat(bet_timestamp)

            end_date = row["end_date"]
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)

            market_created = row["market_created"]
            if isinstance(market_created, str):
                market_created = datetime.fromisoformat(market_created)

            # Timing features
            if end_date and bet_timestamp:
                features.hours_to_resolution = max(0, (end_date - bet_timestamp).total_seconds() / 3600)

            features.hour_of_day = bet_timestamp.hour if bet_timestamp else 0
            features.day_of_week = bet_timestamp.weekday() if bet_timestamp else 0
            features.is_weekend = 1 if features.day_of_week >= 5 else 0
            features.is_low_liquidity_hour = 1 if 0 <= features.hour_of_day < 6 else 0

            if market_created and bet_timestamp:
                features.hours_since_market_creation = (bet_timestamp - market_created).total_seconds() / 3600
                features.is_early_bet = 1 if features.hours_since_market_creation < 48 else 0

            # Size features
            amount = float(row["amount"] or 0)
            price = float(row["price"] or 0)
            features.bet_amount = amount
            features.bet_cost_basis = amount * price
            features.log_bet_amount = np.log1p(amount) if ML_AVAILABLE else 0

            market_volume = float(row["market_volume"] or 0)
            features.market_volume = market_volume
            features.market_liquidity = float(row["liquidity"] or 0)
            features.is_niche_market = 1 if market_volume < 100000 else 0

            if market_volume > 0:
                features.relative_to_market_volume = (amount * price) / market_volume

            # Get average bet size for this market
            cursor = conn.execute(
                """
                SELECT AVG(amount * price) as avg_bet
                FROM bets
                WHERE market_id = ?
                """,
                (row["market_id"],)
            )
            avg_row = cursor.fetchone()
            avg_bet = float(avg_row["avg_bet"] or 1) if avg_row else 1
            features.relative_to_avg_bet = (amount * price) / avg_bet if avg_bet > 0 else 0

            # Trader history features
            wallet = row["wallet_address"]
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_bets,
                    SUM(amount * price) as total_volume,
                    AVG(amount) as avg_bet,
                    MIN(timestamp) as first_bet,
                    MAX(timestamp) as last_bet
                FROM bets
                WHERE wallet_address = ? AND timestamp < ?
                """,
                (wallet, bet_timestamp)
            )
            hist_row = cursor.fetchone()

            if hist_row:
                features.trader_total_bets = hist_row["total_bets"] or 0
                features.trader_total_volume = float(hist_row["total_volume"] or 0)
                features.trader_avg_bet_size = float(hist_row["avg_bet"] or 0)

                first_bet = hist_row["first_bet"]
                last_bet = hist_row["last_bet"]

                if first_bet:
                    if isinstance(first_bet, str):
                        first_bet = datetime.fromisoformat(first_bet)
                    features.trader_days_active = (bet_timestamp - first_bet).days

                if last_bet:
                    if isinstance(last_bet, str):
                        last_bet = datetime.fromisoformat(last_bet)
                    features.days_since_last_bet = (bet_timestamp - last_bet).days
                    features.is_dormant_reactivation = 1 if features.days_since_last_bet > 30 else 0

            # Calculate trader win rate on resolved markets
            cursor = conn.execute(
                """
                SELECT
                    b.outcome_bet,
                    m.outcome as market_outcome
                FROM bets b
                JOIN markets m ON b.market_id = m.market_id
                WHERE b.wallet_address = ?
                    AND m.resolved = 1
                    AND b.side = 'BUY'
                    AND b.timestamp < ?
                """,
                (wallet, bet_timestamp)
            )
            resolved_bets = cursor.fetchall()

            wins = 0
            total = 0
            for rb in resolved_bets:
                if rb["market_outcome"]:
                    total += 1
                    bet_outcome = (rb["outcome_bet"] or "").lower()
                    market_outcome = rb["market_outcome"].lower()
                    if bet_outcome == market_outcome or \
                       (bet_outcome in ["yes", "true"] and market_outcome in ["yes", "true"]) or \
                       (bet_outcome in ["no", "false"] and market_outcome in ["no", "false"]):
                        wins += 1

            features.trader_win_rate = (wins / total * 100) if total > 0 else 50.0

            # Probability features
            if row["outcome_prices"]:
                try:
                    prices = json.loads(row["outcome_prices"])
                    if isinstance(prices, list) and len(prices) >= 2:
                        outcome = (row["outcome_bet"] or "").lower()
                        if outcome in ["yes", "true"]:
                            features.implied_probability = float(prices[0])
                        else:
                            features.implied_probability = float(prices[1]) if len(prices) > 1 else 1 - float(prices[0])

                        features.is_betting_underdog = 1 if features.implied_probability < 0.3 else 0
                        features.odds_deviation = abs(features.implied_probability - 0.5)
                except (json.JSONDecodeError, TypeError, IndexError):
                    pass

            # Outcome features (for training)
            if row["resolved"] and row["market_outcome"]:
                features.outcome_known = 1
                bet_outcome = (row["outcome_bet"] or "").lower()
                market_outcome = row["market_outcome"].lower()
                features.bet_won = 1 if bet_outcome == market_outcome or \
                    (bet_outcome in ["yes", "true"] and market_outcome in ["yes", "true"]) else 0

            return features

    def extract_batch(
        self,
        bet_ids: list[str],
        include_labels: bool = False
    ) -> tuple[list[list], list[str], Optional[list[int]]]:
        """
        Extract features for multiple bets.

        Args:
            bet_ids: List of bet IDs.
            include_labels: Whether to include outcome labels.

        Returns:
            Tuple of (feature_matrix, bet_ids, labels).
        """
        features_list = []
        valid_bet_ids = []
        labels = [] if include_labels else None

        for bet_id in bet_ids:
            features = self.extract_features(bet_id)
            if features:
                features_list.append(features.to_array())
                valid_bet_ids.append(bet_id)
                if include_labels:
                    labels.append(features.bet_won)

        return features_list, valid_bet_ids, labels


class MLInsiderDetector:
    """
    Machine Learning-based insider detection.

    Trains a classifier on historical betting data to predict
    suspicious patterns that may indicate insider information.
    """

    def __init__(
        self,
        database: Optional[Database] = None,
        model_type: str = "random_forest"
    ):
        """
        Initialize ML detector.

        Args:
            database: Database instance.
            model_type: "random_forest" or "xgboost".
        """
        if not ML_AVAILABLE:
            raise ImportError("scikit-learn and numpy required. Install with: pip install scikit-learn numpy")

        self.db = database or db
        self.model_type = model_type
        self.feature_extractor = FeatureExtractor(database=self.db)
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance: dict[str, float] = {}

    def _create_model(self):
        """Create the ML model."""
        if self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective="binary:logistic",
                eval_metric="auc",
                use_label_encoder=False,
                random_state=42,
            )
        else:
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

    def create_training_dataset(
        self,
        suspicious_threshold: float = 80.0,
        limit: int = 10000
    ) -> tuple[list[list], list[int], list[str]]:
        """
        Create training dataset from historical data.

        Labels bets as suspicious based on rule-based anomaly score
        as a proxy for actual insider trading (which we can't observe).

        Args:
            suspicious_threshold: Anomaly score threshold for positive class.
            limit: Maximum bets to include.

        Returns:
            Tuple of (features, labels, bet_ids).
        """
        from .insider_detection import InsiderDetector

        detector = InsiderDetector(database=self.db)

        with self.db.get_connection() as conn:
            # Get bets from resolved markets (so we have outcome data)
            cursor = conn.execute(
                """
                SELECT b.bet_id
                FROM bets b
                JOIN markets m ON b.market_id = m.market_id
                WHERE m.resolved = 1
                ORDER BY b.timestamp DESC
                LIMIT ?
                """,
                (limit,)
            )
            bet_ids = [row["bet_id"] for row in cursor.fetchall()]

        features_list = []
        labels = []
        valid_bet_ids = []

        for bet_id in bet_ids:
            try:
                # Extract features
                features = self.feature_extractor.extract_features(bet_id)
                if not features:
                    continue

                # Get anomaly score as label proxy
                suspicious_bet = detector.calculate_anomaly_score(bet_id)

                # Label: 1 if suspicious AND bet won (potential insider)
                # This creates labeled data for the pattern: suspicious behavior + correct prediction
                is_suspicious = suspicious_bet.anomaly_score >= suspicious_threshold
                label = 1 if is_suspicious and features.bet_won else 0

                features_list.append(features.to_array())
                labels.append(label)
                valid_bet_ids.append(bet_id)

            except Exception as e:
                logger.debug(f"Error processing bet {bet_id}: {e}")
                continue

        return features_list, labels, valid_bet_ids

    def train(
        self,
        features: Optional[list[list]] = None,
        labels: Optional[list[int]] = None,
        test_size: float = 0.2,
        auto_create_dataset: bool = True
    ) -> dict:
        """
        Train the ML model.

        Args:
            features: Feature matrix.
            labels: Labels (1 = suspicious, 0 = normal).
            test_size: Fraction for test set.
            auto_create_dataset: Create dataset from database if not provided.

        Returns:
            Training metrics.
        """
        if features is None or labels is None:
            if auto_create_dataset:
                logger.info("Creating training dataset from database...")
                features, labels, _ = self.create_training_dataset()
            else:
                raise ValueError("Must provide features and labels or set auto_create_dataset=True")

        if len(features) < 100:
            raise ValueError(f"Insufficient training data: {len(features)} samples. Need at least 100.")

        X = np.array(features)
        y = np.array(labels)

        # Check class balance
        positive_rate = sum(y) / len(y)
        logger.info(f"Training data: {len(y)} samples, {positive_rate*100:.1f}% positive class")

        if positive_rate < 0.01:
            logger.warning("Very few positive examples. Model may not learn well.")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        metrics = {
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "positive_rate": positive_rate,
        }

        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        except ValueError:
            metrics["roc_auc"] = None

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring="roc_auc")
        metrics["cv_mean_auc"] = np.mean(cv_scores)
        metrics["cv_std_auc"] = np.std(cv_scores)

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            feature_names = BetFeatures.feature_names()
            importance = self.model.feature_importances_
            self.feature_importance = dict(zip(feature_names, importance))
            metrics["top_features"] = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

        # Classification report
        metrics["classification_report"] = classification_report(y_test, y_pred)
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

        logger.info(f"Model trained. ROC-AUC: {metrics.get('roc_auc', 'N/A')}")

        return metrics

    def predict(self, bet_ids: list[str]) -> list[dict]:
        """
        Predict insider probability for bets.

        Args:
            bet_ids: List of bet IDs to predict.

        Returns:
            List of predictions with probabilities.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        features_list, valid_bet_ids, _ = self.feature_extractor.extract_batch(bet_ids)

        if not features_list:
            return []

        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)

        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = self.model.predict(X_scaled)

        results = []
        for bet_id, prob, pred in zip(valid_bet_ids, probabilities, predictions):
            results.append({
                "bet_id": bet_id,
                "insider_probability": float(prob),
                "is_suspicious": bool(pred),
                "confidence": float(abs(prob - 0.5) * 2),  # 0-1 scale
            })

        return results

    def predict_single(self, bet_id: str) -> Optional[dict]:
        """
        Predict for a single bet.

        Args:
            bet_id: Bet ID.

        Returns:
            Prediction dict or None.
        """
        results = self.predict([bet_id])
        return results[0] if results else None

    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.

        Args:
            filepath: Path to save model.
        """
        if not self.is_trained:
            raise ValueError("No trained model to save.")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_importance": self.feature_importance,
            "model_type": self.model_type,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk.

        Args:
            filepath: Path to model file.
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_importance = model_data.get("feature_importance", {})
        self.model_type = model_data.get("model_type", "random_forest")
        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")

    def get_feature_importance(self) -> list[tuple[str, float]]:
        """
        Get ranked feature importance.

        Returns:
            List of (feature_name, importance) tuples.
        """
        return sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )


def train_and_save_model(
    output_path: str = "insider_model.pkl",
    model_type: str = "random_forest"
) -> dict:
    """
    Convenience function to train and save a model.

    Args:
        output_path: Where to save the model.
        model_type: "random_forest" or "xgboost".

    Returns:
        Training metrics.
    """
    detector = MLInsiderDetector(model_type=model_type)
    metrics = detector.train(auto_create_dataset=True)
    detector.save_model(output_path)
    return metrics
