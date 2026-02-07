"""Scheduler for periodic retraining evaluation and drift checks."""

import logging
import signal
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.config import settings
from src.monitoring.retraining import RetrainingPipeline

logger = logging.getLogger(__name__)


def evaluate_and_retrain():
    """
    Scheduled job: evaluate triggers and retrain if needed.
    Designed to be called periodically (e.g., every 24 hours).
    """
    logger.info("Scheduled retraining evaluation started...")

    pipeline = RetrainingPipeline()

    trigger_result = pipeline.should_retrain(
        drift_report=None,
        performance_report=None,
        new_data_count=0,
    )

    if trigger_result["should_retrain"]:
        logger.info("Retraining triggered: %s", trigger_result["reasons"])
        result = pipeline.retrain()
        logger.info("Retraining result: %s", result)
    else:
        logger.info("No retraining needed at this time")


def run_scheduler():
    """Start the blocking scheduler for periodic retraining checks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    scheduler = BlockingScheduler()

    scheduler.add_job(
        evaluate_and_retrain,
        trigger=IntervalTrigger(hours=24),
        id="retraining_check",
        name="Periodic retraining evaluation",
        replace_existing=True,
    )

    def shutdown(signum, frame):
        logger.info("Scheduler shutting down...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    logger.info(
        "Scheduler started. Checking every 24h. "
        "Retrain threshold: %dh",
        settings.retraining_schedule_hours,
    )
    scheduler.start()


if __name__ == "__main__":
    run_scheduler()
