import logging
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from src.config import settings
from src.data.ingest import load_and_preprocess
from src.features.build_features import create_features
from src.models.train_model import main as train_main
from src.models.detect_anomalies import detect_anomalies
from src.visualization.plot_results import plot_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_pipeline():
    logging.info("Starting End-to-End Pipeline...")

    # 1. Ingestion
    input_xml = str(settings.data_dir / "1.xml")
    if not Path(input_xml).exists():
        xmls = list(settings.data_dir.glob(settings.raw_xml_pattern))
        if xmls:
            input_xml = str(xmls[0])
            logging.info(f"Using found XML: {input_xml}")
        else:
            logging.error("No XML file found in %s", settings.data_dir)
            return

    daily_sales_path = str(settings.get_daily_sales_path())
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    load_and_preprocess(input_xml, daily_sales_path)

    # 2. Features
    features_path = str(settings.get_features_path())
    create_features(daily_sales_path, features_path)

    # 3. Training (with optional MLflow registration)
    models_dir = str(settings.models_dir)
    train_main(features_path, models_dir, register_with_mlflow=True)

    # 4. Anomaly Detection
    anomalies_path = str(settings.get_anomalies_path())
    detect_anomalies(features_path, anomalies_path)

    # 5. Visualization
    dashboard_path = "forecast_dashboard.html"
    best_model_path = f"{models_dir}/best_model.pkl"
    plot_results(anomalies_path, best_model_path, dashboard_path)

    logging.info("Pipeline completed successfully.")
    logging.info(f"Check results:\n- Model: {best_model_path}\n- Dashboard: {dashboard_path}")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
