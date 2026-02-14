"""Tests for rule-based feature extraction module."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from src.feature_extraction import FeatureExtractor, ProductFeatures


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> dict:
    """Load extraction config from config.yaml."""
    with open("config.yaml", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    return full_config["extraction"]


@pytest.fixture
def extractor(config: dict) -> FeatureExtractor:
    """Create FeatureExtractor from config."""
    return FeatureExtractor.from_config(config)


# ---------------------------------------------------------------------------
# ExtractColor
# ---------------------------------------------------------------------------


class TestExtractColor:
    def test_czarny(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_color("CZARNY PORTFEL SKÓRZANY") == "czarny"

    def test_czarne(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_color("CZARNE BOTKI NUBUKOWE") == "czarny"

    def test_taupe(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_color("TAUPE KLASYCZNE KOWBOJKI") == "taupe"

    def test_srebrne(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_color("SREBRNE SZPILKI W SZPIC") == "srebrny"

    def test_kamelowe(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_color("KAMELOWE BOTKI 36") == "kamelowy"

    def test_no_color(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_color("BOTKI SKÓRZANE 38") is None

    def test_case_insensitive(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_color("czarny portfel") == "czarny"

    def test_empty_string(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_color("") is None


# ---------------------------------------------------------------------------
# ExtractMaterial
# ---------------------------------------------------------------------------


class TestExtractMaterial:
    def test_skorzane(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_material("BOTKI SKÓRZANE 38") == "skorzany"

    def test_zamszowe(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_material("KOWBOJKI ZAMSZOWE 41") == "zamszowy"

    def test_nubukowe(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_material("CZARNE NUBUKOWE BOTKI") == "nubukowy"

    def test_lakierowane(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_material("PORTFEL LAKIEROWANY") == "lakierowany"

    def test_no_material(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_material("SNEAKERSY 37") is None

    def test_welurowe(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_material("BOTKI WELUROWE 39") == "welurowy"


# ---------------------------------------------------------------------------
# ExtractSize
# ---------------------------------------------------------------------------


class TestExtractSize:
    def test_size_at_end(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_size("BOTKI SKÓRZANE 41") == 41

    def test_size_36(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_size("KOWBOJKI ZAMSZOWE 36") == 36

    def test_size_50(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_size("BOTKI 50") == 50

    def test_size_35(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_size("BOTKI 35") == 35

    def test_no_size(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_size("PORTFEL MONNARI SKÓRZANY") is None

    def test_size_with_dash(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_size("Sneakersy z motywem - 37") == 37

    def test_ignores_numbers_outside_range(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_size("MODEL 100 BOTKI 39") == 39

    def test_prefers_last_match(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_size("BOTKI 36 37") == 37


# ---------------------------------------------------------------------------
# ExtractProductType
# ---------------------------------------------------------------------------


class TestExtractProductType:
    def test_botki(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_product_type("CZARNE BOTKI SKÓRZANE 38") == "botki"

    def test_kowbojki(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_product_type("TAUPE KOWBOJKI ZAMSZOWE 41") == "kowbojki"

    def test_szpilki(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_product_type("SREBRNE SZPILKI 38") == "szpilki"

    def test_portfel(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_product_type("CZARNY PORTFEL MONNARI") == "portfel"

    def test_torebka(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_product_type("STYLOWA TOREBKA DAMSKA") == "torebka"

    def test_sneakersy(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_product_type("BIAŁE SNEAKERSY 40") == "sneakersy"

    def test_kozaki(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_product_type("CZARNE KOZAKI SKÓRZANE 39") == "kozaki"

    def test_sandaly(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_product_type("BEŻOWE SANDAŁY 38") == "sandaly"

    def test_no_type(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_product_type("SKÓRZANE DAMSKIE 38") is None

    def test_mixed_case(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_product_type("Sneakersy z motywem") == "sneakersy"

    def test_glany(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_product_type("CZARNE GLANY SKÓRZANE 39") == "glany"


# ---------------------------------------------------------------------------
# ExtractBrand
# ---------------------------------------------------------------------------


class TestExtractBrand:
    def test_from_sku_opt(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_brand("KOWBOJKI", sku="OPT-611-OC-CAP-41") == "Optimo"

    def test_from_sku_mon(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_brand("PORTFEL", sku="MON-PUR0162-020") == "Monnari"

    def test_from_sku_sl(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_brand("BOTKI", sku="SL-TR928-CZARNY") == "Sergio Leone"

    def test_from_sku_art(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_brand("SNEAKERSY", sku="SNK23-060-37") == "Art"

    def test_from_sku_bota(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_brand("TRAPERY", sku="bota19-016-38") == "Art"

    def test_from_name_monnari(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_brand("PORTFEL MONNARI SKÓRZANY") == "Monnari"

    def test_from_name_sergio_leone(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_brand("BOTKI SERGIO LEONE 40") == "Sergio Leone"

    def test_sku_priority_over_name(self, extractor: FeatureExtractor) -> None:
        result = extractor.extract_brand("PORTFEL MONNARI", sku="OPT-123")
        assert result == "Optimo"

    def test_no_brand(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_brand("BOTKI SKÓRZANE 38") is None

    def test_none_sku(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_brand("BOTKI SKÓRZANE 38", sku=None) is None


# ---------------------------------------------------------------------------
# ExtractSeason
# ---------------------------------------------------------------------------


class TestExtractSeason:
    def test_warm_cieple(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_season("KOWBOJKI CIEPŁE 41") == "warm"

    def test_warm_ocieplane(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_season("BOTKI OCIEPLANE 39") == "warm"

    def test_warm_zimowe(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_season("BOTKI ZIMOWE 40") == "warm"

    def test_cold_azurowe(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_season("SANDAŁY AŻUROWE 38") == "cold"

    def test_no_season(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract_season("BOTKI SKÓRZANE 38") is None


# ---------------------------------------------------------------------------
# ExtractAll (integration)
# ---------------------------------------------------------------------------


class TestExtractAll:
    def test_full_extraction_kowbojki(self, extractor: FeatureExtractor) -> None:
        features = extractor.extract_all(
            "TAUPE KLASYCZNE KOWBOJKI ZA KOSTKĘ NA ZAMEK SKÓRZANE ZAMSZOWE CIEPŁE 41",
            sku="OPT-611-OC-CAP-41",
        )
        assert features.color == "taupe"
        assert features.material in ("skorzany", "zamszowy")
        assert features.size == 41
        assert features.product_type == "kowbojki"
        assert features.brand == "Optimo"
        assert features.season == "warm"

    def test_full_extraction_portfel(self, extractor: FeatureExtractor) -> None:
        features = extractor.extract_all(
            "CZARNY MAŁY PORTFEL MONNARI PORTMONETKA SKÓRZANY LAKIEROWANY",
            sku="MON-PUR0162-020-24",
        )
        assert features.color == "czarny"
        assert features.material in ("skorzany", "lakierowany")
        assert features.size is None
        assert features.product_type == "portfel"
        assert features.brand == "Monnari"

    def test_full_extraction_sneakersy(self, extractor: FeatureExtractor) -> None:
        features = extractor.extract_all(
            "Sneakersy z muzycznym motywem - 37",
            sku="SNK23-060-37",
        )
        assert features.size == 37
        assert features.product_type == "sneakersy"
        assert features.brand == "Art"

    def test_empty_text(self, extractor: FeatureExtractor) -> None:
        features = extractor.extract_all("")
        assert features == ProductFeatures()

    def test_none_text(self, extractor: FeatureExtractor) -> None:
        features = extractor.extract_all(None)
        assert features == ProductFeatures()


# ---------------------------------------------------------------------------
# ExtractDataframe
# ---------------------------------------------------------------------------


class TestExtractDataframe:
    def test_adds_all_columns(self, extractor: FeatureExtractor) -> None:
        df = pd.DataFrame({
            "name": ["CZARNE BOTKI SKÓRZANE 38"],
            "products_sku": ["SL-123"],
        })
        result = extractor.extract_dataframe(df)
        for col in ["color", "material", "size", "product_type", "brand", "season"]:
            assert col in result.columns

    def test_correct_extraction(self, extractor: FeatureExtractor) -> None:
        df = pd.DataFrame({
            "name": ["CZARNE BOTKI SKÓRZANE 38"],
            "products_sku": ["SL-TR928-CZARNY"],
        })
        result = extractor.extract_dataframe(df)
        assert result["color"].iloc[0] == "czarny"
        assert result["product_type"].iloc[0] == "botki"
        assert result["brand"].iloc[0] == "Sergio Leone"
        assert result["size"].iloc[0] == 38

    def test_handles_missing_name(self, extractor: FeatureExtractor) -> None:
        df = pd.DataFrame({
            "name": [None],
            "products_sku": ["OPT-123"],
        })
        result = extractor.extract_dataframe(df)
        assert result["color"].iloc[0] is None
        assert result["product_type"].iloc[0] is None

    def test_handles_missing_sku(self, extractor: FeatureExtractor) -> None:
        df = pd.DataFrame({
            "name": ["CZARNE BOTKI MONNARI 38"],
        })
        result = extractor.extract_dataframe(df)
        assert result["brand"].iloc[0] == "Monnari"

    def test_does_not_modify_original(self, extractor: FeatureExtractor) -> None:
        df = pd.DataFrame({
            "name": ["BOTKI 38"],
            "products_sku": ["X-1"],
        })
        result = extractor.extract_dataframe(df)
        assert "color" not in df.columns
        assert "color" in result.columns

    def test_multiple_rows(self, extractor: FeatureExtractor) -> None:
        df = pd.DataFrame({
            "name": [
                "CZARNE BOTKI 38",
                "BIAŁE SZPILKI 39",
                "PORTFEL SKÓRZANY",
            ],
            "products_sku": ["SL-1", "OPT-2", "MON-3"],
        })
        result = extractor.extract_dataframe(df)
        assert len(result) == 3
        assert result["color"].iloc[0] == "czarny"
        assert result["color"].iloc[1] == "bialy"
        assert pd.isna(result["color"].iloc[2])
