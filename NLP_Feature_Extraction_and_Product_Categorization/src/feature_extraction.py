"""Rule-based feature extraction from Polish product names.

Extracts structured attributes (color, material, size, product type, brand,
season) using dictionary matching and regex patterns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProductFeatures:
    """Extracted features from a single product name."""

    color: str | None = None
    material: str | None = None
    size: int | None = None
    product_type: str | None = None
    brand: str | None = None
    season: str | None = None


# ---------------------------------------------------------------------------
# Size regex
# ---------------------------------------------------------------------------

SIZE_PATTERN = re.compile(r"\b(3[5-9]|4[0-9]|50)\b")


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Rule-based feature extraction from Polish product names.

    Loads dictionaries from config and matches them against product text.
    """

    def __init__(
        self,
        colors: dict[str, list[str]],
        materials: dict[str, list[str]],
        product_types: dict[str, list[str]],
        season_indicators: dict[str, list[str]],
        brand_sku_prefixes: dict[str, str],
    ) -> None:
        """Initialize with extraction dictionaries.

        Args:
            colors: Canonical color -> list of surface forms.
            materials: Canonical material -> list of surface forms.
            product_types: Canonical type -> list of surface forms.
            season_indicators: Season category -> list of indicator words.
            brand_sku_prefixes: SKU prefix -> brand name.
        """
        self.color_map = self._build_lookup(colors)
        self.material_map = self._build_lookup(materials)
        self.product_type_map = self._build_lookup(product_types)
        self.season_map = self._build_season_lookup(season_indicators)
        self.brand_sku_prefixes = brand_sku_prefixes

    @staticmethod
    def _build_lookup(mapping: dict[str, list[str]]) -> dict[str, str]:
        """Build surface_form -> canonical_name lookup from config mapping."""
        lookup: dict[str, str] = {}
        for canonical, forms in mapping.items():
            for form in forms:
                lookup[form.upper()] = canonical
        return lookup

    @staticmethod
    def _build_season_lookup(
        season_indicators: dict[str, list[str]],
    ) -> dict[str, str]:
        """Build indicator_word -> season_category lookup."""
        lookup: dict[str, str] = {}
        for season, indicators in season_indicators.items():
            for word in indicators:
                lookup[word.upper()] = season
        return lookup

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FeatureExtractor:
        """Create FeatureExtractor from the 'extraction' section of config.yaml.

        Args:
            config: The 'extraction' dict from config.yaml.

        Returns:
            Configured FeatureExtractor instance.
        """
        return cls(
            colors=config.get("colors", {}),
            materials=config.get("materials", {}),
            product_types=config.get("product_types", {}),
            season_indicators=config.get("season_indicators", {}),
            brand_sku_prefixes=config.get("brand_sku_prefixes", {}),
        )

    def extract_color(self, text: str) -> str | None:
        """Match first color from dictionary against text tokens.

        Args:
            text: Uppercase product name.

        Returns:
            Canonical color name or None.
        """
        text_upper = text.upper()
        for form, canonical in self.color_map.items():
            if form in text_upper.split():
                return canonical
        return None

    def extract_material(self, text: str) -> str | None:
        """Match material from dictionary against text tokens.

        Args:
            text: Uppercase product name.

        Returns:
            Canonical material name or None.
        """
        text_upper = text.upper()
        for form, canonical in self.material_map.items():
            if form in text_upper.split():
                return canonical
        return None

    def extract_size(self, text: str) -> int | None:
        """Extract shoe/clothing size (35-50) from text.

        Prefers the last numeric match in the valid range, as sizes are
        typically placed at the end of product names.

        Args:
            text: Product name string.

        Returns:
            Integer size or None.
        """
        matches = SIZE_PATTERN.findall(text)
        if matches:
            return int(matches[-1])
        return None

    def extract_product_type(self, text: str) -> str | None:
        """Match product type keyword against text tokens.

        Args:
            text: Uppercase product name.

        Returns:
            Canonical product type or None.
        """
        text_upper = text.upper()
        for form, canonical in self.product_type_map.items():
            if form in text_upper.split():
                return canonical
        return None

    def extract_brand(
        self, text: str, sku: str | None = None
    ) -> str | None:
        """Extract brand from SKU prefix or product name.

        SKU prefix takes priority over name-based matching.

        Args:
            text: Product name string.
            sku: SKU code string or None.

        Returns:
            Brand name or None.
        """
        if sku:
            sku_str = str(sku)
            for prefix, brand in self.brand_sku_prefixes.items():
                if sku_str.startswith(prefix):
                    return brand

        text_upper = text.upper()
        brand_keywords = {
            "OPTIMO": "Optimo",
            "MONNARI": "Monnari",
            "SERGIO LEONE": "Sergio Leone",
            "SEASTAR": "Seastar",
            "BUTDAM": "Butdam",
        }
        for keyword, brand in brand_keywords.items():
            if keyword in text_upper:
                return brand

        return None

    def extract_season(self, text: str) -> str | None:
        """Match season/warmth indicators in text.

        Args:
            text: Uppercase product name.

        Returns:
            Season category ('warm', 'cold') or None.
        """
        text_upper = text.upper()
        for form, season in self.season_map.items():
            if form in text_upper.split():
                return season
        return None

    def extract_all(
        self, text: str, sku: str | None = None
    ) -> ProductFeatures:
        """Apply all extractors and return ProductFeatures.

        Args:
            text: Product name string.
            sku: SKU code string or None.

        Returns:
            ProductFeatures with all extracted attributes.
        """
        if not text:
            return ProductFeatures()

        return ProductFeatures(
            color=self.extract_color(text),
            material=self.extract_material(text),
            size=self.extract_size(text),
            product_type=self.extract_product_type(text),
            brand=self.extract_brand(text, sku),
            season=self.extract_season(text),
        )

    def extract_dataframe(
        self,
        df: pd.DataFrame,
        name_col: str = "name",
        sku_col: str = "products_sku",
    ) -> pd.DataFrame:
        """Apply extraction to full DataFrame, adding feature columns.

        Adds columns: color, material, size, product_type, brand, season.

        Args:
            df: Input DataFrame.
            name_col: Column with product names.
            sku_col: Column with SKU codes.

        Returns:
            DataFrame with added feature columns.
        """
        df = df.copy()

        features_list: list[ProductFeatures] = []
        for _, row in df.iterrows():
            name = str(row.get(name_col) or "")
            sku = row.get(sku_col) if sku_col in df.columns else None
            features_list.append(self.extract_all(name, sku))

        df["color"] = [f.color for f in features_list]
        df["material"] = [f.material for f in features_list]
        df["size"] = [f.size for f in features_list]
        df["product_type"] = [f.product_type for f in features_list]
        df["brand"] = [f.brand for f in features_list]
        df["season"] = [f.season for f in features_list]

        # Log extraction coverage
        for col in ["color", "material", "size", "product_type", "brand", "season"]:
            coverage = df[col].notna().mean() * 100
            logger.info("Extraction coverage - %s: %.1f%%", col, coverage)

        return df
