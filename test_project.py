"""Small automated checks for the data safeguards and the deployed model pipeline."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from train_model import AMENITIES, AREA, BEDROOMS, LOCATION, PRICE, RESALE, load_and_validate, make_pipeline

ROOT = Path(__file__).resolve().parent


class ProjectDataTests(unittest.TestCase):
    def test_source_data_is_validated_and_exact_duplicates_removed(self):
        data, audit = load_and_validate(ROOT / "Hyderabad.csv")
        self.assertEqual(len(data), audit["training_rows"])
        self.assertEqual(audit["exact_duplicate_rows_removed"], 520)
        self.assertGreater(len(data), 1000)
        self.assertTrue((data[PRICE] > 0).all())
        self.assertTrue((data[AREA] > 0).all())
        self.assertTrue(data[LOCATION].str.strip().ne("").all())

    def test_amenity_unknown_is_not_converted_to_no(self):
        data, _ = load_and_validate(ROOT / "Hyderabad.csv")
        self.assertIn("Not specified", set(data[AMENITIES[0]]))
        self.assertIn("No", set(data[AMENITIES[0]]))
        self.assertIn("Yes", set(data[AMENITIES[0]]))

    def test_wrong_csv_fails_with_missing_column_message(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "wrong.csv"
            pd.DataFrame({"Price": [1], "Area": [1000]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_and_validate(path)

    def test_pipeline_can_fit_and_predict(self):
        data, _ = load_and_validate(ROOT / "Hyderabad.csv")
        sample = data.sample(180, random_state=8)
        model = make_pipeline("Ridge").fit(sample[[AREA, BEDROOMS, LOCATION, RESALE, *AMENITIES]], sample[PRICE])
        predictions = model.predict(sample[[AREA, BEDROOMS, LOCATION, RESALE, *AMENITIES]].head(5))
        self.assertEqual(len(predictions), 5)
        self.assertTrue(pd.Series(predictions).notna().all())
        self.assertTrue((predictions > 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
