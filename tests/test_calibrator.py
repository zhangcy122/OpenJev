import pytest
from openjevpro.calibrator import TemperatureCalibrator

class TestCalibrator:
    def test_default_calibration(self):
        calib = TemperatureCalibrator(temperature=1.25)
        probs = calib.calibrate({"A": 2.0, "B": 1.0})
        assert "A" in probs and "B" in probs
        assert probs["A"] > probs["B"]
        assert round(sum(probs.values()), 5) == 1.0

    def test_fit_temperature_convergence(self):
        calib = TemperatureCalibrator(temperature=1.0)
        
        # Confident logits matching targets
        logits_list = [
            {"cat": 5.0, "dog": 1.0, "bird": 0.5},
            {"cat": 0.5, "dog": 6.0, "bird": 1.0},
            {"cat": 1.0, "dog": 1.5, "bird": 7.0},
            {"cat": 4.5, "dog": 1.2, "bird": 0.8},
        ]
        targets = ["cat", "dog", "bird", "cat"]
        
        fitted_t = calib.fit(logits_list, targets)
        assert 0.1 <= fitted_t <= 10.0
        assert calib.temperature == fitted_t
        
        # Calibrated output matches updated temperature
        probs = calib.calibrate(logits_list[0])
        assert probs["cat"] > 0.5

    def test_fit_temperature_insufficient_samples_raises(self):
        calib = TemperatureCalibrator(temperature=1.25)
        with pytest.raises(ValueError):
            calib.fit([], [])
            
        with pytest.raises(ValueError):
            calib.fit([{"cat": 1.0}], ["cat"])

    def test_fit_temperature_mismatched_lengths_raises(self):
        calib = TemperatureCalibrator(temperature=1.25)
        with pytest.raises(ValueError):
            calib.fit([{"cat": 1.0}, {"dog": 2.0}], ["cat"])
