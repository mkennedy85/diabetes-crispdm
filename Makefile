PY=python

setup:
	$(PY) -m pip install -r requirements.txt

baselines:
	$(PY) -m btds.baselines --data_path data/diabetes_binary_health_indicators_BRFSS2015.csv --out_dir reports

train:
	$(PY) -m btds.train --data_path data/diabetes_binary_health_indicators_BRFSS2015.csv --target Diabetes_binary --out_dir reports

eval:
	$(PY) -m btds.eval --data_path data/diabetes_binary_health_indicators_BRFSS2015.csv --target Diabetes_binary --out_dir reports

report:
	@echo "CRISP-DM writeup at reports/report.md; figures in reports/figures"
