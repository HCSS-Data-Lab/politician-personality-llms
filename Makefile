PYTHON ?= python

.PHONY: test audit criterion analysis baselines ideology language figures package

test:
	$(PYTHON) -m pytest

audit:
	$(PYTHON) scripts/audit_release.py

criterion:
	$(PYTHON) scripts/build_inputs.py

analysis:
	$(PYTHON) scripts/run_analysis.py

baselines:
	$(PYTHON) scripts/identity_baselines.py

ideology:
	$(PYTHON) scripts/ideology.py

language:
	$(PYTHON) scripts/language_checks.py

figures:
	$(PYTHON) scripts/make_figure_overview_single.py
	$(PYTHON) scripts/make_figure_convergence_bias.py

package:
	$(PYTHON) scripts/package_release.py
