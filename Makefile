# The generated files and their checks; CI runs `make check`.
.PHONY: all check

all:
	python3 lab/catalog.py
	python3 lab/hardware.py
	python3 lab/export_plugin.py

check:
	python3 -m unittest discover -s lab -p test_check_recipe.py
	python3 -m unittest discover -s lab -p test_slotstream.py
	python3 -m unittest discover -s lab -p test_proxy_card.py
	python3 -m unittest discover -s lab -p test_export_plugin.py
	python3 -m unittest discover -s lab -p test_gate_answers.py
	python3 -m unittest discover -s lab -p test_speed_evidence.py
	python3 lab/lab.py check
	python3 -m unittest discover -s lab -p test_run_evidence.py
	python3 lab/catalog.py --check
	python3 lab/hardware.py --check
	python3 lab/export_plugin.py --check
	sdk/test.sh
