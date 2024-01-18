#!/bin/bash
source env/bin/activate
python -m flask --app webapp.py --debug run
