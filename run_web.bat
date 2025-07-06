@echo off
echo Starting backend (Flask)...
cd E:\Download\Deep_Normal\PUBLIC
start cmd /k python WEBDEEP.py

cd E:\Download\Deep_Normal\PUBLIC\webdeep-frontend
echo Starting frontend (React)...
npm start
