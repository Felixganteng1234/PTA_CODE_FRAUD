@echo off
cd /d "%~dp0"
set "FRAUD_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "FRAUD_PYTHON=.venv\Scripts\python.exe"
"%FRAUD_PYTHON%" -m fraud_lstm run --config configs/smoke.json --output results/smoke_baru
if errorlevel 1 goto gagal
echo Uji teknis selesai. Lihat results\smoke_baru\LAPORAN_HASIL.md.
pause
exit /b 0
:gagal
echo Proses gagal. Jika folder hasil sudah ada, gunakan --output folder_baru lewat terminal.
pause
exit /b 1
