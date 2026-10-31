@echo off
cd /d "%~dp0"
set "FRAUD_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "FRAUD_PYTHON=.venv\Scripts\python.exe"
"%FRAUD_PYTHON%" -m fraud_lstm run --config configs/full.json
if errorlevel 1 goto gagal
echo Penelitian selesai. Lihat results\full\LAPORAN_HASIL.md.
pause
exit /b 0
:gagal
echo Proses gagal. Periksa pesan di atas. Jangan menimpa hasil eksperimen sebelumnya.
pause
exit /b 1
