@echo off
cd /d "%~dp0"
set "FRAUD_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "FRAUD_PYTHON=.venv\Scripts\python.exe"
echo Menjalankan seluruh data dengan SATU konfigurasi hyperparameter yang ditetapkan sebelumnya.
echo Mode ini menghemat komputasi, tetapi bukan pencarian hyperparameter TPE.
"%FRAUD_PYTHON%" -m fraud_lstm run --config configs/full_single_trial.json
if errorlevel 1 goto gagal
echo Proses selesai. Lihat results\full_single_trial\LAPORAN_HASIL.md.
pause
exit /b 0
:gagal
echo Proses gagal atau terhenti. Untuk melanjutkan, jalankan perintah berikut:
echo "%FRAUD_PYTHON%" -m fraud_lstm run --config configs/full_single_trial.json --resume
pause
exit /b 1
