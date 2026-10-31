@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto gagal
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto gagal
echo Persiapan selesai. Jalankan JALANKAN_UJI_CEPAT.bat atau JALANKAN_PENELITIAN.bat.
pause
exit /b 0
:gagal
echo Persiapan gagal. Periksa pesan di atas dan koneksi internet.
pause
exit /b 1
