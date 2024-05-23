::  Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
::
::  Licensed under the Apache License, Version 2.0 (the "License");
::  you may not use this file except in compliance with the License.
::  You may obtain a copy of the License at
::
::  http://www.apache.org/licenses/LICENSE-2.0
::
::  Unless required by applicable law or agreed to in writing, software
::  distributed under the License is distributed on an "AS IS" BASIS,
::  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
::  See the License for the specific language governing permissions and
::  limitations under the License.

@ECHO OFF

:: get current working dir
SET cwd=%cd%

:: install python modules
SET script_dir=%~dp0

:: get commandline option
SET full_install=0
SET force_reinstall=""
SET llvm_path=""
SET mingw_w64_path=""
SET skip_check_cert=

ECHO ============ checking params and inputs =============================

:loop
IF NOT "%1"=="" (
    IF "%1"=="--full" (
        SET full_install=1
        SHIFT
    ) ELSE IF "%1"=="-k"  (
        SET skip_check_cert=-k
        SHIFT
    ) ELSE IF "%1"=="--force-reinstall"  (
        SET force_reinstall="--force-reinstall"
        SHIFT
    ) ELSE IF "%1"=="--llvm"  (
        SET llvm_path="%~2"
        SHIFT & SHIFT
    ) ELSE IF "%1"=="--mingw"  (
        SET mingw_w64_path="%~2"
        SHIFT & SHIFT
    ) ELSE (
        ECHO unknown option: %1, exiting...
		EXIT /B 1
    )
    GOTO :loop
)

:: read config.ini file content
SET CONFIG_FILE_PARAM=%script_dir%\config.ini
FOR /f "delims=" %%i IN ('type "%CONFIG_FILE_PARAM%"^| find /i "="') DO (
    SET %%i
)

:: correct md5 of llvm and mingw install package
SET llvm_correct_md5=9bba445c3dbf99dac07e492ff2c55851
SET mingw_correct_md5=55c00ca779471df6faf1c9320e49b5a9

:: try find llvm package in current folder
SET llvm_file_name=LLVM-12.0.0-win64.exe
IF %llvm_path%=="" IF EXIST %llvm_file_name% (
    SET llvm_path=%llvm_file_name%
)

:: get MD5 of existing llvm install package
IF NOT %llvm_path%=="" (
    FOR /F %%i IN ('certutil -hashfile %llvm_path% MD5 ^| findstr /V MD5 ^| findstr /V CertUtil') DO SET llvm_md5=%%i
)

:: check MD5 of existing llvm install package
IF DEFINED llvm_md5 (
    IF NOT "%llvm_correct_md5%"=="%llvm_md5%" (
        ECHO the llvm package at %llvm_path% is incomplete due to wrong MD5, will skip using it and download new one.
        SET llvm_path=""
    )
)

:: try find mingw-w64 package in current folder
SET mingw_w64_file_name=x86_64-8.1.0-release-posix-seh-rt_v6-rev0.7z
IF %mingw_w64_path%=="" IF exist %mingw_w64_file_name% (
    SET mingw_w64_path=%mingw_w64_file_name%
)

:: get MD5 of existing mingw-w64 install package
IF NOT %mingw_w64_path%=="" (
    FOR /F %%i IN ('certutil -hashfile %mingw_w64_path% MD5 ^| findstr /V MD5 ^| findstr /V CertUtil') DO SET mingw_md5=%%i
)

:: check MD5 of existing mingw-w64 install package
IF DEFINED mingw_md5 (
    IF NOT "%mingw_md5%"=="%mingw_correct_md5%" (
        ECHO the mingw-w64 package at %mingw_w64_path% is incomplete due to wrong MD5, will skip using it and download new one.
        SET mingw_w64_path=""
    )
)

:: go to C:\, so that "import app_analyze" won't import the one in current transplt dir
C: && cd C:\

:: get python3 executable filename
SET PYTHON3=
python3 -V 2>nul ^| findstr /C:"Python 3." >nul && ( SET PYTHON3=python3 )

IF NOT DEFINED PYTHON3 (
    python -V 2>nul ^| findstr /C:"Python 3." >nul && ( SET PYTHON3=python )
)

IF NOT DEFINED PYTHON3 (
    ECHO "Error: python3 is not installed"
    %cwd:~0,2% && cd "%cwd%" && EXIT /B 1
)

:: get dir of installed app_analyze module
SET app_analyze_pos=
FOR /F "delims=" %%i IN ('%PYTHON3% -c "import app_analyze; print(app_analyze.__path__[0])"') DO ( SET app_analyze_pos=%%i)
IF NOT DEFINED app_analyze_pos (
    ECHO Cannot find python module: transplt, please make sure it is correctly installed.
    %cwd:~0,2% && cd "%cwd%" && EXIT /B 1
)

:: download and unzip config.zip & headers.zip
ECHO ============ downloading ait transplt config and headers ============
%app_analyze_pos:~0,2% && cd "%app_analyze_pos%"
curl %skip_check_cert% %download_config_zip_link% -o config.zip
IF NOT %errorlevel%==0 ( %cwd:~0,2% && cd "%cwd%" && EXIT /B 1 )
tar -p -zxf config.zip
IF NOT %errorlevel%==0 ( %cwd:~0,2% && cd "%cwd%" && EXIT /B 1 )
del config.zip
curl %skip_check_cert% %download_headers_zip_link% -o headers.zip
IF NOT %errorlevel%==0 ( %cwd:~0,2% && cd "%cwd%" && EXIT /B 1 )
tar -p -zxf headers.zip
IF NOT %errorlevel%==0 ( %cwd:~0,2% && cd "%cwd%" && EXIT /B 1 )
del headers.zip

:: go to original working dir
%cwd:~0,2% && cd "%cwd%"

:: if not full install, exit now.
IF %full_install%==0 (
    EXIT /B 0
)

:: get 7zip
SET zip_exists=0
FOR /F "delims=" %%I IN ("7z.exe") DO (
    IF EXIST %%~$PATH:I (
        rem 7zip existing in system PATH
        SET zip_loc="7z.exe"
        SET zip_exists=1
    ) ELSE (
        rem download a new 7zip
        curl %skip_check_cert% -LJo "%TEMP%\7z2301-x64.msi" %download_7zip_link%
        msiexec /a "%TEMP%\7z2301-x64.msi" /qb TARGETDIR="%TEMP%\7zip"
        SET zip_loc="%TEMP%\7zip\Files\7-Zip\7z.exe"
        del 7z2301-x64.msi
    )
)

net.exe session 1>NUL 2>NUL && (
    SET pkg_install_path=C:\Program Files
    SET is_admin=1
) || (
    SET pkg_install_path=%USERPROFILE%\AppData\Local
    SET is_admin=0
)
:: download and install LLVM
IF %llvm_path%=="" (
    ECHO ============ downloading llvm installation package ==================
    curl %skip_check_cert% --connect-timeout 30 -LJo %llvm_file_name% %download_llvm_link%
    IF NOT %errorlevel%==0 (
        ECHO Downloading llvm failed, please try again or manually download it.
        %cwd:~0,2% && cd "%cwd%" && EXIT /B 1
    )
)

:: check md5 of downloaded llvm package
IF %llvm_path%=="" (
    SET llvm_md5=""
    FOR /F %%i IN ('certutil -hashfile %llvm_file_name% MD5 ^| findstr /V MD5 ^| findstr /V CertUtil') DO SET llvm_md5=%%i
    IF %llvm_md5%==%llvm_correct_md5% (
        SET llvm_path=%llvm_file_name%
    ) ELSE (
        ECHO The downloaded llvm installation file is incomplete, please try again or manually download it.
        %cwd:~0,2% && cd "%cwd%" && EXIT /B 1
    )
)

:: unzip llvm package to installing path
ECHO ============ unzip llvm installation package ========================
SET llvm_install_path=%pkg_install_path%\LLVM
%zip_loc% x -y %llvm_path% -o"%llvm_install_path%"
ECHO LLVM is installed to: %llvm_install_path%

ECHO ============ sets llvm system environment ===========================
:: add LLVM include path to system environment
if %is_admin% == 1 (
    setx /m "CPLUS_INCLUDE_PATH" "%llvm_install_path%\lib\clang\12.0.0\include"
) ELSE (
    setx "CPLUS_INCLUDE_PATH" "%llvm_install_path%\lib\clang\12.0.0\include"
)
SET CPLUS_INCLUDE_PATH=%llvm_install_path%\lib\clang\12.0.0\include

SET path_value1=
:: add LLVM bin folder to system Path environment if it's not added.
if %is_admin% == 1 (
    for /f "usebackq tokens=2,*" %%A in (`reg query "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH`) do set path_value1=%%B
) ELSE (
    for /f "usebackq tokens=2,*" %%A in (`reg query HKCU\Environment /v PATH`) do set path_value1=%%B
)
IF DEFINED path_value1 (
    IF "%path_value1:~-1%" == ";" (
        SET "path_value1=%path_value1:~0,-1%"
    )
)
IF DEFINED path_value1 (
    if %is_admin% == 1 (
        ECHO "%path_value1%" | findstr /C:"%llvm_install_path%\bin" >nul || (
            setx /m "Path" "%path_value1:"=%;%llvm_install_path%\bin"
        )
    ) ELSE (
        ECHO "%path_value1%" | findstr /C:"%llvm_install_path%\bin" >nul || (
            setx "Path" "%path_value1:"=%;%llvm_install_path%\bin"
        )
    )
)

SET Path=%Path:"=%
IF "%Path:~-1%" == ";" (
    SET "Path=%Path:~0,-1%"
)
SET "Path=%Path%;%llvm_install_path%\bin"

:: download and install mingw-w64
IF %mingw_w64_path%=="" (
    ECHO ============ downloading mingw-w64 installation package =============
    curl %skip_check_cert% --connect-timeout 30 -LJo %mingw_w64_file_name% %download_mingw_link%
    IF NOT %errorlevel%==0 (
        ECHO Downloading mingw-w64 failed, please try again or manually download it.
        %cwd:~0,2% && cd "%cwd%" && EXIT /B 1
    )
)

:: check md5 of downloaded mingw-w64 package
IF %mingw_w64_path%=="" (
    SET mingw_md5=""
    FOR /F %%i IN ('certutil -hashfile %mingw_w64_file_name% MD5 ^| findstr /V MD5 ^| findstr /V CertUtil') DO SET mingw_md5=%%i
    IF %mingw_md5%==%mingw_correct_md5% (
        SET mingw_w64_path=%mingw_w64_file_name%
    ) ELSE (
        ECHO The downloaded mingw-w64 installation file is incomplete, please try again or manually download it.
        %cwd:~0,2% && cd "%cwd%" && EXIT /B 1
    )
)

:: unzip mingw-w64 package to installing path
ECHO ============ unzip mingw-w64 installation package ===================
SET mingw_w64_install_path=%pkg_install_path%
%zip_loc% x -y %mingw_w64_path% -o"%mingw_w64_install_path%"
ECHO MinGW-W64 is installed to: %llvm_install_path%\mingw

ECHO ============ sets mingw-w64 system environment ======================
SET path_value2=
:: add mingw-w64 bin folder to system Path environment
if %is_admin% == 1 (
    for /f "usebackq tokens=2,*" %%A in (`reg query "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH`) do set path_value2=%%B
) ELSE (
    for /f "usebackq tokens=2,*" %%A in (`reg query HKCU\Environment /v PATH`) do set path_value2=%%B
)
IF DEFINED path_value2 (
    IF "%path_value2:~-1%" == ";" (
        SET "path_value2=%path_value2:~0,-1%"
    )
)
IF DEFINED path_value2 (
    if %is_admin% == 1 (
        ECHO "%path_value2%" | findstr /C:"%mingw_w64_install_path%\mingw64\bin" >nul || (
            setx /m "Path" "%path_value2:"=%;%mingw_w64_install_path%\mingw64\bin"
        )
    ) ELSE (
        ECHO "%path_value2%" | findstr /C:"%mingw_w64_install_path%\mingw64\bin" >nul || (
            setx "Path" "%path_value2:"=%;%mingw_w64_install_path%\mingw64\bin"
        )
    )
)

SET Path=%Path:"=%
IF "%Path:~-1%" == ";" (
    SET "Path=%Path:~0,-1%"
)
SET "Path=%Path%;%mingw_w64_install_path%\mingw64\bin"


ECHO ============ downloading patch file float.h of mingw-w64 ============
curl %skip_check_cert% --connect-timeout 30 -LJo "%mingw_w64_install_path%\mingw64\x86_64-w64-mingw32\include\float.h" %download_float_h_link%
IF NOT %errorlevel%==0 (
    ECHO WARNING: downloading mingw patch file float.h failed. This may cause ait transplt meets error when scanning projects.
    ECHO Please manually download it from %download_float_h_link%
    ECHO and then replace the file at "%mingw_w64_install_path%\mingw64\x86_64-w64-mingw32\include\float.h" with it
)

:: try to delete downloaded 7zip files
IF zip_exists==0 (
    RD /S /Q "%cwd%\7zip"
)

:: go to original working dir
%cwd:~0,2% && cd "%cwd%"

:: finished.
EXIT /B 0
