New-Item -ItemType Directory -Force -Path "models" | Out-Null
Invoke-WebRequest -Uri "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel" -OutFile "models\colorization_release_v2.caffemodel"
Write-Host "Descargado: models\colorization_release_v2.caffemodel"
