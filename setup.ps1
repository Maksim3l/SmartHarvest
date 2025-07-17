Write-Host "Setting up the environment for the model..."
python -m venv Model_env
& "Model_env\Scripts\Activate.ps1"

Write-Host "Installing required packages..."
pip install -r model_requirements.txt

Write-Host "Setting up the pictures and masks..."
python MakeMaskInstanceSemnatic.py


$install_cuda = Read-Host "Want to install CUDA? (y/n)"
if ($install_cuda -eq "y" -or $install_cuda -eq "Y") {
    Write-Host "Installing CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
} else {
    Write-Host "Skipping CUDA installation."
}

Write-Host "Setup complete. Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")