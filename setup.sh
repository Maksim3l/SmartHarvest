echo "Setting up the environment for the model..."
python -m venv model_env
source model_env/Scripts/activate

echo "Installing required packages..."
pip install -r model_requirements.txt

echo "Setting up the pictures and masks..."
python MakeMaskInstanceSemnatic.py

read -p "Want to install CUDA? (y/n): " install_cuda

if [ "$install_cuda" == "y" ]; then
    echo "Installing CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Skipping CUDA installation."
fi

echo "Setup complete. Press any key to exit..."
read -n 1 -s