# Create the virtual environment
uv venv

# Activate it (Mac/Linux)
source .venv/bin/activate
# Activate it (Windows)
# .venv\Scripts\activate

# Install the exact packages
uv pip install -r requirements.txt