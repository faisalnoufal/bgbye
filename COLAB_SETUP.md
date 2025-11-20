# 🚀 Running BGBye on Google Colab

This guide explains how to run the BGBye background removal API on Google Colab for free.

## 📋 Prerequisites

1. **Google Account** - For Google Colab access
2. **HuggingFace Account** - For RMBG-2.0 model access
   - Sign up: https://huggingface.co/join
   - Get token: https://huggingface.co/settings/tokens
   - Request access: https://huggingface.co/briaai/RMBG-2.0
3. **Ngrok Account** (Optional but recommended)
   - Sign up: https://dashboard.ngrok.com/signup
   - Get auth token: https://dashboard.ngrok.com/get-started/your-authtoken

---

## 🎯 Quick Start

### Option 1: Upload Notebook (Recommended)

1. **Download the notebook**: [`bgbye_colab.ipynb`](bgbye_colab.ipynb)
2. Go to [Google Colab](https://colab.research.google.com/)
3. Click **File → Upload notebook**
4. Upload `bgbye_colab.ipynb`
5. Run cells in order (1 → 2 → 3 → 4 → 5)

### Option 2: Copy-Paste Code

Create a new notebook in [Google Colab](https://colab.research.google.com/) and copy-paste these cells:

---

## 📝 Code Cells

### Cell 1: Clone Repository

```python
# Clone the repository
!git clone https://github.com/faisalnoufal/bgbye.git
%cd bgbye/server

print("\n✅ Repository cloned successfully!")
```

---

### Cell 2: Install Dependencies

```python
# Install core dependencies
!pip install -q fastapi uvicorn python-multipart
!pip install -q pillow numpy requests

# Install PyTorch (CPU version for Colab)
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install ML models and utilities
!pip install -q carvekit transformers huggingface-hub

# Install ngrok for public URL
!pip install -q pyngrok

# Install nest-asyncio for Colab compatibility
!pip install -q nest-asyncio

print("\n✅ All dependencies installed successfully!")
```

---

### Cell 3: HuggingFace Authentication

```python
from huggingface_hub import login

# This will prompt for your HuggingFace token
login()

print("\n✅ HuggingFace authentication successful!")
```

**Before running:** Get your token from https://huggingface.co/settings/tokens

---

### Cell 4: Setup Ngrok Tunnel

```python
from pyngrok import ngrok

# Optional: Add your ngrok authtoken for better limits
# ngrok.set_auth_token("YOUR_NGROK_TOKEN")

# Kill any existing tunnels
ngrok.kill()

# Create tunnel on port 9876
public_url = ngrok.connect(9876, bind_tls=True)

print("\n" + "="*60)
print("🌐 PUBLIC API URL:")
print(f"   {public_url}")
print("\n📝 API ENDPOINT:")
print(f"   {public_url}/remove_background/")
print("\n📖 API DOCUMENTATION:")
print(f"   {public_url}/docs")
print("="*60)
print("\n✅ Copy the public URL above to use in your frontend!")
```

---

### Cell 5: Start Server (Keep Running)

```python
import nest_asyncio
import uvicorn
from threading import Thread
import time

# Allow nested event loops (required for Colab)
nest_asyncio.apply()

# Function to run the server
def run_server():
    uvicorn.run("server:app", host="0.0.0.0", port=9876, log_level="info")

# Start server in background thread
server_thread = Thread(target=run_server, daemon=True)
server_thread.start()

print("\n⏳ Starting server and loading models...")
print("   (First-time model downloads may take 2-3 minutes)\n")
time.sleep(10)

print("\n" + "="*60)
print("✅ SERVER IS RUNNING!")
print("="*60)
print(f"\n🔗 Public URL: {public_url}")
print(f"\n📝 API Endpoint: {public_url}/remove_background/")
print("\n⚠️  IMPORTANT: Keep this cell running!")
print("="*60)

# Keep alive loop
try:
    while True:
        time.sleep(60)
        print("⏰ Server is running...", end='\r')
except KeyboardInterrupt:
    print("\n⛔ Server stopped.")
```

**⚠️ DO NOT STOP THIS CELL** - It keeps your server running!

---

### Cell 6: Test API (Optional)

```python
import requests
from google.colab import files
from PIL import Image
from IPython.display import display
import io

# Upload a test image
print("📤 Upload an image to test:")
uploaded = files.upload()

if uploaded:
    filename = list(uploaded.keys())[0]
    image_bytes = uploaded[filename]
    
    # Display original
    print("\n📷 Original Image:")
    display(Image.open(io.BytesIO(image_bytes)))
    
    # Process with API
    print("\n⏳ Processing with RMBG-2.0...")
    response = requests.post(
        f"{public_url}/remove_background/",
        files={"file": (filename, image_bytes)},
        data={"method": "rmbg"}
    )
    
    if response.status_code == 200:
        print("\n✅ Success! Processed Image:")
        result_image = Image.open(io.BytesIO(response.content))
        display(result_image)
        
        # Save and download result
        result_image.save("result.png")
        files.download("result.png")
        print("\n💾 Downloaded as result.png")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)
```

---

## 🔧 Connect to Your Frontend

### Update Local `.env` File

After Cell 4 gives you the ngrok URL, update your local `.env`:

```env
REACT_APP_TRACER_URL=https://xxxx-xx-xxx-xxx-xxx.ngrok-free.app
REACT_APP_BASNET_URL=https://xxxx-xx-xxx-xxx-xxx.ngrok-free.app
REACT_APP_RMBG_URL=https://xxxx-xx-xxx-xxx-xxx.ngrok-free.app
```

### Restart Your React App

```bash
npm start
```

Your local frontend will now use the Colab backend! 🎉

---

## 📊 Available Models

| Model | Method Name | Best For | Speed |
|-------|-------------|----------|-------|
| **RMBG-2.0** | `rmbg` | Vehicles, complex scenes | Fast |
| **Tracer-B7** | `tracer` | Excellent edges | Medium |
| **BASNet** | `basnet` | General purpose | Fast |

---

## 🌐 API Usage

### POST `/remove_background/`

**Parameters:**
- `file`: Image file (required)
- `method`: Model name - `rmbg`, `tracer`, or `basnet` (required)

**Example with cURL:**

```bash
curl -X POST "https://your-ngrok-url.ngrok-free.app/remove_background/" \
  -F "file=@image.jpg" \
  -F "method=rmbg" \
  --output result.png
```

**Example with Python:**

```python
import requests

response = requests.post(
    "https://your-ngrok-url.ngrok-free.app/remove_background/",
    files={"file": open("image.jpg", "rb")},
    data={"method": "rmbg"}
)

with open("result.png", "wb") as f:
    f.write(response.content)
```

---

## ⚠️ Important Notes

### Session Limits
- **Colab Free**: ~12 hours max runtime, then session expires
- **Solution**: Re-run all cells to restart (takes ~5 minutes)

### Ngrok Limits
- **Free Tier**: 40 connections/minute
- **URL Changes**: New URL every time you restart
- **Solution**: Get free auth token for stable URLs

### Performance
- **First Load**: 2-3 minutes (downloads models)
- **Subsequent**: Instant (models cached)
- **Processing**: 2-5 seconds per image

### Costs
- **Colab**: FREE ✅
- **Ngrok**: FREE (with limits) ✅
- **HuggingFace**: FREE ✅

---

## 🐛 Troubleshooting

### "Module not found" Error
**Solution**: Re-run Cell 2 (Install Dependencies)

### "HuggingFace token invalid"
**Solution**: 
1. Check token at https://huggingface.co/settings/tokens
2. Ensure you requested access to RMBG-2.0
3. Re-run Cell 3

### "Connection refused" Error
**Solution**: Cell 5 (server) stopped. Re-run it.

### "Ngrok tunnel expired"
**Solution**: Free ngrok sessions expire after 2 hours. Re-run Cell 4 and Cell 5, then update your `.env` with the new URL.

### Models loading slow
**Solution**: First-time downloads are slow. Wait 2-3 minutes. Subsequent runs are instant.

---

## 🎓 Tips & Best Practices

1. **Save your work**: Download processed images before session expires
2. **Use auth tokens**: Add ngrok auth token for stable URLs
3. **Monitor usage**: Check ngrok dashboard at http://127.0.0.1:4040
4. **Test locally first**: Use Cell 6 to test before connecting frontend
5. **Keep Cell 5 running**: Don't stop the server cell while in use

---

## 📚 Additional Resources

- **API Documentation**: Visit `your-ngrok-url/docs` for interactive API docs
- **Ngrok Dashboard**: http://127.0.0.1:4040 (while running)
- **Model Info**:
  - [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
  - [Carvekit Models](https://github.com/OPHoperHPO/image-background-remove-tool)

---

## 🎉 Success!

Your BGBye API is now running in the cloud for free! Share your results and enjoy unlimited background removal! 🚀
