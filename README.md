# 🎵 AI Music Studio

> AI-powered music generation web app — create, discover, and share AI-generated music with lyrics, humming recognition, and an interactive community gallery.

**[🚀 Live Demo (Render.com)](#deployment) | [📖 Features](#features) | [🛠 Tech Stack](#tech-stack)**

---

## Features

| Feature | Description |
|---|---|
| 🎵 **Generate Music** | Text-to-music using Meta MusicGen (300M params) |
| 🎤 **Music + Lyrics** | AI lyrics in English, Telugu, Hindi, Tamil & mixed styles |
| 🔀 **Remix Songs** | Upload audio → continue/remix with a new style |
| 🎬 **Video BGM** | Upload a video → auto-generate duration-matched background music |
| 🔍 **Song Discovery** | Hum or type lyrics → identify songs via AudD fingerprinting |
| 🌐 **Community Gallery** | Browse & play tracks from all users |
| ✨ **Prompt Enhancer** | AI rewrites your description into professional music terms |
| 🎙️ **Voice Input** | Speak your music description instead of typing |
| 🎮 **Music Quiz** | Interactive music trivia game |

---

## Tech Stack

- **Backend:** Python, Flask, Gunicorn
- **AI Models:** [Meta MusicGen](https://github.com/facebookresearch/audiocraft) (music), [Groq LLaMA 3.3 70B](https://groq.com) (lyrics/text)
- **Audio:** PyTorch, torchaudio, MoviePy, pydub
- **Frontend:** Vanilla JS, WaveSurfer.js, Font Awesome
- **Database:** SQLite (via `database.py`)
- **Hosting:** [Render.com](https://render.com)

---

## Deployment

### Deploy to Render.com (Recommended)

1. **Fork / push this repo to your GitHub account.**

2. **Go to [render.com](https://render.com) → New → Web Service → Connect your repo.**

3. Render will auto-detect `render.yaml`. Set these environment variables in the Render dashboard:

   | Variable | Description |
   |---|---|
   | `GROQ_API_KEY` | Free at [console.groq.com](https://console.groq.com) |
   | `AUDD_API_TOKEN` | Free at [audd.io](https://audd.io) (optional, for song recognition) |

4. Click **Deploy**. Render installs dependencies and starts Gunicorn automatically.

> ⚠️ **Note:** MusicGen requires significant RAM (~4 GB). Use at least a **Standard** Render plan for reliable generation.

### Run Locally

```bash
# 1. Clone
git clone https://github.com/saibhanush/AI-Music-Studio-project-.git
cd AI-Music-Studio-project-

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export GROQ_API_KEY=your_groq_key_here
export AUDD_API_TOKEN=your_audd_token_here   # optional

# 5. Run
python app.py
```

Open `http://localhost:5000` — the showcase landing page loads first.  
Sign up with any email or phone number; the OTP is printed to the terminal in dev mode.

---

## Pages

| Route | Description |
|---|---|
| `/` | Public showcase / college project landing page |
| `/login` | Sign-in with OTP (email or phone) |
| `/studio` | Main music generation studio *(login required)* |
| `/discovery` | Song discovery & lyric search *(login required)* |
| `/gallery` | Community gallery *(login required)* |
| `/profile` | User profile & track history *(login required)* |

---

## Project Structure

```
AI-Music-Studio-project-/
├── app.py           # Flask routes & AI generation logic
├── database.py      # SQLite helpers (users & tracks)
├── requirements.txt
├── Procfile         # Gunicorn start command
├── render.yaml      # Render.com deploy config
├── templates/
│   ├── showcase.html   # Public landing page
│   ├── login.html
│   ├── index.html      # Studio
│   ├── discovery.html
│   ├── gallery.html
│   └── profile.html
└── static/
    ├── style.css
    ├── script.js
    └── generated/      # AI-generated audio files
```

---

*Built as a college project — AI & Full-Stack Web Development*
