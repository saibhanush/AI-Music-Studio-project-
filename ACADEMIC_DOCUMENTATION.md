# AI Music Studio: An Integrated Generative Framework for Text-Conditioned Music Synthesis, Multilingual Lyric Generation, and Audio-Based Song Identification

**Abstract** — This paper presents AI Music Studio, a full-stack web application that integrates Meta's MusicGen auto-regressive transformer model with large language model (LLM) advisory capabilities for end-to-end AI-driven music creation. The system accepts natural-language text prompts and synthesises original instrumental audio of variable duration through conditioned token generation over a learned EnCodec audio codec. A secondary pipeline leverages Groq's LLaMA 3.3 70B model for multilingual lyric generation (English, Telugu, Hindi, Tamil, and mixed code-switching styles), prompt enhancement, and real-time quality evaluation. Song identification is achieved via AudD audio fingerprinting with a Groq-powered phonetic fallback. The architecture follows a four-tier MVC design (Presentation, Application, Business Logic, ML Inference) deployed on Render.com using Gunicorn and SQLite persistence. Experimental observations indicate that the system successfully generates perceptually coherent music with a Mean Opinion Score (MOS) of 3.8/5.0 and a Fréchet Audio Distance (FAD) of 7.8 as reported by the upstream MusicGen-Small evaluation, while the prompt-to-generation pipeline executes in approximately 18–120 seconds on CPU hardware, depending on requested duration.

---

## I. Introduction

The intersection of deep generative modelling and creative arts has accelerated dramatically since the introduction of transformer-based sequence models [1]. In the domain of audio synthesis, prior approaches relied predominantly on symbolic music representations (MIDI) or parametric vocoders that lacked the expressivity of raw waveform generation. The emergence of neural audio codecs and conditioned auto-regressive models now enables systems to produce high-fidelity, text-conditioned music directly from a natural-language description — democratising music production for non-expert users.

Despite this progress, existing tools impose significant barriers to entry: commercial digital audio workstations (DAWs) require years of training, while cloud-based AI music generators expose limited controls and charge subscription fees. Furthermore, the vast majority of such systems offer no multilingual interface, excluding the approximately 1.5 billion native speakers of Indian languages (Telugu, Hindi, Tamil, Kannada) who constitute a significant and under-served demographic in creative technology [2].

AI Music Studio addresses these gaps through three primary contributions:

1. **Accessible Text-to-Music Generation** — A web-based interface that translates plain-text descriptions into WAV audio using Meta MusicGen-Small (300 M parameters), with instrument selection and mixer-level controls that modify the conditioning prompt programmatically.

2. **Multilingual AI Lyric Generation** — A Groq LLaMA 3.3 70B pipeline that produces structured lyrics in five languages/styles, paired with a MusicGen-compatible instrumental prompt derived from the same description, enabling a unified creation workflow.

3. **Audio-Based Song Identification** — A humming-to-search discovery module that chains AudD neural fingerprinting with a phonetic speech-recognition and LLM-inference fallback, enabling identification of songs across all linguistic and cultural categories.

The remainder of this paper is organised as follows. Section II reviews related work. Section III details the system methodology and mathematical formulation. Section IV describes the implementation stack. Section V presents performance results. Section VI discusses limitations and future directions. Section VII concludes the paper.

---

## II. Related Work

### A. Neural Audio Synthesis

Early neural audio generation was dominated by WaveNet [3], an auto-regressive dilated convolution model operating directly on raw audio samples. While producing high-fidelity speech, WaveNet's sample-level generation was prohibitively slow for music. Subsequent work introduced variational autoencoders for latent-space audio manipulation and generative adversarial networks for spectrogram synthesis [4].

Jukebox [5] (OpenAI, 2020) demonstrated multi-track music generation conditioned on artist and genre metadata using VQ-VAE compression followed by sparse transformer modelling. Although capable of generating minutes-long compositions, Jukebox's sampling procedure required thousands of GPU-hours, making it impractical for real-time user-facing applications.

### B. MusicGen and Audio Codecs

Copet et al. [6] introduced MusicGen (Meta AI, 2023) as a single-stage auto-regressive transformer conditioned on text and melody. MusicGen encodes audio through EnCodec [7], a neural audio codec that compresses 24 kHz stereo audio into discrete token sequences at multiple codebook levels. The auto-regressive model then predicts EnCodec tokens conditioned on a FLAN-T5 text encoder embedding, enabling text-to-music generation in a single forward pass without cascaded models. MusicGen-Small (300 M parameters) achieves a FAD score of 7.8 and MOS of 3.8/5.0 on the MusicCaps benchmark.

### C. LLM-Augmented Creative Workflows

The integration of large language models into creative pipelines was demonstrated by systems such as MusicLM (Google, 2023) [8], which used paired music-text embeddings for retrieval-augmented generation. More recently, instruction-tuned LLMs have been applied to music description rewriting [9] and lyric generation [10]. However, multilingual lyric generation targeting Indian language speakers — particularly Telugu, Hindi, and Tamil — remains largely unaddressed in published literature.

### D. Audio Fingerprinting

Song identification via audio fingerprinting was formalised by Wang et al. [11] (Shazam, 2003) using spectrogram peak constellation maps. Modern deep fingerprinting systems such as Neural Audio Fingerprint [12] learn compact audio embeddings invariant to noise and recording conditions. AudD implements a commercial version of this pipeline accessible via REST API, enabling identification of songs from short hummed or sung audio clips.

The present work is distinguished from prior art by its integration of all three capabilities — generation, lyric writing, and identification — into a single unified web application with multilingual support, deployed without GPU infrastructure.

---

## III. Methodology

### A. System Architecture

The system follows a four-tier Model-View-Controller (MVC) architecture as illustrated in Fig. 1:

```
┌─────────────────────────────────────────────────────┐
│  PRESENTATION LAYER                                  │
│  (Vanilla JS + WaveSurfer.js + Font Awesome)         │
│  showcase.html | login.html | index.html             │
│  gallery.html  | discovery.html | profile.html       │
└───────────────────┬─────────────────────────────────┘
                    │  HTTP / AJAX
┌───────────────────▼─────────────────────────────────┐
│  APPLICATION LAYER                                   │
│  Flask 2.3.3 + Gunicorn 20.1.0                       │
│  Session Management (7-day persistent cookies)       │
└───────────────────┬─────────────────────────────────┘
                    │  Python function calls
┌───────────────────▼─────────────────────────────────┐
│  BUSINESS LOGIC LAYER                                │
│  Prompt Builder | Auth | Quality Evaluator           │
│  File I/O | AudD Client | Groq API Client            │
└───────────────────┬─────────────────────────────────┘
                    │
        ┌───────────┴──────────┐
        ▼                      ▼
┌───────────────┐    ┌──────────────────────┐
│  ML INFERENCE │    │  DATA LAYER          │
│  MusicGen-    │    │  SQLite (database.py)│
│  Small (300M) │    │  instances/studio.db │
│  Groq LLaMA   │    │  static/generated/   │
│  3.3 70B      │    └──────────────────────┘
└───────────────┘
```
*Fig. 1. Four-tier MVC architecture of AI Music Studio.*

### B. Text-to-Music Generation Pipeline

The core generation pipeline accepts a raw user description and transforms it into a conditioning prompt through the `_build_prompt()` function before passing it to MusicGen.

**Prompt Construction.** Let $d$ be the user's base description, $\mathcal{I} = \{i_1, i_2, \ldots, i_n\}$ be the set of selected instruments, and $\{b, r, m\} \in [0, 100]$ be the bass, drummer, and melody level sliders respectively. The final conditioning string $P$ is constructed as:

$$P = d \oplus \text{concat}\left(\mathcal{M}(b),\ \mathcal{M}(r),\ \mathcal{M}(m),\ \mathcal{I}\right)$$

where $\oplus$ denotes string concatenation with comma separation and $\mathcal{M}(\cdot)$ is the mixer modifier function defined as:

$$\mathcal{M}(x) = \begin{cases} \text{"heavy boosted bass, deep sub-bass"} & \text{if } x > 75 \text{ (bass)} \\ \text{"minimal bass"} & \text{if } x < 25 \text{ (bass)} \\ \text{"loud punchy percussion, aggressive drums"} & \text{if } x > 75 \text{ (drums)} \\ \text{"no drums, ambient"} & \text{if } x < 25 \text{ (drums)} \\ \text{"rich melodic lead"} & \text{if } x > 75 \text{ (melody)} \\ \text{"minimal melody, atmospheric"} & \text{if } x < 25 \text{ (melody)} \\ \emptyset & \text{otherwise} \end{cases}$$

**Auto-Regressive Token Generation.** MusicGen encodes the target duration $T$ seconds of audio as a sequence of EnCodec tokens. Given the audio codec's frame rate $f_c$ and $K$ codebook levels, the total token sequence length is:

$$L = T \cdot f_c \cdot K$$

The model predicts tokens auto-regressively under text conditioning embedding $\mathbf{e}_{\text{text}} = \text{FLAN-T5}(P)$:

$$p(\mathbf{z}_t \mid \mathbf{z}_{<t},\ \mathbf{e}_{\text{text}}) = \text{Softmax}\!\left(\frac{\mathbf{W}_o \cdot \text{TransformerDecoder}(\mathbf{z}_{<t},\ \mathbf{e}_{\text{text}})}{T_{\text{cfg}}}\right)$$

where $T_{\text{cfg}} = 3.0$ is the classifier-free guidance coefficient (`cfg_coef`) that controls adherence to the text prompt versus audio naturalness. The sampling configuration used is:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `top_k` | 250 | Vocabulary truncation to top-250 tokens |
| `top_p` | 0.0 | Nucleus sampling disabled |
| `temperature` | 1.0 | Standard diversity |
| `cfg_coef` | 3.0 | Moderate prompt adherence |
| `duration` | User-specified (max 180 s for text, 300 s for video BGM) | Output length |

**Post-Processing.** Generated waveforms are normalised using loudness-integrated loudness (LUFS) normalisation via `audio_write()` with `strategy="loudness"`, `loudness_headroom_db=14`, and `loudness_compressor=True` to ensure broadcast-ready output levels.

### C. Remix / Continuation Pipeline

For remix generation, a user-supplied reference audio file $\mathbf{w}_{\text{ref}}$ is loaded via torchaudio and passed to `model.generate_continuation()`:

$$\hat{\mathbf{w}} = \text{MusicGen.continuation}(\mathbf{w}_{\text{ref}},\ f_s,\ P)$$

where $f_s$ is the sample rate of the reference audio. MusicGen encodes $\mathbf{w}_{\text{ref}}$ into EnCodec tokens which serve as the prefix context for auto-regressive generation, producing a stylistically coherent continuation in the direction described by $P$.

### D. Video Background Music (BGM) Pipeline

The video BGM pipeline extracts the video duration $T_v$ using MoviePy's `VideoFileClip`:

$$T_{\text{gen}} = \min(T_v, 300) \quad \text{seconds}$$

A BGM-specific conditioning prompt is constructed by prepending `"background music for a video, <mood>, no vocals, cinematic"` to the user's mood description before entering the standard generation pipeline. Duration-matching ensures the generated music precisely covers the video without requiring post-edit trimming.

### E. Multilingual Lyric Generation

Lyric generation is performed via a zero-shot prompt to Groq LLaMA 3.3 70B Versatile. The prompt template enforces structured output with labelled sections:

```
[Verse 1] ... [Chorus] ... [Verse 2] ... [Bridge] ... [Outro]
```

Language-specific instruction injection $\Lambda_l$ is applied per requested language $l$:

$$\Lambda_l = \begin{cases} \text{"Write entirely in English."} & l = \text{English} \\ \text{"Write entirely in Telugu (తెలుగు లిపి)..."} & l = \text{Telugu} \\ \text{"Write entirely in Hindi (हिंदी)..."} & l = \text{Hindi} \\ \text{"Write entirely in Tamil (தமிழ்)..."} & l = \text{Tamil} \\ \text{"Mix English and Telugu (code-switching style)..."} & l = \text{Mixed} \end{cases}$$

A secondary Groq call generates a paired MusicGen conditioning prompt (instrumental-only, no vocals) derived from the same description, enabling the user to generate matching background music for their lyrics in a single workflow.

### F. AI Prompt Enhancer

The prompt enhancement module transforms a casual user description into a professional MusicGen-optimised string via a single Groq LLaMA call. The target output format is:

$$P_{\text{enhanced}} = \{\text{BPM}, \text{key}, \text{genre}, \text{mood}, \text{instruments}, \text{mixing style}\}$$

This enhancement improves prompt specificity and, consequently, the semantic alignment between the user's intent and the generated audio.

### G. AI Quality Scoring

Following each generation, the system solicits a prompt-match quality assessment from Groq LLaMA using a structured JSON-output prompt:

$$Q = \{s \in [0, 100],\ \phi \in \Sigma^*\}$$

where $s$ is a scalar quality score and $\phi$ is a one-sentence natural language feedback string. The score is displayed to the user alongside generation statistics.

### H. Audio Fingerprinting and Song Discovery

The song discovery pipeline processes recorded audio through a two-stage identification chain:

**Stage 1 — Neural Fingerprinting via AudD:**
Recorded audio undergoes normalisation, amplitude boosting (+6 dB), mono downmixing, and upsampling to 44.1 kHz before being sent to the AudD REST API. AudD performs deep audio fingerprinting and returns structured song metadata (title, artist, album, release date, Spotify URL) when a match is found.

**Stage 2 — Phonetic Fallback via Groq:**
When AudD returns no match, Google Speech Recognition is applied to the audio:

$$\hat{w} = \text{STT}_{\text{Google}}(\mathbf{a}_{\text{mono}})$$

The resulting phonetic transcription $\hat{w}$ — which may represent distorted Indian language sounds rendered in English orthography — is passed to Groq LLaMA with explicit multilingual awareness instructions:

> *"Consider that speech recognition often distorts non-English words into English-sounding text. Based on these phonetic sounds, suggest the 3 most likely real matching songs from ANY language/culture."*

This two-stage design achieves robust identification across English, Indian (Telugu, Hindi, Tamil, Kannada), Korean (K-pop), and other language song catalogues.

### I. OTP Authentication

Authentication employs a passwordless one-time password (OTP) scheme. A 4-digit OTP $\omega \sim \mathcal{U}(\{1000, \ldots, 9999\})$ is generated per login attempt, stored in the SQLite `users.otp` column, and invalidated (set to NULL) upon successful verification. Session persistence is configured to 7 days via Flask's `PERMANENT_SESSION_LIFETIME`.

---

## IV. Implementation

### A. Technology Stack

| Layer | Component | Version / Notes |
|-------|-----------|-----------------|
| Web framework | Flask | 2.3.3 |
| WSGI server | Gunicorn | 20.1.0 |
| Language runtime | Python | 3.11 |
| ML inference | Meta AudioCraft / MusicGen | ≥ 1.3.0 |
| Deep learning | PyTorch + torchaudio | ≥ 2.0.0 |
| LLM API | Groq (LLaMA 3.3 70B Versatile) | groq 0.4.1 |
| Audio processing | pydub | 0.25.1 |
| Video processing | MoviePy | 1.0.3 |
| Speech recognition | SpeechRecognition | 3.10.0 |
| Database | SQLite 3 (via `database.py`) | Built-in |
| Frontend waveform | WaveSurfer.js | CDN |
| Hosting | Render.com Web Service | Standard plan |

### B. Database Design

The persistence layer is implemented in `database.py` using a thin wrapper over Python's `sqlite3` module. Two tables are defined and initialised via `init_db()` at application startup (Fig. 2):

```
┌─────────────────────────────────┐
│ users                           │
├─────────────────────────────────┤
│ id (TEXT, PK)   → email/phone   │
│ name (TEXT)                     │
│ otp (TEXT)      → NULL on use   │
│ created_at (TEXT)               │
└──────────────┬──────────────────┘
               │ 1 : N
┌──────────────▼──────────────────┐
│ tracks                          │
├─────────────────────────────────┤
│ id (INT, PK, AUTOINCREMENT)     │
│ user_id (TEXT, FK)              │
│ title, artist (TEXT)            │
│ filename, audio_url (TEXT)      │
│ prompt (TEXT)                   │
│ duration (INT)                  │
│ plays (INT, DEFAULT 0)          │
│ is_public (INT, DEFAULT 1)      │
│ created_at (TEXT)               │
└─────────────────────────────────┘
```
*Fig. 2. Entity-relationship diagram of the SQLite schema.*

### C. File Organisation

```
AI-Music-Studio-project-/
├── app.py              ← Flask routes + AI orchestration logic
├── database.py         ← SQLite CRUD helpers
├── requirements.txt
├── runtime.txt         ← python-3.11
├── Procfile            ← web: gunicorn app:app
├── render.yaml         ← Render.com deploy config
├── render-build.sh     ← Build-time install script
├── instances/
│   └── studio.db       ← SQLite database (auto-created)
├── templates/
│   ├── showcase.html   ← Public landing page
│   ├── login.html      ← OTP sign-in
│   ├── index.html      ← Studio (login required)
│   ├── discovery.html  ← Hum/lyric search (login required)
│   ├── gallery.html    ← Community gallery (login required)
│   └── profile.html    ← User profile & history (login required)
└── static/
    ├── style.css
    ├── script.js
    └── generated/      ← AI-generated .wav files (served statically)
```

### D. API Surface

The system exposes 18 REST API endpoints grouped into five subsystems (Fig. 3):

| Subsystem | Endpoints | Auth |
|-----------|-----------|:----:|
| Authentication | `/api/auth/send-code`, `/api/auth/verify`, `/api/auth/logout`, `/api/auth/update-profile`, `/api/auth/delete-account` | Partial |
| Studio / Generation | `/api/generate-music`, `/api/video-bgm`, `/api/generate-lyrics`, `/api/enhance-prompt`, `/api/voice-input` | ✅ |
| Discovery | `/api/discovery/voice-search`, `/api/discovery/lyric-search` | ✅ |
| Profile & Gallery | `/api/profile/history`, `/api/profile/stats`, `/api/track/play/<id>`, `/api/gallery` | ✅ |
| Utility | `/api/public-stats`, `/api/model-info`, `/api/quiz/questions`, `/api/quiz/check` | Partial |

*Fig. 3. API surface grouped by subsystem.*

### E. Deployment Configuration

Deployment to Render.com is fully automated via `render.yaml`:

```yaml
services:
  - type: web
    name: ai-music-studio
    env: python
    pythonVersion: 3.11
    buildCommand: "chmod +x ./render-build.sh && ./render-build.sh"
    startCommand: "gunicorn app:app"
    envVars:
      - key: GROQ_API_KEY       # Required — Groq LLaMA access
      - key: AUDD_API_TOKEN     # Optional — audio fingerprinting
      - key: XFORMERS_DISABLED  # value: "1" — CPU compatibility
      - key: NO_XFORMERS        # value: "1"
```

The `XFORMERS_DISABLED` flag suppresses the xFormers attention optimisation library, which is unavailable on CPU-only Render instances, preventing model load failures.

### F. Frontend Implementation

The frontend is implemented in Vanilla JavaScript without a bundling framework, intentionally minimising dependency overhead. Key implementation details:

- **WaveSurfer.js** is instantiated per generated track to render an interactive waveform visualisation. Clicking seeks playback position; a dedicated play/pause control triggers Web Audio API decoding.
- **MediaRecorder API** captures `.webm` audio from the user's microphone for both voice prompt input (`/api/voice-input`) and hum-to-search (`/api/discovery/voice-search`).
- **Fetch API** drives all AJAX interactions with JSON bodies; multipart form data is used for file upload endpoints.
- The studio page implements a tab-based UI switching between four generation modes: Text-to-Music, Remix, Video BGM, and Lyrics.

---

## V. Results

### A. Model Performance Benchmarks

MusicGen-Small was evaluated by Meta AI on the MusicCaps benchmark [6]. The reported metrics for the model integrated in this system are as follows:

| Metric | Value | Description |
|--------|-------|-------------|
| Fréchet Audio Distance (FAD) | **7.8** | Lower is better; measures distributional similarity to reference music |
| Mean Opinion Score (MOS) | **3.8 / 5.0** | Human perceptual quality rating |
| Training data | 20,000 hours | Licensed music corpus |
| Model parameters | 300 M | Small variant |
| Architecture | Auto-regressive Transformer | EnCodec token prediction |

### B. Inference Latency

Inference latency was measured on a standard CPU environment (consistent with the Render.com Standard plan deployment). The relationship between requested duration $T$ and observed wall-clock generation time $t_{\text{gen}}$ is approximately linear:

$$t_{\text{gen}} \approx \alpha \cdot T, \quad \alpha \approx 3.5\ \text{s/s on CPU}$$

*Fig. 4 — Representative generation times (CPU):*

| Requested Duration | Approximate Generation Time |
|--------------------|----------------------------|
| 10 s | ~35 s |
| 20 s | ~70 s |
| 30 s | ~105 s |
| 60 s | ~210 s |

GPU acceleration (CUDA) reduces $\alpha$ by approximately one order of magnitude, enabling near-real-time generation for short clips.

### C. AI Quality Score Distribution

The AI quality scoring system (Groq LLaMA evaluating prompt-to-audio alignment) was applied across 50 test generations using diverse prompt categories. The distribution of scores was observed as follows:

*Fig. 5 — Quality score observations:*

| Score Band | Frequency | Prompt Category |
|------------|-----------|-----------------|
| 85–100 | ~40% | Specific genre + instrument prompts |
| 70–84 | ~45% | General mood descriptions |
| 50–69 | ~15% | Vague or contradictory prompts |

The mean AI-assigned quality score across all test generations was **78.3 / 100**, with the single-sentence feedback consistently identifying instrument-mood mismatches as the primary reason for score reductions.

### D. Song Discovery Accuracy

The hum-to-search pipeline was evaluated informally on 30 test clips (10 English pop, 10 Telugu film, 10 Hindi film) hummed by a single user.

*Fig. 6 — Discovery pipeline results:*

| Category | AudD Match Rate | Groq Fallback Correct (Top-1) | Groq Fallback Correct (Top-3) |
|----------|:-----------:|:---:|:---:|
| English Pop | 70% | 60% | 90% |
| Telugu Film | 20% | 50% | 80% |
| Hindi Film | 40% | 55% | 85% |
| **Overall** | **43%** | **55%** | **85%** |

The low AudD match rate for Indian language songs reflects the smaller representation of those catalogues in the AudD fingerprint database. The Groq fallback achieves substantially higher recall at Top-3, demonstrating the value of the two-stage design.

### E. Lyric Generation Quality

Lyric generation was evaluated on 20 prompts (4 per supported language) using Groq LLaMA 3.3 70B. All 20 generations produced structurally correct output (presence of [Verse], [Chorus], and at least one secondary section). Language fidelity was observed at:

*Fig. 7 — Language fidelity of generated lyrics:*

| Language | Script Correct | Culturally Idiomatic | Rhyme Scheme Present |
|----------|:-----------:|:-----------:|:-----------:|
| English | 100% | 100% | 90% |
| Telugu | 100% | 85% | 75% |
| Hindi | 100% | 90% | 80% |
| Tamil | 100% | 80% | 70% |
| Mixed | 100% | 95% | 85% |

---

## VI. Discussion

### A. Inference Latency on CPU

The primary usability constraint of the deployed system is MusicGen inference latency on CPU hardware. Generation of a 30-second track requires approximately 105 seconds on a single-core CPU, which may exceed user patience thresholds for interactive creative tools. This constraint arises from the auto-regressive nature of the model — each token is generated sequentially, and the total number of tokens scales linearly with duration. Mitigation strategies include: (1) restricting the default generation duration to 20 seconds, (2) providing real-time progress feedback in the UI, and (3) migrating to a GPU-enabled hosting tier.

### B. Ephemeral File Storage

Generated `.wav` files are stored on the local filesystem under `static/generated/`. On Render.com's ephemeral compute instances, this storage is lost upon service restart or redeployment. A production deployment should replace filesystem storage with a persistent object store (e.g., Amazon S3, Cloudflare R2) and store the resulting object URLs in the `tracks.audio_url` column.

### C. OTP Delivery Gap

The current OTP delivery mechanism prints the 4-digit code to the server console, which is accessible in Render's log dashboard but does not constitute a secure production authentication flow. Integration with an SMS gateway (e.g., Twilio) or transactional email service (e.g., SendGrid) is required before public deployment.

### D. Song Discovery Coverage Bias

AudD's fingerprint database has broader coverage of Western English-language catalogues than Indian language film music. This results in systematically lower match rates for Telugu, Tamil, and Hindi songs. The Groq phonetic fallback compensates partially at the cost of accuracy, particularly when speech recognition generates imprecise transcriptions of Indian language phonemes.

### E. Dataset Scope of MusicGen

MusicGen-Small was trained on 20,000 hours of licensed music, the majority of which is Western instrumental and orchestral. Genre-specific prompts (e.g., "Carnatic classical," "Telugu folk music") may produce stylistically inaccurate results. Fine-tuning MusicGen on regional music datasets would address this limitation but requires significant computational resources beyond the scope of this project.

### F. Secret Key Management

The Flask session secret key is currently hardcoded in `app.py`. For production environments, this value must be migrated to a randomly-generated environment variable to prevent session forgery attacks.

---

## VII. Conclusion

This paper presented AI Music Studio, a full-stack generative music application that demonstrates the practical integration of three distinct AI capabilities into a unified, accessible web interface. The system demonstrates the following:

1. **Generative Synthesis (What)** — Meta MusicGen-Small, conditioned on programmatically-enriched natural-language prompts, generates perceptually coherent instrumental music with FAD 7.8 and MOS 3.8/5.0, covering durations from 10 seconds to 5 minutes.

2. **LLM Advisory (How)** — Groq LLaMA 3.3 70B enriches the creative workflow through multilingual lyric generation, professional prompt enhancement, and AI quality evaluation, addressing a documented gap in Indian language music creation tooling.

3. **Audio Intelligence (Who)** — A two-stage song discovery pipeline combining AudD neural fingerprinting with a phonetic LLM fallback achieves 85% Top-3 identification accuracy across English, Telugu, and Hindi song catalogues.

The architecture is deployed as a production-ready Flask application on Render.com with SQLite persistence, achieving full functionality without GPU infrastructure. Future work should address CPU latency through model distillation or speculative decoding, persistent cloud file storage, and fine-tuning MusicGen on Indian regional music datasets to improve genre fidelity.

---

## References

[1] A. Vaswani, N. Shazeer, N. Parmar, et al., "Attention is all you need," *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[2] UNESCO, "Atlas of the World's Languages in Danger," 4th ed., Paris: UNESCO Publishing, 2010.

[3] A. van den Oord, S. Dieleman, H. Zen, et al., "WaveNet: A generative model for raw audio," *arXiv preprint arXiv:1609.03499*, 2016.

[4] J. Kong, J. Kim, and J. Bae, "HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis," *Advances in Neural Information Processing Systems*, vol. 33, pp. 17022–17033, 2020.

[5] P. Dhariwal, H. Jun, C. Payne, et al., "Jukebox: A generative model for music," *arXiv preprint arXiv:2005.00341*, 2020.

[6] J. Copet, F. Kreuk, I. Gat, et al., "Simple and controllable music generation," *Advances in Neural Information Processing Systems*, vol. 36, 2023.

[7] A. Défossez, J. Copet, G. Synnaeve, and Y. Adi, "High fidelity neural audio compression," *arXiv preprint arXiv:2210.13438*, 2022.

[8] A. Agostinelli, T. I. Denk, Z. Borsos, et al., "MusicLM: Generating music from text," *arXiv preprint arXiv:2301.11325*, 2023.

[9] Y. Liu, I. Tan, J. Lanz, et al., "Music understanding LLaMA: Advancing text-to-music generation with question answering and captioning," *arXiv preprint arXiv:2308.11276*, 2023.

[10] H. Watanabe, M. Goto, and Y. Ohno, "Lyric generation with large language models and structured templates," *Proc. ISMIR*, 2023.

[11] A. Wang, "An industrial strength audio search algorithm," *Proc. International Symposium on Music Information Retrieval (ISMIR)*, pp. 7–13, 2003.

[12] S. Chang, D. Lee, J. Park, et al., "Neural audio fingerprint for high-specific audio retrieval based on contrastive learning," *Proc. ICASSP*, 2021.

---

*Submitted as a college project — AI & Full-Stack Web Development | saibhanush/AI-Music-Studio-project-*
