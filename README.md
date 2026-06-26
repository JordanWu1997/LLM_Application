# LLM Application

- Application for Large Language Model (LLM) on local machine

## App List

- [X] Ollama client
  - [X] Model Management
  - [X] Chat/Generation
    - [X] w/ Texts
    - [X] w/ Images
    - [X] w/ Files (txt, csv, etc.)
- [X] Voice assistant
  - [X] STT Input
  - [X] TTS Output
    - [X] Integrate w/ `espeak` or TTS tools
    - [X] Kokoro
    - [X] Qwen-TTS

### Text

- [X] PDF summarizer
- [X] Git commit message generator
- [X] Logger/Journal analyzer
- [X] Markdown note keyword/tag generator
- [X] Translate bot w/ `translategemma`
  - [X] Auto detect input language w/ langdetect
- [X] Code reviewer
  - Here I just integrate ollama models into my dev workflow with [vim-ai](https://github.com/madox2/vim-ai)
    - My [Vim-Tmux workflow](https://github.com/JordanWu1997/Vim_Tmux_Config)
- [X] Python code function docstring and type hinting generator
- [X] Shell assistant
  - [X] Error fixer
- [ ] SQL assistant
- [ ] Note/Diary organizer
  - [ ] LLM Wiki

### Vision

- [X] Video narrator (also support USB camera input)
  - [X] Ollama
  - [X] vLLM for better performance
- [X] OCR w/ VLM w/ `glm-ocr`
- [X] Object detection integrated VLM classification
  - Check my repo: [CV_Application](https://github.com/JordanWu1997/CV_Application)
- [ ] Media tagger

## RAG List

- [ ] Knowledge Graph for [My Personal knowledge](https://github.com/JordanWu1997/Knowledge_Base)

## Agent List

- [X] Tool Use
  - [X] Weather
  - [ ] News
  - [ ] Web Search
- [X] Hermes Agent
  - [X] Personal Assistant
  - [X] Connect to Discord
  - [X] Connect to Google Workplace
- [ ] OpenClaw
  - [ ] Task Worker
- [ ] Agentic Coding
  - [ ] Planer
  - [ ] Code builder
  - [ ] Code tester/reviewer
