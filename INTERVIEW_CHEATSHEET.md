# Interview cheatsheet

Run from the project root in PowerShell. Create or populate `.env` from `.env.example` for live providers. Keep keys local.

## 1. Prepare a new show

```powershell
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force data/new_show | Out-Null
Copy-Item 'C:/path/to/episode_pack/*.txt' data/new_show/
$config = Get-Content config.yaml -Raw
$config.Replace('parsed_dir: parsed', 'parsed_dir: parsed/new_show').Replace('output_dir: outputs', 'output_dir: outputs/new_show').Replace('raw_log_dir: outputs/llm_raw', 'raw_log_dir: outputs/new_show/llm_raw').Replace('cache_dir: .cache/llm', 'cache_dir: .cache/new_show') | Set-Content -Encoding utf8 data/new_show/config.yaml
```

Use one UTF-8 TXT per episode, with an episode header, timed scenes, and CAMERA, ACTION or speaker blocks. Separate work directories prevent a new show from replacing parsed episodes or reusing the wrong story bible.

## 2–4. Plan, run and resume

```powershell
$runArgs = @('--config','data/new_show/config.yaml','--input','data/new_show','--tag','new_show','--batch-size','5','--max-calls-gemini','16','--max-calls-groq','60')
.venv\Scripts\python.exe run.py @runArgs --dry-run
.venv\Scripts\python.exe run.py @runArgs
# After the provider resets its quota, resume with the same arguments.
.venv\Scripts\python.exe run.py @runArgs
```

## 5. Swap models

```powershell
.venv\Scripts\python.exe run.py @runArgs --batch-size 1 --candidate-provider groq --candidate-model openai/gpt-oss-120b --judge-provider groq --judge-model openai/gpt-oss-120b
```

For a fresh provider comparison use a new tag. Existing candidate files resume by tag. Edit `models.bible`, `models.trailers` and `models.editor` in the run's YAML to change the other stages.

## 6. Open saved results locally

```powershell
$env:KUKU_CONFIG = 'data/new_show/config.yaml'
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open the local URL and select **Open existing run → new_show**. For a run made with the default config, clear `KUKU_CONFIG` before launching.

## 7. Run offline evaluation

```powershell
.venv\Scripts\python.exe -m src.eval run --config data/new_show/config.yaml --tag new_show --gold data/new_show/gold.csv
```

Use the schema in `eval/README.md`. Set `evaluation.gold_episodes` in the run config to this show's labeled episodes, or clear the list to use episodes declared by the gold CSV.

## Troubleshooting

1. A Gemini 503 or 504 can be transient. Let the configured retry and backoff finish.
2. If large batches keep failing, rerun with `--batch-size 1`.
3. After a daily quota stop, wait for reset and resume with the same tag.
4. If the local call budget is exhausted, rerun with the intended per-provider caps.
5. For network blocked, check DNS and connectivity from a local terminal.
6. To test a provider once, use `.venv\Scripts\python.exe -m src.llm --ping --provider groq`.
7. Check `.env` key names locally when a provider reports missing credentials.
8. If no episodes parse, check UTF-8, the episode header, and scene timestamp lines.
9. If cuts are missing, check CAMERA headers and Start/End lines in the parser warnings.
10. For offline UI verification, choose Demo or select mock for an uploaded test pack.
