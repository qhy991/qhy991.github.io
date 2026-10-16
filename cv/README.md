# Public CV

`content.json` owns the Chinese and English resume text. `portrait.jpg` is the photo supplied by Haiyan Qin. `build.py` generates `zh/CV.pdf` and `en/CV.pdf`; website and GitHub download links point to these stable paths.

Edit both language entries and the `updated` date in `content.json`, then rebuild:

```bash
python3 -m pip install -r cv/requirements.txt
python3 cv/build.py
```

The renderer uses Arial Unicode and STHeiti Medium from macOS by default. On another system, supply Chinese TrueType fonts with regular and bold faces:

```bash
python3 cv/build.py --font-zh /path/to/regular.ttf --font-zh-bold /path/to/bold.ttf
```

Fonts are embedded in the PDFs. Each version must remain one A4 page. The renderer stops if text exceeds the page; shorten the text before reducing readability. Render and inspect both PDFs after each content or layout change, and verify the download links before publishing.

The initial text comes from Haiyan Qin's supplied resume and his ApxInf core-developer statement. ApxInf's project description links to its [owning repository](https://github.com/infinigence/ApxInf). EngineTailor's title and accepted status follow the [DAI 2026 official list](https://adai.ai/dai/2026/accepted-papers.html). The Qwen3-4B result retains its same-engine B200 and concurrency-16 scope, as described in the [public technical note](https://qhy991.github.io/2026/08/09/enginetailor-minimal-inference-engines/). The GLM-5.3 result retains the user-supplied 630 tokens/s statement without inferred test conditions.
