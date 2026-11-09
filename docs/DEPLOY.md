# Deploying NeuroBridge

Two public addresses, both free and both served from this GitHub repository.

| What | Where | Host |
|---|---|---|
| Research platform (Streamlit app) | https://neurobridge.streamlit.app | Streamlit Community Cloud |
| Website (scroll journey) | https://stefaniazoibb.github.io/neurobridge/ | GitHub Pages |

## Research platform: Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with the GitHub account `stefaniazoibb`.
2. Choose **Create app**, then **Deploy a public app from GitHub**.
3. Fill in the form:
   - **Repository:** `stefaniazoibb/neurobridge`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL:** `neurobridge`. The website and README link to `neurobridge.streamlit.app`. If you pick another name, update those links.
4. Under **Advanced settings**, set **Python version** to `3.11`. Leave the secrets empty (see below).
5. Choose **Deploy**.

Installation from `requirements.txt` takes a few minutes. After that the app redeploys on every push to `main`.

**On the first visit** the app downloads about 170 MB of public data: TCGA, CGGA, GEO and 10x Visium. The first investigation therefore takes a few minutes. Later runs reuse the cache until the app goes to sleep. Community Cloud puts idle apps to sleep, and anyone visiting can wake it.

**Claude assistance stays off on the public app.** If you added `ANTHROPIC_API_KEY` to the app's secrets, any visitor could spend your API credit. Without a key, the platform runs fully deterministically. Every statistic is identical, because the language model never produces numbers.

**PDF export** needs a Chrome binary to render figures, which Community Cloud does not provide. The PDF is still produced, without figure images. The Markdown export and all figures inside the app are unaffected.

## Website: GitHub Pages

1. Open the repository on GitHub, then **Settings → Pages**.
2. Under **Build and deployment**, set the source to **Deploy from a branch**, the branch to `main`, and the folder to `/site`.
   - If the folder menu offers only `/ (root)` and `/docs`, use a GitHub Actions workflow that uploads `site/` with `actions/upload-pages-artifact` and `actions/deploy-pages`.
3. Save. The site appears at https://stefaniazoibb.github.io/neurobridge/ within a minute or two.

## Repository "About" box

In the repository's About settings (the gear icon), set **Website** to `https://neurobridge.streamlit.app`. The link then appears on the repository page.
