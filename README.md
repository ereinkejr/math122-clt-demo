# CLT Visualizer (Voilà)

An interactive Central Limit Theorem demo rendered with Voilà.
Choose a source distribution, adjust sample sizes, and watch the sampling distribution of the mean approach normality.

## Files
- `clt_demo.py` — simulation + plotting utilities
- `app.ipynb` — minimal notebook that launches the interactive UI
- `requirements.txt` — environment for Binder/Voilà

## Launch on Binder (replace `YOUR_USERNAME/YOUR_REPO`)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ereinkejr/math122-clt-voila/HEAD?urlpath=voila/render/app.ipynb)

## Embed in a webpage
```html
<iframe
   src="https://mybinder.org/v2/gh/ereinkejr/math122-clt-voila/HEAD?urlpath=voila/render/app.ipynb"
  width="100%"
  height="800"
  frameborder="0">
</iframe>
```

## Local run (optional)
```bash
pip install -r requirements.txt
voila app.ipynb
```
This will serve the app at http://localhost:8866 by default.
[![Launch CLT Visualizer on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ereinkejr/math122-clt-voila/HEAD?urlpath=voila/render/app.ipynb)
