# jeongseoko.github.io

Personal research site of Jungsuk Oh (오정석), built with Jekyll and served by GitHub Pages from the `homepage` branch. The 2022 blog is kept on the `blog-2022` branch.

## Editing

| What | Where |
|---|---|
| Name, position, links, education | `_data/profile.yml` |
| Research goal and the three research directions (`research_goal`, `goal`, `pillars`) | `_data/profile.yml` |
| Research illustrations and the convergence diagram | `_includes/illus/*.svg`, `_includes/convergence.html` |
| Publications (shared by both languages) | `_data/publications.yml` |
| Projects (shared by both languages) | `_data/projects.yml` |
| Menu and button labels | `_data/i18n.yml` |
| Bio | `index.md` (English), `ko/index.md` (Korean) |
| LSC research page | `research/lsc.md`, `ko/research/lsc.md`, table numbers in `_data/lsc_results.yml` |
| SPEED research page | `research/speed.md`, `ko/research/speed.md`, table numbers in `_data/speed_results.yml` |
| Translation inference page (MFDS) | `research/translation-inference.md`, `ko/research/translation-inference.md` |
| Page figures | `_includes/figures/*.svg` |
| CV | `cv/cv.tex`, then run `script/build-cv` (needs XeLaTeX and the Inter font) to update `cv/JungsukOh_CV.pdf` |
| Design | `assets/css/main.css` |

To show a profile photo, put it in `assets/img/` and set `photo:` in `_data/profile.yml`.

A new English page and its Korean version must share the same `ref` in their front matter; the EN/KO switch links pages by `ref`.

## Local preview and checks

```bash
bundle install
bundle exec jekyll serve      # http://127.0.0.1:4000
script/test                   # build + content tests + internal link check
```
