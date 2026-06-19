# Academic Homepage

This is the academic homepage for Haiyan Qin, featuring a modern pixel-art design with interactive elements.

## About

A static HTML website showcasing AI-assisted hardware design research, featuring:
- Pixel art aesthetic with modern animations
- Interactive visualizations and data displays
- Research projects and publications
- Conference presentations and talks

## Features

- 🎮 Pixel-art inspired design
- 📊 Interactive charts and animations
- 🎯 Modern responsive layout
- 🚀 Fast loading with CDN resources
- 📱 Mobile-friendly interface
- 🌍 Visitor map with optional global aggregation (JSONBin.io)

## Visitor Map (Global Sync)

The site includes an interactive world map showing visitor locations. To enable **global aggregation** (all visitors, not just your current session):

1. Create a free account at [jsonbin.io](https://jsonbin.io)
2. Create a new bin with initial content: `{ "visitors": [] }`
3. Copy the **Bin ID** and **X-Master-Key** (Access Key)
4. In this GitHub repo, go to **Settings → Secrets and variables → Actions** and add:
   - `JSONBIN_BIN_ID` — your bin ID
   - `JSONBIN_ACCESS_KEY` — your X-Master-Key
5. Push to `master` — the deploy workflow injects credentials automatically

For local testing, paste the same values into `assets/js/visitor-map-config.js`.

## Technology Stack

- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Animations**: Anime.js, p5.js, Matter.js, PIXI.js
- **Charts**: ECharts
- **Hosting**: GitHub Pages

## Contact

- **Email**: haiyanq@buaa.edu.cn
- **Google Scholar**: [Profile](https://scholar.google.com/citations?user=zzmYq9QAAAAJ&hl=en)
- **GitHub**: [qhy991](https://github.com/qhy991)

## License

This project is open source and available under the MIT License.