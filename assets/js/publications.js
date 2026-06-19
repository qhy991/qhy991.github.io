/**
 * Render publications from assets/data/publications.json
 */
(function () {
    'use strict';

    function getAssetBase() {
        const link = document.querySelector('link[href*="fancy.css"]');
        if (link) {
            const href = link.getAttribute('href');
            const idx = href.indexOf('assets/');
            if (idx >= 0) return href.slice(0, idx);
        }
        return '/';
    }

    function badgeClass(type) {
        if (type === 'journal') return 'badge-journal';
        return 'badge-conference';
    }

    function renderAuthors(authors) {
        return authors.map(a =>
            `<span class="author-tag${a.highlight ? ' highlight' : ''}">${a.name}</span>`
        ).join('\n');
    }

    function renderKeywords(keywords) {
        const styles = [
            'bg-cyan-500/20 text-cyan-400',
            'bg-purple-500/20 text-purple-400',
            'bg-green-500/20 text-green-400',
            'bg-yellow-500/20 text-yellow-400',
        ];
        return keywords.map((kw, i) =>
            `<span class="px-2 py-1 ${styles[i % styles.length]} rounded text-xs">${kw}</span>`
        ).join('\n');
    }

    function renderLinks(links) {
        if (!links.length) return '';
        return links.map(l =>
            `<a href="${l.url}" class="download-btn" target="_blank" rel="noopener">${l.label}</a>`
        ).join('\n');
    }

    function renderPublication(pub) {
        const badges = pub.badges.map((b, i) =>
            `<span class="achievement-badge ${i === 0 ? badgeClass(pub.type) : 'badge-accepted'}">${b}</span>`
        ).join('\n');

        return `
        <div class="timeline-item">
            <div class="publication-card">
                <div class="flex flex-wrap items-center mb-4">
                    ${badges}
                    <span class="achievement-badge badge-accepted">${pub.year}</span>
                </div>
                <h3 class="text-xl font-bold mb-3 text-cyan-400">${pub.title}</h3>
                <p class="text-sm text-purple-400 mb-3 mono-font">${pub.venue}</p>
                <div class="mb-4">${renderAuthors(pub.authors)}</div>
                <p class="text-slate-400 mb-4 leading-relaxed">${pub.abstract}</p>
                <div class="mb-4">
                    <h4 class="text-sm font-semibold text-purple-400 mb-2">Keywords</h4>
                    <div class="flex flex-wrap gap-2">${renderKeywords(pub.keywords)}</div>
                </div>
                <div class="citation-stats">
                    <div class="citation-stat">
                        <div class="citation-stat-value">${pub.citations}</div>
                        <div class="citation-stat-label">Citations</div>
                    </div>
                    <div class="citation-stat">
                        <div class="citation-stat-value">${pub.year}</div>
                        <div class="citation-stat-label">Year</div>
                    </div>
                </div>
                ${pub.links.length ? `<div class="mt-4 flex flex-wrap gap-2">${renderLinks(pub.links)}</div>` : ''}
            </div>
        </div>`;
    }

    function updateMetrics(metrics) {
        const map = {
            papers: metrics.papers,
            citations: metrics.citations,
            hIndex: metrics.hIndex,
            topVenues: metrics.topVenues,
        };
        document.querySelectorAll('[data-metric]').forEach(el => {
            const key = el.dataset.metric;
            if (map[key] !== undefined) el.textContent = map[key];
        });
    }

    async function init() {
        const container = document.getElementById('publications-timeline');
        if (!container) return;

        try {
            const res = await fetch(`${getAssetBase()}assets/data/publications.json`);
            const data = await res.json();
            updateMetrics(data.metrics);
            container.innerHTML = data.publications.map(renderPublication).join('');
            document.dispatchEvent(new CustomEvent('publications-rendered'));
        } catch (e) {
            container.innerHTML = '<p class="text-slate-400 text-center">Failed to load publications.</p>';
            console.error(e);
        }
    }

    document.addEventListener('DOMContentLoaded', init);
})();
