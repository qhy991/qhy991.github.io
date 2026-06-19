/**
 * Visitor Map — ECharts world map with geolocation
 * Works on static GitHub Pages; optional JSONBin.io for global aggregation.
 */
(function () {
    'use strict';

    const GEO_API = 'https://ipwho.is/';
    const WORLD_JSON = 'https://fastly.jsdelivr.net/npm/echarts@5.4.3/map/json/world.json';
    const ECHARTS_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js';
    const STORAGE_KEY = 'hq_visitor_session';
    const DEDUP_HOURS = 24;

    class VisitorMap {
        constructor(containerId) {
            this.container = document.getElementById(containerId);
            if (!this.container) return;

            this.chart = null;
            this.visitors = [];
            this.currentVisitor = null;
            this.stats = { total: 0, countries: 0, cities: 0 };

            this.init();
        }

        async init() {
            this.renderShell();
            await this.loadEcharts();
            await this.loadWorldMap();
            await this.loadVisitorData();
            const synced = await this.registerCurrentVisit();
            if (synced) await this.loadVisitorData();
            this.initChart();
            this.updateStats();
            this.renderSyncBadge(synced);
            window.addEventListener('resize', () => this.chart?.resize());
        }

        renderShell() {
            this.container.innerHTML = `
                <div class="visitor-map-wrap">
                    <div id="visitor-map-chart"></div>
                    <div class="visitor-map-sidebar">
                        <div class="vm-stat-card">
                            <div class="vm-stat-value" id="vm-stat-total">—</div>
                            <div class="vm-stat-label">Total Visits</div>
                        </div>
                        <div class="vm-stat-card">
                            <div class="vm-stat-value" id="vm-stat-countries">—</div>
                            <div class="vm-stat-label">Countries</div>
                        </div>
                        <div class="vm-stat-card">
                            <div class="vm-stat-value" id="vm-stat-cities">—</div>
                            <div class="vm-stat-label">Cities</div>
                        </div>
                        <div class="vm-you-card" id="vm-you-card" style="display:none">
                            <div class="vm-you-label">📍 You are here</div>
                            <div class="vm-you-location" id="vm-you-location"></div>
                        </div>
                        <div class="vm-recent" id="vm-recent-list"></div>
                        <div class="vm-sync-badge" id="vm-sync-badge"></div>
                    </div>
                </div>`;
            this.chartEl = document.getElementById('visitor-map-chart');
        }

        loadEcharts() {
            if (window.echarts) return Promise.resolve();
            return new Promise((resolve, reject) => {
                const s = document.createElement('script');
                s.src = ECHARTS_CDN;
                s.onload = resolve;
                s.onerror = reject;
                document.head.appendChild(s);
            });
        }

        async loadWorldMap() {
            if (echarts.getMap('world')) return;
            const res = await fetch(WORLD_JSON);
            const geoJson = await res.json();
            echarts.registerMap('world', geoJson);
        }

        getConfig() {
            return window.VISITOR_MAP_CONFIG || {};
        }

        isGlobalSyncEnabled() {
            const cfg = this.getConfig();
            return !!(cfg.jsonBinId && cfg.jsonBinAccessKey);
        }

        getAssetBase() {
            const link = document.querySelector('link[href*="fancy.css"]');
            if (link) {
                const href = link.getAttribute('href');
                const idx = href.indexOf('assets/');
                if (idx >= 0) return href.slice(0, idx);
            }
            return '/';
        }

        async loadVisitorData() {
            const sources = [];

            try {
                const res = await fetch(`${this.getAssetBase()}assets/data/visitors.json`);
                if (res.ok) {
                    const data = await res.json();
                    sources.push(...(data.visitors || []));
                }
            } catch (_) { /* ignore */ }

            // JSONBin aggregate
            const cfg = this.getConfig();
            if (cfg.jsonBinId && cfg.jsonBinAccessKey) {
                try {
                    const res = await fetch(`https://api.jsonbin.io/v3/b/${cfg.jsonBinId}/latest`, {
                        headers: { 'X-Master-Key': cfg.jsonBinAccessKey },
                    });
                    if (res.ok) {
                        const data = await res.json();
                        sources.push(...(data.record?.visitors || []));
                    }
                } catch (_) { /* ignore */ }
            }

            this.visitors = this.deduplicateVisits(sources);
        }

        deduplicateVisits(list) {
            const seen = new Set();
            return list.filter(v => {
                if (!v.lat || !v.lng) return false;
                const key = `${v.lat.toFixed(1)},${v.lng.toFixed(1)},${v.city || ''}`;
                if (seen.has(key)) return false;
                seen.add(key);
                return true;
            });
        }

        async fetchGeo() {
            try {
                const res = await fetch(GEO_API);
                if (!res.ok) return null;
                const data = await res.json();
                if (!data.success) return null;
                return {
                    lat: data.latitude,
                    lng: data.longitude,
                    city: data.city,
                    country: data.country,
                    countryCode: data.country_code,
                    ip: data.ip,
                    ts: Date.now(),
                };
            } catch (_) {
                return null;
            }
        }

        isSessionRecorded(geo) {
            try {
                const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
                if (stored.ip === geo.ip && Date.now() - stored.ts < DEDUP_HOURS * 3600000) {
                    return true;
                }
            } catch (_) { /* ignore */ }
            return false;
        }

        markSessionRecorded(geo) {
            localStorage.setItem(STORAGE_KEY, JSON.stringify({ ip: geo.ip, ts: Date.now() }));
        }

        async registerCurrentVisit() {
            const geo = await this.fetchGeo();
            if (!geo) return false;

            this.currentVisitor = geo;

            if (!this.isSessionRecorded(geo)) {
                this.markSessionRecorded(geo);
                const synced = await this.persistVisit(geo);
                if (!synced) {
                    this.visitors.push(geo);
                    this.visitors = this.deduplicateVisits(this.visitors);
                }
                return synced;
            }
            return false;
        }

        async persistVisit(geo) {
            const cfg = this.getConfig();
            if (!cfg.jsonBinId || !cfg.jsonBinAccessKey) return false;

            try {
                const res = await fetch(`https://api.jsonbin.io/v3/b/${cfg.jsonBinId}/latest`, {
                    headers: { 'X-Master-Key': cfg.jsonBinAccessKey },
                });
                let visitors = [];
                if (res.ok) {
                    const data = await res.json();
                    visitors = data.record?.visitors || [];
                } else if (res.status === 404) {
                    // Bin empty or new — start fresh
                    visitors = [];
                } else {
                    return false;
                }

                const entry = {
                    lat: geo.lat,
                    lng: geo.lng,
                    city: geo.city,
                    country: geo.country,
                    countryCode: geo.countryCode,
                    ts: geo.ts,
                };

                // Dedupe: same city within 24h
                const dayAgo = Date.now() - DEDUP_HOURS * 3600000;
                const dup = visitors.some(v =>
                    v.city === entry.city &&
                    v.countryCode === entry.countryCode &&
                    (v.ts || 0) > dayAgo
                );
                if (!dup) visitors.push(entry);
                if (visitors.length > 500) visitors = visitors.slice(-500);

                const putRes = await fetch(`https://api.jsonbin.io/v3/b/${cfg.jsonBinId}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Master-Key': cfg.jsonBinAccessKey,
                    },
                    body: JSON.stringify({ visitors }),
                });
                return putRes.ok;
            } catch (_) {
                return false;
            }
        }

        renderSyncBadge(syncedThisSession) {
            const el = document.getElementById('vm-sync-badge');
            const subtitle = document.getElementById('vm-subtitle-sync');
            if (!el) return;

            if (this.isGlobalSyncEnabled()) {
                el.className = 'vm-sync-badge vm-sync-on';
                el.textContent = syncedThisSession
                    ? '🌐 Global sync · visit recorded'
                    : '🌐 Global sync · active';
                if (subtitle) subtitle.textContent = 'Visitor locations aggregated via JSONBin.io';
            } else {
                el.className = 'vm-sync-badge vm-sync-off';
                el.textContent = '📍 Showing your location only';
                if (subtitle) subtitle.textContent = 'Add JSONBIN_BIN_ID secret to enable global aggregation';
            }
        }

        initChart() {
            this.chart = echarts.init(this.chartEl, 'dark');

            const scatterData = this.visitors.map(v => ({
                name: `${v.city || 'Unknown'}, ${v.country || ''}`,
                value: [v.lng, v.lat, 1],
            }));

            const currentData = this.currentVisitor
                ? [{ name: `You: ${this.currentVisitor.city}`, value: [this.currentVisitor.lng, this.currentVisitor.lat, 2] }]
                : [];

            const option = {
                backgroundColor: 'transparent',
                tooltip: {
                    trigger: 'item',
                    backgroundColor: 'rgba(15,23,42,0.95)',
                    borderColor: '#06b6d4',
                    textStyle: { color: '#f1f5f9', fontSize: 12 },
                    formatter: (p) => p.name,
                },
                geo: {
                    map: 'world',
                    roam: true,
                    zoom: 1.2,
                    center: [10, 20],
                    itemStyle: {
                        areaColor: '#1e293b',
                        borderColor: '#334155',
                        borderWidth: 0.5,
                    },
                    emphasis: {
                        itemStyle: { areaColor: '#334155' },
                        label: { show: false },
                    },
                    silent: true,
                },
                series: [
                    {
                        name: 'Visitors',
                        type: 'scatter',
                        coordinateSystem: 'geo',
                        data: scatterData,
                        symbolSize: (val) => 6 + val[2] * 2,
                        itemStyle: {
                            color: '#8b5cf6',
                            shadowBlur: 10,
                            shadowColor: 'rgba(139,92,246,0.6)',
                        },
                        emphasis: {
                            scale: 1.8,
                            itemStyle: { color: '#a78bfa' },
                        },
                    },
                    {
                        name: 'You',
                        type: 'effectScatter',
                        coordinateSystem: 'geo',
                        data: currentData,
                        symbolSize: 14,
                        showEffectOn: 'render',
                        rippleEffect: {
                            brushType: 'stroke',
                            scale: 3,
                            period: 4,
                        },
                        itemStyle: {
                            color: '#06b6d4',
                            shadowBlur: 20,
                            shadowColor: 'rgba(6,182,212,0.8)',
                        },
                        zlevel: 2,
                    },
                ],
            };

            this.chart.setOption(option);
        }

        updateStats() {
            const countries = new Set(this.visitors.map(v => v.countryCode || v.country).filter(Boolean));
            const cities = new Set(this.visitors.map(v => v.city).filter(Boolean));

            this.stats = {
                total: this.visitors.length,
                countries: countries.size,
                cities: cities.size,
            };

            document.getElementById('vm-stat-total').textContent = this.stats.total;
            document.getElementById('vm-stat-countries').textContent = this.stats.countries;
            document.getElementById('vm-stat-cities').textContent = this.stats.cities;

            if (this.currentVisitor) {
                const card = document.getElementById('vm-you-card');
                card.style.display = 'block';
                document.getElementById('vm-you-location').textContent =
                    `${this.currentVisitor.city}, ${this.currentVisitor.country}`;
            }

            this.renderRecentList();
        }

        renderRecentList() {
            const list = document.getElementById('vm-recent-list');
            const recent = [...this.visitors]
                .sort((a, b) => (b.ts || 0) - (a.ts || 0))
                .slice(0, 8);

            if (!recent.length) {
                list.innerHTML = '<div class="vm-recent-empty">Be the first visitor on the map!</div>';
                return;
            }

            list.innerHTML = '<div class="vm-recent-title">Recent Visitors</div>' +
                recent.map(v => `
                    <div class="vm-recent-item">
                        <span class="vm-flag">${this.countryFlag(v.countryCode)}</span>
                        <span class="vm-recent-loc">${v.city || 'Unknown'}, ${v.country || ''}</span>
                    </div>`).join('');
        }

        countryFlag(code) {
            if (!code || code.length !== 2) return '🌍';
            const pts = [...code.toUpperCase()].map(c => 0x1F1E6 + c.charCodeAt(0) - 65);
            return String.fromCodePoint(...pts);
        }
    }

    function injectSection() {
        if (document.getElementById('visitor-map-section')) return;

        const section = document.createElement('section');
        section.id = 'visitor-map-section';
        section.className = 'content-section visitor-map-section';
        section.innerHTML = `
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div class="text-center mb-12">
                    <h2 class="text-4xl font-bold mb-4">
                        <span class="text-cyan-400">Visitor</span>
                        <span class="text-purple-400">Map</span>
                    </h2>
                    <p class="text-lg text-slate-400">See where visitors are exploring from around the world</p>
                    <p class="text-sm text-slate-500 mt-2" id="vm-subtitle-sync"></p>
                </div>
                <div id="visitor-map-container"></div>
            </div>`;

        const footer = document.querySelector('footer');
        if (footer) {
            footer.parentNode.insertBefore(section, footer);
        } else {
            document.body.appendChild(section);
        }

        new VisitorMap('visitor-map-container');
    }

    document.addEventListener('DOMContentLoaded', injectSection);
})();
