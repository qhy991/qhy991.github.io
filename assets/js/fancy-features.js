/**
 * Fancy Features — Command Palette, Scroll Progress, Tilt Cards, Counters, etc.
 */

(function () {
    'use strict';

    const PAGES = [
        { label: 'Home', desc: 'Homepage & terminal', url: 'index.html', icon: '🏠', keywords: 'home 首页' },
        { label: 'About', desc: 'About me & timeline', url: 'about.html', icon: '👤', keywords: 'about 关于' },
        { label: 'Blog', desc: 'Technical blog posts', url: 'blog.html', icon: '📝', keywords: 'blog 博客' },
        { label: 'Portfolio', desc: 'Projects & demos', url: 'portfolio.html', icon: '💼', keywords: 'portfolio 项目' },
        { label: 'Publications', desc: 'Papers & citations', url: 'publications.html', icon: '📚', keywords: 'publications 论文' },
    ];

    const ACTIONS = [
        { label: 'View Achievements', desc: 'Open achievement panel', icon: '🏆', action: () => window.achievementSystem?.showAchievementPanel(), keywords: 'achievements 成就' },
        { label: 'Konami Code Hint', desc: 'Easter egg hint', icon: '🎮', action: () => alert('Try: ↑ ↑ ↓ ↓ ← → ← → B A'), keywords: 'konami easter egg 彩蛋' },
        { label: 'Copy Email', desc: 'haiyanq@buaa.edu.cn', icon: '📧', action: () => copyText('haiyanq@buaa.edu.cn'), keywords: 'email contact 联系' },
        { label: 'GitHub Profile', desc: 'github.com/qhy991', icon: '💻', action: () => window.open('https://github.com/qhy991', '_blank'), keywords: 'github' },
        { label: 'Google Scholar', desc: 'Academic profile', icon: '🎓', action: () => window.open('https://scholar.google.com/citations?user=zzmYq9QAAAAJ&hl=en', '_blank'), keywords: 'scholar 学术' },
        { label: 'Scroll to Top', desc: 'Jump to page top', icon: '⬆️', action: () => window.scrollTo({ top: 0, behavior: 'smooth' }), keywords: 'top scroll 顶部' },
    ];

    function copyText(text) {
        navigator.clipboard.writeText(text).then(() => showToast('Copied to clipboard!'));
    }

    function showToast(msg) {
        const t = document.createElement('div');
        t.textContent = msg;
        t.style.cssText = 'position:fixed;top:80px;right:20px;background:rgba(16,185,129,0.9);color:#fff;padding:10px 18px;border-radius:8px;z-index:10001;font-size:14px;font-family:JetBrains Mono,monospace;animation:fadeIn 0.3s';
        document.body.appendChild(t);
        setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity 0.3s'; setTimeout(() => t.remove(), 300); }, 2000);
    }

    function resolveUrl(path) {
        const base = document.querySelector('nav a[href*="index"]')?.getAttribute('href')?.replace(/index\.html$/, '') || '';
        if (path.startsWith('http') || path.startsWith('/')) return path;
        const depth = (window.location.pathname.match(/\//g) || []).length;
        const prefix = window.location.pathname.includes('_posts') ? '../'.repeat(2) : '';
        return prefix + path;
    }

    // ── Scroll Progress ──
    function initScrollProgress() {
        const bar = document.createElement('div');
        bar.id = 'scroll-progress';
        document.body.prepend(bar);

        window.addEventListener('scroll', () => {
            const scrollTop = window.scrollY;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            bar.style.width = docHeight > 0 ? `${(scrollTop / docHeight) * 100}%` : '0%';
        }, { passive: true });
    }

    // ── Mouse Spotlight ──
    function initMouseSpotlight() {
        const spot = document.createElement('div');
        spot.id = 'mouse-spotlight';
        document.body.prepend(spot);

        let mx = 0, my = 0, cx = 0, cy = 0;
        document.addEventListener('mousemove', (e) => { mx = e.clientX; my = e.clientY; }, { passive: true });

        function animate() {
            cx += (mx - cx) * 0.08;
            cy += (my - cy) * 0.08;
            spot.style.left = cx + 'px';
            spot.style.top = cy + 'px';
            requestAnimationFrame(animate);
        }
        animate();
    }

    // ── Back to Top ──
    function initBackToTop() {
        const btn = document.createElement('button');
        btn.id = 'back-to-top';
        btn.innerHTML = '<svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18"/></svg>';
        btn.title = 'Back to top';
        btn.setAttribute('aria-label', 'Back to top');
        document.body.appendChild(btn);

        window.addEventListener('scroll', () => {
            btn.classList.toggle('visible', window.scrollY > 400);
        }, { passive: true });

        btn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
    }

    // ── Command Palette ──
    function initCommandPalette() {
        const overlay = document.createElement('div');
        overlay.id = 'command-palette-overlay';
        overlay.innerHTML = `
            <div id="command-palette" role="dialog" aria-label="Command palette">
                <div class="cp-search-wrap">
                    <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
                    <input id="cp-input" type="text" placeholder="Search pages, actions..." autocomplete="off" spellcheck="false">
                    <span class="cp-hint">esc</span>
                </div>
                <div id="cp-results"></div>
                <div class="cp-footer">
                    <span>↑↓ navigate</span>
                    <span>↵ select</span>
                    <span>esc close</span>
                </div>
            </div>`;
        document.body.appendChild(overlay);

        const input = overlay.querySelector('#cp-input');
        const results = overlay.querySelector('#cp-results');
        let activeIndex = 0;
        let filtered = [];

        function getAllItems() {
            return [
                ...PAGES.map(p => ({ ...p, type: 'page' })),
                ...ACTIONS.map(a => ({ ...a, type: 'action' })),
            ];
        }

        function filterItems(query) {
            const q = query.toLowerCase().trim();
            if (!q) return getAllItems();
            return getAllItems().filter(item =>
                item.label.toLowerCase().includes(q) ||
                (item.desc && item.desc.toLowerCase().includes(q)) ||
                (item.keywords && item.keywords.toLowerCase().includes(q))
            );
        }

        function renderItems(items) {
            filtered = items;
            activeIndex = 0;
            results.innerHTML = items.map((item, i) => `
                <div class="cp-item${i === 0 ? ' active' : ''}" data-index="${i}">
                    <span class="cp-item-icon">${item.icon}</span>
                    <div class="cp-item-text">
                        <div class="cp-item-label">${item.label}</div>
                        <div class="cp-item-desc">${item.desc || ''}</div>
                    </div>
                    ${item.type === 'page' ? '<span class="cp-item-kbd">↵</span>' : ''}
                </div>`).join('');

            results.querySelectorAll('.cp-item').forEach(el => {
                el.addEventListener('click', () => selectItem(+el.dataset.index));
                el.addEventListener('mouseenter', () => {
                    activeIndex = +el.dataset.index;
                    updateActive();
                });
            });
        }

        function updateActive() {
            results.querySelectorAll('.cp-item').forEach((el, i) => {
                el.classList.toggle('active', i === activeIndex);
            });
            const active = results.querySelector('.cp-item.active');
            if (active) active.scrollIntoView({ block: 'nearest' });
        }

        function selectItem(index) {
            const item = filtered[index];
            if (!item) return;
            close();
            if (item.type === 'page') {
                window.location.href = resolveUrl(item.url);
            } else if (item.action) {
                item.action();
            }
        }

        function open() {
            overlay.classList.add('open');
            input.value = '';
            renderItems(getAllItems());
            setTimeout(() => input.focus(), 50);
        }

        function close() {
            overlay.classList.remove('open');
        }

        input.addEventListener('input', () => renderItems(filterItems(input.value)));
        input.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowDown') { e.preventDefault(); activeIndex = Math.min(activeIndex + 1, filtered.length - 1); updateActive(); }
            else if (e.key === 'ArrowUp') { e.preventDefault(); activeIndex = Math.max(activeIndex - 1, 0); updateActive(); }
            else if (e.key === 'Enter') { e.preventDefault(); selectItem(activeIndex); }
            else if (e.key === 'Escape') { e.preventDefault(); close(); }
        });

        overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

        document.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                overlay.classList.contains('open') ? close() : open();
            }
            if (e.key === 'Escape' && overlay.classList.contains('open')) close();
        });

        // Nav hint button
        const nav = document.querySelector('nav .flex.items-baseline') || document.querySelector('nav .hidden.md\\:block > div');
        if (nav) {
            const hint = document.createElement('button');
            hint.className = 'nav-cmd-hint';
            hint.innerHTML = navigator.platform.includes('Mac') ? '⌘K' : 'Ctrl+K';
            hint.title = 'Command Palette';
            hint.addEventListener('click', open);
            nav.appendChild(hint);
        }

        window.openCommandPalette = open;
    }

    // ── 3D Tilt Cards ──
    function initTiltCards() {
        document.querySelectorAll('.skill-card, .stats-card, .blog-card, .achievement-badge').forEach(card => {
            if (card.dataset.tiltInit) return;
            card.dataset.tiltInit = '1';
            card.classList.add('tilt-card');
            card.style.position = card.style.position || 'relative';

            const shine = document.createElement('div');
            shine.className = 'tilt-shine';
            card.appendChild(shine);

            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = (e.clientX - rect.left) / rect.width - 0.5;
                const y = (e.clientY - rect.top) / rect.height - 0.5;
                card.style.transform = `perspective(600px) rotateY(${x * 12}deg) rotateX(${-y * 12}deg) scale(1.02)`;
            });

            card.addEventListener('mouseleave', () => {
                card.style.transform = '';
            });
        });
    }

    window.reinitFancyCards = initTiltCards;

    // ── Animated Counters ──
    function initCounters() {
        const counters = document.querySelectorAll('[data-count]');
        if (!counters.length) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (!entry.isIntersecting) return;
                const el = entry.target;
                const target = parseFloat(el.dataset.count);
                const suffix = el.dataset.suffix || '';
                const prefix = el.dataset.prefix || '';
                const duration = 1500;
                const start = performance.now();

                function tick(now) {
                    const progress = Math.min((now - start) / duration, 1);
                    const eased = 1 - Math.pow(1 - progress, 3);
                    const val = Math.round(target * eased);
                    el.textContent = prefix + val + suffix;
                    if (progress < 1) requestAnimationFrame(tick);
                }
                requestAnimationFrame(tick);
                observer.unobserve(el);
            });
        }, { threshold: 0.5 });

        counters.forEach(c => observer.observe(c));
    }

    // ── Blog Post Enhancements ──
    function initBlogEnhancements() {
        const article = document.querySelector('.blog-content, article.prose, main article');
        if (!article) return;

        // Reading progress (left bar)
        const rp = document.createElement('div');
        rp.id = 'reading-progress';
        document.body.appendChild(rp);

        window.addEventListener('scroll', () => {
            const rect = article.getBoundingClientRect();
            const total = article.offsetHeight;
            const scrolled = Math.max(0, -rect.top);
            const pct = Math.min(100, (scrolled / total) * 100);
            rp.style.height = pct + '%';
        }, { passive: true });

        // Table of contents
        const headings = article.querySelectorAll('h2, h3');
        if (headings.length >= 2) {
            const toc = document.createElement('nav');
            toc.id = 'toc-sidebar';
            toc.innerHTML = '<h4>Contents</h4>';
            const list = document.createElement('div');

            headings.forEach((h, i) => {
                const id = h.id || `heading-${i}`;
                h.id = id;
                const a = document.createElement('a');
                a.href = `#${id}`;
                a.textContent = h.textContent;
                a.className = h.tagName === 'H3' ? 'toc-h3' : '';
                a.addEventListener('click', (e) => {
                    e.preventDefault();
                    h.scrollIntoView({ behavior: 'smooth', block: 'start' });
                });
                list.appendChild(a);
            });

            toc.appendChild(list);
            document.body.appendChild(toc);

            const tocLinks = toc.querySelectorAll('a');
            const tocObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        tocLinks.forEach(l => l.classList.remove('active'));
                        const link = toc.querySelector(`a[href="#${entry.target.id}"]`);
                        if (link) link.classList.add('active');
                    }
                });
            }, { rootMargin: '-80px 0px -60% 0px' });

            headings.forEach(h => tocObserver.observe(h));
        }

        // Copy code buttons
        article.querySelectorAll('pre').forEach(pre => {
            const btn = document.createElement('button');
            btn.className = 'copy-code-btn';
            btn.textContent = 'Copy';
            btn.addEventListener('click', () => {
                const code = pre.querySelector('code')?.textContent || pre.textContent;
                navigator.clipboard.writeText(code.trim()).then(() => {
                    btn.textContent = 'Copied!';
                    btn.classList.add('copied');
                    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
                });
            });
            pre.appendChild(btn);
        });
    }

    // ── Init ──
    document.addEventListener('DOMContentLoaded', () => {
        initScrollProgress();
        initMouseSpotlight();
        initBackToTop();
        initCommandPalette();
        initTiltCards();
        initCounters();
        initBlogEnhancements();
    });
})();
