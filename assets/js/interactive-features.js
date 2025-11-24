/**
 * Konami Code Easter Egg & Enhanced Typewriter Effect
 * Fun interactive features for the website
 */

// ============================================
// KONAMI CODE EASTER EGG
// ============================================
class KonamiCode {
    constructor() {
        this.pattern = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown',
            'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight',
            'b', 'a'];
        this.current = 0;
        this.activated = false;

        this.init();
    }

    init() {
        document.addEventListener('keydown', (e) => this.handleKeyPress(e));
    }

    handleKeyPress(e) {
        const key = e.key;

        if (key === this.pattern[this.current]) {
            this.current++;

            if (this.current === this.pattern.length) {
                this.activate();
                this.current = 0;
            }
        } else {
            this.current = 0;
        }
    }

    activate() {
        if (this.activated) {
            this.deactivate();
            return;
        }

        this.activated = true;

        // 触发成就
        if (window.achievementSystem) {
            window.achievementSystem.trackEasterEgg();
        }

        // 播放音效
        this.playSound();

        // 显示Matrix效果
        this.showMatrixRain();

        // 显示提示
        this.showMessage();
    }

    playSound() {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const notes = [523.25, 659.25, 783.99, 1046.50, 1318.51]; // C, E, G, C, E

        notes.forEach((freq, index) => {
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);

            oscillator.frequency.value = freq;
            oscillator.type = 'square';
            gainNode.gain.value = 0.1;

            const startTime = audioContext.currentTime + (index * 0.1);
            oscillator.start(startTime);
            oscillator.stop(startTime + 0.1);
        });
    }

    showMatrixRain() {
        const canvas = document.createElement('canvas');
        canvas.id = 'matrix-rain';
        canvas.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 9998;
            pointer-events: none;
        `;
        document.body.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%^&*(){}[]<>/\\|~`';
        const fontSize = 14;
        const columns = canvas.width / fontSize;
        const drops = Array(Math.floor(columns)).fill(1);

        const draw = () => {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = '#0F0';
            ctx.font = fontSize + 'px monospace';

            for (let i = 0; i < drops.length; i++) {
                const text = chars[Math.floor(Math.random() * chars.length)];
                ctx.fillText(text, i * fontSize, drops[i] * fontSize);

                if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                    drops[i] = 0;
                }
                drops[i]++;
            }
        };

        const interval = setInterval(draw, 33);

        // 30秒后自动关闭
        setTimeout(() => {
            clearInterval(interval);
            canvas.remove();
            this.activated = false;
        }, 30000);
    }

    showMessage() {
        const message = document.createElement('div');
        message.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 255, 0, 0.1);
            border: 2px solid #0F0;
            padding: 30px 50px;
            border-radius: 10px;
            z-index: 9999;
            font-family: 'JetBrains Mono', monospace;
            color: #0F0;
            text-align: center;
            box-shadow: 0 0 50px rgba(0, 255, 0, 0.5);
            animation: glitch 0.3s infinite;
        `;

        message.innerHTML = `
            <div style="font-size: 24px; font-weight: bold; margin-bottom: 10px;">
                🎮 KONAMI CODE ACTIVATED! 🎮
            </div>
            <div style="font-size: 14px; opacity: 0.8;">
                Welcome to the Matrix, Neo...
            </div>
        `;

        document.body.appendChild(message);

        setTimeout(() => {
            message.style.opacity = '0';
            message.style.transition = 'opacity 1s';
            setTimeout(() => message.remove(), 1000);
        }, 3000);
    }

    deactivate() {
        this.activated = false;
        const canvas = document.getElementById('matrix-rain');
        if (canvas) canvas.remove();
    }
}

// ============================================
// ENHANCED TYPEWRITER EFFECT
// ============================================
class EnhancedTypewriter {
    constructor(element, phrases, options = {}) {
        this.element = element;
        this.phrases = phrases;
        this.currentPhraseIndex = 0;
        this.currentText = '';
        this.isDeleting = false;
        this.typeSpeed = options.typeSpeed || 100;
        this.deleteSpeed = options.deleteSpeed || 50;
        this.pauseTime = options.pauseTime || 2000;

        this.init();
    }

    init() {
        this.type();
    }

    type() {
        const currentPhrase = this.phrases[this.currentPhraseIndex];

        if (this.isDeleting) {
            this.currentText = currentPhrase.substring(0, this.currentText.length - 1);
        } else {
            this.currentText = currentPhrase.substring(0, this.currentText.length + 1);
        }

        this.element.textContent = this.currentText;

        let delta = this.isDeleting ? this.deleteSpeed : this.typeSpeed;

        if (!this.isDeleting && this.currentText === currentPhrase) {
            delta = this.pauseTime;
            this.isDeleting = true;
        } else if (this.isDeleting && this.currentText === '') {
            this.isDeleting = false;
            this.currentPhraseIndex = (this.currentPhraseIndex + 1) % this.phrases.length;
            delta = 500;
        }

        setTimeout(() => this.type(), delta);
    }
}

// ============================================
// GITHUB ACTIVITY FEED
// ============================================
class GitHubActivityFeed {
    constructor(username, containerId) {
        this.username = username;
        this.container = document.getElementById(containerId);
        this.apiUrl = `https://api.github.com/users/${username}/events/public`;
        this.cacheKey = `github_activity_${username}`;
        this.cacheTime = 10 * 60 * 1000; // 10 minutes

        this.init();
    }

    async init() {
        const cachedData = this.getCache();

        if (cachedData) {
            this.render(cachedData);
        } else {
            await this.fetchActivity();
        }
    }

    getCache() {
        const cached = localStorage.getItem(this.cacheKey);
        if (!cached) return null;

        const { data, timestamp } = JSON.parse(cached);
        const now = Date.now();

        if (now - timestamp > this.cacheTime) {
            localStorage.removeItem(this.cacheKey);
            return null;
        }

        return data;
    }

    setCache(data) {
        const cacheData = {
            data,
            timestamp: Date.now()
        };
        localStorage.setItem(this.cacheKey, JSON.stringify(cacheData));
    }

    async fetchActivity() {
        try {
            const response = await fetch(this.apiUrl);
            const data = await response.json();

            this.setCache(data);
            this.render(data);
        } catch (error) {
            console.error('Failed to fetch GitHub activity:', error);
            this.renderError();
        }
    }

    render(events) {
        if (!this.container) return;

        const recentEvents = events.slice(0, 5);

        this.container.innerHTML = `
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 12px; padding: 24px;">
                <h3 style="color: #06b6d4; font-size: 20px; font-weight: 700; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                    <svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                    Recent GitHub Activity
                </h3>
                <div style="space-y: 12px;">
                    ${recentEvents.map(event => this.renderEvent(event)).join('')}
                </div>
                <a href="https://github.com/${this.username}" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; margin-top: 16px; color: #06b6d4; font-size: 14px; text-decoration: none; hover:text-decoration: underline;">
                    View all activity on GitHub →
                </a>
            </div>
        `;
    }

    renderEvent(event) {
        const icon = this.getEventIcon(event.type);
        const description = this.getEventDescription(event);
        const time = this.getRelativeTime(event.created_at);

        return `
            <div style="padding: 12px; background: rgba(15, 23, 42, 0.5); border-radius: 8px; border-left: 3px solid #8b5cf6;">
                <div style="display: flex; align-items: start; gap: 12px;">
                    <div style="font-size: 20px;">${icon}</div>
                    <div style="flex: 1;">
                        <div style="color: #f1f5f9; font-size: 14px; margin-bottom: 4px;">
                            ${description}
                        </div>
                        <div style="color: #94a3b8; font-size: 12px;">
                            ${time}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getEventIcon(type) {
        const icons = {
            'PushEvent': '📝',
            'CreateEvent': '✨',
            'DeleteEvent': '🗑️',
            'ForkEvent': '🍴',
            'IssuesEvent': '🐛',
            'PullRequestEvent': '🔀',
            'WatchEvent': '⭐',
            'ReleaseEvent': '🚀'
        };
        return icons[type] || '📌';
    }

    getEventDescription(event) {
        const repo = event.repo.name;

        switch (event.type) {
            case 'PushEvent':
                const commits = event.payload.commits?.length || 0;
                return `Pushed ${commits} commit${commits > 1 ? 's' : ''} to <strong>${repo}</strong>`;
            case 'CreateEvent':
                return `Created ${event.payload.ref_type} in <strong>${repo}</strong>`;
            case 'ForkEvent':
                return `Forked <strong>${repo}</strong>`;
            case 'WatchEvent':
                return `Starred <strong>${repo}</strong>`;
            case 'PullRequestEvent':
                return `${event.payload.action} pull request in <strong>${repo}</strong>`;
            case 'IssuesEvent':
                return `${event.payload.action} issue in <strong>${repo}</strong>`;
            default:
                return `Activity in <strong>${repo}</strong>`;
        }
    }

    getRelativeTime(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);

        if (diffMins < 1) return 'just now';
        if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
        if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
        if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
        return date.toLocaleDateString();
    }

    renderError() {
        if (!this.container) return;

        this.container.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #94a3b8;">
                <div style="font-size: 40px; margin-bottom: 12px;">😅</div>
                <div>Unable to load GitHub activity</div>
            </div>
        `;
    }
}

// 添加Glitch动画
const style = document.createElement('style');
style.textContent = `
    @keyframes glitch {
        0% {
            transform: translate(-50%, -50%);
        }
        20% {
            transform: translate(-52%, -48%);
        }
        40% {
            transform: translate(-48%, -52%);
        }
        60% {
            transform: translate(-52%, -50%);
        }
        80% {
            transform: translate(-50%, -48%);
        }
        100% {
            transform: translate(-50%, -50%);
        }
    }
`;
document.head.appendChild(style);

// 导出
window.KonamiCode = KonamiCode;
window.EnhancedTypewriter = EnhancedTypewriter;
window.GitHubActivityFeed = GitHubActivityFeed;
