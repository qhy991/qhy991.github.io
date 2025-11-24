/**
 * Achievement System for Personal Website
 * Gamification feature to track visitor interactions
 */

class AchievementSystem {
    constructor() {
        this.achievements = {
            explorer: {
                id: 'explorer',
                name: '🔍 探索者',
                description: '访问所有主要页面',
                icon: '🔍',
                requirement: 5,
                progress: 0,
                unlocked: false,
                pages: new Set()
            },
            scholar: {
                id: 'scholar',
                name: '📚 学者',
                description: '阅读5篇博客文章',
                icon: '📚',
                requirement: 5,
                progress: 0,
                unlocked: false
            },
            geek: {
                id: 'geek',
                name: '💻 极客',
                description: '使用终端命令10次',
                icon: '💻',
                requirement: 10,
                progress: 0,
                unlocked: false
            },
            eggHunter: {
                id: 'eggHunter',
                name: '🎮 彩蛋猎人',
                description: '发现隐藏彩蛋',
                icon: '🎮',
                requirement: 1,
                progress: 0,
                unlocked: false
            },
            nightOwl: {
                id: 'nightOwl',
                name: '🦉 夜猫子',
                description: '在凌晨(0:00-6:00)访问网站',
                icon: '🦉',
                requirement: 1,
                progress: 0,
                unlocked: false
            },
            earlyBird: {
                id: 'earlyBird',
                name: '🐦 早起鸟',
                description: '在清晨(5:00-7:00)访问网站',
                icon: '🐦',
                requirement: 1,
                progress: 0,
                unlocked: false
            },
            dedicated: {
                id: 'dedicated',
                name: '⭐ 忠实粉丝',
                description: '访问网站超过10次',
                icon: '⭐',
                requirement: 10,
                progress: 0,
                unlocked: false
            },
            speedReader: {
                id: 'speedReader',
                name: '⚡ 速读者',
                description: '在5分钟内阅读3篇文章',
                icon: '⚡',
                requirement: 3,
                progress: 0,
                unlocked: false,
                startTime: null
            }
        };

        this.loadProgress();
        this.checkTimeBasedAchievements();
        this.trackVisit();
        this.createNotificationContainer();
    }

    // 加载进度
    loadProgress() {
        const saved = localStorage.getItem('achievements');
        if (saved) {
            const savedData = JSON.parse(saved);
            Object.keys(savedData).forEach(key => {
                if (this.achievements[key]) {
                    this.achievements[key] = { ...this.achievements[key], ...savedData[key] };
                }
            });
        }
    }

    // 保存进度
    saveProgress() {
        const data = {};
        Object.keys(this.achievements).forEach(key => {
            data[key] = {
                progress: this.achievements[key].progress,
                unlocked: this.achievements[key].unlocked
            };
        });
        localStorage.setItem('achievements', JSON.stringify(data));
    }

    // 检查时间相关成就
    checkTimeBasedAchievements() {
        const hour = new Date().getHours();

        // 夜猫子 (0:00-6:00)
        if (hour >= 0 && hour < 6) {
            this.unlock('nightOwl');
        }

        // 早起鸟 (5:00-7:00)
        if (hour >= 5 && hour < 7) {
            this.unlock('earlyBird');
        }
    }

    // 追踪访问
    trackVisit() {
        let visits = parseInt(localStorage.getItem('visitCount') || '0');
        visits++;
        localStorage.setItem('visitCount', visits.toString());

        if (visits >= 10) {
            this.unlock('dedicated');
        }
    }

    // 追踪页面访问
    trackPageVisit(pageName) {
        const achievement = this.achievements.explorer;
        achievement.pages.add(pageName);
        achievement.progress = achievement.pages.size;

        if (achievement.progress >= achievement.requirement) {
            this.unlock('explorer');
        }

        this.saveProgress();
    }

    // 追踪博客阅读
    trackBlogRead() {
        this.incrementProgress('scholar');
    }

    // 追踪终端使用
    trackTerminalUse() {
        this.incrementProgress('geek');
    }

    // 追踪彩蛋发现
    trackEasterEgg() {
        this.unlock('eggHunter');
    }

    // 增加进度
    incrementProgress(achievementId) {
        const achievement = this.achievements[achievementId];
        if (!achievement || achievement.unlocked) return;

        achievement.progress++;

        if (achievement.progress >= achievement.requirement) {
            this.unlock(achievementId);
        }

        this.saveProgress();
    }

    // 解锁成就
    unlock(achievementId) {
        const achievement = this.achievements[achievementId];
        if (!achievement || achievement.unlocked) return;

        achievement.unlocked = true;
        achievement.progress = achievement.requirement;
        this.saveProgress();
        this.showNotification(achievement);
        this.playUnlockSound();
    }

    // 创建通知容器
    createNotificationContainer() {
        if (document.getElementById('achievement-notifications')) return;

        const container = document.createElement('div');
        container.id = 'achievement-notifications';
        container.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            z-index: 9999;
            pointer-events: none;
        `;
        document.body.appendChild(container);
    }

    // 显示通知
    showNotification(achievement) {
        const notification = document.createElement('div');
        notification.className = 'achievement-notification';
        notification.style.cssText = `
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.95), rgba(6, 182, 212, 0.95));
            border: 2px solid rgba(6, 182, 212, 0.5);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 20px rgba(6, 182, 212, 0.4);
            backdrop-filter: blur(10px);
            pointer-events: auto;
            cursor: pointer;
            transform: translateX(400px);
            transition: all 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
            animation: slideIn 0.5s forwards, pulse 2s infinite;
        `;

        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="font-size: 32px;">${achievement.icon}</div>
                <div style="flex: 1;">
                    <div style="font-weight: 600; color: white; font-size: 14px; margin-bottom: 4px;">
                        🎉 成就解锁!
                    </div>
                    <div style="font-weight: 700; color: white; font-size: 16px; margin-bottom: 2px;">
                        ${achievement.name}
                    </div>
                    <div style="color: rgba(255, 255, 255, 0.9); font-size: 12px;">
                        ${achievement.description}
                    </div>
                </div>
            </div>
        `;

        const container = document.getElementById('achievement-notifications');
        container.appendChild(notification);

        // 动画
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);

        // 点击关闭
        notification.addEventListener('click', () => {
            notification.style.transform = 'translateX(400px)';
            setTimeout(() => notification.remove(), 500);
        });

        // 自动关闭
        setTimeout(() => {
            notification.style.transform = 'translateX(400px)';
            setTimeout(() => notification.remove(), 500);
        }, 5000);
    }

    // 播放解锁音效
    playUnlockSound() {
        // 创建简单的8-bit音效
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.frequency.value = 800;
        oscillator.type = 'square';
        gainNode.gain.value = 0.1;

        oscillator.start();

        setTimeout(() => {
            oscillator.frequency.value = 1000;
        }, 100);

        setTimeout(() => {
            oscillator.frequency.value = 1200;
        }, 200);

        setTimeout(() => {
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
            oscillator.stop(audioContext.currentTime + 0.5);
        }, 300);
    }

    // 获取所有成就
    getAllAchievements() {
        return Object.values(this.achievements);
    }

    // 获取已解锁成就
    getUnlockedAchievements() {
        return Object.values(this.achievements).filter(a => a.unlocked);
    }

    // 获取完成百分比
    getCompletionPercentage() {
        const total = Object.keys(this.achievements).length;
        const unlocked = this.getUnlockedAchievements().length;
        return Math.round((unlocked / total) * 100);
    }

    // 显示成就面板
    showAchievementPanel() {
        const panel = document.createElement('div');
        panel.id = 'achievement-panel';
        panel.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(15, 23, 42, 0.98);
            border: 2px solid rgba(6, 182, 212, 0.5);
            border-radius: 16px;
            padding: 32px;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            z-index: 10000;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        `;

        const completion = this.getCompletionPercentage();

        panel.innerHTML = `
            <div style="margin-bottom: 24px;">
                <h2 style="color: #06b6d4; font-size: 28px; font-weight: 700; margin-bottom: 8px;">
                    🏆 成就系统
                </h2>
                <div style="color: #94a3b8; font-size: 14px;">
                    完成度: ${completion}% (${this.getUnlockedAchievements().length}/${Object.keys(this.achievements).length})
                </div>
                <div style="background: rgba(6, 182, 212, 0.1); height: 8px; border-radius: 4px; margin-top: 12px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #06b6d4, #8b5cf6); height: 100%; width: ${completion}%; transition: width 0.5s;"></div>
                </div>
            </div>
            
            <div style="display: grid; gap: 12px;">
                ${Object.values(this.achievements).map(achievement => `
                    <div style="
                        background: ${achievement.unlocked ? 'rgba(6, 182, 212, 0.1)' : 'rgba(51, 65, 85, 0.3)'};
                        border: 1px solid ${achievement.unlocked ? 'rgba(6, 182, 212, 0.3)' : 'rgba(71, 85, 105, 0.3)'};
                        border-radius: 8px;
                        padding: 16px;
                        display: flex;
                        align-items: center;
                        gap: 16px;
                        ${achievement.unlocked ? '' : 'opacity: 0.6;'}
                    ">
                        <div style="font-size: 32px; ${achievement.unlocked ? '' : 'filter: grayscale(100%);'}">
                            ${achievement.icon}
                        </div>
                        <div style="flex: 1;">
                            <div style="color: ${achievement.unlocked ? '#06b6d4' : '#94a3b8'}; font-weight: 600; margin-bottom: 4px;">
                                ${achievement.name}
                            </div>
                            <div style="color: #94a3b8; font-size: 13px; margin-bottom: 8px;">
                                ${achievement.description}
                            </div>
                            ${!achievement.unlocked ? `
                                <div style="background: rgba(71, 85, 105, 0.3); height: 4px; border-radius: 2px; overflow: hidden;">
                                    <div style="background: #8b5cf6; height: 100%; width: ${(achievement.progress / achievement.requirement) * 100}%;"></div>
                                </div>
                                <div style="color: #64748b; font-size: 11px; margin-top: 4px;">
                                    ${achievement.progress}/${achievement.requirement}
                                </div>
                            ` : `
                                <div style="color: #10b981; font-size: 12px; font-weight: 600;">
                                    ✓ 已解锁
                                </div>
                            `}
                        </div>
                    </div>
                `).join('')}
            </div>
            
            <button id="close-achievement-panel" style="
                margin-top: 24px;
                width: 100%;
                padding: 12px;
                background: rgba(6, 182, 212, 0.2);
                border: 1px solid rgba(6, 182, 212, 0.3);
                border-radius: 8px;
                color: #06b6d4;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            " onmouseover="this.style.background='rgba(6, 182, 212, 0.3)'" onmouseout="this.style.background='rgba(6, 182, 212, 0.2)'">
                关闭
            </button>
        `;

        // 添加背景遮罩
        const overlay = document.createElement('div');
        overlay.id = 'achievement-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            z-index: 9999;
            backdrop-filter: blur(4px);
        `;

        document.body.appendChild(overlay);
        document.body.appendChild(panel);

        // 关闭按钮
        document.getElementById('close-achievement-panel').addEventListener('click', () => {
            panel.remove();
            overlay.remove();
        });

        // 点击遮罩关闭
        overlay.addEventListener('click', () => {
            panel.remove();
            overlay.remove();
        });
    }
}

// 添加CSS动画
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 20px rgba(6, 182, 212, 0.4);
        }
        50% {
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 30px rgba(6, 182, 212, 0.6);
        }
    }
`;
document.head.appendChild(style);

// 导出全局实例
window.achievementSystem = new AchievementSystem();
