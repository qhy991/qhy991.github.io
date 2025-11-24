/**
 * Main Initialization Script for Interactive Features
 * Coordinates all the fun stuff on the website
 */

document.addEventListener('DOMContentLoaded', function () {
    console.log('🚀 Initializing interactive features...');

    // ============================================
    // 1. ACHIEVEMENT SYSTEM
    // ============================================
    if (window.achievementSystem) {
        console.log('✅ Achievement System loaded');

        // Track page visits
        const currentPage = window.location.pathname.split('/').pop() || 'index.html';
        window.achievementSystem.trackPageVisit(currentPage);

        // Add achievement button to navigation
        addAchievementButton();
    }

    // ============================================
    // 2. KONAMI CODE EASTER EGG
    // ============================================
    if (window.KonamiCode) {
        new KonamiCode();
        console.log('✅ Konami Code activated (try: ↑↑↓↓←→←→BA)');
    }

    // ============================================
    // 3. ENHANCED TYPEWRITER EFFECT
    // ============================================
    const typewriterElement = document.querySelector('.typewriter');
    if (typewriterElement && window.EnhancedTypewriter) {
        const phrases = [
            'AI Infra Researcher',
            'Hardware Designer',
            'CUDA Enthusiast',
            'Open Source Contributor',
            'Lifelong Learner',
            'Tech Explorer'
        ];

        new EnhancedTypewriter(typewriterElement, phrases, {
            typeSpeed: 80,
            deleteSpeed: 40,
            pauseTime: 2000
        });
        console.log('✅ Enhanced Typewriter initialized');
    }

    // ============================================
    // 4. GITHUB ACTIVITY FEED
    // ============================================
    const githubContainer = document.getElementById('github-activity');
    if (githubContainer && window.GitHubActivityFeed) {
        new GitHubActivityFeed('qhy991', 'github-activity');
        console.log('✅ GitHub Activity Feed loaded');
    }

    // ============================================
    // 5. SKILL TREE VISUALIZATION
    // ============================================
    const skillTreeContainer = document.getElementById('skill-tree-container');
    if (skillTreeContainer && window.SkillTree) {
        new SkillTree('skill-tree-container');
        console.log('✅ Skill Tree initialized');
    }

    // ============================================
    // TERMINAL ENHANCEMENTS
    // ============================================
    enhanceTerminal();

    // ============================================
    // BLOG READING TRACKER
    // ============================================
    if (currentPage.includes('blog') || currentPage.includes('post')) {
        trackBlogReading();
    }

    console.log('🎉 All interactive features initialized!');
});

// ============================================
// HELPER FUNCTIONS
// ============================================

function addAchievementButton() {
    const nav = document.querySelector('nav .flex.items-baseline');
    if (!nav) return;

    const achievementBtn = document.createElement('button');
    achievementBtn.innerHTML = '🏆';
    achievementBtn.title = 'View Achievements';
    achievementBtn.style.cssText = `
        background: rgba(6, 182, 212, 0.2);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 8px;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 18px;
        transition: all 0.3s;
        margin-left: 16px;
    `;

    achievementBtn.addEventListener('mouseenter', () => {
        achievementBtn.style.background = 'rgba(6, 182, 212, 0.3)';
        achievementBtn.style.transform = 'scale(1.1)';
    });

    achievementBtn.addEventListener('mouseleave', () => {
        achievementBtn.style.background = 'rgba(6, 182, 212, 0.2)';
        achievementBtn.style.transform = 'scale(1)';
    });

    achievementBtn.addEventListener('click', () => {
        if (window.achievementSystem) {
            window.achievementSystem.showAchievementPanel();
        }
    });

    nav.appendChild(achievementBtn);
}

function enhanceTerminal() {
    const commandInput = document.getElementById('command-input');
    const terminalBody = document.getElementById('terminal-body');

    if (!commandInput || !terminalBody) return;

    const commands = {
        help: () => {
            return `Available commands:
  help       - Show this help message
  about      - About me
  skills     - List my skills
  projects   - View my projects
  contact    - Contact information
  clear      - Clear terminal
  konami     - Hint about easter egg
  achievements - View achievements`;
        },
        about: () => {
            return `Haiyan Qin - AI Infra Researcher
Graduate student at Beihang University
Research: AI-assisted hardware design, neural network deployment`;
        },
        skills: () => {
            return `Core Skills:
  • Large Language Models (LLM)
  • CUDA Programming
  • Verilog/RTL Design
  • Neural Network Optimization
  • Reinforcement Learning`;
        },
        projects: () => {
            window.location.href = 'portfolio.html';
            return 'Redirecting to portfolio...';
        },
        contact: () => {
            return `Contact Information:
  Email: haiyanq@buaa.edu.cn
  GitHub: github.com/qhy991
  Scholar: scholar.google.com/citations?user=zzmYq9QAAAAJ`;
        },
        clear: () => {
            terminalBody.innerHTML = '';
            return '';
        },
        konami: () => {
            return 'Try pressing: ↑ ↑ ↓ ↓ ← → ← → B A';
        },
        achievements: () => {
            if (window.achievementSystem) {
                window.achievementSystem.showAchievementPanel();
                return 'Opening achievements panel...';
            }
            return 'Achievement system not loaded';
        }
    };

    commandInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const command = commandInput.value.trim().toLowerCase();

            if (command) {
                // Display command
                const commandLine = document.createElement('div');
                commandLine.className = 'command-line';
                commandLine.innerHTML = `<span class="prompt">></span> ${command}`;
                terminalBody.insertBefore(commandLine, terminalBody.lastElementChild);

                // Execute command
                const output = commands[command] ? commands[command]() : `Command not found: ${command}. Type 'help' for available commands.`;

                if (output) {
                    const outputDiv = document.createElement('div');
                    outputDiv.className = 'output';
                    outputDiv.style.whiteSpace = 'pre-line';
                    outputDiv.textContent = output;
                    terminalBody.insertBefore(outputDiv, terminalBody.lastElementChild);
                }

                // Track terminal usage for achievement
                if (window.achievementSystem) {
                    window.achievementSystem.trackTerminalUse();
                }

                // Clear input
                commandInput.value = '';

                // Scroll to bottom
                terminalBody.scrollTop = terminalBody.scrollHeight;
            }
        }
    });
}

function trackBlogReading() {
    // Track when user spends time on blog post
    let startTime = Date.now();
    let tracked = false;

    window.addEventListener('beforeunload', () => {
        const timeSpent = (Date.now() - startTime) / 1000; // seconds

        // If user spent more than 30 seconds, count as read
        if (timeSpent > 30 && !tracked && window.achievementSystem) {
            window.achievementSystem.trackBlogRead();
            tracked = true;
        }
    });

    // Also track on scroll (user engaged with content)
    let scrollTracked = false;
    window.addEventListener('scroll', () => {
        if (!scrollTracked && window.scrollY > 500) {
            if (window.achievementSystem) {
                window.achievementSystem.trackBlogRead();
            }
            scrollTracked = true;
        }
    });
}

// Add some fun console messages
console.log('%c🎮 Welcome to my website!', 'font-size: 20px; color: #06b6d4; font-weight: bold;');
console.log('%cTry the Konami Code: ↑↑↓↓←→←→BA', 'color: #8b5cf6;');
console.log('%cCheck out the achievements system! 🏆', 'color: #10b981;');
