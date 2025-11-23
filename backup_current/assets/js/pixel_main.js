// Main JavaScript file for Haiyan Qin's academic homepage
// Implements pixel-art style interactions and animations

class AcademicHomepage {
    constructor() {
        this.terminalCommands = {
            'help': this.showHelp.bind(this),
            'about': this.showAbout.bind(this),
            'skills': this.showSkills.bind(this),
            'projects': this.showProjects.bind(this),
            'publications': this.showPublications.bind(this),
            'contact': this.showContact.bind(this),
            'clear': this.clearTerminal.bind(this),
            'date': this.showDate.bind(this),
            'whoami': this.showWhoami.bind(this)
        };
        
        this.commandHistory = [];
        this.historyIndex = -1;
        this.init();
    }
    
    init() {
        this.setupParticleBackground();
        this.setupTerminal();
        this.setupSkillsRadar();
        this.setupScrollAnimations();
        this.setupMobileMenu();
        this.setupTypewriter();
    }
    
    // Particle background using p5.js
    setupParticleBackground() {
        new p5((p) => {
            let particles = [];
            const numParticles = 50;
            
            p.setup = () => {
                const canvas = p.createCanvas(p.windowWidth, p.windowHeight);
                canvas.parent('particle-bg');
                
                // Create particles
                for (let i = 0; i < numParticles; i++) {
                    particles.push({
                        x: p.random(p.width),
                        y: p.random(p.height),
                        vx: p.random(-0.5, 0.5),
                        vy: p.random(-0.5, 0.5),
                        size: p.random(2, 6),
                        opacity: p.random(0.3, 0.8)
                    });
                }
            };
            
            p.draw = () => {
                p.clear();
                
                // Update and draw particles
                particles.forEach(particle => {
                    // Update position
                    particle.x += particle.vx;
                    particle.y += particle.vy;
                    
                    // Wrap around edges
                    if (particle.x < 0) particle.x = p.width;
                    if (particle.x > p.width) particle.x = 0;
                    if (particle.y < 0) particle.y = p.height;
                    if (particle.y > p.height) particle.y = 0;
                    
                    // Draw particle
                    p.fill(6, 182, 212, particle.opacity * 255);
                    p.noStroke();
                    p.rect(particle.x, particle.y, particle.size, particle.size);
                });
                
                // Draw connections
                for (let i = 0; i < particles.length; i++) {
                    for (let j = i + 1; j < particles.length; j++) {
                        const dist = p.dist(particles[i].x, particles[i].y, 
                                          particles[j].x, particles[j].y);
                        if (dist < 100) {
                            p.stroke(139, 92, 246, (1 - dist / 100) * 50);
                            p.strokeWeight(1);
                            p.line(particles[i].x, particles[i].y, 
                                   particles[j].x, particles[j].y);
                        }
                    }
                }
            };
            
            p.windowResized = () => {
                p.resizeCanvas(p.windowWidth, p.windowHeight);
            };
        });
    }
    
    // Terminal functionality
    setupTerminal() {
        const input = document.getElementById('command-input');
        const terminalBody = document.getElementById('terminal-body');
        
        if (!input || !terminalBody) return;
        
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.executeCommand(input.value.trim());
                this.commandHistory.push(input.value.trim());
                this.historyIndex = this.commandHistory.length;
                input.value = '';
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (this.historyIndex > 0) {
                    this.historyIndex--;
                    input.value = this.commandHistory[this.historyIndex] || '';
                }
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (this.historyIndex < this.commandHistory.length - 1) {
                    this.historyIndex++;
                    input.value = this.commandHistory[this.historyIndex] || '';
                } else {
                    this.historyIndex = this.commandHistory.length;
                    input.value = '';
                }
            }
        });
        
        // Auto-focus on terminal input
        input.focus();
    }
    
    executeCommand(command) {
        const terminalBody = document.getElementById('terminal-body');
        const input = document.getElementById('command-input');
        
        // Add command to terminal
        const commandLine = document.createElement('div');
        commandLine.className = 'output';
        commandLine.innerHTML = `<span class="prompt">></span> ${command}`;
        terminalBody.insertBefore(commandLine, input.parentElement);
        
        // Execute command
        if (this.terminalCommands[command]) {
            this.terminalCommands[command]();
        } else if (command) {
            this.addTerminalOutput(`命令未找到: ${command}. 输入 'help' 查看可用命令。`);
        }
        
        // Scroll to bottom
        terminalBody.scrollTop = terminalBody.scrollHeight;
    }
    
    addTerminalOutput(text, className = 'output') {
        const terminalBody = document.getElementById('terminal-body');
        const input = document.getElementById('command-input');
        
        const output = document.createElement('div');
        output.className = className;
        output.textContent = text;
        terminalBody.insertBefore(output, input.parentElement);
        
        // Animate text appearance
        anime({
            targets: output,
            opacity: [0, 1],
            translateY: [-10, 0],
            duration: 300,
            easing: 'easeOutQuad'
        });
    }
    
    // Terminal commands
    showHelp() {
        const helpText = [
            '可用命令:',
            '  about       - 显示个人信息',
            '  skills      - 显示研究领域',
            '  projects    - 显示项目列表',
            '  publications- 显示论文列表',
            '  contact     - 显示联系方式',
            '  date        - 显示当前日期',
            '  whoami      - 显示当前用户',
            '  clear       - 清屏',
            '  help        - 显示此帮助信息',
            '',
            '使用上下箭头键浏览命令历史'
        ];
        
        helpText.forEach(line => {
            this.addTerminalOutput(line);
        });
    }
    
    showAbout() {
        const aboutText = [
            '秦海岩 (Haiyan Qin)',
            '===================',
            '北京航空航天大学研究生',
            '研究领域: AI辅助硬件设计、神经网络部署、电路优化',
            '',
            '学术成就:',
            '- 7篇学术论文',
            '- 5次引用',
            '- H指数: 1',
            '- 30+开源项目',
            '',
            '专注于LLM电路生成和高效神经网络研究'
        ];
        
        aboutText.forEach(line => {
            this.addTerminalOutput(line);
        });
    }
    
    showSkills() {
        const skillsText = [
            '研究领域 (熟练度):',
            '===================',
            '🤖 AI辅助电路设计     ████████░░ 85%',
            '🧠 神经网络部署       ████████░░ 80%',
            '💾 存内计算          ███████░░░ 75%',
            '📊 电路优化          ████████░░ 82%',
            '🔧 Verilog/HDL       █████████░ 90%',
            '⚡ CUDA/GPU          ████████░░ 80%',
            '🐍 Python/ML         █████████░ 88%',
            '',
            '持续学习和研究中...'
        ];
        
        skillsText.forEach(line => {
            this.addTerminalOutput(line);
        });
    }
    
    showProjects() {
        const projectsText = [
            '主要项目:',
            '===================',
            '1. Awesome-LLM-Circuit-Agent',
            '   - 基于LLM的RTL生成和模拟电路生成',
            '   - ⭐ 2 stars, 1 fork',
            '',
            '2. Awesome-LLM-Kernel-Agent',
            '   - 内核生成的LLM智能体研究',
            '   - ⭐ 3 stars',
            '',
            '3. Digital-CIM',
            '   - Verilog实现的数字存内计算',
            '   - ⭐ 1 star, 1 fork',
            '',
            '更多项目请访问: https://github.com/qhy991'
        ];
        
        projectsText.forEach(line => {
            this.addTerminalOutput(line);
        });
    }
    
    showPublications() {
        const publicationsText = [
            '主要论文:',
            '===================',
            '1. "ReasoningV: Efficient Verilog Code Generation with Adaptive Hybrid Reasoning Model"',
            '   - ICAIS 2025 (已接收)',
            '   - 自适应混合推理模型的高效Verilog代码生成',
            '',
            '2. "Multi-agent Approaches for Circuit Design Optimization"',
            '   - DAC 2025 (已接收)',
            '   - 电路设计优化的多智能体方法',
            '',
            '总计: 7篇论文，5次引用',
            '',
            'Google Scholar: https://scholar.google.com/citations?user=zzmYq9QAAAAJ&hl=en'
        ];
        
        publicationsText.forEach(line => {
            this.addTerminalOutput(line);
        });
    }
    
    showContact() {
        const contactText = [
            '联系方式:',
            '===================',
            '📧 Email: haiyanq@buaa.edu.cn',
            '🎓 Google Scholar: zzmYq9QAAAAJ',
            '💻 GitHub: qhy991',
            '🏫 机构: 北京航空航天大学',
            '',
            '欢迎学术交流与合作！',
            '',
            '可用服务:',
            '- 研究合作',
            '- 学术评审',
            '- 会议参与'
        ];
        
        contactText.forEach(line => {
            this.addTerminalOutput(line);
        });
    }
    
    clearTerminal() {
        const terminalBody = document.getElementById('terminal-body');
        const input = document.getElementById('command-input');
        
        // Clear all content except input
        const children = Array.from(terminalBody.children);
        children.forEach(child => {
            if (child !== input.parentElement) {
                child.remove();
            }
        });
    }
    
    showDate() {
        const now = new Date();
        const dateStr = now.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        this.addTerminalOutput(`当前时间: ${dateStr}`);
    }
    
    showWhoami() {
        this.addTerminalOutput('当前用户: Haiyan_Qin (秦海岩)');
        this.addTerminalOutput('身份: AI硬件设计研究员');
        this.addTerminalOutput('机构: 北京航空航天大学');
    }
    
    // Skills radar chart
    setupSkillsRadar() {
        const chartDom = document.getElementById('skills-radar');
        if (!chartDom) return;
        
        const myChart = echarts.init(chartDom);
        
        const option = {
            backgroundColor: 'transparent',
            radar: {
                indicator: [
                    { name: 'AI电路设计', max: 100 },
                    { name: '神经网络', max: 100 },
                    { name: '存内计算', max: 100 },
                    { name: '电路优化', max: 100 },
                    { name: 'Verilog/HDL', max: 100 },
                    { name: 'CUDA/GPU', max: 100 },
                    { name: 'Python/ML', max: 100 },
                    { name: '学术研究', max: 100 }
                ],
                shape: 'polygon',
                splitNumber: 4,
                axisName: {
                    color: '#94a3b8',
                    fontSize: 12
                },
                splitLine: {
                    lineStyle: {
                        color: '#334155'
                    }
                },
                splitArea: {
                    show: false
                },
                axisLine: {
                    lineStyle: {
                        color: '#475569'
                    }
                }
            },
            series: [{
                name: '技能水平',
                type: 'radar',
                data: [{
                    value: [85, 80, 75, 82, 90, 80, 88, 85],
                    name: '当前水平',
                    areaStyle: {
                        color: 'rgba(6, 182, 212, 0.2)'
                    },
                    lineStyle: {
                        color: '#06b6d4',
                        width: 2
                    },
                    itemStyle: {
                        color: '#06b6d4'
                    }
                }],
                animationDuration: 2000,
                animationEasing: 'cubicOut'
            }]
        };
        
        myChart.setOption(option);
        
        // Resize chart on window resize
        window.addEventListener('resize', () => {
            myChart.resize();
        });
    }
    
    // Scroll animations
    setupScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);
        
        // Observe skill cards
        document.querySelectorAll('.skill-card').forEach(card => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            observer.observe(card);
        });
    }
    
    // Mobile menu
    setupMobileMenu() {
        const menuBtn = document.getElementById('mobile-menu-btn');
        const mobileMenu = document.getElementById('mobile-menu');
        
        if (menuBtn && mobileMenu) {
            menuBtn.addEventListener('click', () => {
                mobileMenu.classList.toggle('hidden');
            });
        }
    }
    
    // Typewriter effect
    setupTypewriter() {
        const typewriterElements = document.querySelectorAll('.typewriter');
        
        typewriterElements.forEach(element => {
            const text = element.textContent;
            element.textContent = '';
            element.style.borderRight = '2px solid #06b6d4';
            
            let i = 0;
            const typeInterval = setInterval(() => {
                if (i < text.length) {
                    element.textContent += text.charAt(i);
                    i++;
                } else {
                    clearInterval(typeInterval);
                    // Remove cursor after typing is complete
                    setTimeout(() => {
                        element.style.borderRight = 'none';
                    }, 1000);
                }
            }, 100);
        });
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AcademicHomepage();
    
    // Add some additional interactive effects
    
    // Achievement badge hover effects
    document.querySelectorAll('.achievement-badge').forEach(badge => {
        badge.addEventListener('mouseenter', () => {
            anime({
                targets: badge,
                scale: 1.1,
                rotate: '5deg',
                duration: 300,
                easing: 'easeOutQuad'
            });
        });
        
        badge.addEventListener('mouseleave', () => {
            anime({
                targets: badge,
                scale: 1,
                rotate: '0deg',
                duration: 300,
                easing: 'easeOutQuad'
            });
        });
    });
    
    // Navigation link hover effects
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('mouseenter', () => {
            anime({
                targets: link,
                translateY: -2,
                duration: 200,
                easing: 'easeOutQuad'
            });
        });
        
        link.addEventListener('mouseleave', () => {
            anime({
                targets: link,
                translateY: 0,
                duration: 200,
                easing: 'easeOutQuad'
            });
        });
    });
    
    // Skill card hover effects
    document.querySelectorAll('.skill-card').forEach(card => {
        card.addEventListener('mouseenter', () => {
            anime({
                targets: card,
                translateY: -5,
                duration: 300,
                easing: 'easeOutQuad'
            });
        });
        
        card.addEventListener('mouseleave', () => {
            anime({
                targets: card,
                translateY: 0,
                duration: 300,
                easing: 'easeOutQuad'
            });
        });
    });
    
    // Floating animation for avatar
    anime({
        targets: '.floating',
        translateY: [-10, 10],
        duration: 4000,
        easing: 'easeInOutSine',
        direction: 'alternate',
        loop: true
    });
    
    // Pulse glow effect
    anime({
        targets: '.pulse-glow',
        boxShadow: [
            '0 0 20px rgba(6, 182, 212, 0.5)',
            '0 0 40px rgba(6, 182, 212, 0.8)',
            '0 0 20px rgba(6, 182, 212, 0.5)'
        ],
        duration: 2000,
        easing: 'easeInOutSine',
        loop: true
    });
    
    // News ticker animation
    const newsTicker = document.querySelector('.news-ticker');
    if (newsTicker) {
        anime({
            targets: newsTicker,
            translateX: ['100%', '-100%'],
            duration: 30000,
            easing: 'linear',
            loop: true
        });
    }
});

// Utility functions
function showComingSoon() {
    alert('功能即将推出，敬请期待！');
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        // Show success message
        const message = document.createElement('div');
        message.textContent = '已复制到剪贴板！';
        message.className = 'fixed top-20 right-4 bg-green-500 text-white px-4 py-2 rounded-lg z-50';
        document.body.appendChild(message);
        
        setTimeout(() => {
            message.remove();
        }, 2000);
    });
}

// Export for use in other files
window.AcademicHomepage = AcademicHomepage;