/**
 * Skill Tree Visualization System
 * Interactive tech stack visualization
 */

class SkillTree {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.skills = this.defineSkills();
        this.init();
    }

    defineSkills() {
        return {
            root: {
                id: 'root',
                name: 'AI Research',
                level: 'expert',
                icon: '🤖',
                description: 'Artificial Intelligence & Machine Learning',
                children: ['llm', 'cv', 'rl']
            },
            llm: {
                id: 'llm',
                name: 'LLM/NLP',
                level: 'expert',
                icon: '💬',
                description: 'Large Language Models & Natural Language Processing',
                projects: ['ReasoningV', 'Chat-CPU'],
                children: ['bert', 'gpt', 'verilog-gen']
            },
            cv: {
                id: 'cv',
                name: 'Computer Vision',
                level: 'intermediate',
                icon: '👁️',
                description: 'Image Processing & Recognition',
                children: ['resnet', 'yolo']
            },
            rl: {
                id: 'rl',
                name: 'Reinforcement Learning',
                level: 'advanced',
                icon: '🎮',
                description: 'Policy Optimization & Decision Making',
                projects: ['GRPO-Clean'],
                children: ['dqn', 'ppo']
            },
            bert: {
                id: 'bert',
                name: 'BERT',
                level: 'advanced',
                icon: '📚',
                description: 'Bidirectional Encoder Representations'
            },
            gpt: {
                id: 'gpt',
                name: 'GPT',
                level: 'expert',
                icon: '✨',
                description: 'Generative Pre-trained Transformer'
            },
            'verilog-gen': {
                id: 'verilog-gen',
                name: 'Verilog Generation',
                level: 'expert',
                icon: '⚡',
                description: 'AI-assisted Hardware Description Language'
            },
            resnet: {
                id: 'resnet',
                name: 'ResNet',
                level: 'intermediate',
                icon: '🏗️',
                description: 'Residual Neural Networks'
            },
            yolo: {
                id: 'yolo',
                name: 'YOLO',
                level: 'intermediate',
                icon: '🎯',
                description: 'Real-time Object Detection'
            },
            dqn: {
                id: 'dqn',
                name: 'DQN',
                level: 'advanced',
                icon: '🧠',
                description: 'Deep Q-Network'
            },
            ppo: {
                id: 'ppo',
                name: 'PPO',
                level: 'expert',
                icon: '🚀',
                description: 'Proximal Policy Optimization'
            }
        };
    }

    init() {
        if (!this.container) return;

        this.container.innerHTML = '';
        this.container.style.cssText = `
            position: relative;
            width: 100%;
            min-height: 600px;
            background: rgba(15, 23, 42, 0.5);
            border-radius: 16px;
            padding: 40px;
            overflow: hidden;
        `;

        this.renderTree();
    }

    renderTree() {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '600');
        svg.style.position = 'absolute';
        svg.style.top = '0';
        svg.style.left = '0';

        // 绘制连接线
        this.drawConnections(svg);

        // 添加节点
        this.renderNodes();

        this.container.appendChild(svg);
    }

    drawConnections(svg) {
        const connections = [
            { from: 'root', to: 'llm', color: '#06b6d4' },
            { from: 'root', to: 'cv', color: '#8b5cf6' },
            { from: 'root', to: 'rl', color: '#10b981' },
            { from: 'llm', to: 'bert', color: '#06b6d4' },
            { from: 'llm', to: 'gpt', color: '#06b6d4' },
            { from: 'llm', to: 'verilog-gen', color: '#06b6d4' },
            { from: 'cv', to: 'resnet', color: '#8b5cf6' },
            { from: 'cv', to: 'yolo', color: '#8b5cf6' },
            { from: 'rl', to: 'dqn', color: '#10b981' },
            { from: 'rl', to: 'ppo', color: '#10b981' }
        ];

        const positions = this.getNodePositions();

        connections.forEach(conn => {
            const from = positions[conn.from];
            const to = positions[conn.to];

            if (!from || !to) return;

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', from.x);
            line.setAttribute('y1', from.y);
            line.setAttribute('x2', to.x);
            line.setAttribute('y2', to.y);
            line.setAttribute('stroke', conn.color);
            line.setAttribute('stroke-width', '2');
            line.setAttribute('opacity', '0.3');
            line.style.transition = 'opacity 0.3s';

            line.addEventListener('mouseenter', () => {
                line.setAttribute('opacity', '0.8');
            });

            line.addEventListener('mouseleave', () => {
                line.setAttribute('opacity', '0.3');
            });

            svg.appendChild(line);
        });
    }

    getNodePositions() {
        const width = this.container.offsetWidth;
        const centerX = width / 2;

        return {
            'root': { x: centerX, y: 80 },
            'llm': { x: centerX - 250, y: 200 },
            'cv': { x: centerX, y: 200 },
            'rl': { x: centerX + 250, y: 200 },
            'bert': { x: centerX - 350, y: 350 },
            'gpt': { x: centerX - 250, y: 350 },
            'verilog-gen': { x: centerX - 150, y: 350 },
            'resnet': { x: centerX - 50, y: 350 },
            'yolo': { x: centerX + 50, y: 350 },
            'dqn': { x: centerX + 200, y: 350 },
            'ppo': { x: centerX + 300, y: 350 }
        };
    }

    renderNodes() {
        const positions = this.getNodePositions();

        Object.keys(this.skills).forEach(skillId => {
            const skill = this.skills[skillId];
            const pos = positions[skillId];

            if (!pos) return;

            const node = this.createNode(skill, pos);
            this.container.appendChild(node);
        });
    }

    createNode(skill, position) {
        const node = document.createElement('div');
        node.className = 'skill-node';
        node.style.cssText = `
            position: absolute;
            left: ${position.x}px;
            top: ${position.y}px;
            transform: translate(-50%, -50%);
            width: 80px;
            height: 80px;
            background: ${this.getLevelColor(skill.level)};
            border: 3px solid ${this.getLevelBorderColor(skill.level)};
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px ${this.getLevelShadow(skill.level)};
            z-index: 10;
        `;

        node.innerHTML = `
            <div style="font-size: 28px; margin-bottom: 4px;">${skill.icon}</div>
            <div style="font-size: 10px; color: white; font-weight: 600; text-align: center; line-height: 1.2;">
                ${skill.name}
            </div>
        `;

        // 悬停效果
        node.addEventListener('mouseenter', () => {
            node.style.transform = 'translate(-50%, -50%) scale(1.2)';
            node.style.zIndex = '100';
            this.showTooltip(skill, position);
        });

        node.addEventListener('mouseleave', () => {
            node.style.transform = 'translate(-50%, -50%) scale(1)';
            node.style.zIndex = '10';
            this.hideTooltip();
        });

        // 点击显示详情
        node.addEventListener('click', () => {
            this.showSkillModal(skill);
        });

        return node;
    }

    getLevelColor(level) {
        const colors = {
            'beginner': 'linear-gradient(135deg, #64748b, #475569)',
            'intermediate': 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
            'advanced': 'linear-gradient(135deg, #06b6d4, #0891b2)',
            'expert': 'linear-gradient(135deg, #10b981, #059669)'
        };
        return colors[level] || colors.intermediate;
    }

    getLevelBorderColor(level) {
        const colors = {
            'beginner': '#94a3b8',
            'intermediate': '#a78bfa',
            'advanced': '#22d3ee',
            'expert': '#34d399'
        };
        return colors[level] || colors.intermediate;
    }

    getLevelShadow(level) {
        const shadows = {
            'beginner': 'rgba(100, 116, 139, 0.3)',
            'intermediate': 'rgba(139, 92, 246, 0.4)',
            'advanced': 'rgba(6, 182, 212, 0.4)',
            'expert': 'rgba(16, 185, 129, 0.4)'
        };
        return shadows[level] || shadows.intermediate;
    }

    showTooltip(skill, position) {
        const tooltip = document.createElement('div');
        tooltip.id = 'skill-tooltip';
        tooltip.style.cssText = `
            position: absolute;
            left: ${position.x}px;
            top: ${position.y - 60}px;
            transform: translateX(-50%);
            background: rgba(15, 23, 42, 0.98);
            border: 1px solid rgba(6, 182, 212, 0.5);
            border-radius: 8px;
            padding: 12px 16px;
            z-index: 1000;
            white-space: nowrap;
            pointer-events: none;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        `;

        tooltip.innerHTML = `
            <div style="color: #06b6d4; font-weight: 600; margin-bottom: 4px;">
                ${skill.name}
            </div>
            <div style="color: #94a3b8; font-size: 12px;">
                ${skill.description}
            </div>
            <div style="color: ${this.getLevelBorderColor(skill.level)}; font-size: 11px; margin-top: 4px; text-transform: uppercase;">
                ${skill.level}
            </div>
        `;

        this.container.appendChild(tooltip);
    }

    hideTooltip() {
        const tooltip = document.getElementById('skill-tooltip');
        if (tooltip) tooltip.remove();
    }

    showSkillModal(skill) {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(15, 23, 42, 0.98);
            border: 2px solid ${this.getLevelBorderColor(skill.level)};
            border-radius: 16px;
            padding: 32px;
            max-width: 500px;
            z-index: 10001;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        `;

        modal.innerHTML = `
            <div style="text-align: center; margin-bottom: 24px;">
                <div style="font-size: 64px; margin-bottom: 16px;">${skill.icon}</div>
                <h2 style="color: ${this.getLevelBorderColor(skill.level)}; font-size: 28px; font-weight: 700; margin-bottom: 8px;">
                    ${skill.name}
                </h2>
                <div style="color: #94a3b8; font-size: 14px; margin-bottom: 12px;">
                    ${skill.description}
                </div>
                <div style="display: inline-block; padding: 6px 16px; background: ${this.getLevelColor(skill.level)}; border-radius: 20px; color: white; font-size: 12px; font-weight: 600; text-transform: uppercase;">
                    ${skill.level}
                </div>
            </div>

            ${skill.projects ? `
                <div style="margin-bottom: 20px;">
                    <h3 style="color: #06b6d4; font-size: 16px; font-weight: 600; margin-bottom: 12px;">
                        Related Projects
                    </h3>
                    <div style="display: flex; flex-wrap: gap: 8px;">
                        ${skill.projects.map(project => `
                            <span style="background: rgba(6, 182, 212, 0.1); border: 1px solid rgba(6, 182, 212, 0.3); padding: 6px 12px; border-radius: 6px; color: #06b6d4; font-size: 13px;">
                                ${project}
                            </span>
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            <button id="close-skill-modal" style="
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
                Close
            </button>
        `;

        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            z-index: 10000;
            backdrop-filter: blur(4px);
        `;

        document.body.appendChild(overlay);
        document.body.appendChild(modal);

        document.getElementById('close-skill-modal').addEventListener('click', () => {
            modal.remove();
            overlay.remove();
        });

        overlay.addEventListener('click', () => {
            modal.remove();
            overlay.remove();
        });
    }
}

// 导出
window.SkillTree = SkillTree;
