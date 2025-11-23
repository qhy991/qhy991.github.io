---
layout: pixel
title: Haiyan Qin - 秦海岩 | AI硬件设计研究员
---

<!-- Hero Section -->
<section class="hero-bg">
    <div class="hero-content">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                <!-- Avatar Section -->
                <div class="text-center lg:text-left">
                    <div class="floating">
                        <img src="{{ '/assets/images/pixel/avatar.png' | relative_url }}" alt="Haiyan Qin" class="w-48 h-48 mx-auto lg:mx-0 rounded-full pixel-border pulse-glow mb-8">
                    </div>
                    <h1 class="text-5xl lg:text-6xl font-bold mb-4">
                        <span class="text-cyan-400">Haiyan</span>
                        <span class="text-purple-400">Qin</span>
                    </h1>
                    <h2 class="text-2xl lg:text-3xl text-slate-300 mb-6 typewriter">
                        秦海岩 - AI硬件设计研究员
                    </h2>
                    <p class="text-lg text-slate-400 mb-8 leading-relaxed">
                        北京航空航天大学研究生，专注于AI辅助硬件设计、神经网络部署和电路优化研究。
                        在LLM电路生成和高效神经网络领域拥有丰富的研究经验。
                    </p>
                    <div class="flex flex-wrap gap-4 justify-center lg:justify-start">
                        <div class="achievement-badge" title="7篇出版物">
                            <span class="text-xs">7PUB</span>
                        </div>
                        <div class="achievement-badge" title="5次引用">
                            <span class="text-xs">5CIT</span>
                        </div>
                        <div class="achievement-badge" title="30+项目">
                            <span class="text-xs">30+PRJ</span>
                        </div>
                        <div class="achievement-badge" title="H指数1">
                            <span class="text-xs">H1</span>
                        </div>
                    </div>
                </div>
                
                <!-- Terminal Section -->
                <div class="terminal glow-effect">
                    <div class="terminal-header">
                        <span class="mono-font">Haiyan_Qin@Research:~$</span>
                    </div>
                    <div class="terminal-body" id="terminal-body">
                        <div class="output">欢迎使用我的学术主页终端！</div>
                        <div class="output">输入 'help' 查看可用命令。</div>
                        <div class="command-line">
                            <span class="prompt">></span>
                            <input type="text" class="command-input" id="command-input" placeholder="输入命令..." autocomplete="off">
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</section>

<!-- Skills Radar Section -->
<section class="content-section bg-slate-900/50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-16">
            <h2 class="text-4xl font-bold mb-4">
                <span class="text-cyan-400">研究</span>
                <span class="text-purple-400">领域</span>
            </h2>
            <p class="text-xl text-slate-400">探索AI与硬件设计的交叉领域</p>
        </div>
        
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <!-- Radar Chart -->
            <div class="skill-card">
                <div id="skills-radar" style="height: 400px;"></div>
            </div>
            
            <!-- Skills Description -->
            <div class="space-y-6">
                <div class="skill-card">
                    <div class="flex items-center mb-4">
                        <div class="w-4 h-4 bg-cyan-400 rounded mr-3"></div>
                        <h3 class="text-xl font-semibold">AI辅助电路设计</h3>
                    </div>
                    <p class="text-slate-400">使用大语言模型生成Verilog代码，自动化RTL设计流程，提高电路设计效率。</p>
                </div>
                
                <div class="skill-card">
                    <div class="flex items-center mb-4">
                        <div class="w-4 h-4 bg-purple-400 rounded mr-3"></div>
                        <h3 class="text-xl font-semibold">神经网络部署</h3>
                    </div>
                    <p class="text-slate-400">在FPGA和嵌入式系统上实现高效的神经网络推理，优化性能和功耗。</p>
                </div>
                
                <div class="skill-card">
                    <div class="flex items-center mb-4">
                        <div class="w-4 h-4 bg-green-400 rounded mr-3"></div>
                        <h3 class="text-xl font-semibold">存内计算</h3>
                    </div>
                    <p class="text-slate-400">研究新型存内计算架构，为AI边缘推理提供高效解决方案。</p>
                </div>
                
                <div class="skill-card">
                    <div class="flex items-center mb-4">
                        <div class="w-4 h-4 bg-yellow-400 rounded mr-3"></div>
                        <h3 class="text-xl font-semibold">电路优化</h3>
                    </div>
                    <p class="text-slate-400">应用多智能体方法和贝叶斯优化进行电路良率分析和性能优化。</p>
                </div>
            </div>
        </div>
    </div>
</section>

<!-- Latest News Section -->
<section class="content-section">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-12">
            <h2 class="text-4xl font-bold mb-4">
                <span class="text-cyan-400">最新</span>
                <span class="text-purple-400">动态</span>
            </h2>
        </div>
        
        <div class="bg-slate-800/50 rounded-lg p-6 border border-cyan-500/30">
            <div class="news-ticker text-lg font-medium mb-4">
                🎉 我们的论文《ReasoningV: Efficient Verilog Code Generation with Adaptive Hybrid Reasoning Model》已被ICAIS 2025接收！
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="skill-card">
                    <div class="flex items-center mb-3">
                        <div class="w-3 h-3 bg-yellow-400 rounded-full mr-3 animate-pulse"></div>
                        <span class="text-sm text-slate-400">2025.11</span>
                    </div>
                    <h3 class="text-lg font-semibold mb-2">ICAIS 2025论文接收</h3>
                    <p class="text-slate-400 text-sm">我们的论文《ReasoningV: Efficient Verilog Code Generation with Adaptive Hybrid Reasoning Model》已被第一届国际AI科学家会议接收！</p>
                </div>
                
                <div class="skill-card">
                    <div class="flex items-center mb-3">
                        <div class="w-3 h-3 bg-green-400 rounded-full mr-3 animate-pulse"></div>
                        <span class="text-sm text-slate-400">2025</span>
                    </div>
                    <h3 class="text-lg font-semibold mb-2">DAC 2025论文接收</h3>
                    <p class="text-slate-400 text-sm">关于多智能体电路设计优化的研究论文已被设计自动化会议DAC 2025接收。</p>
                </div>
            </div>
        </div>
    </div>
</section>