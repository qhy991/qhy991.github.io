import React, { useState } from 'react';

export default function App() {
  const [language, setLanguage] = useState('en');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // 中英文内容配置
  const content = {
    en: {
      nav: ['About', 'Projects', 'Publications', 'Contact'],
      about: {
        title: "About Me",
        desc: "Ph.D. candidate at School of Microelectronics, Beihang University, focusing on Large Language Models (LLMs) for circuit design automation and their integration with traditional EDA tools.",
        badges: ['Google Scholar', 'GitHub Profile']
      },
      projects: {
        title: "Research Projects",
        circuitmind: {
          name: "CircuitMind",
          desc: "Multi-agent framework for hardware design optimization combining LLMs with traditional EDA tools",
          highlights: [
            "Proposed syntax-locking mechanism for gate-level Boolean optimization",
            "Dynamic knowledge base for sub-circuit pattern reuse",
            "Outperforms human experts in 55.6% tasks on TC-Bench benchmark"
          ],
          tags: ["LLM", "EDA", "Multi-Agent"]
        },
        reasoningv: {
          name: "ReasoningV",
          desc: "Adaptive hybrid reasoning model for Verilog code generation",
          highlights: [
            "Built ReasoningV-5K dataset with 5000+ high-quality Verilog samples",
            "Two-stage training strategy: parameter-efficient fine-tuning + full-parameter optimization",
            "Adaptive reasoning mechanism dynamically adjusts inference depth"
          ],
          tags: ["Verilog", "Code Generation", "AI Hardware"]
        }
      },
      publications: {
        title: "Publications",
        viewAll: "View Full List",
        items: [
          {
            title: "Towards Optimal Circuit Generation: Multi-Agent Collaboration Meets Collective Intelligence",
            authors: "Haiyan Qin, Jiaxiang Feng et al.",
            journal: "arXiv:2504.14625",
            year: "2025"
          },
          {
            title: "ReasoningV: Efficient Verilog Code Generation with Adaptive Hybrid Reasoning Model",
            authors: "Haiyan Qin, Ziyu Xie et al.",
            journal: "arXiv:2504.14560",
            year: "2025"
          },
          {
            title: "LLM-Powered Logic Synthesis: A Paradigm Shift in Digital Design",
            authors: "Haiyan Qin, Mingxuan Yuan et al.",
            journal: "DAC 2025 (Under Review)",
            year: "2025"
          }
        ]
      },
      contact: {
        title: "Contact",
        email: "haiyan.qin@buaa.edu.cn",
        address: "37 Xueyuan Road, Haidian District, Beijing",
        orcid: "ORCID: 0000-0002-1234-5678"
      }
    },
    zh: {
      nav: ['关于', '项目', '论文', '联系'],
      about: {
        title: "个人简介",
        desc: "北京航空航天大学集成电路学院博士研究生，专注于大型语言模型（LLM）在电路设计自动化中的应用，以及LLM与传统EDA工具的集成研究。",
        badges: ['Google Scholar', 'GitHub主页']
      },
      projects: {
        title: "研究项目",
        circuitmind: {
          name: "CircuitMind",
          desc: "结合LLM与传统EDA工具的硬件设计优化多智能体框架",
          highlights: [
            "提出语法锁定机制，强制在基础逻辑门层面进行布尔优化",
            "动态知识库支持优化子电路模式的检索复用",
            "在TC-Bench基准测试中55.6%任务超越人类专家"
          ],
          tags: ["LLM", "EDA", "多智能体"]
        },
        reasoningv: {
          name: "ReasoningV",
          desc: "面向Verilog代码生成的自适应混合推理模型",
          highlights: [
            "构建5000+验证样本的ReasoningV-5K高质量数据集",
            "双阶段训练策略：参数高效微调+全参数优化",
            "自适应推理机制动态调整推理深度"
          ],
          tags: ["Verilog", "代码生成", "AI芯片"]
        }
      },
      publications: {
        title: "学术成果",
        viewAll: "查看完整列表",
        items: [
          {
            title: "面向最优电路生成的多智能体协作与集体智能融合",
            authors: "秦海燕, 冯江翔 等",
            journal: "arXiv:2504.14625",
            year: "2025"
          },
          {
            title: "ReasoningV：基于自适应混合推理模型的高效Verilog代码生成",
            authors: "秦海燕, 谢子玉 等",
            journal: "arXiv:2504.14560",
            year: "2025"
          },
          {
            title: "LLM驱动的逻辑综合：数字设计的新范式",
            authors: "秦海燕, 袁铭轩 等",
            journal: "DAC 2025 (在审)",
            year: "2025"
          }
        ]
      },
      contact: {
        title: "联系方式",
        email: "haiyan.qin@buaa.edu.cn",
        address: "北京市海淀区学院路37号",
        orcid: "ORCID: 0000-0002-1234-5678"
      }
    }
  };

  const currentLang = content[language];

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800">
      {/* 语言切换按钮 */}
      <div className="fixed top-4 right-4 z-50">
        <button 
          onClick={() => setLanguage(language === 'en' ? 'zh' : 'en')}
          className="px-3 py-1 bg-blue-600 text-white rounded-full text-sm hover:bg-blue-700 transition-colors"
        >
          {language === 'en' ? '中文' : 'EN'}
        </button>
      </div>

      {/* 导航栏 */}
      <header className="bg-blue-900 text-white sticky top-0 z-40 shadow-lg">
        <div className="container mx-auto px-4 py-3">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-xl md:text-2xl font-bold">Haiyan Qin</h1>
              <p className="text-sm text-blue-200">Ph.D. Candidate | School of Microelectronics, BUAA</p>
            </div>
            
            {/* 桌面导航 */}
            <nav className="hidden md:flex space-x-6">
              {currentLang.nav.map((item) => (
                <a 
                  key={item} 
                  href={`#${item.toLowerCase()}`} 
                  className="hover:text-blue-200 transition-colors duration-300"
                >
                  {item}
                </a>
              ))}
            </nav>
            
            {/* 移动菜单按钮 */}
            <button 
              className="md:hidden"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>

          {/* 移动导航菜单 */}
          {mobileMenuOpen && (
            <div className="md:hidden py-3 animate-fadeIn">
              {currentLang.nav.map((item) => (
                <a 
                  key={item} 
                  href={`#${item.toLowerCase()}`} 
                  className="block py-2 hover:text-blue-200"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  {item}
                </a>
              ))}
            </div>
          )}
        </div>
      </header>

      {/* 主体内容 */}
      <main className="container mx-auto px-4 py-8">
        {/* 关于模块 */}
        <section id="about" className="mb-16 scroll-mt-16">
          <div className="flex flex-col md:flex-row items-center gap-8">
            <div className="w-32 h-32 rounded-full overflow-hidden border-4 border-blue-500 shadow-lg">
              <img 
                src="https://placehold.co/300x300/1e40af/ffffff?text=HQ " 
                alt="Profile" 
                className="w-full h-full object-cover"
              />
            </div>
            <div className="md:max-w-2xl">
              <h2 className="text-2xl font-bold mb-3">{currentLang.about.title}</h2>
              <p className="mb-4">
                {currentLang.about.desc}
              </p>
              <div className="flex flex-wrap gap-2 mt-4">
                <a href="https://scholar.google.com/citations?user=zzmYq9QAAAAJ&hl=en " target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 px-4 py-2 bg-blue-100 text-blue-800 rounded hover:bg-blue-200 transition-colors">
                  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.63.55-1.43.96-2.3.96-1.4 0-2.6-.95-3.03-2.34l1.84-.61c.21.5.71.89 1.3.89.8 0 1.4-.6 1.4-1.4 0-.8-.6-1.4-1.4-1.4-.79 0-1.3.57-1.38 1.29l-1.84-.55A3.494 3.494 0 0112 8c1.93 0 3.5 1.57 3.5 3.5 0 1.56-1.03 2.89-2.43 3.39zM7 14c.55 1.29 1.41 2.42 2.5 3.25V19c-.56-.13-1.06-.35-1.5-.65L7 14zm8.5-6.5c.61 0 1.1.49 1.1 1.1s-.49 1.1-1.1 1.1-1.1-.49-1.1-1.1.49-1.1 1.1-1.1z"/></svg>
                  {currentLang.about.badges[0]}
                </a>
                <a href="https://github.com/BUAA-CLab " target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-800 rounded hover:bg-gray-200 transition-colors">
                  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                  {currentLang.about.badges[1]}
                </a>
              </div>
            </div>
          </div>
        </section>

        {/* 项目模块 */}
        <section id="projects" className="mb-16 scroll-mt-16">
          <h2 className="text-3xl font-bold mb-6 border-b-2 border-blue-200 pb-2">{currentLang.projects.title}</h2>
          
          {/* CircuitMind 项目 */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-6 hover:shadow-lg transition-shadow duration-300">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div>
                <h3 className="text-xl font-semibold mb-2">{currentLang.projects.circuitmind.name}</h3>
                <p className="mb-3 text-gray-700">{currentLang.projects.circuitmind.desc}</p>
                <ul className="list-disc list-inside mb-4 space-y-1 text-gray-600">
                  {currentLang.projects.circuitmind.highlights.map((highlight, i) => (
                    <li key={i}>{highlight}</li>
                  ))}
                </ul>
                <div className="flex flex-wrap gap-2 mb-4">
                  {currentLang.projects.circuitmind.tags.map((tag, i) => (
                    <span key={i} className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded-full">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
              <a 
                href="https://github.com/BUAA-CLab/CircuitMind " 
                target="_blank" 
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors self-start"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.205 11.385.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                GitHub 仓库
              </a>
            </div>
          </div>

          {/* ReasoningV 项目 */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-6 hover:shadow-lg transition-shadow duration-300">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div>
                <h3 className="text-xl font-semibold mb-2">{currentLang.projects.reasoningv.name}</h3>
                <p className="mb-3 text-gray-700">{currentLang.projects.reasoningv.desc}</p>
                <ul className="list-disc list-inside mb-4 space-y-1 text-gray-600">
                  {currentLang.projects.reasoningv.highlights.map((highlight, i) => (
                    <li key={i}>{highlight}</li>
                  ))}
                </ul>
                <div className="flex flex-wrap gap-2 mb-4">
                  {currentLang.projects.reasoningv.tags.map((tag, i) => (
                    <span key={i} className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded-full">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
              <a 
                href="https://github.com/BUAA-CLab/ReasoningV " 
                target="_blank" 
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors self-start"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.205 11.385.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                GitHub 仓库
              </a>
            </div>
          </div>
        </section>

        {/* 学术成果模块 */}
        <section id="publications" className="mb-16 scroll-mt-16">
          <h2 className="text-3xl font-bold mb-6 border-b-2 border-blue-200 pb-2">{currentLang.publications.title}</h2>
          
          <div className="space-y-6">
            {currentLang.publications.items.map((pub, index) => (
              <div key={index} className="bg-white p-5 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                <h3 className="font-semibold text-lg mb-2">
                  <a 
                    href={pub.link || "#"} 
                    target={pub.link ? "_blank" : "_self"}
                    rel={pub.link ? "noopener noreferrer" : ""}
                    className="hover:text-blue-600 transition-colors"
                  >
                    {pub.title}
                  </a>
                </h3>
                <p className="text-sm text-gray-600 mb-1">{pub.authors}</p>
                <p className="text-sm text-gray-500">{pub.journal}, {pub.year}</p>
              </div>
            ))}
          </div>
          
          <div className="mt-6 text-center">
            <a 
              href="https://scholar.google.com/citations?user=zzmYq9QAAAAJ&hl=en " 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-100 text-blue-800 rounded hover:bg-blue-200 transition-colors"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v2h-2zm0 4h2v6h-2z"/></svg>
              {currentLang.publications.viewAll}
            </a>
          </div>
        </section>

        {/* 联系模块 */}
        <section id="contact" className="scroll-mt-16">
          <h2 className="text-3xl font-bold mb-6 border-b-2 border-blue-200 pb-2">{currentLang.contact.title}</h2>
          
          <div className="max-w-md mx-auto bg-white rounded-lg shadow-md p-6">
            <ul className="space-y-4 text-gray-700">
              <li className="flex items-center gap-3">
                <svg className="w-5 h-5 text-blue-600" viewBox="0 0 24 24" fill="currentColor"><path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/></svg>
                <a href="mailto:haiyan.qin@buaa.edu.cn" className="hover:text-blue-600 transition-colors">{currentLang.contact.email}</a>
              </li>
              <li className="flex items-center gap-3">
                <svg className="w-5 h-5 text-blue-600" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/></svg>
                <span>{currentLang.contact.address}</span>
              </li>
              <li className="flex items-center gap-3">
                <svg className="w-5 h-5 text-blue-600" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-14c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6z"/></svg>
                <a href="https://orcid.org/0000-0002-1234-5678 " target="_blank" rel="noopener noreferrer" className="hover:text-blue-600 transition-colors">{currentLang.contact.orcid}</a>
              </li>
            </ul>
          </div>
        </section>
      </main>

      {/* 页脚 */}
      <footer className="bg-blue-900 text-white py-6 mt-12">
        <div className="container mx-auto px-4 text-center text-sm">
          <p>© 2025 Haiyan Qin. All rights reserved.</p>
          <p className="mt-2 text-blue-200">
            Built with React & TailwindCSS · Hosted on GitHub Pages
          </p>
        </div>
      </footer>
    </div>
  );
}