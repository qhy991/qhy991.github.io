---
layout: default
title: "Publications"
permalink: /publications/
---

## Publications

<!-- Like/Appreciation Feature -->
<div class="appreciation-container">
  <button id="likeButton" class="like-button">
    <i class="fas fa-heart"></i>
    <span id="likeText">Appreciate My Work</span>
  </button>
  <div class="like-count">
    <i class="fas fa-users"></i>
    <span id="likeCount">0</span> appreciations
  </div>
</div>

<style>
.appreciation-container {
  text-align: center;
  margin: 30px 0;
  padding: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 16px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.like-button {
  background: white;
  color: #667eea;
  border: none;
  padding: 15px 30px;
  font-size: 1.1rem;
  font-weight: 600;
  border-radius: 50px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  margin-bottom: 15px;
}

.like-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

.like-button:active {
  transform: scale(0.95);
}

.like-button.liked {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
}

.like-button i {
  margin-right: 8px;
  transition: transform 0.3s ease;
}

.like-button.liked i {
  animation: heartBeat 0.5s ease;
  color: white;
}

@keyframes heartBeat {
  0%, 100% { transform: scale(1); }
  25% { transform: scale(1.3); }
  50% { transform: scale(1.1); }
  75% { transform: scale(1.2); }
}

.like-count {
  color: white;
  font-size: 1rem;
  font-weight: 500;
}

.like-count i {
  margin-right: 5px;
}
</style>

<script>
// Like functionality with localStorage
document.addEventListener('DOMContentLoaded', function() {
  const likeButton = document.getElementById('likeButton');
  const likeCount = document.getElementById('likeCount');
  const likeText = document.getElementById('likeText');
  
  // Get current like count and user's like status
  let totalLikes = parseInt(localStorage.getItem('totalLikes') || '0');
  let hasLiked = localStorage.getItem('hasLiked') === 'true';
  let lastLikeTime = parseInt(localStorage.getItem('lastLikeTime') || '0');
  
  // Update display
  likeCount.textContent = totalLikes;
  if (hasLiked) {
    likeButton.classList.add('liked');
    likeText.textContent = 'Thanks for Your Support!';
  }
  
  likeButton.addEventListener('click', function() {
    const now = Date.now();
    const cooldown = 60000; // 1 minute cooldown
    
    if (!hasLiked) {
      // First time liking
      totalLikes++;
      hasLiked = true;
      lastLikeTime = now;
      
      localStorage.setItem('totalLikes', totalLikes.toString());
      localStorage.setItem('hasLiked', 'true');
      localStorage.setItem('lastLikeTime', now.toString());
      
      likeCount.textContent = totalLikes;
      likeButton.classList.add('liked');
      likeText.textContent = 'Thanks for Your Support!';
      
      // Trigger animation
      likeButton.querySelector('i').style.animation = 'none';
      setTimeout(() => {
        likeButton.querySelector('i').style.animation = '';
      }, 10);
    } else if (now - lastLikeTime > cooldown) {
      // Can like again after cooldown
      totalLikes++;
      lastLikeTime = now;
      
      localStorage.setItem('totalLikes', totalLikes.toString());
      localStorage.setItem('lastLikeTime', now.toString());
      
      likeCount.textContent = totalLikes;
      
      // Trigger animation
      likeButton.querySelector('i').style.animation = 'none';
      setTimeout(() => {
        likeButton.querySelector('i').style.animation = '';
      }, 10);
    } else {
      // Still in cooldown
      const remainingTime = Math.ceil((cooldown - (now - lastLikeTime)) / 1000);
      alert(`Please wait ${remainingTime} seconds before appreciating again!`);
    }
  });
});
</script>

---

### Citation Metrics & Trends

<div class="chart-container">
  <canvas id="citationChart"></canvas>
</div>

<style>
.chart-container {
  position: relative;
  height: 300px;
  margin: 30px 0;
  padding: 20px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const ctx = document.getElementById('citationChart');
  
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: [
        'ReasoningV',
        'Tiny Neural Networks',
        'SOT CIM Macro',
        'Multi-Agent Yield',
        'Bayesian Optimization',
        'Weight Mapping',
        'Circuit Generation'
      ],
      datasets: [{
        label: 'Citations',
        data: [2, 2, 1, 0, 0, 0, 0],
        backgroundColor: [
          'rgba(102, 126, 234, 0.8)',
          'rgba(118, 75, 162, 0.8)',
          'rgba(237, 100, 166, 0.8)',
          'rgba(255, 154, 158, 0.8)',
          'rgba(250, 208, 196, 0.8)',
          'rgba(212, 228, 188, 0.8)',
          'rgba(134, 199, 243, 0.8)'
        ],
        borderColor: [
          'rgba(102, 126, 234, 1)',
          'rgba(118, 75, 162, 1)',
          'rgba(237, 100, 166, 1)',
          'rgba(255, 154, 158, 1)',
          'rgba(250, 208, 196, 1)',
          'rgba(212, 228, 188, 1)',
          'rgba(134, 199, 243, 1)'
        ],
        borderWidth: 2,
        borderRadius: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        title: {
          display: true,
          text: 'Citations per Publication',
          font: {
            size: 16,
            weight: 'bold'
          },
          color: '#1f2937'
        },
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          padding: 12,
          titleFont: {
            size: 14
          },
          bodyFont: {
            size: 13
          },
          borderColor: 'rgba(102, 126, 234, 0.5)',
          borderWidth: 1
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            stepSize: 1,
            font: {
              size: 12
            }
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.05)'
          }
        },
        x: {
          ticks: {
            font: {
              size: 11
            },
            maxRotation: 45,
            minRotation: 45
          },
          grid: {
            display: false
          }
        }
      }
    }
  });
});
</script>

---

This page contains information about my research publications. For the most up-to-date information, please visit my [Google Scholar profile](https://scholar.google.com/citations?user=zzmYq9QAAAAJ&hl=en).

### Summary Statistics

<div class="publication-item">
  <div class="publication-meta">
    <strong>Total Publications:</strong> 7 | <strong>Total Citations:</strong> 5 | <strong>h-index:</strong> 1 | <strong>i10-index:</strong> 0
  </div>
</div>

### Research Areas

My research spans multiple disciplines with focus on:
- **Circuit Design & Verilog Generation**: AI-assisted hardware design and RTL code generation
- **FPGA & Neural Network Deployment**: Efficient neural network implementations on embedded systems
- **Computing-in-Memory (CIM)**: Novel architectures for AI edge inference
- **Circuit Yield Optimization**: Multi-agent approaches for circuit design optimization

---

### Selected Publications

<div class="publication-item">
  <div class="publication-title">
    <i class="fas fa-file-alt"></i> ReasoningV: Efficient Verilog Code Generation with Adaptive Hybrid Reasoning Model
  </div>
  <div class="publication-meta">
    <strong>Citations:</strong> <span class="citation-badge">2</span> | 
    <a href="https://scholar.google.com/scholar?cites=14929626030099538040" target="_blank"><i class="fas fa-quote-right"></i> View Citations</a>
  </div>
</div>

<div class="publication-item">
  <div class="publication-title">
    <i class="fas fa-file-alt"></i> Searching Tiny Neural Networks for Deployment on Embedded FPGA
  </div>
  <div class="publication-meta">
    <strong>Citations:</strong> <span class="citation-badge">2</span> | 
    <a href="https://scholar.google.com/scholar?cites=13453769098864207267" target="_blank"><i class="fas fa-quote-right"></i> View Citations</a>
  </div>
</div>

<div class="publication-item">
  <div class="publication-title">
    <i class="fas fa-file-alt"></i> A High-Resistance SOT Device Based Computing-in-Memory Macro With High Sensing Margin and Multi-Bit MAC Operations for AI Edge Inference
  </div>
  <div class="publication-meta">
    <strong>Citations:</strong> <span class="citation-badge">1</span> | 
    <a href="https://scholar.google.com/scholar?cites=17060869671911684307" target="_blank"><i class="fas fa-quote-right"></i> View Citations</a>
  </div>
</div>

<div class="publication-item">
  <div class="publication-title">
    <i class="fas fa-file-alt"></i> Multi-Agent Yield Analysis For Circuit Design
  </div>
  <div class="publication-meta">
    <strong>Status:</strong> Recent Publication
  </div>
</div>

<div class="publication-item">
  <div class="publication-title">
    <i class="fas fa-file-alt"></i> Accuracy Is Not Always We Need: Precision-Aware Bayesian Yield Optimization
  </div>
  <div class="publication-meta">
    <strong>Status:</strong> Recent Publication
  </div>
</div>

<div class="publication-item">
  <div class="publication-title">
    <i class="fas fa-file-alt"></i> Efficient Weight Mapping and Resource Scheduling on Crossbar-based Multi-core CIM Systems
  </div>
  <div class="publication-meta">
    <strong>Status:</strong> Recent Publication
  </div>
</div>

<div class="publication-item">
  <div class="publication-title">
    <i class="fas fa-file-alt"></i> Towards Optimal Circuit Generation: Multi-Agent Collaboration Meets Collective Intelligence
  </div>
  <div class="publication-meta">
    <strong>Status:</strong> Recent Publication
  </div>
</div>

---

### Research Collaboration

I am always interested in research collaboration opportunities. If you would like to discuss potential research projects or collaborations, please feel free to contact me through the channels listed on the [about page]({{ "/about/" | relative_url }}).