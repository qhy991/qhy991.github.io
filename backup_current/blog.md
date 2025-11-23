---
layout: default
title: "Blog"
permalink: /blog/
---

## Blog

Welcome to my technical blog! Here I share research notes, tutorials, and insights on AI-assisted hardware design, CUDA programming, and more.

---

### <i class="fas fa-tags"></i> Categories

<div class="blog-categories">
  <a href="#ai-hardware" class="category-tag"><i class="fas fa-microchip"></i> AI & Hardware</a>
  <a href="#cuda" class="category-tag"><i class="fas fa-bolt"></i> CUDA</a>
  <a href="#research" class="category-tag"><i class="fas fa-flask"></i> Research</a>
  <a href="#tutorials" class="category-tag"><i class="fas fa-book-open"></i> Tutorials</a>
</div>

<style>
.blog-categories {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 20px 0;
}
.category-tag {
  display: inline-flex;
  align-items: center;
  padding: 8px 16px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white !important;
  border-radius: 20px;
  font-size: 0.9rem;
  text-decoration: none !important;
  transition: transform 0.2s, box-shadow 0.2s;
}
.category-tag:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}
.category-tag i {
  margin-right: 6px;
}
.blog-post {
  padding: 20px;
  margin: 20px 0;
  background: var(--bg-light);
  border-radius: 12px;
  border-left: 4px solid var(--primary-color);
  transition: all 0.3s ease;
}
.blog-post:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}
.blog-post-title {
  font-weight: 600;
  font-size: 1.1rem;
  color: var(--text-color);
  margin-bottom: 8px;
}
.blog-post-meta {
  color: var(--text-light);
  font-size: 0.85rem;
  margin-bottom: 10px;
}
.blog-post-excerpt {
  color: var(--text-color);
  font-size: 0.95rem;
  line-height: 1.6;
}
</style>

---

### <i class="fas fa-pencil-alt"></i> Recent Posts

<div class="blog-post" id="ai-hardware">
  <div class="blog-post-title">
    <i class="fas fa-microchip"></i> Understanding LLM-based Verilog Code Generation
  </div>
  <div class="blog-post-meta">
    <i class="fas fa-calendar"></i> Coming Soon | <i class="fas fa-tag"></i> AI & Hardware
  </div>
  <div class="blog-post-excerpt">
    An introduction to how Large Language Models can assist in generating hardware description code, with a focus on our ReasoningV approach.
  </div>
</div>

<div class="blog-post" id="cuda">
  <div class="blog-post-title">
    <i class="fas fa-bolt"></i> CUDA Optimization Techniques for Deep Learning
  </div>
  <div class="blog-post-meta">
    <i class="fas fa-calendar"></i> Coming Soon | <i class="fas fa-tag"></i> CUDA
  </div>
  <div class="blog-post-excerpt">
    Key optimization strategies for writing efficient CUDA kernels, including memory coalescing, shared memory usage, and warp-level primitives.
  </div>
</div>

<div class="blog-post" id="research">
  <div class="blog-post-title">
    <i class="fas fa-flask"></i> Computing-in-Memory: A New Paradigm for AI Inference
  </div>
  <div class="blog-post-meta">
    <i class="fas fa-calendar"></i> Coming Soon | <i class="fas fa-tag"></i> Research
  </div>
  <div class="blog-post-excerpt">
    Exploring how Computing-in-Memory (CIM) architectures can overcome the von Neumann bottleneck for efficient AI edge inference.
  </div>
</div>

<div class="blog-post" id="tutorials">
  <div class="blog-post-title">
    <i class="fas fa-book-open"></i> Getting Started with FPGA-based Neural Network Deployment
  </div>
  <div class="blog-post-meta">
    <i class="fas fa-calendar"></i> Coming Soon | <i class="fas fa-tag"></i> Tutorials
  </div>
  <div class="blog-post-excerpt">
    A step-by-step guide to deploying quantized neural networks on embedded FPGAs, from model optimization to hardware implementation.
  </div>
</div>

---

### <i class="fas fa-external-link-alt"></i> External Resources

For more technical notes and code examples, check out:

- <i class="fab fa-github"></i> [Blog-Note Repository](https://github.com/qhy991/Blog-Note) - Research notes and Jupyter notebooks
- <i class="fab fa-github"></i> [Book-CUDA](https://github.com/qhy991/Book-CUDA) - CUDA programming learning materials
- <i class="fab fa-github"></i> [Book-CUDA-OPT](https://github.com/qhy991/Book-CUDA-OPT) - CUDA optimization techniques

---

*More posts coming soon! Stay tuned for updates on AI-assisted hardware design and efficient computing.*
