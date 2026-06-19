/**
 * GEMM Matrix Multiplication Visualizer
 * Animated tile-based matrix multiply — themed for AI Infra research
 */

class GEMMVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.N = 8;
        this.tileSize = 0;
        this.phase = 0;
        this.activeRow = 0;
        this.activeCol = 0;
        this.accumulated = Array.from({ length: 8 }, () => Array(8).fill(0));
        this.frameCount = 0;
        this.gflops = 0;

        this.matrixA = this.randomMatrix(this.N);
        this.matrixB = this.randomMatrix(this.N);

        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.animate();
    }

    randomMatrix(n) {
        return Array.from({ length: n }, () =>
            Array.from({ length: n }, () => Math.floor(Math.random() * 9))
        );
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = 280 * dpr;
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = '280px';
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.scale(dpr, dpr);
        this.displayW = rect.width;
        this.displayH = 280;
        this.tileSize = Math.min((this.displayW - 80) / (this.N * 3 + 2), 28);
    }

    drawMatrix(x, y, matrix, label, highlightRow, highlightCol, color) {
        const ts = this.tileSize;
        const ctx = this.ctx;

        ctx.fillStyle = '#64748b';
        ctx.font = '11px JetBrains Mono, monospace';
        ctx.fillText(label, x, y - 6);

        for (let i = 0; i < this.N; i++) {
            for (let j = 0; j < this.N; j++) {
                const px = x + j * (ts + 2);
                const py = y + i * (ts + 2);
                const isHL = (highlightRow !== undefined && i === highlightRow) ||
                             (highlightCol !== undefined && j === highlightCol);

                ctx.fillStyle = isHL ? color + '55' : 'rgba(30,41,59,0.9)';
                ctx.strokeStyle = isHL ? color : '#334155';
                ctx.lineWidth = isHL ? 2 : 1;
                ctx.beginPath();
                ctx.roundRect(px, py, ts, ts, 3);
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = isHL ? color : '#94a3b8';
                ctx.font = `${Math.max(9, ts * 0.4)}px JetBrains Mono, monospace`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(String(matrix[i][j]), px + ts / 2, py + ts / 2);
            }
        }
    }

    drawResultMatrix(x, y) {
        const ts = this.tileSize;
        const ctx = this.ctx;

        ctx.fillStyle = '#64748b';
        ctx.font = '11px JetBrains Mono, monospace';
        ctx.textAlign = 'left';
        ctx.fillText('C = A × B', x, y - 6);

        for (let i = 0; i < this.N; i++) {
            for (let j = 0; j < this.N; j++) {
                const px = x + j * (ts + 2);
                const py = y + i * (ts + 2);
                const computed = i < this.activeRow || (i === this.activeRow && j <= this.activeCol);
                const isActive = i === this.activeRow && j === this.activeCol;

                ctx.fillStyle = isActive ? 'rgba(6,182,212,0.4)' :
                    computed ? 'rgba(16,185,129,0.25)' : 'rgba(30,41,59,0.9)';
                ctx.strokeStyle = isActive ? '#06b6d4' : computed ? '#10b981' : '#334155';
                ctx.lineWidth = isActive ? 2 : 1;
                ctx.beginPath();
                ctx.roundRect(px, py, ts, ts, 3);
                ctx.fill();
                ctx.stroke();

                if (computed) {
                    ctx.fillStyle = isActive ? '#06b6d4' : '#10b981';
                    ctx.font = `${Math.max(8, ts * 0.35)}px JetBrains Mono, monospace`;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(String(this.accumulated[i][j]), px + ts / 2, py + ts / 2);
                }
            }
        }
    }

    step() {
        if (this.activeCol >= this.N) {
            this.activeCol = 0;
            this.activeRow++;
        }
        if (this.activeRow >= this.N) {
            this.activeRow = 0;
            this.activeCol = 0;
            this.accumulated = Array.from({ length: this.N }, () => Array(this.N).fill(0));
            this.matrixA = this.randomMatrix(this.N);
            this.matrixB = this.randomMatrix(this.N);
            return;
        }

        let sum = 0;
        for (let k = 0; k < this.N; k++) {
            sum += this.matrixA[this.activeRow][k] * this.matrixB[k][this.activeCol];
        }
        this.accumulated[this.activeRow][this.activeCol] = sum;
        this.activeCol++;
        this.frameCount++;
        this.gflops = (2 * this.N * this.N * this.N * (this.frameCount / 60)) / 1e9;
    }

    animate() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.displayW, this.displayH);

        // Background grid
        ctx.strokeStyle = 'rgba(51,65,85,0.3)';
        ctx.lineWidth = 0.5;
        for (let x = 0; x < this.displayW; x += 20) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, this.displayH); ctx.stroke();
        }
        for (let y = 0; y < this.displayH; y += 20) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(this.displayW, y); ctx.stroke();
        }

        const ts = this.tileSize;
        const matH = this.N * (ts + 2);
        const startY = (this.displayH - matH) / 2 + 10;

        this.drawMatrix(16, startY, this.matrixA, 'A', this.activeRow, undefined, '#06b6d4');
        this.drawMatrix(this.displayW / 2 - matH / 2, startY, this.matrixB, 'B', undefined, this.activeCol, '#8b5cf6');
        this.drawResultMatrix(this.displayW - matH - 16, startY);

        // Multiply symbol
        ctx.fillStyle = '#64748b';
        ctx.font = '20px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('×', this.displayW / 2 - matH / 2 - 20, startY + matH / 2);
        ctx.fillText('=', this.displayW - matH - 32, startY + matH / 2);

        // Update stats display
        const opsEl = document.getElementById('gemm-ops');
        const tileEl = document.getElementById('gemm-tile');
        if (opsEl) opsEl.textContent = (this.frameCount * this.N * this.N * 2).toLocaleString();
        if (tileEl) tileEl.textContent = `(${this.activeRow}, ${Math.max(0, this.activeCol - 1)})`;

        if (this.frameCount % 3 === 0) this.step();
        requestAnimationFrame(() => this.animate());
    }
}

document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('gemm-canvas')) {
        new GEMMVisualizer('gemm-canvas');
    }
});
