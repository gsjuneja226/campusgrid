/**
 * CampusGrid Frontend Controller
 * Handshakes with Flask-SocketIO, controls glassmorphic UI tabs, triggers jobs,
 * compiles active execution chunk progress grids, and displays real-time telemetry.
 */

// Initialize SocketIO Client
const socket = io();

// Initialize Theme Selection
const themeToggleBtn = document.getElementById('theme-toggle-btn');
const themeIcon = document.getElementById('theme-icon');
const themeText = document.getElementById('theme-text');

let currentTheme = localStorage.getItem('theme') || 'dark';
document.body.className = currentTheme + '-theme';
updateThemeUI(currentTheme);

if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
        currentTheme = (currentTheme === 'dark') ? 'light' : 'dark';
        document.body.className = currentTheme + '-theme';
        localStorage.setItem('theme', currentTheme);
        updateThemeUI(currentTheme);
    });
}

function updateThemeUI(theme) {
    if (theme === 'dark') {
        if (themeIcon) themeIcon.textContent = '☀️';
        if (themeText) themeText.textContent = 'Light Mode';
        if (themeToggleBtn) themeToggleBtn.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
    } else {
        if (themeIcon) themeIcon.textContent = '🌙';
        if (themeText) themeText.textContent = 'Dark Mode';
        if (themeToggleBtn) themeToggleBtn.style.backgroundColor = 'rgba(0, 0, 0, 0.03)';
    }
}

// UI Elements Tracking
const navItems = document.querySelectorAll('.nav-item');
const tabContents = document.querySelectorAll('.tab-content');
const pageTitle = document.getElementById('page-title');
const liveClock = document.getElementById('live-clock');

// Job Form elements
const jobForm = document.getElementById('job-submission-form');
const jobTypeSelect = document.getElementById('job-type-select');
const chunksSelect = document.getElementById('chunks-select');
const btnSubmitJob = document.getElementById('btn-submit-job');

// Dashboard active job panels
const jobEmptyState = document.getElementById('job-empty-state');
const jobActiveState = document.getElementById('job-active-state');
const activeJobTitle = document.getElementById('active-job-title');
const activeJobBadge = document.getElementById('active-job-badge');
const activeJobElapsed = document.getElementById('active-job-elapsed');
const activeJobRemaining = document.getElementById('active-job-remaining');
const activeJobProgressVal = document.getElementById('active-job-progress-val');
const activeJobChunksRatio = document.getElementById('active-job-chunks-ratio');
const activeJobProgressBar = document.getElementById('active-job-progress-bar');
const activeJobLiveSpeedup = document.getElementById('active-job-live-speedup');
const activeJobChunksList = document.getElementById('active-job-chunks-list');

// Job complete panel
const finalResultCard = document.getElementById('final-result-card');
const resultJobTitle = document.getElementById('result-job-title');
const resultParallelTime = document.getElementById('result-parallel-time');
const resultSingleTime = document.getElementById('result-single-time');
const resultSpeedup = document.getElementById('result-speedup');
const resultConsoleOutput = document.getElementById('result-console-output');
const btnDownloadResult = document.getElementById('btn-download-result');

// Telemetry counters & grids
const workersCardsContainer = document.getElementById('workers-cards-container');
const workersEmptyState = document.getElementById('workers-empty-state');
const nodesOnlineBadge = document.getElementById('nodes-online-badge');
const workersCountBadge = document.getElementById('workers-count-badge');
const clusterCores = document.getElementById('cluster-cores');
const clusterRam = document.getElementById('cluster-ram');

// Storage & Specs tab elements
const storageNodesCount = document.getElementById('storage-nodes-count');
const storageCoresCount = document.getElementById('storage-cores-count');
const storageRamCount = document.getElementById('storage-ram-count');
const storageDiskCount = document.getElementById('storage-disk-count');
const storageTableBody = document.getElementById('storage-table-body');
const historyTableBody = document.getElementById('history-table-body');

// New UI Elements for Queuing, Aborting and Benchmarking
const btnCancelJob = document.getElementById('btn-cancel-job');
const jobQueueCard = document.getElementById('job-queue-card');
const queueCountBadge = document.getElementById('queue-count-badge');
const queueEmptyState = document.getElementById('queue-empty-state');
const queueList = document.getElementById('queue-list');

const leaderboardCard = document.getElementById('leaderboard-card');
const leaderboardActiveNodes = document.getElementById('leaderboard-active-nodes');
const leaderboardEmptyState = document.getElementById('leaderboard-empty-state');
const leaderboardRanks = document.getElementById('leaderboard-ranks');

// Initialize Chart.js Cluster Telemetry Chart
let telemetryChart = null;
const chartCtx = document.getElementById('cluster-telemetry-chart');
const chartAvgCpu = document.getElementById('chart-avg-cpu');
const chartAvgRam = document.getElementById('chart-avg-ram');

// Maintain historical data for charting (up to 30 data points = 60 seconds at 2s intervals)
const maxChartPoints = 30;
const chartLabels = Array(maxChartPoints).fill('');
const cpuData = Array(maxChartPoints).fill(0);
const ramData = Array(maxChartPoints).fill(0);

if (chartCtx) {
    const ctx = chartCtx.getContext('2d');
    telemetryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartLabels,
            datasets: [
                {
                    label: 'Cluster Avg CPU (%)',
                    data: cpuData,
                    borderColor: '#7c4dff',
                    backgroundColor: 'rgba(124, 77, 255, 0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Cluster Avg RAM (%)',
                    data: ramData,
                    borderColor: '#2979ff',
                    backgroundColor: 'rgba(41, 121, 255, 0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#4b5563',
                        font: { family: 'Outfit', size: 11 }
                    }
                }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        color: '#6b7280',
                        font: { family: 'JetBrains Mono', size: 10 }
                    }
                }
            }
        }
    });
}

// ----------------------------------------------------
// 1. Tab Switching & UI Navigation
// ----------------------------------------------------

navItems.forEach(item => {
    item.addEventListener('click', () => {
        const tabName = item.getAttribute('data-tab');

        // Update active class on nav links
        navItems.forEach(nav => nav.classList.remove('active'));
        item.classList.add('active');

        // Display matched tab content
        tabContents.forEach(content => {
            if (content.id === `tab-${tabName}`) {
                content.classList.remove('hidden');
            } else {
                content.classList.add('hidden');
            }
        });

        // Update navbar header title
        const titles = {
            'dashboard': 'Cluster Dashboard Control Panel',
            'compute': 'Submit Standard Compute Job',
            'rendering': 'Rendering Hub',
            'ai': 'AI & LLM Hub',
            'workers': 'Laboratory Computer Nodes Telemetry',
            'history': 'Cluster Job History Ledger',
            'storage': 'Hardware Specifications & Storage Overview'
        };
        pageTitle.textContent = titles[tabName] || 'CampusGrid';

        // If switching to history, pull logs
        if (tabName === 'history') {
            loadJobHistory();
        }
    });
});

// Toggle Job Type Parameters Subforms
jobTypeSelect.addEventListener('change', (e) => {
    const selectedJob = e.target.value;

    // Hide all parameter blocks
    document.querySelectorAll('.parameter-section').forEach(section => {
        section.classList.add('hidden');
    });

    // Display selected block
    const targetSection = document.getElementById(`params-${selectedJob}`);
    if (targetSection) {
        targetSection.classList.remove('hidden');
    }

    // Keep chunks selection element visible for all jobs (including custom scripts)
    const chunksGroup = chunksSelect.closest('.form-row');
    chunksGroup.classList.remove('hidden');
});

// Live Digital Clock
function updateClock() {
    const now = new Date();
    const hrs = String(now.getHours()).padStart(2, '0');
    const mins = String(now.getMinutes()).padStart(2, '0');
    const secs = String(now.getSeconds()).padStart(2, '0');
    liveClock.textContent = `${hrs}:${mins}:${secs}`;
}
setInterval(updateClock, 1000);
updateClock();

// ----------------------------------------------------
// 2. Job Submissions (REST Calls)
// ----------------------------------------------------

jobForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Collect form data
    const formData = new FormData(jobForm);
    const jobType = formData.get('job_type');
    const isCustom = (jobType === 'custom_script');

    // Pre-validate custom script file choice
    if (isCustom) {
        const fileInput = document.getElementById('custom-file-input');
        if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
            alert('Please select a Python script file (.py) to execute on the cluster.');
            return;
        }
    }

    // Auto-switch tabs for Rendering jobs so user sees the output panel
    if (jobType === 'distributed_render' || jobType === 'render_frames') {
        const renderTabBtn = document.querySelector('.nav-item[data-tab="rendering"]');
        if (renderTabBtn) renderTabBtn.click();
    }

    // Disable submit button during call
    btnSubmitJob.disabled = true;
    btnSubmitJob.querySelector('.btn-text').textContent = 'Submitting Job...';

    // Reset previous final results view
    finalResultCard.classList.add('hidden');

    try {
        let response;
        if (isCustom) {
            // Multipart upload for custom script file
            response = await fetch('/api/run_job', {
                method: 'POST',
                body: formData
            });
        } else {
            // Standard JSON submission for existing jobs
            const payload = {};
            formData.forEach((value, key) => {
                payload[key] = value;
            });
            response = await fetch('/api/run_job', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
        }

        const data = await response.json();

        if (!data.success) {
            alert(`Execution Rejected: ${data.error}`);
        } else {
            console.log(`Job scheduled successfully with ID: ${data.job_id}`);
        }
    } catch (err) {
        alert(`Failed to send job command: ${err.message}`);
    } finally {
        btnSubmitJob.disabled = false;
        btnSubmitJob.querySelector('.btn-text').textContent = 'Execute on Cluster Grid';
    }
});

// ----------------------------------------------------
// 3. Telemetry Sockets & Dynamic UI Renderers
// ----------------------------------------------------

// Handle workers stats list broadcasts
socket.on('workers_list', (workers) => {
    const onlineCount = workers.length;

    // 1. DYNAMIC TELEMETRY REAL-TIME CHART UPDATES
    let avgCPU = 0;
    let avgRAM = 0;
    let workersWithStats = 0;
    workers.forEach(w => {
        if (w.stats && w.stats.cpu_usage !== undefined) {
            avgCPU += w.stats.cpu_usage;
            avgRAM += w.stats.ram_usage;
            workersWithStats++;
        }
    });
    if (workersWithStats > 0) {
        avgCPU = Math.round(avgCPU / workersWithStats);
        avgRAM = Math.round(avgRAM / workersWithStats);
    }
    
    // Update live metrics on chart card
    if (chartAvgCpu) chartAvgCpu.textContent = `${avgCPU}%`;
    if (chartAvgRam) chartAvgRam.textContent = `${avgRAM}%`;
    
    // Push new point and shift old point in datasets
    if (telemetryChart) {
        cpuData.shift();
        cpuData.push(avgCPU);
        ramData.shift();
        ramData.push(avgRAM);
        telemetryChart.update('none'); // Update without animation lag
    }

    // Update active counters and badges
    nodesOnlineBadge.textContent = `${onlineCount} Worker${onlineCount !== 1 ? 's' : ''} Online`;
    workersCountBadge.textContent = onlineCount;
    storageNodesCount.textContent = onlineCount;

    if (onlineCount === 0) {
        // Show empty states
        workersEmptyState.classList.remove('hidden');
        // Clear specs
        clusterCores.textContent = '0';
        clusterRam.textContent = '0 GB';
        storageCoresCount.textContent = '0';
        storageRamCount.textContent = '0 GB';
        storageDiskCount.textContent = '0.0 TB';

        // Clear lists
        document.querySelectorAll('.worker-card-node').forEach(el => el.remove());
        storageTableBody.innerHTML = `<tr><td colspan="7" class="text-center">No lab computers registered.</td></tr>`;
        
        // Hide leaderboard
        leaderboardActiveNodes.textContent = '0 Nodes Benchmarked';
        leaderboardEmptyState.classList.remove('hidden');
        leaderboardRanks.classList.add('hidden');
        return;
    }

    workersEmptyState.classList.add('hidden');

    // Compute aggregations across online hardware
    let totalCores = 0;
    let totalRAM = 0.0;
    let totalDisk = 0.0;
    let freeDisk = 0.0;

    workers.forEach(w => {
        totalCores += w.specs.cores || 0;
        totalRAM += w.specs.ram_total || 0;
        totalDisk += w.specs.disk_total || 0;
        freeDisk += w.specs.disk_free || 0;
    });

    clusterCores.textContent = totalCores;
    clusterRam.textContent = `${totalRAM.toFixed(1)} GB`;

    storageCoresCount.textContent = totalCores;
    storageRamCount.textContent = `${totalRAM.toFixed(1)} GB`;
    // Convert to TB if large, else GB
    if (totalDisk >= 1024) {
        storageDiskCount.textContent = `${(totalDisk / 1024).toFixed(2)} TB`;
    } else {
        storageDiskCount.textContent = `${totalDisk.toFixed(0)} GB`;
    }

    // 2. RENDER TAB 2: ACTIVE COMPUTERS GRID WITH STRESS TEST ACTIONS
    // Clear container except empty state
    const cards = workersCardsContainer.querySelectorAll('.worker-card');
    cards.forEach(c => c.remove());

    workers.forEach(w => {
        const cpuVal = w.stats.cpu_usage !== undefined ? w.stats.cpu_usage : 0;
        const ramVal = w.stats.ram_usage !== undefined ? w.stats.ram_usage : 0;
        const diskVal = w.stats.disk_usage !== undefined ? w.stats.disk_usage : 0;

        const isBenchmarking = w.status === 'benchmarking';
        const benchmarkBtnText = isBenchmarking ? '⌛ Testing...' : '⚡ Stress Test';
        const benchmarkBtnDisabled = isBenchmarking ? 'disabled' : '';
        const gflopsText = w.specs.gflops !== undefined ? `${w.specs.gflops.toFixed(1)} GFLOPS` : 'No GFLOPS score';
        const gflopsClass = w.specs.gflops !== undefined ? 'gflops-badge rated' : 'gflops-badge';

        let latencyClass = 'latency-good';
        if (w.latency > 150) latencyClass = 'latency-poor';
        else if (w.latency > 60) latencyClass = 'latency-medium';
        const latencyText = w.latency ? `${w.latency.toFixed(0)} ms` : '0 ms';
        const donatedCoresText = `Cores: ${w.specs.donated_cores || w.specs.cores}/${w.specs.cores}`;

        const cardHtml = `
            <div class="worker-card animate-fade-in ${w.status}">
                <div class="worker-card-header">
                    <div class="worker-node-info">
                        <h4 style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                            <span>🖥️ ${w.worker_id}</span>
                            <span class="latency-badge ${latencyClass}" style="font-size: 10px; font-weight: 700;">${latencyText}</span>
                        </h4>
                        <span>IP: ${w.ip}</span>
                    </div>
                    <span class="worker-status-dot ${w.status}"></span>
                </div>
                <div class="worker-telemetry">
                    <div class="telemetry-row">
                        <div class="telemetry-label-row">
                            <span>CPU Processor</span>
                            <span class="val">${cpuVal}%</span>
                        </div>
                        <div class="telemetry-bar-bg">
                            <div class="telemetry-bar-fill cpu" style="width: ${cpuVal}%;"></div>
                        </div>
                    </div>
                    <div class="telemetry-row">
                        <div class="telemetry-label-row">
                            <span>Memory Load</span>
                            <span class="val">${ramVal}%</span>
                        </div>
                        <div class="telemetry-bar-bg">
                            <div class="telemetry-bar-fill ram" style="width: ${ramVal}%;"></div>
                        </div>
                    </div>
                </div>
                <div class="worker-bench-row" style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; border-top: 1px dashed rgba(255, 255, 255, 0.04); padding-top: 10px;">
                    <span class="${gflopsClass}" style="font-size: 11px; font-weight: 700; color: #00e676; font-family: var(--font-mono);">${gflopsText}</span>
                    <button class="btn btn-stress" data-worker-id="${w.worker_id}" ${benchmarkBtnDisabled} style="padding: 4px 8px; font-size: 11px; background-color: rgba(124, 77, 255, 0.15); color: var(--color-primary); border: 1px solid rgba(124, 77, 255, 0.2); border-radius: 6px; font-weight: 600; cursor: pointer; transition: background-color 0.2s;">
                        ${benchmarkBtnText}
                    </button>
                </div>
                <div class="worker-specs-brief">
                    <span>${donatedCoresText}</span>
                    <span>${w.specs.ram_total} GB RAM</span>
                    <span>${w.specs.os.split(' ')[0]}</span>
                </div>
            </div>
        `;
        workersCardsContainer.insertAdjacentHTML('beforeend', cardHtml);
    });

    // Bind click events to stress test buttons
    workersCardsContainer.querySelectorAll('.btn-stress').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const wId = btn.getAttribute('data-worker-id');
            socket.emit('trigger_benchmark', { worker_id: wId });
        });
    });

    // 3. COMPILE & SORT GFLOPS LEADERBOARD
    const benchmarkedWorkers = workers
        .filter(w => w.specs.gflops !== undefined && w.specs.gflops > 0)
        .sort((a, b) => b.specs.gflops - a.specs.gflops);
        
    leaderboardActiveNodes.textContent = `${benchmarkedWorkers.length} Node${benchmarkedWorkers.length !== 1 ? 's' : ''} Rated`;
    
    if (benchmarkedWorkers.length === 0) {
        leaderboardEmptyState.classList.remove('hidden');
        leaderboardRanks.classList.add('hidden');
    } else {
        leaderboardEmptyState.classList.add('hidden');
        leaderboardRanks.classList.remove('hidden');
        
        let leaderboardHtml = '';
        benchmarkedWorkers.forEach((w, index) => {
            let medal = '💻';
            if (index === 0) medal = '🥇';
            else if (index === 1) medal = '🥈';
            else if (index === 2) medal = '🥉';
            
            let tier = 'Tier B';
            let tierColor = 'var(--text-secondary)';
            if (w.specs.gflops > 20) {
                tier = 'Tier S [Supercomputer]';
                tierColor = 'var(--color-warning)';
            } else if (w.specs.gflops > 10) {
                tier = 'Tier A [High Performance]';
                tierColor = 'var(--color-success)';
            }
            
            leaderboardHtml += `
                <div class="leaderboard-item animate-fade-in" style="background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border-color); padding: 12px 16px; border-radius: 10px; display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 18px;">${medal}</span>
                        <div>
                            <h4 style="font-size: 14px; font-weight: 700;">Rank #${index + 1}: ${w.worker_id}</h4>
                            <span style="font-size: 11px; color: ${tierColor}; font-weight: 600;">${tier}</span>
                        </div>
                    </div>
                    <span style="font-family: var(--font-mono); font-size: 14px; font-weight: 800; color: #00e676;">${w.specs.gflops.toFixed(1)} GFLOPS</span>
                </div>
            `;
        });
        leaderboardRanks.innerHTML = leaderboardHtml;
    }

    // 4. RENDER TAB 4: STORAGE & SPECS SHEET TABLE
    let tableHtml = '';
    workers.forEach(w => {
        const diskUsedPct = w.specs.disk_total > 0
            ? Math.round(((w.specs.disk_total - w.specs.disk_free) / w.specs.disk_total) * 100)
            : 0;

        let latencyClass = 'latency-good';
        if (w.latency > 150) latencyClass = 'latency-poor';
        else if (w.latency > 60) latencyClass = 'latency-medium';
        const latencyText = w.latency ? `${w.latency.toFixed(0)} ms` : '0 ms';
        const coresText = `${w.specs.donated_cores || w.specs.cores} / ${w.specs.cores}`;

        tableHtml += `
            <tr>
                <td>🖥️ ${w.worker_id}</td>
                <td>${w.ip}</td>
                <td>${w.specs.os}</td>
                <td>${coresText} Cores</td>
                <td>${w.specs.ram_total} GB RAM</td>
                <td>${(w.specs.disk_total - w.specs.disk_free).toFixed(1)} / ${w.specs.disk_total.toFixed(0)} GB Free</td>
                <td>
                    <div class="disk-meter-container">
                        <div class="disk-meter-bg">
                            <div class="disk-meter-fill" style="width: ${diskUsedPct}%;"></div>
                        </div>
                        <span class="disk-meter-pct">${diskUsedPct}%</span>
                    </div>
                </td>
                <td><span class="latency-badge ${latencyClass}">${latencyText}</span></td>
            </tr>
        `;
    });
    storageTableBody.innerHTML = tableHtml;
});

// Handle real-time job state broadcasts
socket.on('job_status', (job) => {
    // 1. ALWAYS RENDER SCHEDULED JOB QUEUE
    const queuedJobs = job.queued_jobs || [];
    queueCountBadge.textContent = `${queuedJobs.length} Queued`;
    
    if (queuedJobs.length === 0) {
        queueEmptyState.classList.remove('hidden');
        queueList.classList.add('hidden');
        queueList.innerHTML = '';
    } else {
        queueEmptyState.classList.add('hidden');
        queueList.classList.remove('hidden');
        
        let queueHtml = '';
        queuedJobs.forEach((q, qIndex) => {
            queueHtml += `
                <div class="queue-item animate-fade-in" style="background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border-color); padding: 10px 14px; border-radius: 8px; display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-family: var(--font-mono); font-size: 12px; color: var(--text-muted);">#${qIndex + 1}</span>
                        <div>
                            <h4 style="font-size: 13px; font-weight: 600;">${q.name}</h4>
                            <span style="font-size: 10px; color: var(--text-muted); font-family: var(--font-mono); text-transform: uppercase;">${q.job_type}</span>
                        </div>
                    </div>
                    <button class="btn btn-cancel-queued" data-job-id="${q.job_id}" style="padding: 4px 8px; font-size: 10px; background-color: rgba(255, 23, 68, 0.1); color: var(--color-danger); border: 1px solid rgba(255, 23, 68, 0.2); border-radius: 6px; font-weight: 600; cursor: pointer; transition: background-color 0.2s;">
                        Remove
                    </button>
                </div>
            `;
        });
        queueList.innerHTML = queueHtml;
        
        // Bind cancel buttons for queued jobs
        queueList.querySelectorAll('.btn-cancel-queued').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const qId = btn.getAttribute('data-job-id');
                btn.disabled = true;
                btn.textContent = 'Removing...';
                try {
                    await fetch('/api/cancel_job', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ job_id: parseInt(qId) })
                    });
                } catch (err) {
                    alert(`Failed to cancel queued job: ${err.message}`);
                    btn.disabled = false;
                    btn.textContent = 'Remove';
                }
            });
        });
    }

    if (!job.active) {
        jobActiveState.classList.add('hidden');
        jobEmptyState.classList.remove('hidden');
        return;
    }

    // Show active job card
    jobEmptyState.classList.add('hidden');
    jobActiveState.classList.remove('hidden');

    // Update labels
    activeJobTitle.textContent = job.name;
    activeJobElapsed.textContent = `${job.elapsed_time}s`;
    activeJobRemaining.textContent = job.est_remaining > 0 ? `${job.est_remaining}s` : 'Calculating...';
    activeJobProgressVal.textContent = `${job.progress}%`;
    activeJobChunksRatio.textContent = `${job.completed_chunks}/${job.total_chunks}`;

    // Set widths & values
    activeJobProgressBar.style.width = `${job.progress}%`;
    activeJobLiveSpeedup.textContent = `${job.speedup_est.toFixed(1)}x`;

    // Render chunks progress lists
    activeJobChunksList.innerHTML = '';

    job.chunks.forEach(c => {
        // Compute ranges text labels
        let details = '';
        if (job.job_type === 'prime_finder') {
            details = `${(c.chunk_data.start / 1000).toFixed(0)}k → ${(c.chunk_data.end / 1000).toFixed(0)}k`;
        } else if (job.job_type === 'monte_carlo_pi') {
            details = `${(c.chunk_data.iterations / 1000000).toFixed(1)}M pts`;
        } else if (job.job_type === 'matrix_multiply') {
            details = `${c.chunk_data.A_chunk.length} Rows`;
        } else if (job.job_type === 'word_count') {
            details = `${c.chunk_data.paragraphs.length} Paragraphs`;
        } else if (job.job_type === 'custom_script') {
            details = c.chunk_data.filename || 'custom_job.py';
        }

        const isComplete = c.status === 'COMPLETED';
        const progressFillClass = isComplete ? 'chunk-progress-fill completed' : 'chunk-progress-fill';
        const statusBadgeClass = `status-badge status-${c.status.toLowerCase()}`;
        const activeNodeText = c.worker_id ? c.worker_id : 'Idle';

        const chunkHtml = `
            <div class="chunk-row animate-fade-in">
                <div class="chunk-col-id">💻 ${activeNodeText}</div>
                <div class="chunk-col-range">Chunk ${c.chunk_index + 1} (${details})</div>
                <div class="chunk-col-progress-container">
                    <div class="chunk-progress-bg">
                        <div class="${progressFillClass}" style="width: ${c.progress}%;"></div>
                    </div>
                    <span class="chunk-pct">${c.progress}%</span>
                </div>
                <div class="chunk-col-status">
                    <span class="${statusBadgeClass}">${c.status}</span>
                </div>
            </div>
        `;
        activeJobChunksList.insertAdjacentHTML('beforeend', chunkHtml);
    });
});

// Handle final computational results completes
socket.on('job_complete', (data) => {
    // Show success banner card
    finalResultCard.classList.remove('hidden');

    // Set labels
    resultJobTitle.textContent = `✅ ${data.name} Complete`;
    resultParallelTime.textContent = `${data.parallel_time}s`;
    resultSingleTime.textContent = `${data.single_time_est}s`;
    resultSpeedup.textContent = `${data.speedup}x`;

    // Download anchors setups
    btnDownloadResult.href = `/api/results/${data.filename}`;

    // Clear any previous dynamic image gallery
    const existingGallery = document.getElementById('result-image-gallery');
    if (existingGallery) {
        existingGallery.remove();
    }

    if (data.job_type === 'blur_images' || data.job_type === 'process_portrait') {
        const imgUrls = data.summary.match(/https?:\/\/[^\s\)]+\.png/gi) || [];
        if (imgUrls.length > 0) {
            const galleryDiv = document.createElement('div');
            galleryDiv.id = 'result-image-gallery';
            galleryDiv.style.marginTop = '20px';
            galleryDiv.style.borderTop = '1px dashed var(--border-color)';
            galleryDiv.style.paddingTop = '16px';

            const title = document.createElement('h4');
            title.textContent = '🖼️ Generated Output Images';
            title.style.marginBottom = '12px';
            title.style.fontSize = '14px';
            title.style.color = 'var(--text-secondary)';
            title.style.textTransform = 'uppercase';
            title.style.fontWeight = '600';
            galleryDiv.appendChild(title);

            const gridDiv = document.createElement('div');
            gridDiv.style.display = 'grid';
            gridDiv.style.gridTemplateColumns = 'repeat(auto-fill, minmax(160px, 1fr))';
            gridDiv.style.gap = '16px';

            imgUrls.forEach(url => {
                const imgContainer = document.createElement('div');
                imgContainer.style.background = 'rgba(0,0,0,0.02)';
                imgContainer.style.border = '1px solid var(--border-color)';
                imgContainer.style.borderRadius = '10px';
                imgContainer.style.padding = '8px';
                imgContainer.style.textAlign = 'center';
                imgContainer.style.display = 'flex';
                imgContainer.style.flexDirection = 'column';
                imgContainer.style.gap = '8px';

                const img = document.createElement('img');
                img.src = url;
                img.alt = 'Rendered result';
                img.style.width = '100%';
                img.style.height = '140px';
                img.style.objectFit = 'contain';
                img.style.borderRadius = '6px';
                img.style.background = 'rgba(0, 0, 0, 0.05)';

                const link = document.createElement('a');
                link.href = url;
                link.target = '_blank';
                link.textContent = url.substring(url.lastIndexOf('/') + 1);
                link.style.fontSize = '11px';
                link.style.textDecoration = 'none';
                link.style.color = 'var(--color-primary)';
                link.style.fontWeight = '600';
                link.style.whiteSpace = 'nowrap';
                link.style.overflow = 'hidden';
                link.style.textOverflow = 'ellipsis';

                imgContainer.appendChild(img);
                imgContainer.appendChild(link);
                gridDiv.appendChild(imgContainer);
            });
            galleryDiv.appendChild(gridDiv);
            resultConsoleOutput.parentNode.insertBefore(galleryDiv, resultConsoleOutput);
        }
    }

    // Print console summary
    resultConsoleOutput.textContent = data.summary;

    // Auto Scroll results card into view
    finalResultCard.scrollIntoView({ behavior: 'smooth' });

    // Pull history logs to refresh tab list
    loadJobHistory();
});

// Handle computational job failure
socket.on('job_failed', (data) => {
    // Show final result card as a failure card
    finalResultCard.classList.remove('hidden');

    // Set labels
    resultJobTitle.textContent = `❌ ${data.name} Failed`;
    resultParallelTime.textContent = `-`;
    resultSingleTime.textContent = `-`;
    resultSpeedup.textContent = `-`;

    // Disable download link
    btnDownloadResult.removeAttribute('href');

    // Print console summary with error details
    resultConsoleOutput.textContent = `Execution Error:\n${data.error}`;

    // Auto Scroll results card into view
    finalResultCard.scrollIntoView({ behavior: 'smooth' });

    // Pull history logs to refresh tab list
    loadJobHistory();
});

// ----------------------------------------------------
// 4. Job History REST Loader
// ----------------------------------------------------

async function loadJobHistory() {
    try {
        const response = await fetch('/api/history');
        const jobs = await response.json();

        if (jobs.length === 0) {
            historyTableBody.innerHTML = `<tr><td colspan="9" class="text-center">No computational job history logged yet.</td></tr>`;
            return;
        }

        let html = '';
        jobs.forEach(j => {
            // Compile values
            const parallelTimeText = j.parallel_time ? `${j.parallel_time.toFixed(2)}s` : '-';
            const singleTimeText = j.single_time_est ? `${j.single_time_est.toFixed(2)}s` : '-';
            const speedupText = j.speedup ? `${j.speedup.toFixed(1)}x` : '-';
            const dateStr = new Date(j.created_at).toLocaleString();

            let actionHtml = '-';
            if (j.status === 'COMPLETED' && j.result_path) {
                actionHtml = `<a href="/api/results/${j.result_path}" class="btn btn-download" style="padding:6px 12px; font-size:11px;">Get Results</a>`;
            } else if (j.status === 'RUNNING') {
                actionHtml = `<span class="badge badge-active">Processing</span>`;
            } else if (j.status === 'FAILED') {
                actionHtml = `<span class="badge" style="background-color:rgba(255,23,68,0.15); color:var(--color-danger);">Failed</span>`;
            }

            html += `
                <tr>
                    <td>#${j.id}</td>
                    <td>${j.name}</td>
                    <td><code class="console-code" style="margin-top:0; font-size:11px; padding:2px 6px;">${j.job_type}</code></td>
                    <td>${j.worker_count} Nodes</td>
                    <td>${parallelTimeText}</td>
                    <td>${singleTimeText}</td>
                    <td class="text-glow-blue" style="font-weight:700;">${speedupText}</td>
                    <td>${dateStr}</td>
                    <td>${actionHtml}</td>
                </tr>
            `;
        });
        historyTableBody.innerHTML = html;
    } catch (err) {
        historyTableBody.innerHTML = `<tr><td colspan="9" class="text-center text-danger">Failed to retrieve historical logs: ${err.message}</td></tr>`;
    }
}

// Perform initial history retrieval on boot
loadJobHistory();

// Bind click event for active job cancel button
if (btnCancelJob) {
    btnCancelJob.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to abort the active grid execution? This will instantly terminate worker tasks.')) {
            return;
        }
        btnCancelJob.disabled = true;
        btnCancelJob.textContent = 'Aborting...';
        try {
            const response = await fetch('/api/cancel_job', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            const data = await response.json();
            if (!data.success) {
                alert(`Cancel failed: ${data.error}`);
                btnCancelJob.disabled = false;
                btnCancelJob.textContent = 'Cancel Run';
            }
        } catch (err) {
            alert(`Failed to cancel active job: ${err.message}`);
            btnCancelJob.disabled = false;
            btnCancelJob.textContent = 'Cancel Run';
        }
    });
}

// System Activity Console Logic
socket.on('system_log', (log) => {
    const consoleEl = document.getElementById('live-log-console');
    if (consoleEl) {
        const timeStr = new Date(log.timestamp * 1000).toLocaleTimeString();
        const logLine = `[${timeStr}] ${log.message}\n`;
        consoleEl.textContent += logLine;
        consoleEl.scrollTop = consoleEl.scrollHeight;
    }
});

const clearConsoleBtn = document.getElementById('clear-console-btn');
if (clearConsoleBtn) {
    clearConsoleBtn.addEventListener('click', () => {
        const consoleEl = document.getElementById('live-log-console');
        if (consoleEl) {
            consoleEl.textContent = '';
        }
    });
}

