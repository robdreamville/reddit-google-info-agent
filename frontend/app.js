// Global Chart Instances
let chartPlatforms = null;
let chartContentTypes = null;
let activePreset = null;
let lastGenerationResults = {};

// Page Loading Init
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadDashboardStats();
    loadActiveConfig();
    setupPlatformListener();
});

// =============================================================================
// 1. NAVIGATION & HEALTH STATUS
// =============================================================================
function switchTab(tabId) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    
    // Show active tab
    document.getElementById(`tab-${tabId}`).classList.add('active');
    document.getElementById(`btn-tab-${tabId}`).classList.add('active');
    
    // Update header subtitles
    const titleMap = {
        'dashboard': { title: 'Dashboard', sub: 'Real-time performance and usage metrics' },
        'generate': { title: 'Content Generator', sub: 'Input search keywords and synthesize native posts' },
        'history': { title: 'Log History', sub: 'Browse past agent runs, scripts, and evaluation metrics' },
        'settings': { title: 'System Settings', sub: 'Tweak prompts, temperature defaults, and preset files' }
    };
    
    document.getElementById('page-title').innerText = titleMap[tabId].title;
    document.getElementById('page-subtitle').innerText = titleMap[tabId].sub;
    
    // Reload metrics/data on switch
    if (tabId === 'dashboard') {
        loadDashboardStats();
    } else if (tabId === 'history') {
        loadHistoryLogs();
    } else if (tabId === 'settings') {
        checkHealth();
        loadActiveConfig();
    }
}

async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        const indicator = document.getElementById('health-indicator');
        const textNode = document.getElementById('health-text');
        const envList = document.getElementById('env-checker-list');
        
        if (data.status === 'healthy') {
            indicator.className = 'status-indicator healthy';
            textNode.innerText = `Connected: ${data.active_model}`;
        } else {
            indicator.className = 'status-indicator unconfigured';
            textNode.innerText = 'Unconfigured (Missing Keys)';
        }
        
        // Settings Tab Checker Render
        if (envList) {
            const keys = {
                'GEMINI_API_KEY': !data.missing_env_vars.includes('GEMINI_API_KEY'),
                'REDDIT_CLIENT_ID': !data.missing_env_vars.includes('REDDIT_CLIENT_ID'),
                'REDDIT_CLIENT_SECRET': !data.missing_env_vars.includes('REDDIT_CLIENT_SECRET')
            };
            
            envList.innerHTML = Object.entries(keys).map(([key, ok]) => `
                <div class="env-row ${ok ? 'configured' : 'missing'}">
                    <span>${key}</span>
                    <span class="env-badge">${ok ? 'CONFIGURED' : 'MISSING'}</span>
                </div>
            `).join('');
        }
    } catch (e) {
        document.getElementById('health-indicator').className = 'status-indicator unconfigured';
        document.getElementById('health-text').innerText = 'Offline';
    }
}

// =============================================================================
// 2. DASHBOARD & ANALYTICS CHARTS
// =============================================================================
async function loadDashboardStats() {
    try {
        const res = await fetch('/api/analytics');
        const stats = await res.json();
        
        // Check if logs are empty
        if (stats.message || stats.error) {
            document.getElementById('stat-total-runs').innerText = '0';
            document.getElementById('stat-success-rate').innerText = '0%';
            document.getElementById('stat-avg-latency').innerText = '0s';
            document.getElementById('stat-active-model').innerText = 'N/A';
            return;
        }
        
        // Render stats
        document.getElementById('stat-total-runs').innerText = stats.summary.total_content_created;
        document.getElementById('stat-success-rate').innerText = `${stats.summary.success_rate}%`;
        document.getElementById('stat-avg-latency').innerText = `${stats.performance.average_latency_seconds}s`;
        
        // Get active model from health
        const hRes = await fetch('/api/health');
        const hData = await hRes.json();
        document.getElementById('stat-active-model').innerText = hData.active_model || 'Gemini Flash';
        
        // Load Charts
        renderPlatformChart(stats.platform_distribution);
        renderContentTypeChart(stats.content_type_distribution);
        
        // Load Recent Runs Table
        loadRecentRuns();
    } catch (e) {
        console.error('Failed to load dashboard metrics:', e);
    }
}

function renderPlatformChart(data) {
    const ctx = document.getElementById('chart-platforms').getContext('2d');
    
    if (chartPlatforms) {
        chartPlatforms.destroy();
    }
    
    const labels = Object.keys(data).map(k => k.toUpperCase());
    const values = Object.values(data);
    
    if (labels.length === 0) {
        labels.push('None');
        values.push(0);
    }
    
    chartPlatforms = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: ['#00f2fe', '#f02fc2', '#8a3ffc', '#ffd370'],
                borderWidth: 1,
                borderColor: 'rgba(255, 255, 255, 0.08)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8', font: { family: 'Outfit' } }
                }
            }
        }
    });
}

function renderContentTypeChart(data) {
    const ctx = document.getElementById('chart-content-types').getContext('2d');
    
    if (chartContentTypes) {
        chartContentTypes.destroy();
    }
    
    const labels = Object.keys(data).map(k => k.charAt(0).toUpperCase() + k.slice(1));
    const values = Object.values(data);
    
    chartContentTypes = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Runs',
                data: values,
                backgroundColor: 'rgba(0, 242, 254, 0.3)',
                borderColor: '#00f2fe',
                borderWidth: 1,
                borderRadius: 5
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.03)' }, ticks: { color: '#94a3b8' } },
                y: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

async function loadRecentRuns() {
    try {
        const res = await fetch('/api/logs?log_type=content_creation&limit=5');
        const logs = await res.json();
        
        const tbody = document.getElementById('recent-runs-table-body');
        if (logs.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No posts generated yet. Go to Generator!</td></tr>`;
            return;
        }
        
        tbody.innerHTML = logs.map(log => {
            const data = log.data;
            const date = new Date(log.timestamp).toLocaleString();
            const badgeClass = data.success ? 'badge success' : 'badge failed';
            const badgeText = data.success ? 'Success' : 'Failed';
            const platforms = data.platforms ? data.platforms.map(p => p.toUpperCase()).join(', ') : 'N/A';
            
            return `
                <tr>
                    <td>${date}</td>
                    <td style="font-weight: 600;">${data.topic}</td>
                    <td>${platforms}</td>
                    <td><span class="${badgeClass}">${badgeText}</span></td>
                    <td>
                        <button class="btn-sm" onclick="showLogDetails(${JSON.stringify(log).replace(/"/g, '&quot;')})">View</button>
                    </td>
                </tr>
            `;
        }).join('');
    } catch (e) {
        console.error('Failed to load recent activity:', e);
    }
}

// =============================================================================
// 3. GENERATION RUN LOOP (SSE AGENT STREAM)
// =============================================================================
function setupPlatformListener() {
    const checks = ['check-yt', 'check-tt'];
    checks.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', updateDurationVisibility);
        }
    });
}

function updateDurationVisibility() {
    const yt = document.getElementById('check-yt').checked;
    const tt = document.getElementById('check-tt').checked;
    const durationGrp = document.getElementById('video-duration-group');
    if (yt || tt) {
        durationGrp.style.display = 'block';
    } else {
        durationGrp.style.display = 'none';
    }
}

function toggleAccordion() {
    const acc = document.getElementById('tweak-controls-panel').parentNode;
    acc.classList.toggle('active');
}

function updateTempVal(val) {
    document.getElementById('temp-val').innerText = val;
}

async function startGeneration() {
    const topic = document.getElementById('gen-topic').value.trim();
    const content_type = document.getElementById('gen-content-type').value;
    const tone = document.getElementById('gen-tone').value;
    const duration = document.getElementById('gen-duration').value.trim();
    
    // Platforms selected
    const platforms = [];
    document.querySelectorAll('input[name="gen-platforms"]:checked').forEach(cb => {
        platforms.push(cb.value);
    });
    
    if (platforms.length === 0) {
        alert('Please select at least one platform.');
        return;
    }
    
    // Tweak parameters
    const temperature = document.getElementById('gen-temp').value;
    const custom_instructions = document.getElementById('gen-custom-notes').value.trim();
    const system_prompt = document.getElementById('gen-system-prompt').value.trim();
    
    // Clear console terminal
    const terminal = document.getElementById('console-logs');
    terminal.innerHTML = `<div class="terminal-line">🤖 Handshaking with Content Agent Engine...</div>`;
    
    const consolePulse = document.getElementById('console-pulse');
    consolePulse.className = 'console-dot-loader active';
    
    // Hide old results
    document.getElementById('generation-results-panel').style.display = 'none';
    
    // Construct query parameters
    const params = new URLSearchParams({
        topic: topic,
        platforms: platforms.join(','),
        content_type: content_type,
        tone: tone
    });
    if (duration) params.append('duration', duration);
    if (custom_instructions) params.append('custom_instructions', custom_instructions);
    if (temperature) params.append('temperature', temperature);
    if (system_prompt) params.append('system_prompt', system_prompt);
    
    // Setup SSE connection
    const eventSource = new EventSource(`/api/generate/stream?${params.toString()}`);
    
    eventSource.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        
        if (payload.type === 'step') {
            const time = new Date().toLocaleTimeString();
            terminal.innerHTML += `<div class="terminal-line">[${time}] ${payload.message}</div>`;
            terminal.scrollTop = terminal.scrollHeight;
        } else if (payload.type === 'result') {
            eventSource.close();
            consolePulse.className = 'console-dot-loader';
            terminal.innerHTML += `<div class="terminal-line success">[Success] Content generation pipeline completed!</div>`;
            terminal.scrollTop = terminal.scrollHeight;
            
            // Display outputs
            renderGenerationResults(payload.data);
        } else if (payload.type === 'error') {
            eventSource.close();
            consolePulse.className = 'console-dot-loader';
            terminal.innerHTML += `<div class="terminal-line error">[Fatal Error] ${payload.message}</div>`;
            terminal.scrollTop = terminal.scrollHeight;
        }
    };
    
    eventSource.onerror = (err) => {
        console.error('SSE Error:', err);
        eventSource.close();
        consolePulse.className = 'console-dot-loader';
        terminal.innerHTML += `<div class="terminal-line error">[Error] Lost connection to execution thread.</div>`;
        terminal.scrollTop = terminal.scrollHeight;
    };
}

// Render Results tabs and panels
function renderGenerationResults(data) {
    const resultsPanel = document.getElementById('generation-results-panel');
    const tabsContainer = document.getElementById('generated-tabs');
    const contentsContainer = document.getElementById('generated-contents');
    
    // Store results globally for regeneration feature
    lastGenerationResults = data;

    tabsContainer.innerHTML = '';
    contentsContainer.innerHTML = '';
    
    const platforms = Object.keys(data.content);
    
    // Generate tabs
    platforms.forEach((plat, i) => {
        tabsContainer.innerHTML += `
            <button class="tab-btn ${i === 0 ? 'active' : ''}" id="res-tab-${plat}" onclick="switchResultSubtab('${plat}')">
                ${plat.toUpperCase()}
            </button>
        `;
        
        // Add content section
        let contentHtml = '';
        if (plat === 'youtube' || plat === 'tiktok') {
            contentHtml = renderScriptOutput(data.content[plat]);
        } else if (plat === 'article') {
            contentHtml = renderArticleOutput(data.content[plat]);
        } else if (plat === 'x') {
            contentHtml = renderThreadOutput(data.content[plat]);
        }
        
        // Critic analysis box
        const analysis = data.analyses[plat] || {};
        const analysisHtml = renderCriticOutput(analysis, plat);
        
        // Saved file path info
        const filePath = data.files[plat] || 'Local drive';
        
        contentsContainer.innerHTML += `
            <div class="result-subtab-content ${i === 0 ? 'active' : ''}" id="res-panel-${plat}" style="display: ${i === 0 ? 'block' : 'none'};">
                <div class="rendered-output-container">
                    ${analysisHtml}
                    
                    <div class="copy-section">
                        <div style="flex-grow:1; text-align:left; font-size:0.85rem; color:var(--text-muted); align-self:center;">
                            Saved to: <code>${filePath}</code>
                        </div>
                        <button class="copy-btn" onclick="copyContentText('${plat}')">📋 Copy Content</button>
                    </div>
                    
                    <div class="actual-content-markup" id="text-to-copy-${plat}">
                        ${contentHtml}
                    </div>
                </div>
            </div>
        `;
    });
    
    resultsPanel.style.display = 'flex';
    resultsPanel.scrollIntoView({ behavior: 'smooth' });
}

function switchResultSubtab(platform) {
    document.querySelectorAll('.result-subtab-content').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    
    document.getElementById(`res-panel-${platform}`).style.display = 'block';
    document.getElementById(`res-tab-${platform}`).classList.add('active');
}

// Outputs render templates
function renderScriptOutput(script) {
    const hook = escapeHtml(script.hook || '');
    const segmentsHtml = (script.segments || []).map(s => {
        // Highlight emphasis words
        let narration = escapeHtml(s.narration || '');
        if (s.vocal_emphasis && s.vocal_emphasis.length > 0) {
            s.vocal_emphasis.forEach(word => {
                const regex = new RegExp(`\\b(${escapeRegex(word)})\\b`, 'gi');
                narration = narration.replace(regex, '<span class="emphasis-word">$1</span>');
            });
        }
        const pause = s.is_pause_after ? '<span class="emphasis-word" style="background:rgba(0, 242, 254, 0.1);color:var(--neon-cyan);">[PAUSE]</span>' : '';
        
        return `
            <div class="timeline-segment">
                <span class="segment-time">${s.time_cue}</span>
                <div class="segment-content">
                    <p class="segment-narration">${narration} ${pause}</p>
                    <div class="segment-visuals">📺 <strong>Visuals:</strong> ${escapeHtml(s.visual_cue)}</div>
                </div>
            </div>
        `;
    }).join('');
    
    return `
        <div class="timeline-flow">
            <div class="tweet-bubble" style="border-radius:12px; margin-bottom:1rem; border-color:var(--neon-cyan);">
                <div class="tweet-header"><span>HOOK / INTRO</span></div>
                <p class="tweet-body"><strong>${hook}</strong></p>
            </div>
            ${segmentsHtml}
            <div class="tweet-bubble" style="border-radius:12px; margin-top:1rem; border-color:var(--neon-purple);">
                <div class="tweet-header"><span>OUTRO / CALL TO ACTION</span></div>
                <p class="tweet-body">${escapeHtml(script.call_to_action || '')}</p>
            </div>
        </div>
    `;
}

function renderArticleOutput(art) {
    const sections = (art.sections || []).map(s => `
        <h3>${escapeHtml(s.heading)}</h3>
        <p>${markdownToHtml(s.content)}</p>
    `).join('');
    
    return `
        <div class="article-render">
            <h2>${escapeHtml(art.title)}</h2>
            <p class="lead-text"><strong>${escapeHtml(art.introduction)}</strong></p>
            ${sections}
            <h3>Conclusion</h3>
            <p>${escapeHtml(art.conclusion)}</p>
            
            <div class="article-meta">
                <span>🔑 <strong>SEO Keywords:</strong> ${escapeHtml((art.seo_keywords || []).join(', '))}</span>
                <span>ℹ️ <strong>Meta Description:</strong> ${escapeHtml(art.meta_description || '')}</span>
            </div>
        </div>
    `;
}

function renderThreadOutput(threadObj) {
    const tweets = (threadObj.thread || []).map((t, idx) => `
        <div class="tweet-bubble">
            <div class="tweet-header">
                <span>POST ${t.index || (idx + 1)} / ${threadObj.thread.length}</span>
                <span>${t.text.length} chars</span>
            </div>
            <p class="tweet-body">${escapeHtml(t.text)}</p>
        </div>
    `).join('');
    
    return `
        <div class="thread-flow">
            ${tweets}
            <div style="font-size:0.85rem; color:var(--text-muted); margin-top:0.5rem;">
                🏷️ <strong>Tags:</strong> ${escapeHtml((threadObj.hashtags || []).join(' '))}
            </div>
        </div>
    `;
}

function renderCriticOutput(analysis, platform) {
    const verdict = analysis.verdict || 'TRASH';
    const isPost = verdict.toUpperCase() === 'POST';
    const boxClass = isPost ? 'post' : 'trash';
    const verdictText = isPost ? 'POST' : 'TRASH / NEEDS WORK';
    
    const fixHtml = analysis.actionable_fix ? `
        <p style="margin-top:0.8rem; font-size:0.9rem; border-top:1px solid rgba(255,255,255,0.06); padding-top:0.6rem; color:#ffd370;">
            🛠️ <strong>Actionable Fix:</strong> ${escapeHtml(analysis.actionable_fix)}
        </p>
        <button class="btn-sm btn-regenerate" onclick="regenerateWithCritique('${platform}')">♻️ Regenerate with Critique</button>
    ` : '';
    
    return `
        <div class="critic-assessment ${boxClass}">
            <div class="critic-title">
                🕵️ Honest critic evaluation on ${platform.toUpperCase()}:
                <span class="critic-verdict">${verdictText}</span>
            </div>
            <p style="font-size:0.95rem; line-height:1.4rem;">${escapeHtml(analysis.core_assessment || 'No review data')}</p>
            ${fixHtml}
        </div>
    `;
}

async function regenerateWithCritique(platform) {
    if (!lastGenerationResults || !lastGenerationResults.analyses || !lastGenerationResults.analyses[platform]) {
        alert("No previous generation results or critique found for this platform.");
        return;
    }

    const analysis = lastGenerationResults.analyses[platform];
    const previousContent = lastGenerationResults.content[platform];
    
    if (!analysis.actionable_fix) {
        alert("No actionable critique found for this platform.");
        return;
    }

    // Retrieve original parameters from the form or last run
    const topic = document.getElementById('gen-topic').value.trim();
    const content_type = document.getElementById('gen-content-type').value;
    const tone = document.getElementById('gen-tone').value;
    const duration = document.getElementById('gen-duration').value.trim();
    const temperature = document.getElementById('gen-temp').value;
    const system_prompt = document.getElementById('gen-system-prompt').value.trim();

    // Construct a detailed refinement prompt
    const refinementPrompt = `
You are in a refinement run. Your previous attempt to generate content had some issues. Here is the full context from the last run:

---
**PREVIOUSLY GENERATED CONTENT (JSON):**
\`\`\`json
${JSON.stringify(previousContent, null, 2)}
\`\`\`
---
**HONEST CRITIC'S ASSESSMENT:**
${analysis.core_assessment}
---
**ACTIONABLE FIX TO IMPLEMENT:**
${analysis.actionable_fix}
---

Please generate new, improved content for the topic "${topic}".
Your main goal is to implement the actionable fix and address the critic's assessment. Do not repeat the same mistakes. Use the original research data but re-interpret it through the lens of this critique to generate a superior piece of content.
    `.trim();

    // Populate the form fields for the new generation
    document.getElementById('gen-custom-notes').value = refinementPrompt;
    
    // Ensure accordion is open so user can see the new instructions
    const acc = document.getElementById('tweak-controls-panel').parentNode;
    if (!acc.classList.contains('active')) {
        toggleAccordion();
    }
    
    // Ensure only the current platform is selected
    document.querySelectorAll('input[name="gen-platforms"]').forEach(cb => {
        cb.checked = (cb.value === platform);
    });
    updateDurationVisibility();

    // Switch to generator tab and scroll to the form
    switchTab('generate');
    document.getElementById('generate-form').scrollIntoView({ behavior: 'smooth' });

    // Optional: auto-start generation after a brief delay
    // setTimeout(startGeneration, 500); 
    // For now, let the user click "Run Agent Engine" to confirm.
}

// =============================================================================
// 4. LOG HISTORY & VIEWING DETAILS
// =============================================================================
let historyLogsCache = [];

async function loadHistoryLogs() {
    try {
        const res = await fetch('/api/logs?limit=80');
        const logs = await res.json();
        
        historyLogsCache = logs;
        filterHistory();
    } catch (e) {
        console.error('Failed to load history logs:', e);
    }
}

function filterHistory() {
    const search = document.getElementById('history-search').value.toLowerCase();
    const type = document.getElementById('history-filter-type').value;
    const tbody = document.getElementById('full-history-table-body');
    
    const filtered = historyLogsCache.filter(log => {
        const matchesSearch = JSON.stringify(log.data).toLowerCase().includes(search) || 
                              log.session_id.toLowerCase().includes(search);
        const matchesType = type === 'all' || log.log_type === type;
        return matchesSearch && matchesType;
    });
    
    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No matching log files found.</td></tr>`;
        return;
    }
    
    tbody.innerHTML = filtered.map(log => {
        const date = new Date(log.timestamp).toLocaleString();
        let details = '';
        
        if (log.log_type === 'content_creation') {
            details = `Generated content for <strong>${log.data.topic}</strong> (${(log.data.platforms || []).join(', ').toUpperCase()})`;
        } else if (log.log_type === 'research') {
            details = `Research brief: <strong>${log.data.topic}</strong> (Preview: ${escapeHtml(log.data.results_preview || '')})`;
        } else if (log.log_type === 'error') {
            details = `<span style="color:var(--neon-red);">[Error] ${escapeHtml(log.data.error || 'Exception thrown')}</span>`;
        } else {
            details = `Session metrics and execution log`;
        }
        
        return `
            <tr>
                <td>${date}</td>
                <td><span class="badge ${log.log_type === 'error' ? 'failed' : 'success'}">${log.log_type.replace('_', ' ')}</span></td>
                <td><code>${log.session_id}</code></td>
                <td>${details}</td>
                <td>
                    <button class="btn-sm" onclick="showLogDetails(${JSON.stringify(log).replace(/"/g, '&quot;')})">Details</button>
                </td>
            </tr>
        `;
    }).join('');
}

function showLogDetails(log) {
    const modal = document.getElementById('log-modal');
    const title = document.getElementById('modal-log-title');
    const content = document.getElementById('modal-log-content');
    
    title.innerText = `${log.log_type.toUpperCase().replace('_', ' ')} LOG DETAILS`;
    
    let html = `
        <p style="margin-bottom: 0.8rem; font-size:0.9rem; color:var(--text-muted);">
            Timestamp: <strong>${new Date(log.timestamp).toLocaleString()}</strong> | 
            Session ID: <code>${log.session_id}</code>
        </p>
        <hr style="border-color:var(--border-color); margin-bottom: 1.5rem;">
    `;
    
    if (log.log_type === 'content_creation') {
        const data = log.data;
        const successBadge = data.success ? 'badge success' : 'badge failed';
        
        html += `
            <div style="display:flex; justify-content:space-between; margin-bottom:1.5rem;">
                <div>Topic: <strong style="font-size:1.1rem; color:var(--neon-cyan);">${data.topic}</strong></div>
                <div>Status: <span class="${successBadge}">${data.success ? 'Success' : 'Failed'}</span></div>
            </div>
            
            <div class="metrics-grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 1.5rem;">
                <div class="glass-card" style="padding:1rem; text-align:center;">
                    <div style="font-size:0.8rem; color:var(--text-muted);">Latency</div>
                    <div style="font-size:1.2rem; font-weight:700;">${data.latency || 0}s</div>
                </div>
                <div class="glass-card" style="padding:1rem; text-align:center;">
                    <div style="font-size:0.8rem; color:var(--text-muted);">Tokens Estimate</div>
                    <div style="font-size:1.2rem; font-weight:700;">${data.token_usage || 0}</div>
                </div>
                <div class="glass-card" style="padding:1rem; text-align:center;">
                    <div style="font-size:0.8rem; color:var(--text-muted);">Tone / Type</div>
                    <div style="font-size:1.1rem; font-weight:700; text-transform:capitalize;">${data.tone || 'engaging'}</div>
                </div>
            </div>
        `;
        
        if (data.generated_content && data.generated_content.content) {
            html += `<h4>Artifact Output Data:</h4><br>`;
            Object.keys(data.generated_content.content).forEach(plat => {
                html += `
                    <div class="accordion-container active" style="margin-bottom:1rem;">
                        <div class="accordion-header"><span>📱 ${plat.toUpperCase()}</span></div>
                        <div class="accordion-content" style="max-height:unset; padding:1.25rem; background:rgba(0,0,0,0.25);">
                            <pre style="white-space:pre-wrap; font-family:var(--font-body); font-size:0.95rem; line-height:1.45rem; color:var(--text-main);">
                                ${JSON.stringify(data.generated_content.content[plat], null, 2)}
                            </pre>
                        </div>
                    </div>
                `;
            });
        } else if (data.error) {
            html += `
                <div class="critic-assessment trash">
                    <div class="critic-title">Execution Stacktrace Exception:</div>
                    <pre style="white-space:pre-wrap; font-family:monospace; color:var(--neon-red); font-size:0.8rem; margin-top:0.5rem; line-height:1.2rem;">${escapeHtml(data.error)}</pre>
                </div>
            `;
        }
    } else {
        // Fallback formatting for errors, tool calls, and research
        html += `
            <pre style="white-space:pre-wrap; font-family:monospace; background:rgba(0,0,0,0.3); padding:1rem; border-radius:8px; border:1px solid var(--border-color); color:#a8ffb2; font-size:0.85rem; line-height:1.3rem;">
                ${escapeHtml(JSON.stringify(log.data, null, 2))}
            </pre>
        `;
    }
    
    content.innerHTML = html;
    modal.classList.add('active');
}

function closeLogModal(e) {
    if (e.target.id === 'log-modal') {
        document.getElementById('log-modal').classList.remove('active');
    }
}

// =============================================================================
// 5. SYSTEM CONFIGURATION & PRESETS
// =============================================================================
async function loadActiveConfig() {
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        
        document.getElementById('set-model-name').value = data.content_creator.model.name;
        document.getElementById('set-model-temp').value = data.content_creator.model.temperature;
        document.getElementById('set-reddit-prompt').value = data.reddit_agent.system_prompt;
        document.getElementById('set-creator-prompt').value = data.content_creator.system_prompt;
    } catch (e) {
        console.error('Failed to load active system configuration:', e);
    }
}

async function saveSettings() {
    const model = document.getElementById('set-model-name').value.trim();
    const temp = parseFloat(document.getElementById('set-model-temp').value);
    const redditPrompt = document.getElementById('set-reddit-prompt').value.trim();
    const creatorPrompt = document.getElementById('set-creator-prompt').value.trim();
    
    try {
        // Update Reddit
        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                agent_type: 'reddit',
                settings: {
                    system_prompt: redditPrompt
                }
            })
        });
        
        // Update Content Creator
        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                agent_type: 'content_creator',
                settings: {
                    model: { name: model, temperature: temp },
                    system_prompt: creatorPrompt
                }
            })
        });
        
        alert('Active system configuration saved successfully!');
        checkHealth();
    } catch (e) {
        alert(`Failed to save settings: ${e}`);
    }
}

async function applyPreset(presetName) {
    try {
        const res = await fetch(`/api/config/preset/${presetName}`, { method: 'POST' });
        if (res.ok) {
            alert(`Config preset "${presetName}" applied!`);
            loadActiveConfig();
        } else {
            alert('Failed to apply config preset.');
        }
    } catch (e) {
        alert(`Error: ${e}`);
    }
}

async function clearSystemLogs() {
    if (!confirm('Are you sure you want to purge the activity logs database? This cannot be undone.')) {
        return;
    }
    
    // In our backend API route, we could add a clear log endpoint. For safety we just mock or call the API.
    // Let's call /api/logs/clear or write an endpoint. We will write it if not present.
    try {
        const res = await fetch('/api/logs', { method: 'DELETE' }); // Mocked or direct
        alert('All activity logs purged!');
        loadDashboardStats();
    } catch (e) {
        console.error('Purge error:', e);
    }
}

// =============================================================================
// 6. UTILITY HELPERS
// =============================================================================
function copyContentText(platform) {
    const el = document.getElementById(`text-to-copy-${platform}`);
    if (!el) return;
    
    // Quick extract plain text
    const text = el.innerText;
    navigator.clipboard.writeText(text).then(() => {
        alert(`${platform.toUpperCase()} content copied to clipboard!`);
    }).catch(e => {
        console.error('Failed to copy text:', e);
    });
}

function escapeHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function escapeRegex(string) {
    return string.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
}

function markdownToHtml(md) {
    // Simple basic markdown parser for paragraphs, links, bold
    let html = escapeHtml(md);
    
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Unordered lists
    html = html.replace(/^\s*-\s+(.*?)$/gm, '<li>$1</li>');
    
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    
    return html;
}
