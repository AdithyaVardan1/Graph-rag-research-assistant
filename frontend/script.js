/**
 * GraphRAG Research Assistant - Frontend Script
 * 
 * Enhanced with:
 * - Entity graph visualization (GraphRAG)
 * - Toggle between paper and entity views
 * - Stats display
 * - Streaming support (optional)
 */

const API_URL = '';

// DOM Elements
const buildBtn = document.getElementById('build-btn');
const topicInput = document.getElementById('topic-input');
const graphContainer = document.getElementById('graph-container');
const messageBox = document.getElementById('message-box');
const summarizeBtn = document.getElementById('summarize-btn');
const hypothesizeBtn = document.getElementById('hypothesize-btn');
const themesBtn = document.getElementById('themes-btn');
const chatInput = document.getElementById('chat-input');
const chatBtn = document.getElementById('chat-btn');
const selectionInfo = document.getElementById('selection-info');
const outputContainer = document.getElementById('output-container');
const outputBox = document.getElementById('output-box');
const outputTitle = document.getElementById('output-title');
const closeOutput = document.getElementById('close-output');
const graphTooltip = document.getElementById('graph-tooltip');
const graphLegend = document.getElementById('graph-legend');
const entityLegend = document.getElementById('entity-legend');
const paperCountSlider = document.getElementById('paper-count');
const paperCountValue = document.getElementById('paper-count-value');
const paperViewBtn = document.getElementById('paper-view-btn');
const entityViewBtn = document.getElementById('entity-view-btn');
const statsPanel = document.getElementById('stats-panel');
const statPapers = document.getElementById('stat-papers');
const statEntities = document.getElementById('stat-entities');
const statRelationships = document.getElementById('stat-relationships');

// Update slider display value
paperCountSlider.addEventListener('input', () => {
    paperCountValue.textContent = paperCountSlider.value;
});

// Graph Settings
const MIN_NODE_SIZE = 20;
const MAX_NODE_SIZE = 50;

// Entity type colors
const ENTITY_COLORS = {
    'CONCEPT': '#8B5CF6',
    'METHOD': '#10B981',
    'ALGORITHM': '#3B82F6',
    'DATASET': '#F59E0B',
    'METRIC': '#EC4899'
};

// State
let paperCy = null;  // Paper graph
let entityCy = null; // Entity graph
let currentView = 'papers';
let currentSessionId = null;
let graphData = null;

// Event Listeners
buildBtn.addEventListener('click', handleBuildGraph);
summarizeBtn.addEventListener('click', () => handleAgentAction('summarize', 'Summary'));
hypothesizeBtn.addEventListener('click', () => handleAgentAction('hypothesize', 'Research Hypothesis'));
themesBtn.addEventListener('click', () => handleAgentAction('extract_themes', 'Key Themes'));
chatBtn.addEventListener('click', () => handleAgentAction('chat', 'Answer'));
closeOutput.addEventListener('click', () => { outputContainer.style.display = 'none'; });

// View toggle
paperViewBtn.addEventListener('click', () => switchView('papers'));
entityViewBtn.addEventListener('click', () => switchView('entities'));

// Enter key handlers
topicInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleBuildGraph();
});
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !chatBtn.disabled) handleAgentAction('chat', 'Answer');
});

/**
 * Switch between paper and entity graph views
 */
function switchView(view) {
    currentView = view;

    paperViewBtn.classList.toggle('active', view === 'papers');
    entityViewBtn.classList.toggle('active', view === 'entities');

    if (view === 'papers') {
        graphLegend.style.display = 'block';
        entityLegend.style.display = 'none';
        if (graphData) {
            renderPaperGraph(graphData);
        }
    } else {
        graphLegend.style.display = 'none';
        entityLegend.style.display = 'block';
        if (graphData && graphData.knowledge_graph) {
            renderEntityGraph(graphData.knowledge_graph);
        }
    }
}

/**
 * Set loading state on a button
 */
function setButtonLoading(button, isLoading) {
    const textSpan = button.querySelector('.btn-text');
    const loadingSpan = button.querySelector('.btn-loading');

    if (isLoading) {
        button.disabled = true;
        if (textSpan) textSpan.style.display = 'none';
        if (loadingSpan) loadingSpan.style.display = 'inline-flex';
    } else {
        if (textSpan) textSpan.style.display = 'inline';
        if (loadingSpan) loadingSpan.style.display = 'none';
    }
}

/**
 * Build the knowledge graph
 */
async function handleBuildGraph() {
    const topic = topicInput.value.trim();
    if (!topic) {
        showMessage('Please enter a research topic.', 'error');
        topicInput.focus();
        return;
    }

    if (topic.length < 2) {
        showMessage('Topic must be at least 2 characters.', 'error');
        return;
    }

    showMessage('Fetching papers and building knowledge graph...', 'loading');
    setButtonLoading(buildBtn, true);
    outputContainer.style.display = 'none';
    statsPanel.style.display = 'none';

    try {
        const response = await fetch(`${API_URL}/api/build_graph`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic, max_results: parseInt(paperCountSlider.value) })
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(error.detail || `Server error: ${response.status}`);
        }

        graphData = await response.json();
        currentSessionId = graphData.session_id;

        if (!graphData.nodes || graphData.nodes.length === 0) {
            showMessage('No papers found for this topic. Try a different search term.', 'warning');
        } else {
            const entityCount = graphData.knowledge_graph?.nodes?.length || 0;
            showMessage(`Found ${graphData.nodes.length} papers, extracted ${entityCount} entities. Click nodes to select.`, 'success');

            // Update stats
            updateStats(graphData);

            // Render based on current view
            if (currentView === 'papers') {
                renderPaperGraph(graphData);
            } else {
                renderEntityGraph(graphData.knowledge_graph);
            }
        }
    } catch (error) {
        console.error("Build graph error:", error);
        showMessage(`Error: ${error.message}`, 'error');
    } finally {
        setButtonLoading(buildBtn, false);
        buildBtn.disabled = false;
    }
}

/**
 * Update stats panel
 */
function updateStats(data) {
    statPapers.textContent = data.nodes?.length || 0;
    statEntities.textContent = data.knowledge_graph?.nodes?.length || 0;
    statRelationships.textContent = data.knowledge_graph?.edges?.length || 0;
    statsPanel.style.display = 'block';
}

/**
 * Execute an agent action
 */
async function handleAgentAction(action, title) {
    const selectedNodes = paperCy?.$('node:selected');
    if (!selectedNodes || selectedNodes.length === 0) {
        showMessage('Please select papers first by clicking on nodes.', 'warning');
        return;
    }

    const selectedPapers = selectedNodes.map(node => node.data('fullData'));
    const query = chatInput.value.trim();

    if (action === 'chat' && !query) {
        showMessage('Please enter a question.', 'warning');
        chatInput.focus();
        return;
    }

    // Show output panel with loading state
    outputContainer.style.display = 'block';
    outputTitle.textContent = title;
    outputBox.textContent = 'Processing your request...';
    outputBox.classList.remove('streaming');

    // Set loading state on the relevant button
    const buttonMap = {
        'summarize': summarizeBtn,
        'hypothesize': hypothesizeBtn,
        'extract_themes': themesBtn,
        'chat': chatBtn
    };
    const activeBtn = buttonMap[action];
    setButtonLoading(activeBtn, true);

    try {
        const response = await fetch(`${API_URL}/api/agent_action`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action,
                selected_papers: selectedPapers,
                query,
                session_id: currentSessionId,
                use_hybrid: true,
                use_reranking: true,
                stream: false
            })
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(error.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        outputBox.textContent = data.response;

        if (action === 'chat') {
            chatInput.value = '';
        }
    } catch (error) {
        console.error("Agent action error:", error);
        outputBox.textContent = `Error: ${error.message}`;
    } finally {
        setButtonLoading(activeBtn, false);
        updateButtonStates();
    }
}

/**
 * Calculate node degree (number of edges) for sizing
 */
function calculateNodeDegrees(edges, threshold = 0) {
    const degrees = {};
    edges.forEach(edge => {
        const weight = edge.weight || 1;
        if (weight > threshold) {
            degrees[edge.source] = (degrees[edge.source] || 0) + 1;
            degrees[edge.target] = (degrees[edge.target] || 0) + 1;
        }
    });
    return degrees;
}

/**
 * Map a value from one range to another
 */
function mapRange(value, inMin, inMax, outMin, outMax) {
    if (inMax === inMin) return (outMin + outMax) / 2;
    return ((value - inMin) / (inMax - inMin)) * (outMax - outMin) + outMin;
}

/**
 * Render the paper similarity graph
 */
function renderPaperGraph(data) {
    if (typeof cytoscape === 'undefined') {
        showMessage('Error: Visualization library not loaded. Please refresh.', 'error');
        return;
    }

    try {
        messageBox.style.display = 'none';

        // Calculate node degrees for sizing
        const degrees = calculateNodeDegrees(data.edges, 0);
        const maxDegree = Math.max(...Object.values(degrees), 1);
        const minDegree = Math.min(...Object.values(degrees), 0);

        // Build elements with computed sizes
        const elements = data.nodes.map(node => {
            const degree = degrees[node.id] || 0;
            const size = mapRange(degree, minDegree, maxDegree, MIN_NODE_SIZE, MAX_NODE_SIZE);
            return {
                group: 'nodes',
                data: {
                    id: node.id,
                    label: node.title,
                    fullData: node,
                    degree: degree,
                    nodeSize: size
                }
            };
        });

        // Add edges
        data.edges.forEach(edge => {
            elements.push({
                group: 'edges',
                data: {
                    source: edge.source,
                    target: edge.target,
                    weight: edge.weight,
                    lineWidth: mapRange(edge.weight, 0.3, 1, 1, 5)
                }
            });
        });

        // Destroy existing graph if any
        if (paperCy) {
            paperCy.destroy();
        }

        paperCy = cytoscape({
            container: graphContainer,
            elements,
            selectionType: 'additive',

            style: [
                {
                    selector: 'node',
                    style: {
                        'background-color': '#4F46E5',
                        'width': 'data(nodeSize)',
                        'height': 'data(nodeSize)',
                        'border-color': '#fff',
                        'border-width': 2,
                        'cursor': 'pointer',
                        'background-opacity': 0.9
                    }
                },
                {
                    selector: 'node[label]',
                    style: {
                        'label': 'data(label)',
                        'font-size': '10px',
                        'color': '#374151',
                        'text-valign': 'bottom',
                        'text-halign': 'center',
                        'text-margin-y': '6px',
                        'text-wrap': 'ellipsis',
                        'text-max-width': '120px',
                        'font-weight': '500',
                        'text-background-color': '#fff',
                        'text-background-opacity': 0.7,
                        'text-background-padding': '2px'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 'data(lineWidth)',
                        'line-color': '#94A3B8',
                        'curve-style': 'bezier',
                        'opacity': 'mapData(weight, 0.3, 1, 0.3, 0.8)'
                    }
                },
                {
                    selector: 'node:selected',
                    style: {
                        'background-color': '#F59E0B',
                        'border-color': '#D97706',
                        'border-width': 3,
                        'z-index': 999
                    }
                },
                {
                    selector: 'node:active',
                    style: {
                        'overlay-opacity': 0.1
                    }
                }
            ],
            layout: {
                name: 'cose',
                idealEdgeLength: 180,
                nodeRepulsion: 500000,
                edgeElasticity: 100,
                gravity: 0.25,
                fit: true,
                padding: 60,
                animate: true,
                animationDuration: 1200,
                randomize: false
            },
            minZoom: 0.2,
            maxZoom: 3.0,
            wheelSensitivity: 0.1
        });

        // Update button states when selection changes
        paperCy.on('select unselect', 'node', updateButtonStates);

        // Node hover - show tooltip with paper details
        paperCy.on('mouseover', 'node', function (evt) {
            const node = evt.target;
            const data = node.data('fullData');
            const degree = node.data('degree');

            const title = data.title || 'Unknown';
            const authors = data.authors ? data.authors.slice(0, 3).join(', ') : 'Unknown';
            const year = data.published ? data.published.substring(0, 4) : 'N/A';

            graphTooltip.innerHTML = `
                <div class="tooltip-title">${title}</div>
                <div class="tooltip-meta">
                    ${authors}${data.authors && data.authors.length > 3 ? ' et al.' : ''}<br>
                    Year: ${year} • Connections: ${degree}
                </div>
            `;
            graphTooltip.style.display = 'block';
        });

        paperCy.on('mousemove', 'node', function (evt) {
            const pos = evt.renderedPosition;
            graphTooltip.style.left = (pos.x + 20) + 'px';
            graphTooltip.style.top = (pos.y + 20) + 'px';
        });

        paperCy.on('mouseout', 'node', function () {
            graphTooltip.style.display = 'none';
        });

        // Show legend
        graphLegend.style.display = 'block';

        console.log(`Paper graph rendered: ${data.nodes.length} nodes, ${data.edges.length} edges`);

    } catch (error) {
        console.error("Graph render error:", error);
        showMessage('Error displaying the graph. Please try again.', 'error');
    }
}

/**
 * Render the entity knowledge graph (GraphRAG)
 */
function renderEntityGraph(kgData) {
    if (typeof cytoscape === 'undefined' || !kgData) {
        showMessage('Error: Visualization library not loaded or no entity data.', 'error');
        return;
    }

    if (!kgData.nodes || kgData.nodes.length === 0) {
        showMessage('No entities extracted yet. Build a knowledge graph first.', 'warning');
        return;
    }

    try {
        messageBox.style.display = 'none';

        // Calculate degrees for sizing
        const degrees = calculateNodeDegrees(kgData.edges, 0);
        const maxDegree = Math.max(...Object.values(degrees), 1);

        // Build elements
        const elements = kgData.nodes.map(node => {
            const degree = degrees[node.id] || 0;
            const size = mapRange(degree, 0, maxDegree, 15, 40);
            const color = ENTITY_COLORS[node.type] || '#6B7280';

            return {
                group: 'nodes',
                data: {
                    id: node.id,
                    label: node.label,
                    type: node.type,
                    paperCount: node.paper_count || 0,
                    nodeSize: size,
                    nodeColor: color
                }
            };
        });

        // Add edges with labels
        kgData.edges.forEach(edge => {
            elements.push({
                group: 'edges',
                data: {
                    source: edge.source,
                    target: edge.target,
                    label: edge.label || ''
                }
            });
        });

        // Destroy existing entity graph if any
        if (entityCy) {
            entityCy.destroy();
        }

        entityCy = cytoscape({
            container: graphContainer,
            elements,

            style: [
                {
                    selector: 'node',
                    style: {
                        'background-color': 'data(nodeColor)',
                        'width': 'data(nodeSize)',
                        'height': 'data(nodeSize)',
                        'label': 'data(label)',
                        'font-size': '9px',
                        'color': '#374151',
                        'text-valign': 'bottom',
                        'text-halign': 'center',
                        'text-margin-y': '4px',
                        'text-wrap': 'ellipsis',
                        'text-max-width': '100px',
                        'border-color': '#fff',
                        'border-width': 2,
                        'cursor': 'pointer'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 1.5,
                        'line-color': '#94A3B8',
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'triangle',
                        'target-arrow-color': '#94A3B8',
                        'arrow-scale': 0.8,
                        'label': 'data(label)',
                        'font-size': '8px',
                        'text-rotation': 'autorotate',
                        'color': '#6B7280'
                    }
                },
                {
                    selector: 'node:selected',
                    style: {
                        'border-color': '#F59E0B',
                        'border-width': 3
                    }
                }
            ],
            layout: {
                name: 'cose',
                idealEdgeLength: 100,
                nodeRepulsion: 200000,
                edgeElasticity: 50,
                gravity: 0.5,
                fit: true,
                padding: 40,
                animate: true,
                animationDuration: 1000
            },
            minZoom: 0.2,
            maxZoom: 3.0,
            wheelSensitivity: 0.1
        });

        // Entity hover
        entityCy.on('mouseover', 'node', function (evt) {
            const node = evt.target;
            graphTooltip.innerHTML = `
                <div class="tooltip-title">${node.data('label')}</div>
                <div class="tooltip-meta">
                    Type: ${node.data('type')}<br>
                    Papers: ${node.data('paperCount')}
                </div>
            `;
            graphTooltip.style.display = 'block';
        });

        entityCy.on('mousemove', 'node', function (evt) {
            const pos = evt.renderedPosition;
            graphTooltip.style.left = (pos.x + 20) + 'px';
            graphTooltip.style.top = (pos.y + 20) + 'px';
        });

        entityCy.on('mouseout', 'node', function () {
            graphTooltip.style.display = 'none';
        });

        entityLegend.style.display = 'block';

        console.log(`Entity graph rendered: ${kgData.nodes.length} entities, ${kgData.edges.length} relationships`);

    } catch (error) {
        console.error("Entity graph render error:", error);
        showMessage('Error displaying entity graph.', 'error');
    }
}

/**
 * Update button states based on selection
 */
function updateButtonStates() {
    const selectedCount = paperCy?.$('node:selected').length || 0;
    const isDisabled = selectedCount === 0;

    summarizeBtn.disabled = isDisabled;
    hypothesizeBtn.disabled = isDisabled;
    themesBtn.disabled = isDisabled;
    chatInput.disabled = isDisabled;
    chatBtn.disabled = isDisabled;

    if (isDisabled) {
        selectionInfo.textContent = 'Click nodes to select papers.';
    } else {
        selectionInfo.textContent = `${selectedCount} paper${selectedCount > 1 ? 's' : ''} selected`;
    }
}

/**
 * Show a message in the graph area
 */
function showMessage(text, type = 'info') {
    messageBox.textContent = text;
    messageBox.style.display = 'flex';
    messageBox.className = `message-${type}`;
}